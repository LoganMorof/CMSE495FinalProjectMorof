from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
DATA_API_BASE_URL = "https://data-api.polymarket.com"
HISTORY_CACHE_DIR = "data/raw/history"


# --------------------------------------
# Shared helpers
# --------------------------------------


def _extract_token_ids(tokens_field) -> List[str]:
    """
    Normalize clobTokenIds into a list of clean token id strings.
    Handles lists, nested lists, and stringified lists like '["id1","id2"]' or 'id1, id2'.
    """
    tokens: List[str] = []

    def clean(s: str) -> str:
        return s.strip().strip('[]"\' ').replace(" ", "")

    if isinstance(tokens_field, str):
        stripped = tokens_field.strip().strip("[]")
        parts = stripped.split(",")
        tokens.extend([clean(p) for p in parts if clean(p)])
    elif isinstance(tokens_field, (list, tuple)):
        for item in tokens_field:
            if isinstance(item, str):
                tokens.append(clean(item))
            elif isinstance(item, (list, tuple)) and item:
                inner = item[0]
                if isinstance(inner, str):
                    tokens.append(clean(inner))

    tokens = [t for t in tokens if t]
    return tokens


def _extract_token_id(tokens_field) -> Optional[str]:
    ids = _extract_token_ids(tokens_field)
    return ids[0] if ids else None


def _request_json(
    method: str,
    url: str,
    *,
    params: Optional[Dict] = None,
    json_body: Optional[object] = None,
    timeout: int = 20,
    max_retries: int = 3,
    backoff_s: float = 0.2,
) -> Optional[object]:
    """HTTP request with simple retries and backoff; returns parsed JSON or None."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json_body,
                timeout=timeout,
            )
            if resp.status_code == 429:
                time.sleep(backoff_s * attempt)
                continue
            if resp.status_code >= 400:
                print(f"Warning: {method} {url} -> {resp.status_code} ({resp.text[:200]})")
                time.sleep(backoff_s * attempt)
                continue
            return resp.json()
        except Exception as exc:
            print(f"Warning: {method} {url} failed (attempt {attempt}): {exc}")
            time.sleep(backoff_s * attempt)
    return None


# --------------------------------------
# Gamma markets
# --------------------------------------


def fetch_markets_page(
    offset: int = 0,
    limit: int = 100,
    closed: bool = True,
    timeout: int = 30,
    extra_params: Optional[Dict] = None,
) -> List[Dict]:
    """Fetch a single page of markets from Gamma."""
    params = {"offset": offset, "limit": limit, "closed": str(closed).lower()}
    if extra_params:
        params.update(extra_params)
    url = f"{GAMMA_BASE_URL}/markets"
    data = _request_json("GET", url, params=params, timeout=timeout)
    if isinstance(data, list):
        return data
    print("Warning: unexpected markets payload (not a list).")
    return []


def fetch_resolved_markets(
    max_markets: int = 500,
    page_size: int = 100,
    sleep_s: float = 0.2,
    start_offset: int = 0,
    timeout: int = 30,
    extra_params: Optional[Dict] = None,
) -> List[Dict]:
    """
    Page through resolved markets from Gamma until max_markets or empty page.
    Defaults to ordering by volume24hrClob desc and enableOrderBook=true to prefer liquid CLOB markets.
    """
    if extra_params is None:
        extra_params = {
            "enableOrderBook": "true",
            "order": "volume24hrClob",
            "ascending": "false",
        }
    markets: List[Dict] = []
    offset = start_offset
    while len(markets) < max_markets:
        page = fetch_markets_page(
            offset=offset, limit=page_size, closed=True, timeout=timeout, extra_params=extra_params
        )
        if not page:
            break
        markets.extend(page)
        offset += page_size
        if sleep_s > 0:
            time.sleep(sleep_s)
    return markets[:max_markets]


def markets_to_dataframe(markets: List[Dict]) -> pd.DataFrame:
    if not markets:
        return pd.DataFrame()
    return pd.json_normalize(markets)


# --------------------------------------
# Price history
# --------------------------------------


def _cache_path(token_id: str) -> str:
    return os.path.join(HISTORY_CACHE_DIR, f"{token_id}.json")


def fetch_price_history(
    token_id: str,
    interval: str = "max",
    fidelity: int = 60,
    timeout: int = 10,
    sleep_s: float = 0.2,
    max_retries: int = 3,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
) -> Dict:
    """
    Fetch price history for a token from CLOB with caching and gentle retries.
    Supports interval-based or explicit start/end timestamp queries. Returns the parsed JSON (dict).
    """
    os.makedirs(HISTORY_CACHE_DIR, exist_ok=True)
    cache_file = _cache_path(token_id)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, dict) and "history" in cached:
                return cached
        except Exception as exc:
            print(f"Warning: failed to read cache for {token_id}: {exc}")

    url = f"{CLOB_BASE_URL}/prices-history"
    params = {"market": token_id, "fidelity": fidelity}
    if start_ts is not None or end_ts is not None:
        if start_ts is not None:
            params["startTs"] = start_ts
        if end_ts is not None:
            params["endTs"] = end_ts
    else:
        params["interval"] = interval

    data = _request_json("GET", url, params=params, timeout=timeout, max_retries=max_retries, backoff_s=sleep_s)
    if not isinstance(data, dict):
        data = {"history": []}

    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as exc:
        print(f"Warning: failed to write cache for {token_id}: {exc}")
    return data


def price_history_to_dataframe(history: object) -> pd.DataFrame:
    """
    Convert price history payload to DataFrame with timestamp (UTC) and price.
    Accepts either dict with key 'history' or list of dicts.
    """
    records: List[Dict]
    if isinstance(history, dict):
        records = history.get("history", []) if isinstance(history.get("history"), list) else []
    elif isinstance(history, list):
        records = history
    else:
        records = []
    if not records:
        return pd.DataFrame(columns=["timestamp", "price"])
    df = pd.DataFrame(records)
    rename_map = {}
    if "t" in df.columns:
        rename_map["t"] = "timestamp"
    if "p" in df.columns:
        rename_map["p"] = "price"
    if rename_map:
        df = df.rename(columns=rename_map)
    cols = [
        c
        for c in ["timestamp", "price", "best_bid", "best_ask", "bid", "ask", "bid_depth", "ask_depth"]
        if c in df.columns
    ]
    df = df[cols] if cols else df
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def find_first_history(
    markets: List[Dict],
    sleep_s: float = 0.2,
    interval: str = "max",
    fidelity: int = 60,
    max_tokens: int = 100,
) -> Tuple[Optional[str], List[Dict]]:
    checked = 0
    for m in markets:
        for tok in _extract_token_ids(m.get("clobTokenIds")):
            if checked >= max_tokens:
                return None, []
            checked += 1
            hist = fetch_price_history(tok, interval=interval, fidelity=fidelity, sleep_s=sleep_s)
            payload = hist.get("history") if isinstance(hist, dict) else None
            if payload:
                return tok, payload
            else:
                print(f"No history for token {tok}; trying next...")
    return None, []


# --------------------------------------
# CLOB orderbook / price endpoints
# --------------------------------------


def fetch_orderbooks(token_ids: List[str]) -> Dict[str, dict]:
    """POST /books to fetch orderbooks for a list of token_ids."""
    url = f"{CLOB_BASE_URL}/books"
    body = [{"token_id": tid} for tid in token_ids]
    data = _request_json("POST", url, json_body=body, timeout=10)
    if not isinstance(data, list):
        return {}
    result: Dict[str, dict] = {}
    for book in data:
        tid = book.get("asset_id") or book.get("token_id")
        if tid:
            result[str(tid)] = book
    return result


def fetch_spreads(token_ids: List[str]) -> Dict[str, float]:
    """POST /spreads to fetch spreads for token_ids."""
    url = f"{CLOB_BASE_URL}/spreads"
    body = [{"token_id": tid} for tid in token_ids]
    data = _request_json("POST", url, json_body=body, timeout=10)
    spreads: Dict[str, float] = {}
    if isinstance(data, dict):
        for tid, val in data.items():
            try:
                spreads[tid] = float(val)
            except Exception:
                spreads[tid] = float("nan")
    return spreads


def fetch_market_prices(token_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """Fetch BUY/SELL prices for token_ids using POST /prices."""
    url = f"{CLOB_BASE_URL}/prices"
    payload = []
    for tid in token_ids:
        payload.append({"token_id": tid, "side": "BUY"})
        payload.append({"token_id": tid, "side": "SELL"})
    data = _request_json("POST", url, json_body=payload, timeout=10)
    result: Dict[str, Dict[str, float]] = {}
    if isinstance(data, dict):
        for tid in token_ids:
            entry = data.get(tid, {}) if isinstance(data.get(tid), dict) else {}
            buy_val = entry.get("BUY")
            sell_val = entry.get("SELL")
            try:
                buy = float(buy_val)
            except Exception:
                buy = float("nan")
            try:
                sell = float(sell_val)
            except Exception:
                sell = float("nan")
            result[tid] = {"BUY": buy, "SELL": sell}
    else:
        for tid in token_ids:
            result[tid] = {"BUY": float("nan"), "SELL": float("nan")}
    return result


def fetch_midpoints(token_ids: List[str]) -> Dict[str, float]:
    """
    Fetch midpoints per token; first attempt derived midpoint from batch /prices,
    then fall back to /midpoint per token if needed.
    """
    mids: Dict[str, float] = {}
    prices_map = fetch_market_prices(token_ids)
    for tid in token_ids:
        prices = prices_map.get(tid)
        if prices is not None:
            buy = prices.get("BUY")
            sell = prices.get("SELL")
            try:
                mids[tid] = (float(buy) + float(sell)) / 2.0
                continue
            except Exception:
                pass
        data = _request_json("GET", f"{CLOB_BASE_URL}/midpoint", params={"token_id": tid}, timeout=10)
        if isinstance(data, dict) and "mid" in data:
            try:
                mids[tid] = float(data["mid"])
            except Exception:
                mids[tid] = float("nan")
    return mids


# --------------------------------------
# Trades (data-api)
# --------------------------------------


def fetch_trades_for_market(
    condition_id: str,
    limit: int = 10000,
    taker_only: bool = True,
    page_size: int = 1000,
    timeout: int = 15,
) -> List[Dict]:
    """
    Fetch trades for a market (conditionId) with pagination.
    Stops when fewer than page_size results are returned or when reaching 'limit'.
    """
    trades: List[Dict] = []
    offset = 0
    base_url = f"{DATA_API_BASE_URL}/trades"
    while len(trades) < limit:
        params = {
            "market": condition_id,
            "limit": min(page_size, limit - len(trades)),
            "offset": offset,
        }
        if taker_only:
            params["takerOnly"] = "true"
        data = _request_json("GET", base_url, params=params, timeout=timeout)
        if not isinstance(data, list) or not data:
            break
        trades.extend(data)
        offset += len(data)
        if len(data) < page_size:
            break
    return trades[:limit]


def trades_to_dataframe(trades: List[Dict]) -> pd.DataFrame:
    """Normalize trades list to DataFrame with typed columns."""
    if not trades:
        return pd.DataFrame(columns=["timestamp", "side", "size", "price", "conditionId"])
    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), unit="s", utc=True, errors="coerce")
    df["side"] = df.get("side", "").astype(str).str.upper()
    df["size"] = pd.to_numeric(df.get("size"), errors="coerce")
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    if "conditionId" not in df.columns and "condition_id" in df.columns:
        df = df.rename(columns={"condition_id": "conditionId"})
    return df


# --------------------------------------
# Holders (data-api)
# --------------------------------------


def fetch_holders_for_markets(
    condition_ids: List[str],
    limit: int = 100,
    min_balance: Optional[float] = None,
    timeout: int = 15,
) -> Dict[str, List[Dict]]:
    """
    Fetch holders for multiple conditionIds via /holders.
    Returns mapping conditionId -> list of holder dicts.
    """
    if not condition_ids:
        return {}
    params = {"market": ",".join(condition_ids), "limit": limit}
    if min_balance is not None:
        params["minBalance"] = min_balance
    url = f"{DATA_API_BASE_URL}/holders"
    data = _request_json("GET", url, params=params, timeout=timeout)
    result: Dict[str, List[Dict]] = {}
    if isinstance(data, list):
        for entry in data:
            cid = entry.get("token") or entry.get("conditionId")
            holders = entry.get("holders", [])
            if cid:
                result[str(cid)] = holders if isinstance(holders, list) else []
    return result


# --------------------------------------
# User-centric endpoints (stubs)
# --------------------------------------


def fetch_positions(user: str, **kwargs) -> List[Dict]:
    """GET /positions for a user."""
    url = f"{DATA_API_BASE_URL}/positions"
    params = {"user": user}
    params.update(kwargs)
    data = _request_json("GET", url, params=params, timeout=15)
    return data if isinstance(data, list) else []


def fetch_closed_positions(user: str, **kwargs) -> List[Dict]:
    """GET /closed-positions for a user."""
    url = f"{DATA_API_BASE_URL}/closed-positions"
    params = {"user": user}
    params.update(kwargs)
    data = _request_json("GET", url, params=params, timeout=15)
    return data if isinstance(data, list) else []


def fetch_user_activity(user: str, **kwargs) -> List[Dict]:
    """GET /activity for a user."""
    url = f"{DATA_API_BASE_URL}/activity"
    params = {"user": user}
    params.update(kwargs)
    data = _request_json("GET", url, params=params, timeout=15)
    return data if isinstance(data, list) else []


def fetch_user_value(user: str, markets: Optional[List[str]] = None, **kwargs) -> Optional[float]:
    """GET /value for a user; returns numeric value or None."""
    url = f"{DATA_API_BASE_URL}/value"
    params = {"user": user}
    if markets:
        params["markets"] = ",".join(markets)
    params.update(kwargs)
    data = _request_json("GET", url, params=params, timeout=15)
    if isinstance(data, list) and data:
        entry = data[0]
        if isinstance(entry, dict) and "value" in entry:
            try:
                return float(entry["value"])
            except Exception:
                return None
    return None


# --------------------------------------
# Demo
# --------------------------------------


if __name__ == "__main__":
    offsets_to_try = [0, 5000, 10000, 15000]
    found = False
    for start in offsets_to_try:
        print(f"\nFetching resolved markets (start_offset={start})...")
        mkts = fetch_resolved_markets(
            max_markets=200, page_size=100, sleep_s=0.2, start_offset=start
        )
        print(f"Fetched {len(mkts)} markets.")
        if mkts:
            tok, hist = find_first_history(
                mkts, sleep_s=0.2, interval="max", fidelity=60, max_tokens=100
            )
            if tok and hist:
                hdf = price_history_to_dataframe({"history": hist})
                print(f"\nFound history for token: {tok}")
                print(f"History points: {len(hist)}; DataFrame shape: {hdf.shape}")
                if not hdf.empty:
                    print(hdf.head())
                found = True
                break
            else:
                print("No history found in this batch; advancing offset...")
        else:
            print("No markets returned; stopping.")
    if not found:
        print("\nNo token with history found in sampled offsets; consider increasing start_offset further.")
