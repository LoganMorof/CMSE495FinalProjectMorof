from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data_fetch import (
    fetch_price_history,
    fetch_resolved_markets,
    markets_to_dataframe,
    price_history_to_dataframe,
    fetch_trades_for_market,
    trades_to_dataframe,
)


def _parse_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
    """Safely parse a timestamp string to pandas.Timestamp (UTC)."""
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return ts if pd.notnull(ts) else None


def _extract_token_ids(tokens_field) -> List[str]:
    """
    Normalize clobTokenIds into a list of clean token id strings.
    Handles stringified lists and nested lists.
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


def _fallback_price_from_market(market: Dict) -> Optional[float]:
    """
    Try to derive a last price from market metadata if history is empty.
    Uses outcomePrices or lastTradePrice if available.
    """
    prices = market.get("outcomePrices") or market.get("outcome_prices")
    if isinstance(prices, (list, tuple)) and prices:
        try:
            return float(prices[0])
        except Exception:
            return None
    ltp = market.get("lastTradePrice")
    try:
        return float(ltp) if ltp is not None else None
    except Exception:
        return None


def _extract_label(market: Dict) -> Optional[int]:
    """
    Extract a binary label. Tries common outcome fields; falls back to lastTradePrice heuristic if market is closed.
    """
    keys = [
        "winningOutcome",
        "resolvedOutcome",
        "winning_outcome",
        "resolved_outcome",
        "winningSide",
        "winning_side",
        "result",
        "outcome",
        "resolution",
    ]
    for k in keys:
        if k in market:
            val = market.get(k)
            if isinstance(val, bool):
                return 1 if val else 0
            if isinstance(val, (int, float)) and val in (0, 1):
                return int(val)
            if isinstance(val, str):
                outcome_clean = val.strip().lower()
                if outcome_clean in {"yes", "y", "1", "true"}:
                    return 1
                if outcome_clean in {"no", "n", "0", "false"}:
                    return 0

    if market.get("closed") is True:
        ltp = _fallback_price_from_market(market)
        if ltp is not None:
            return 1 if ltp >= 0.5 else 0
    return None


def _extract_condition_id(market: Dict) -> Optional[str]:
    """Attempt to pull a conditionId/condition_id from market payload."""
    candidates = [
        market.get("conditionId"),
        market.get("condition_id"),
    ]
    for cand in candidates:
        if cand:
            return str(cand)
    list_cands = market.get("conditionIds") or market.get("condition_ids")
    if isinstance(list_cands, (list, tuple)) and list_cands:
        return str(list_cands[0])
    return None


def compute_market_features(
    market: Dict,
    history_df: pd.DataFrame,
    trades_df: Optional[pd.DataFrame] = None,
    snapshot_offset_hours: int = 3,
    min_history_points: int = 1,
) -> Optional[Dict]:
    """
    Compute snapshot features for a single market using hour-based offsets, strict pre-snapshot history,
    and optional trades/order-flow data. Returns feature dict including target 'y', or None if not enough data/label.
    """
    df = history_df.copy() if history_df is not None else pd.DataFrame()
    if "timestamp" not in df.columns and "t" in df.columns:
        df = df.rename(columns={"t": "timestamp"})
    if "price" not in df.columns and "p" in df.columns:
        df = df.rename(columns={"p": "price"})
    if "timestamp" not in df.columns or "price" not in df.columns:
        df = pd.DataFrame(columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df = df.dropna(subset=["timestamp", "price"])

    resolution_time = (
        _parse_timestamp(market.get("endDateIso"))
        or _parse_timestamp(market.get("closeTime"))
        or _parse_timestamp(market.get("endTime"))
    )
    start_time = _parse_timestamp(market.get("startDateIso")) or _parse_timestamp(market.get("startTime"))
    if resolution_time is None or start_time is None:
        return None

    snapshot_time = resolution_time - pd.Timedelta(hours=snapshot_offset_hours)

    df_sorted = df.sort_values("timestamp")
    df_hist = df_sorted[df_sorted["timestamp"] <= snapshot_time]
    if df_hist.empty or len(df_hist) < min_history_points:
        return None

    tail_n = 50
    tail = df_hist.tail(tail_n)
    if tail.empty or len(tail) < min_history_points:
        return None

    prices = tail["price"]
    last_price = prices.iloc[-1]
    recent_ma = prices.mean()
    recent_vol = prices.std(ddof=0)
    recent_trend = last_price - prices.iloc[0]
    recent_range = prices.max() - prices.min()

    life_days = (resolution_time - start_time).total_seconds() / 86400.0
    age_days = (snapshot_time - start_time).total_seconds() / 86400.0
    time_to_resolution_days = (resolution_time - snapshot_time).total_seconds() / 86400.0
    frac_life_elapsed = age_days / life_days if life_days > 0 else np.nan

    bid_ask_pairs = [("best_bid", "best_ask"), ("bid", "ask")]
    bid_col = ask_col = None
    for b_col, a_col in bid_ask_pairs:
        if b_col in df_hist.columns and a_col in df_hist.columns:
            bid_col, ask_col = b_col, a_col
            break
    spread = np.nan
    mid_price = last_price
    if bid_col and ask_col:
        rows_with_ba = df_hist.dropna(subset=[bid_col, ask_col])
        if not rows_with_ba.empty:
            last_row_ba = rows_with_ba.iloc[-1]
            spread = last_row_ba[ask_col] - last_row_ba[bid_col]
            mid_price = (last_row_ba[bid_col] + last_row_ba[ask_col]) / 2.0

    trade_windows = [5, 15, 60]
    trades_counts = {}
    for w in trade_windows:
        window_start = snapshot_time - pd.Timedelta(minutes=w)
        sub = df_hist[df_hist["timestamp"] > window_start]
        trades_counts[w] = len(sub)

    time_since_last_trade_minutes = np.nan
    if not df_hist.empty:
        last_ts = df_hist["timestamp"].iloc[-1]
        time_since_last_trade_minutes = (snapshot_time - last_ts).total_seconds() / 60.0

    up_move_ratio_tail = np.nan
    deltas = prices.diff().dropna()
    if len(deltas) > 0:
        up_move_ratio_tail = (deltas > 0).mean()

    price_minus_ma = last_price - recent_ma if pd.notnull(recent_ma) else np.nan

    final_price = np.nan
    df_up_to_resolution = df_sorted[df_sorted["timestamp"] <= resolution_time]
    if not df_up_to_resolution.empty:
        final_price = df_up_to_resolution["price"].iloc[-1]

    def _empty_trade_features() -> Dict[str, float]:
        feats: Dict[str, float] = {
            "cum_buy_volume": 0.0,
            "cum_sell_volume": 0.0,
            "net_cum_order_flow": 0.0,
            "flow_ratio": np.nan,
        }
        for w in trade_windows:
            feats[f"trade_count_{w}m"] = 0
            feats[f"buy_volume_{w}m"] = 0.0
            feats[f"sell_volume_{w}m"] = 0.0
            feats[f"net_order_flow_{w}m"] = 0.0
            feats[f"avg_trade_size_{w}m"] = np.nan
        return feats

    trade_feats: Dict[str, float]
    if trades_df is None or trades_df.empty:
        trade_feats = _empty_trade_features()
    else:
        tdf = trades_df.copy()
        tdf = tdf.dropna(subset=["timestamp", "size", "price"])
        trades_hist = tdf[tdf["timestamp"] <= snapshot_time]
        if trades_hist.empty:
            trade_feats = _empty_trade_features()
        else:
            cum_buy_volume = trades_hist.loc[trades_hist["side"] == "BUY", "size"].sum()
            cum_sell_volume = trades_hist.loc[trades_hist["side"] == "SELL", "size"].sum()
            net_cum_order_flow = cum_buy_volume - cum_sell_volume
            denom = cum_buy_volume + cum_sell_volume
            flow_ratio = cum_buy_volume / denom if denom > 0 else np.nan
            trade_feats = {
                "cum_buy_volume": float(cum_buy_volume),
                "cum_sell_volume": float(cum_sell_volume),
                "net_cum_order_flow": float(net_cum_order_flow),
                "flow_ratio": flow_ratio,
            }
            for w in trade_windows:
                window_start = snapshot_time - pd.Timedelta(minutes=w)
                sub_trades = trades_hist[trades_hist["timestamp"] > window_start]
                trade_count = len(sub_trades)
                buy_vol = sub_trades.loc[sub_trades["side"] == "BUY", "size"].sum()
                sell_vol = sub_trades.loc[sub_trades["side"] == "SELL", "size"].sum()
                net_flow = buy_vol - sell_vol
                avg_trade_size = (buy_vol + sell_vol) / trade_count if trade_count > 0 else np.nan
                trade_feats[f"trade_count_{w}m"] = int(trade_count)
                trade_feats[f"buy_volume_{w}m"] = float(buy_vol)
                trade_feats[f"sell_volume_{w}m"] = float(sell_vol)
                trade_feats[f"net_order_flow_{w}m"] = float(net_flow)
                trade_feats[f"avg_trade_size_{w}m"] = avg_trade_size

    y = _extract_label(market)
    if y is None:
        return None

    return {
        "market_id": market.get("id"),
        "last_price": last_price,
        "recent_ma": recent_ma,
        "recent_vol": recent_vol,
        "recent_trend": recent_trend,
        "recent_range": recent_range,
        "age_days": age_days,
        "time_to_resolution_days": time_to_resolution_days,
        "frac_life_elapsed": frac_life_elapsed,
        "category": market.get("category"),
        "y": y,
        "final_price": final_price,
        "spread": spread,
        "mid_price": mid_price,
        "trades_5m": trades_counts[5],
        "trades_15m": trades_counts[15],
        "trades_60m": trades_counts[60],
        "time_since_last_trade_minutes": time_since_last_trade_minutes,
        "up_move_ratio_tail": up_move_ratio_tail,
        "price_minus_ma": price_minus_ma,
        **trade_feats,
    }


def build_training_dataset(
    markets: List[Dict],
    snapshot_offset_hours_list: Sequence[int] = (48, 24, 12, 6, 3, 1),
    max_markets: Optional[int] = None,
    min_history_points: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build a training DataFrame by fetching histories, computing features, and assembling rows.
    Each market can contribute multiple rows (one per snapshot_offset_hours entry).
    """
    rows: List[Dict] = []
    to_iterate = markets[:max_markets] if max_markets else markets

    skipped_no_token = 0
    skipped_label_or_data = 0

    for market in to_iterate:
        token_ids = _extract_token_ids(market.get("clobTokenIds"))
        if not token_ids:
            skipped_no_token += 1
            continue

        condition_id = _extract_condition_id(market)
        trades_df = None
        if condition_id:
            trades = fetch_trades_for_market(condition_id=condition_id, limit=10000, taker_only=True)
            trades_df = trades_to_dataframe(trades) if trades else None

        added_for_market = False
        for tok in token_ids:
            history = fetch_price_history(tok, interval="max", fidelity=60)
            hdf = price_history_to_dataframe(history)
            for offset_hours in snapshot_offset_hours_list:
                feats = compute_market_features(
                    market=market,
                    history_df=hdf,
                    trades_df=trades_df,
                    snapshot_offset_hours=offset_hours,
                    min_history_points=min_history_points,
                )
                if feats is not None:
                    feats["snapshot_offset_hours"] = offset_hours
                    rows.append(feats)
                    added_for_market = True
            if added_for_market:
                break
        if not added_for_market:
            skipped_label_or_data += 1

    if verbose:
        print(
            f"Processed markets: {len(to_iterate)}, kept rows: {len(rows)}, "
            f"no_token: {skipped_no_token}, no_label_or_data: {skipped_label_or_data}"
        )

    return pd.DataFrame(rows)


def save_training_dataset_to_csv(
    df: pd.DataFrame,
    path: str = "data/processed/training_data.csv",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    print("Fetching resolved markets...")
    markets = fetch_resolved_markets(
        max_markets=1500,
        page_size=100,
        sleep_s=0.2,
    )
    print(f"Fetched {len(markets)} markets.")
    _mdf = markets_to_dataframe(markets)
    if not _mdf.empty:
        print("Sample markets:")
        print(_mdf.head())

    print("\nBuilding training dataset...")
    df = build_training_dataset(
        markets=markets,
        snapshot_offset_hours_list=(48, 24, 12, 6, 3, 1),
        max_markets=None,
        min_history_points=1,
        verbose=True,
    )
    print(f"Training DataFrame shape: {df.shape}")
    if not df.empty and "y" in df.columns:
        print("Label distribution:")
        print(df["y"].value_counts())

    output_path = "data/processed/training_data.csv"
    save_training_dataset_to_csv(df, path=output_path)
    print(f"\nSaved training data to: {output_path}")
