from __future__ import annotations

import ast
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data_fetch import (
    fetch_price_history,
    fetch_resolved_markets,
    markets_to_dataframe,
    price_history_to_dataframe,
)


def _parse_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
    """Safely parse a timestamp string to pandas.Timestamp (UTC)."""
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return ts if pd.notnull(ts) else None


def _extract_label(market: Dict) -> Optional[int]:
    """
    Try to extract a binary resolved label from common Polymarket fields.
    Returns 1 for YES, 0 for NO, or None if not found.
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
    # TODO: adjust if actual schema exposes a different resolved field.
    return None


def _extract_token_id(tokens_field) -> Optional[str]:
    """
    Extract a token id from clobTokenIds which may be a list, nested, or stringified.
    """
    candidates: List[str] = []

    if isinstance(tokens_field, str):
        try:
            parsed = ast.literal_eval(tokens_field)
            tokens_field = parsed
        except Exception:
            stripped = tokens_field.strip().strip('[]"\'')
            parts = [p.strip().strip('[]"\'') for p in stripped.split(",")]
            candidates.extend(parts)

    if isinstance(tokens_field, (list, tuple)):
        for item in tokens_field:
            if isinstance(item, str):
                candidates.append(item.strip().strip('[]"\''))
            elif isinstance(item, (list, tuple)) and item:
                inner = item[0]
                if isinstance(inner, str):
                    candidates.append(inner.strip().strip('[]"\''))
    for c in candidates:
        if c and c not in {"[", "]", '"', "'"}:
            return c
    return None


def _fallback_price_from_market(market: Dict) -> Optional[float]:
    """
    Try to derive a last price from market metadata if history is empty.
    Uses outcomePrices first element if available.
    """
    prices = market.get("outcomePrices") or market.get("outcome_prices")
    if isinstance(prices, (list, tuple)) and prices:
        try:
            return float(prices[0])
        except Exception:
            return None
    return None


def compute_market_features(
    market: Dict,
    history_df: pd.DataFrame,
    snapshot_offset_days: int = 3,
    min_history_points: int = 0,
) -> Optional[Dict]:
    """
    Compute snapshot features for a single market using its price history and metadata.

    Returns a feature dict including target 'y', or None if not enough data/label.
    If there is insufficient history, falls back to metadata price (outcomePrices) when possible.
    """
    df = history_df.copy()
    if "timestamp" not in df.columns and "t" in df.columns:
        df = df.rename(columns={"t": "timestamp"})
    if "price" not in df.columns and "p" in df.columns:
        df = df.rename(columns={"p": "price"})

    # Parse times
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    resolution_time = (
        _parse_timestamp(market.get("endDateIso"))
        or _parse_timestamp(market.get("closeTime"))
        or _parse_timestamp(market.get("endTime"))
    )
    start_time = _parse_timestamp(market.get("startDateIso")) or _parse_timestamp(market.get("startTime"))
    if resolution_time is None or start_time is None:
        return None

    snapshot_time = resolution_time - pd.Timedelta(days=snapshot_offset_days)
    df = df[df["timestamp"] <= snapshot_time].sort_values("timestamp")

    if "price" not in df.columns or df.empty or len(df) < min_history_points:
        fallback_price = _fallback_price_from_market(market)
        if fallback_price is None:
            return None
        last_price = fallback_price
        ma_7d = np.nan
        vol_7d = np.nan
        vol_30d = np.nan
        price_trend_7d = np.nan
    else:
        def window_stats(days: int) -> pd.DataFrame:
            window_start = snapshot_time - pd.Timedelta(days=days)
            return df[df["timestamp"] >= window_start]

        last_price = df["price"].iloc[-1]
        w7 = window_stats(7)
        w30 = window_stats(30)
        ma_7d = w7["price"].mean() if not w7.empty else np.nan
        vol_7d = w7["price"].std(ddof=0) if not w7.empty else np.nan
        vol_30d = w30["price"].std(ddof=0) if not w30.empty else np.nan
        price_7d_ago = w7["price"].iloc[0] if len(w7) > 0 else np.nan
        price_trend_7d = last_price - price_7d_ago if pd.notnull(price_7d_ago) else np.nan

    age_days = (snapshot_time - start_time).total_seconds() / 86400.0
    time_to_resolution_days = (resolution_time - snapshot_time).total_seconds() / 86400.0

    y = _extract_label(market)
    if y is None:
        return None

    return {
        "market_id": market.get("id"),
        "snapshot_offset_days": snapshot_offset_days,
        "last_price": last_price,
        "ma_7d": ma_7d,
        "price_trend_7d": price_trend_7d,
        "vol_7d": vol_7d,
        "vol_30d": vol_30d,
        "age_days": age_days,
        "time_to_resolution_days": time_to_resolution_days,
        "category": market.get("category"),
        "y": y,
    }


def build_training_dataset(
    markets: List[Dict],
    snapshot_offsets: Sequence[int] = (3, 7, 14),
    max_markets: Optional[int] = None,
    min_history_points: int = 0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build a training DataFrame by fetching histories, computing features, and assembling rows.
    Generates multiple snapshots per market to expand the dataset.
    """
    rows: List[Dict] = []
    to_iterate = markets[:max_markets] if max_markets else markets

    skipped_no_token = 0
    skipped_label = 0

    for market in to_iterate:
        token_id = _extract_token_id(market.get("clobTokenIds"))
        if not token_id:
            skipped_no_token += 1
            continue

        history = fetch_price_history(token_id=token_id, interval="all", fidelity=60)
        history_df = price_history_to_dataframe(history)

        for offset in snapshot_offsets:
            feats = compute_market_features(
                market=market,
                history_df=history_df,
                snapshot_offset_days=offset,
                min_history_points=min_history_points,
            )
            if feats is not None:
                rows.append(feats)
            else:
                skipped_label += 1

    if verbose:
        print(
            f"Processed markets: {len(to_iterate)}, rows kept: {len(rows)}, "
            f"no_token: {skipped_no_token}, skipped_no_label_or_data: {skipped_label}"
        )

    return pd.DataFrame(rows)


def save_training_dataset_to_csv(
    df: pd.DataFrame,
    path: str = "data/processed/training_data.csv",
) -> None:
    """Save the training DataFrame to CSV, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    # Smoke test: fetch markets, build dataset, inspect, and save.
    print("Fetching resolved markets...")
    markets = fetch_resolved_markets(max_markets=1200, page_size=200, sleep_s=0.05)
    print(f"Fetched {len(markets)} markets.")
    _mdf = markets_to_dataframe(markets)
    if not _mdf.empty:
        print("Sample markets:")
        print(_mdf.head())

    print("\nBuilding training dataset...")
    df = build_training_dataset(
        markets=markets,
        snapshot_offsets=(3, 7, 14),
        max_markets=None,
        min_history_points=0,  # allow metadata fallback
        verbose=True,
    )
    print(f"Training DataFrame shape: {df.shape}")
    if not df.empty:
        print(df.head())
        if "y" in df.columns:
            print("Label distribution:")
            print(df["y"].value_counts())

    output_path = "data/processed/training_data.csv"
    save_training_dataset_to_csv(df, path=output_path)
    print(f"\nSaved training data to: {output_path}")
