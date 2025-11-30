from __future__ import annotations

import os

import numpy as np
import pandas as pd

INPUT_PATH = "data/processed/training_data.csv"
OUTPUT_PATH = "data/processed/training_data_clean.csv"


def clean_dataset() -> None:
    df = pd.read_csv(INPUT_PATH)

    # Drop columns that are identifiers or overly missing.
    cols_to_drop = [
        "market_id",
        "category",
        "spread",
        "avg_trade_size_5m",
        "avg_trade_size_15m",
        "avg_trade_size_60m",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Impute specific columns with median.
    for col in ("flow_ratio", "up_move_ratio_tail"):
        if col in df.columns:
            median_val = df[col].median(skipna=True)
            df[col] = df[col].fillna(median_val)

    # Replace inf/-inf with NaN.
    df = df.replace([np.inf, -np.inf], np.nan)

    # Final median imputation for any remaining numeric NaNs.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median(skipna=True)
            df[col] = df[col].fillna(median_val)

    # Save cleaned dataset.
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # Reporting.
    print(f"Saved cleaned dataset → {OUTPUT_PATH}")
    print(f"Final shape: {df.shape}")
    print(df.isna().sum())


if __name__ == "__main__":
    clean_dataset()
