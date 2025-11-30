from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

INPUT_PATH = "data/processed/training_data_clean.csv"
MODEL_PATH = "models/xgboost_optimized_misprice.joblib"


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load cleaned dataset, build mispricing target, and return X, y."""
    if not os.path.exists(INPUT_PATH):
        print(f"Error: input file not found at {INPUT_PATH}")
        raise SystemExit(1)
    df = pd.read_csv(INPUT_PATH)

    required_cols = {"last_price", "final_price"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: required columns missing: {missing}")
        raise SystemExit(1)

    gap = df["final_price"] - df["last_price"]
    y = (gap.abs() >= 0.15).astype(int)
    df["y_misprice"] = y

    X = df.drop(columns=["y", "final_price", "y_misprice"], errors="ignore")

    print("Mispricing label distribution (y_misprice):")
    print(y.value_counts(normalize=False))
    print(y.value_counts(normalize=True))
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified train/test split."""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def load_model(path: str) -> Any:
    """Load trained model from disk."""
    if not os.path.exists(path):
        print(f"Error: model file not found at {path}")
        raise SystemExit(1)
    return joblib.load(path)


def threshold_sweep(y_true: pd.Series, y_proba: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Evaluate metrics across a range of thresholds."""
    rows: list[Dict[str, float]] = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def print_best_points(df_thr: pd.DataFrame, y_true: pd.Series, y_proba: np.ndarray) -> None:
    """Print best thresholds and detailed reports."""
    idx_f1 = df_thr["f1"].idxmax()
    row_f1 = df_thr.loc[idx_f1]
    print("\nBest F1 threshold:")
    print(row_f1)

    y_pred_f1 = (y_proba >= row_f1["threshold"]).astype(int)
    print("\nConfusion matrix (best F1):")
    print(confusion_matrix(y_true, y_pred_f1))
    print("Classification report (best F1):")
    print(classification_report(y_true, y_pred_f1, zero_division=0))

    df_candidate = df_thr[df_thr["recall"] >= 0.50]
    if df_candidate.empty:
        print("\nNo threshold achieved recall >= 0.50.")
        return

    idx_prec = df_candidate["precision"].idxmax()
    row_prec = df_candidate.loc[idx_prec]
    print("\nBest precision with recall >= 0.50:")
    print(row_prec)

    y_pred_prec = (y_proba >= row_prec["threshold"]).astype(int)
    print("\nConfusion matrix (best precision w/ recall>=0.50):")
    print(confusion_matrix(y_true, y_pred_prec))
    print("Classification report (best precision w/ recall>=0.50):")
    print(classification_report(y_true, y_pred_prec, zero_division=0))


def main() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = load_model(MODEL_PATH)
    y_proba = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.2, 0.8, 31)
    df_thr = threshold_sweep(y_test, y_proba, thresholds)

    os.makedirs("data/processed", exist_ok=True)
    df_thr.to_csv("data/processed/threshold_sweep_xgb_misprice.csv", index=False)
    print("\nSaved threshold sweep CSV → data/processed/threshold_sweep_xgb_misprice.csv")

    print("\nThreshold sweep metrics:")
    print(df_thr.sort_values("threshold").to_string(index=False))

    print_best_points(df_thr, y_test, y_proba)


if __name__ == "__main__":
    main()
