# src/explore_data.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    data_path = "data/processed/training_data.csv"

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} does not exist. Run src.features first.")
        return

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    print("\n===== BASIC INFO =====")
    print(df.info())

    print("\n===== SHAPE =====")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n===== COLUMN NAMES =====")
    print(df.columns.tolist())

    print("\n===== LABEL DISTRIBUTION (y) =====")
    print(df["y"].value_counts())
    print(df["y"].value_counts(normalize=True))

    print("\n===== MISSING VALUES (per column) =====")
    print(df.isna().sum())

    print("\n===== MISSING VALUE FRACTION (per column) =====")
    print(df.isna().mean())

    print("\n===== SUMMARY STATISTICS (numeric columns) =====")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe())

    # Optional: correlation heatmap for numeric features
    try:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap="viridis")
        plt.title("Correlation Heatmap (Numeric Features)")
        plt.tight_layout()
        output_path = "data/processed/correlation_heatmap.png"
        plt.savefig(output_path)
        plt.close()
        print(f"\nSaved correlation heatmap → {output_path}")
    except Exception as e:
        print(f"Could not generate correlation heatmap: {e}")

    # Optional: simple histograms for numeric features
    hist_dir = "data/processed/histograms"
    os.makedirs(hist_dir, exist_ok=True)
    for col in numeric_cols:
        try:
            plt.figure(figsize=(6, 4))
            df[col].hist(bins=40)
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            out_path = os.path.join(hist_dir, f"{col}_hist.png")
            plt.savefig(out_path)
            plt.close()
        except Exception as e:
            print(f"Could not create histogram for {col}: {e}")

    print(f"\nSaved histograms → {hist_dir}")
    print("\n===== DONE =====")


if __name__ == "__main__":
    main()
