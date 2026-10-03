"""Reproducible EDA figures and data-quality tables for the project report.

Outputs (reports/)
    eda_descriptive_statistics.csv     eda_missing_values.csv
    eda_data_quality_summary.json      (rows, zero kWh, duplicates, outliers, skew ...)
    eda_daily_energy_trend.png         eda_monthly_energy.png
    eda_correlation_heatmap.png        eda_kwh_distribution.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent))
from common import PROCESSED, REPORTS, TARGET, write_json  # noqa: E402

# Columns that would hide the real drivers of energy use in the heatmap.
HEATMAP_EXCLUDE_PREFIXES = ("kwh_lag", "kwh_rolling", "lag_energy")


def data_quality_summary(df: pd.DataFrame) -> dict:
    """Numbers used in the 'Data Cleaning Results' section of the report."""
    kwh = df["kwh"]
    q1, q3 = kwh.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = int(((kwh < low) | (kwh > high)).sum())
    zero = int((kwh == 0).sum())
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "houses": int(df["house_id"].nunique()) if "house_id" in df else None,
        "date_start": df["timestamp"].min(),
        "date_end": df["timestamp"].max(),
        "zero_kwh_records": zero,
        "zero_kwh_percent": round(zero / len(df) * 100, 3),
        "duplicate_records": int(df.duplicated().sum()),
        "duplicate_house_timestamp": int(df.duplicated(["house_id", "timestamp"]).sum())
        if "house_id" in df else None,
        "outlier_method": "IQR rule (1.5 x IQR) on kwh",
        "outlier_lower_bound": round(float(low), 3),
        "outlier_upper_bound": round(float(high), 3),
        "outlier_records": outliers,
        "outlier_percent": round(outliers / len(df) * 100, 3),
        "kwh_min": round(float(kwh.min()), 3),
        "kwh_max": round(float(kwh.max()), 3),
        "kwh_mean": round(float(kwh.mean()), 3),
        "kwh_median": round(float(kwh.median()), 3),
        "kwh_std": round(float(kwh.std()), 3),
        "kwh_skewness": round(float(kwh.skew()), 3),
        "kwh_excess_kurtosis_pandas": round(float(kwh.kurt()), 3),
    }


def save_figure(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(REPORTS / name, dpi=200)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(PROCESSED / "energy_model_data.csv", parse_dates=["timestamp"])
    sns.set_theme(style="whitegrid", context="notebook")

    df.describe(include="all").transpose().to_csv(REPORTS / "eda_descriptive_statistics.csv")
    (df.isna().sum().sort_values(ascending=False)
       .to_csv(REPORTS / "eda_missing_values.csv", header=["missing_count"]))

    summary = data_quality_summary(df)
    write_json(REPORTS / "eda_data_quality_summary.json", summary)

    # 1. mean daily consumption
    daily = df.groupby("timestamp")["kwh"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=daily, x="timestamp", y="kwh", ax=ax, color="#1d6996")
    ax.set(title="Mean Daily Household Energy Consumption", ylabel="kWh", xlabel="Date")
    save_figure(fig, "eda_daily_energy_trend.png")

    # 2. monthly distribution
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=df, x="month", y="kwh", ax=ax, color="#76b7b2", showfliers=False)
    ax.set(title="Monthly Energy Consumption Distribution", ylabel="kWh", xlabel="Month")
    save_figure(fig, "eda_monthly_energy.png")

    # 3. kWh distribution with mean / median
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["kwh"], bins=60, kde=True, ax=ax, color="#4c78a8")
    ax.axvline(summary["kwh_mean"], color="#e45756", linestyle="--", label=f"Mean {summary['kwh_mean']:.2f}")
    ax.axvline(summary["kwh_median"], color="#54a24b", linestyle=":", label=f"Median {summary['kwh_median']:.2f}")
    ax.set(title="Distribution of Daily Energy Consumption", xlabel="kWh")
    ax.legend()
    save_figure(fig, "eda_kwh_distribution.png")

    # 4. correlation heatmap of external drivers (no IDs, target or lag copies of kwh)
    numeric = df.select_dtypes("number")
    keep = [c for c in numeric.columns
            if c not in ("house_id", TARGET) and not c.startswith(HEATMAP_EXCLUDE_PREFIXES)]
    top = numeric[keep].corr()["kwh"].abs().sort_values(ascending=False).head(15).index
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric[top].corr(), cmap="vlag", center=0, ax=ax)
    ax.set_title("Top Features Correlated with Energy Use")
    save_figure(fig, "eda_correlation_heatmap.png")

    print("EDA reports and figures saved in reports/.")
    print(f"Rows {summary['rows']:,} | zero kWh {summary['zero_kwh_records']} "
          f"({summary['zero_kwh_percent']}%) | duplicates {summary['duplicate_records']} | "
          f"outliers {summary['outlier_records']} ({summary['outlier_percent']}%, IQR rule)")


if __name__ == "__main__":
    main()
