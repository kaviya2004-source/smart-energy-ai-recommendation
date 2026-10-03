"""
Prepare the original energy dataset:
- Load original CSV/XLSX
- Standardize column names
- Clean invalid timestamp / kWh / House ID
- Keep occupancy unchanged
- Filter latest five years
- Handle missing values
- Remove duplicate records
- Create temporal + lag features
- Create next-day target
- Chronological train/validation/test split
- Save processed modelling dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from common import PROCESSED, RANDOM_STATE, TARGET, write_json


def clean_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
    )


def load_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError("Only csv and xlsx files are supported.")


def build_dataset(raw: pd.DataFrame) -> pd.DataFrame:

    print("\n" + "=" * 60)
    print("LOADING ORIGINAL DATASET")
    print("=" * 60)

    print("Original shape:", raw.shape)

    # ---------------------------------------------------------
    # 1. STANDARDISE COLUMN NAMES
    # ---------------------------------------------------------
    df = raw.rename(columns=clean_name).copy()

    print("\nColumns after standardisation:")
    print(df.columns.tolist())

    # ---------------------------------------------------------
    # 2. REQUIRED COLUMNS
    # ---------------------------------------------------------
    required = ["timestamp", "house_id", "kwh"]

    missing_required = [
        col for col in required
        if col not in df.columns
    ]

    if missing_required:
        raise ValueError(
            f"Missing required columns: {missing_required}"
        )

    # ---------------------------------------------------------
    # 3. TIMESTAMP
    # ---------------------------------------------------------
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # 4. NUMERIC CONVERSION
    # ---------------------------------------------------------
    df["house_id"] = pd.to_numeric(
        df["house_id"],
        errors="coerce"
    )

    df["kwh"] = pd.to_numeric(
        df["kwh"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # 5. REMOVE INVALID CORE RECORDS
    # ---------------------------------------------------------
    before = len(df)

    df = df.dropna(
        subset=["timestamp", "house_id", "kwh"]
    )

    removed_invalid = before - len(df)

    print(
        f"\nInvalid timestamp/kWh/House ID removed: "
        f"{removed_invalid:,}"
    )

    # ---------------------------------------------------------
    # 6. REMOVE DUPLICATES
    # ---------------------------------------------------------
    before = len(df)

    df = df.drop_duplicates(
        subset=["house_id", "timestamp"]
    )

    removed_duplicates = before - len(df)

    print(
        f"Duplicate house-timestamp records removed: "
        f"{removed_duplicates:,}"
    )

    # ---------------------------------------------------------
    # 7. SORT BEFORE FIVE-YEAR FILTER
    # ---------------------------------------------------------
    df = df.sort_values(
        ["timestamp", "house_id"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 8. FIVE-YEAR FILTER
    # ---------------------------------------------------------
    original_start = df["timestamp"].min()
    original_end = df["timestamp"].max()

    cutoff = original_end - pd.DateOffset(years=5)

    df = df[
        df["timestamp"] >= cutoff
    ].copy()

    df = df.sort_values(
        ["house_id", "timestamp"]
    ).reset_index(drop=True)

    print("\nFive-year filtering:")
    print("Original start :", original_start)
    print("Original end   :", original_end)
    print("Five-year start:", cutoff)
    print("Five-year end  :", df["timestamp"].max())
    print("Rows           :", f"{len(df):,}")

    # ---------------------------------------------------------
    # 9. NEGATIVE KWH CHECK
    # ---------------------------------------------------------
    negative_count = int((df["kwh"] < 0).sum())

    print(
        f"\nNegative kWh records: {negative_count:,}"
    )

    if negative_count > 0:
        df = df[df["kwh"] >= 0].copy()

    # ---------------------------------------------------------
    # 10. ZERO KWH CHECK
    # ---------------------------------------------------------
    zero_count = int((df["kwh"] == 0).sum())

    print(
        f"Zero kWh records: {zero_count:,}"
    )

    # Zero values are valid observations.
    # They are NOT removed.

    # ---------------------------------------------------------
    # 11. MISSING VALUE INVESTIGATION
    # ---------------------------------------------------------
    missing = df.isna().sum()
    missing = missing[missing > 0]

    print("\nMissing-value investigation:")

    if len(missing) == 0:
        print("No missing values.")
    else:
        print(missing)

    # ---------------------------------------------------------
    # 12. MISSING VALUE HANDLING
    # ---------------------------------------------------------

    # Numeric columns:
    # First use house-wise median,
    # then global median if still missing.
    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    for col in numeric_cols:

        if col == "kwh":
            continue

        df[col] = df.groupby(
            "house_id"
        )[col].transform(
            lambda s: s.fillna(s.median())
        )

        df[col] = df[col].fillna(
            df[col].median()
        )

    # Categorical columns:
    # Use mode, otherwise "unknown".
    categorical_cols = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    for col in categorical_cols:

        if col == "timestamp":
            continue

        mode = df[col].mode(dropna=True)

        if len(mode) > 0:
            df[col] = df[col].fillna(mode.iloc[0])
        else:
            df[col] = df[col].fillna("unknown")

    # ---------------------------------------------------------
    # 13. VERIFY MISSING VALUES
    # ---------------------------------------------------------
    remaining_missing = int(
        df.isna().sum().sum()
    )

    print(
        f"\nRemaining missing values: "
        f"{remaining_missing}"
    )

    # ---------------------------------------------------------
    # 14. TEMPORAL FEATURES
    # ---------------------------------------------------------
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day_of_month"] = df["timestamp"].dt.day
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # ---------------------------------------------------------
    # 15. LAG FEATURES
    # ---------------------------------------------------------
    df["kwh_lag_1"] = (
        df.groupby("house_id")["kwh"]
        .shift(1)
    )

    df["kwh_lag_7"] = (
        df.groupby("house_id")["kwh"]
        .shift(7)
    )

    df["kwh_rolling_7"] = (
        df.groupby("house_id")["kwh"]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                7,
                min_periods=3
            )
            .mean()
        )
    )

    # ---------------------------------------------------------
    # 16. NEXT-DAY TARGET
    # ---------------------------------------------------------
    df[TARGET] = (
        df.groupby("house_id")["kwh"]
        .shift(-1)
    )

    # ---------------------------------------------------------
    # 17. REMOVE ONLY ROWS THAT CANNOT FORM A TARGET
    # ---------------------------------------------------------
    before = len(df)

    df = df.dropna(
        subset=[
            "kwh_lag_1",
            "kwh_lag_7",
            "kwh_rolling_7",
            TARGET
        ]
    )

    removed_model_rows = before - len(df)

    print(
        f"\nRows removed for unavailable "
        f"lag/target values: {removed_model_rows:,}"
    )

    # ---------------------------------------------------------
    # 18. FINAL SORT
    # ---------------------------------------------------------
    df = df.sort_values(
        ["timestamp", "house_id"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 19. CHRONOLOGICAL SPLIT
    # ---------------------------------------------------------
    unique_dates = np.sort(
        df["timestamp"].dt.normalize().unique()
    )

    train_index = int(
        len(unique_dates) * 0.70
    )

    valid_index = int(
        len(unique_dates) * 0.85
    )

    train_end = unique_dates[train_index]
    valid_end = unique_dates[valid_index]

    df["split"] = np.where(
        df["timestamp"].dt.normalize() <= train_end,
        "train",
        np.where(
            df["timestamp"].dt.normalize() <= valid_end,
            "validation",
            "test"
        )
    )

    # ---------------------------------------------------------
    # 20. FINAL VALIDATION
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED")
    print("=" * 60)

    print("Rows    :", f"{len(df):,}")
    print("Columns :", len(df.columns))
    print(
        "Houses  :",
        df["house_id"].nunique()
    )

    print(
        "Date start:",
        df["timestamp"].min()
    )

    print(
        "Date end  :",
        df["timestamp"].max()
    )

    print(
        "Occupancy :",
        "YES" if "occupancy" in df.columns else "NO"
    )

    print("\nSplit counts:")
    print(df["split"].value_counts())

    print("\nFinal missing values:")
    print(
        df.isna().sum()
        .loc[lambda x: x > 0]
    )

    return df


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Original CSV or XLSX dataset"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    raw = load_file(input_path)

    df = build_dataset(raw)

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        PROCESSED / "energy_model_data.csv"
    )

    profile_file = (
        PROCESSED / "data_profile.json"
    )

    df.to_csv(
        output_file,
        index=False
    )

    write_json(
        profile_file,
        {
            "rows": len(df),
            "columns": len(df.columns),
            "date_start": str(
                df["timestamp"].min()
            ),
            "date_end": str(
                df["timestamp"].max()
            ),
            "houses": int(
                df["house_id"].nunique()
            ),
            "occupancy_present":
                "occupancy" in df.columns,
            "split_counts":
                df["split"]
                .value_counts()
                .to_dict(),
            "target": TARGET
        }
    )

    print("\nSaved:")
    print(output_file)
    print(profile_file)


if __name__ == "__main__":
    main()