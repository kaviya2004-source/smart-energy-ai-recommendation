"""Feature engineering for the smart-building energy dataset.

Input : data/processed/energy_model_data.csv
Output: data/processed/energy_features.csv      (the input file is NOT modified)

What it adds (only when the source columns exist)
    temporal    week_of_year, quarter, season (numeric 1-4, same code as the
                deployed pipeline), month / day-of-week / day-of-year sin & cos
    weather     temperature / humidity / pressure / wind ranges, solar variability
    building    area per room, area per floor
    appliances  total appliance load, active appliance count
    lags        lag_energy_difference, lag_energy_ratio
    occupancy   interaction features ONLY if USE_OCCUPANCY_FEATURES = True

Design rule of the project: current human presence comes from YOLOv8, so the
historical forecasting model does not use occupancy by default.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from common import PROCESSED, season_number  # noqa: E402

INPUT_FILE = PROCESSED / "energy_model_data.csv"
OUTPUT_FILE = PROCESSED / "energy_features.csv"

USE_OCCUPANCY_FEATURES = False

APPLIANCE_COLUMNS = [
    "dishwasher", "washing_machine", "boiler", "electric_heater", "fan", "ac",
    "oven", "microwave", "eac", "fridge", "refrigratore", "tv", "play", "shofaz",
]

RANGE_FEATURES = {          # new column: (max column, min column)
    "temperature_range": ("airtc_max", "airtc_min"),
    "humidity_range": ("rh_max", "rh_min"),
    "pressure_range": ("bp_mbar_max", "bp_mbar_min"),
    "wind_speed_range": ("ws_ms_avg_max", "ws_ms_avg_min"),
    "wind_gust_range": ("ws_gust_max_max", "ws_gust_max_min"),
    "solar_variability": ("slrkw_avg_max", "slrkw_avg_mean"),
}


def has(df: pd.DataFrame, *columns: str) -> bool:
    return all(c in df.columns for c in columns)


def add_calendar_columns(df: pd.DataFrame) -> None:
    """Create basic calendar columns only if the source file lacks them."""
    ts = df["timestamp"]
    defaults = {
        "year": ts.dt.year,
        "month": ts.dt.month,
        "day_of_week": ts.dt.dayofweek,
        "day_of_year": ts.dt.dayofyear,
    }
    for column, values in defaults.items():
        if column not in df.columns:
            df[column] = values
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)


def engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Add the engineered columns. Returns (dataframe, names of columns written)."""
    written: list[str] = []

    def put(name: str, values) -> None:
        df[name] = values
        if name not in written:
            written.append(name)

    add_calendar_columns(df)

    # 1. temporal
    put("week_of_year", df["timestamp"].dt.isocalendar().week.astype(int))
    put("quarter", df["timestamp"].dt.quarter)
    put("season", df["month"].map(season_number))

    # 2. cyclical time
    for name, values, period in (
        ("month", df["month"], 12),
        ("day_of_week", df["day_of_week"], 7),
        ("day_of_year", df["day_of_year"], 365.25),
    ):
        put(f"{name}_sin", np.sin(2 * np.pi * values / period))
        put(f"{name}_cos", np.cos(2 * np.pi * values / period))

    # 3. weather ranges and solar variability
    for new, (upper, lower) in RANGE_FEATURES.items():
        if has(df, upper, lower):
            put(new, df[upper] - df[lower])

    # 4. building size
    if has(df, "area", "noof_rooms"):
        put("area_per_room", df["area"] / df["noof_rooms"].replace(0, np.nan))
    if has(df, "area", "noof_floors"):
        put("area_per_floor", df["area"] / df["noof_floors"].replace(0, np.nan))

    # 5. appliances
    appliances = [c for c in APPLIANCE_COLUMNS if c in df.columns]
    if appliances:
        put("total_appliance_load", df[appliances].sum(axis=1))
        put("active_appliance_count", (df[appliances] > 0).sum(axis=1))

    # 6. occupancy interactions (off by default - see module docstring)
    if USE_OCCUPANCY_FEATURES and "occupancy" in df.columns:
        if "area" in df.columns:
            put("occupancy_area_interaction", df["occupancy"] * df["area"])
        if "total_appliance_load" in df.columns:
            put("occupancy_appliance_interaction", df["occupancy"] * df["total_appliance_load"])

    # 7. lag interaction
    if has(df, "kwh_lag_1", "kwh_lag_7"):
        put("lag_energy_difference", df["kwh_lag_1"] - df["kwh_lag_7"])
        put("lag_energy_ratio", df["kwh_lag_1"] / df["kwh_lag_7"].replace(0, np.nan))
    return df, written


def main() -> None:
    print("=" * 65)
    print("FEATURE ENGINEERING")
    print("=" * 65)

    df = pd.read_csv(INPUT_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["house_id", "timestamp"]).reset_index(drop=True)
    print(f"Input shape : {df.shape}")

    original_columns = list(df.columns)
    original_numeric = df.select_dtypes(include=np.number).columns.tolist()

    df, new_columns = engineer(df)
    added = [c for c in df.columns if c not in original_columns]

    # Ratios can create inf/NaN. Clean ONLY the newly created columns.
    df[new_columns] = df[new_columns].replace([np.inf, -np.inf], np.nan)
    new_numeric = df[new_columns].select_dtypes(include=np.number).columns
    df[new_numeric] = df[new_numeric].fillna(0)

    # Original columns are never silently zero-filled; just report problems.
    leftover = df[original_numeric].isna().sum()
    leftover = leftover[leftover > 0]
    if not leftover.empty:
        print("\nWARNING: original columns still contain missing values "
              "(they were NOT filled):")
        print(leftover.to_string())

    before = len(df)
    df = df.drop_duplicates(subset=["house_id", "timestamp"], keep="first")
    duplicates_removed = before - len(df)
    df = df.sort_values(["house_id", "timestamp"]).reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 65)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 65)
    print(f"Rows                 : {len(df):,}")
    print(f"Columns              : {len(df.columns)} ({len(added)} added, "
          f"{len([c for c in new_columns if c in original_columns])} recomputed)")
    print(f"Houses               : {df['house_id'].nunique()}")
    print(f"Date range           : {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Occupancy in data    : {'YES' if 'occupancy' in df.columns else 'NO'}"
          f" | occupancy features used: {USE_OCCUPANCY_FEATURES}")
    print(f"Duplicates removed   : {duplicates_removed}")
    print("\nEngineered columns (* = already existed, recomputed):")
    for column in new_columns:
        print(f"  {'*' if column in original_columns else '+'} {column}")
    print(f"\nSaved to: {OUTPUT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
