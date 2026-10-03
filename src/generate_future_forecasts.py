"""Generate recursive future energy forecasts with the deployed LSTM.

Usage (from the project root):
    python src/generate_future_forecasts.py                 # latest date -> 31-Dec-2027
    python src/generate_future_forecasts.py 30              # next 30 days only
    python src/generate_future_forecasts.py --weather latest

How it works
    The LSTM predicts NEXT-DAY kWh from the last `sequence_days` daily rows.
    For multi-day forecasts the prediction is fed back as tomorrow's history
    (recursive forecasting). Because errors accumulate, long horizons are
    SCENARIO PROJECTIONS, not validated accuracy claims.

Weather modes
    climatology (default) : future weather = average of that calendar day in the
                            historical data, so seasonality is preserved.
    latest                : future weather stays at the last observed values.

Outputs (reports/)
    future_energy_forecasts.csv, future_forecast_summary.csv,
    future_forecast_metadata.json
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FINAL_MODEL, PROCESSED, REPORTS, energy_status, forecast_saving_kwh, season_number, write_json,
)
from lstm_net import detect_lag_convention, load_lstm_checkpoint  # noqa: E402

WEATHER_PREFIXES = ("airtc", "rh_", "bp_", "ws_", "slrkw")
DERIVED_WEATHER = {
    "temperature_range", "humidity_range", "pressure_range",
    "wind_speed_range", "wind_gust_range", "solar_variability",
}
END_DATE = pd.Timestamp("2027-12-31")


# ---------------------------------------------------------------------------
# Lag convention: detect_lag_convention() lives in lstm_net.py (shared with the dashboard)
# ---------------------------------------------------------------------------
def lag_values(history: list, conv: dict) -> tuple:
    """Lag features for the row whose own kWh is history[-1]."""
    lag1 = history[-1 - conv["lag1_back"]]
    lag7 = history[-1 - conv["lag7_back"]]
    window = history[-7:] if conv["rolling_inclusive"] else history[-8:-1]
    return lag1, lag7, float(np.mean(window))


# ---------------------------------------------------------------------------
# Building a synthetic future row
# ---------------------------------------------------------------------------
def build_weather_climatology(df: pd.DataFrame, weather_cols: list) -> pd.DataFrame:
    day = df["timestamp"].dt.dayofyear.clip(upper=365)
    table = df.groupby(day)[weather_cols].mean()
    return table.reindex(range(1, 366)).interpolate(limit_direction="both")


def update_temporal_features(row: pd.Series, date: pd.Timestamp) -> pd.Series:
    row = row.copy()
    values = {
        "timestamp": date,
        "year": date.year,
        "month": date.month,
        "day_of_week": date.dayofweek,
        "is_weekend": int(date.dayofweek >= 5),
        "day_of_year": date.dayofyear,
        "week_of_year": int(date.isocalendar().week),
        "quarter": date.quarter,
        "season": season_number(date.month),
        "month_sin": np.sin(2 * np.pi * date.month / 12),
        "month_cos": np.cos(2 * np.pi * date.month / 12),
        "day_of_week_sin": np.sin(2 * np.pi * date.dayofweek / 7),
        "day_of_week_cos": np.cos(2 * np.pi * date.dayofweek / 7),
        "day_of_year_sin": np.sin(2 * np.pi * date.dayofyear / 365.25),
        "day_of_year_cos": np.cos(2 * np.pi * date.dayofyear / 365.25),
    }
    for column, value in values.items():
        if column in row.index:
            row[column] = value
    return row


def make_row(template: pd.Series, date: pd.Timestamp, history: list, conv: dict,
             climatology: pd.DataFrame | None, weather_cols: list) -> pd.Series:
    """Feature row for `date`, whose own kWh is history[-1]."""
    row = update_temporal_features(template, date)

    if climatology is not None:
        for column in weather_cols:
            row[column] = climatology.at[min(date.dayofyear, 365), column]

    lag1, lag7, rolling = lag_values(history, conv)
    for column, value in (("kwh_lag_1", lag1), ("kwh_lag_7", lag7), ("kwh_rolling_7", rolling)):
        if column in row.index:
            row[column] = value
    if "lag_energy_difference" in row.index:
        row["lag_energy_difference"] = lag1 - lag7
    if "lag_energy_ratio" in row.index:
        row["lag_energy_ratio"] = lag1 / lag7 if lag7 else 0.0
    if "kwh" in row.index:
        row["kwh"] = history[-1]
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("days", nargs="?", type=int, help="forecast horizon in days (default: until 31-Dec-2027)")
    parser.add_argument("--weather", choices=["climatology", "latest"], default="climatology")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(PROCESSED / "energy_model_data.csv", parse_dates=["timestamp"])
    if df.empty:
        raise RuntimeError("energy_model_data.csv is empty.")
    df = df.sort_values(["house_id", "timestamp"]).reset_index(drop=True)

    bundle = load_lstm_checkpoint()
    features, sequence_days = bundle.features, bundle.sequence_days
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise RuntimeError(f"Columns required by the LSTM are missing from the data: {missing}")

    conv = detect_lag_convention(df)
    print(f"Lag convention: lag_1 back={conv['lag1_back']}, lag_7 back={conv['lag7_back']}, "
          f"rolling includes today={conv['rolling_inclusive']}")

    weather_cols = [c for c in features
                    if c.startswith(WEATHER_PREFIXES) or c in DERIVED_WEATHER]
    climatology = None
    if args.weather == "climatology" and weather_cols:
        climatology = build_weather_climatology(df, weather_cols)
    weather_mode = "climatology" if climatology is not None else "latest"
    print(f"Weather mode: {weather_mode} ({len(weather_cols)} weather columns)")

    latest_observed = df["timestamp"].max().normalize()
    if args.days is not None:
        if args.days < 1:
            raise ValueError("Forecast days must be at least 1.")
        days = args.days
    else:
        days = max(1, (END_DATE - latest_observed).days)

    need_rows = max(sequence_days, 8)
    outputs = []
    houses = df["house_id"].unique()

    for number, house_id in enumerate(houses, start=1):
        group = df[df["house_id"] == house_id].tail(max(sequence_days, 14) + 1)
        if len(group) < need_rows:
            print(f"Skipping house {house_id}: only {len(group)} rows.")
            continue

        history = [float(v) for v in group["kwh"]]
        template = group.iloc[-1]
        current_date = pd.Timestamp(template["timestamp"])

        # The window is the last `sequence_days` REAL, consecutive daily rows.
        window = [group.iloc[i][features].astype(float).to_numpy()
                  for i in range(len(group) - sequence_days, len(group))]

        for step in range(1, days + 1):
            predicted = bundle.predict(pd.DataFrame(np.array(window), columns=features))

            baseline = float(np.mean(history[-7:]))
            status = energy_status(predicted, baseline)
            forecast_date = current_date + pd.Timedelta(days=1)

            outputs.append({
                "house_id": house_id,
                "forecast_date": forecast_date,
                "horizon_day": step,
                "predicted_kwh": round(predicted, 4),
                "baseline_kwh": round(baseline, 4),
                "energy_status": status,
                "estimated_saving_kwh": round(forecast_saving_kwh(predicted, baseline, status), 3),
            })

            # Recursive step: the prediction becomes tomorrow's history.
            history.append(predicted)
            row = make_row(template, forecast_date, history, conv, climatology, weather_cols)
            window = window[1:] + [row[features].astype(float).to_numpy()]
            template = row
            current_date = forecast_date

        if number % 10 == 0 or number == len(houses):
            print(f"  processed {number}/{len(houses)} houses")

    out = pd.DataFrame(outputs)
    if out.empty:
        raise RuntimeError("No future forecast rows were generated.")
    out["forecast_date"] = pd.to_datetime(out["forecast_date"])
    out = out.sort_values(["house_id", "forecast_date"]).reset_index(drop=True)

    forecast_path = REPORTS / "future_energy_forecasts.csv"
    out.to_csv(forecast_path, index=False)
    (
        out.groupby("forecast_date", as_index=False)
        .agg(predicted_kwh=("predicted_kwh", "mean"),
             total_predicted_kwh=("predicted_kwh", "sum"))
        .to_csv(REPORTS / "future_forecast_summary.csv", index=False)
    )

    write_json(REPORTS / "future_forecast_metadata.json", {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": f"{FINAL_MODEL} (models/lstm.pt)",
        "sequence_days": sequence_days,
        "last_observed_date": latest_observed.date(),
        "first_forecast_date": out["forecast_date"].min().date(),
        "last_forecast_date": out["forecast_date"].max().date(),
        "houses": int(out["house_id"].nunique()),
        "rows": int(len(out)),
        "weather_mode": weather_mode,
        "lag_convention": conv,
        "note": ("Recursive multi-day forecasts are scenario projections. Only the "
                 "next-day forecast is validated on the held-out test set; uncertainty "
                 "grows with horizon_day."),
    })

    counts = out["forecast_date"].dt.year.value_counts().sort_index()
    print(f"Saved {len(out):,} forecast rows using the deployed LSTM")
    print(f"Forecast range: {out['forecast_date'].min():%Y-%m-%d} to {out['forecast_date'].max():%Y-%m-%d}")
    for year, count in counts.items():
        print(f"  {year}: {int(count):,} rows")
    print(f"Output: {forecast_path}")


if __name__ == "__main__":
    main()
