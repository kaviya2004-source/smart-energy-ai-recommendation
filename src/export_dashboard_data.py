"""Create clean, Power BI-ready tables from the completed project outputs.

Missing optional inputs are skipped with a message instead of crashing.
Output folder: reports/powerbi/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FINAL_MODEL_FILE, PROCESSED, REPORTS, YOLO_REPORTS, load_json, season_name,
)

POWERBI = REPORTS / "powerbi"
POWERBI.mkdir(exist_ok=True)

# report files copied unchanged (name -> required?)
REPORT_FILES = {
    "all_model_comparison.csv": True,
    "recommendations.csv": True,
    "permutation_importance.csv": False,
    "shap_global_importance.csv": False,
    "dl_model_comparison.csv": False,
    "future_energy_forecasts.csv": False,
    "future_forecast_summary.csv": False,
}


def summarise(energy: pd.DataFrame, by: str, filename: str) -> None:
    (energy.groupby(by, as_index=False)
           .agg(mean_kwh=("kwh", "mean"), total_kwh=("kwh", "sum"), records=("kwh", "size"))
           .to_csv(POWERBI / filename, index=False))


def main() -> None:
    energy = pd.read_csv(PROCESSED / "energy_model_data.csv", parse_dates=["timestamp"])
    energy["year"] = energy["timestamp"].dt.year
    energy["month"] = energy["timestamp"].dt.to_period("M").astype(str)
    energy["season"] = energy["timestamp"].dt.month.map(season_name)

    summarise(energy, "timestamp", "energy_daily_summary.csv")
    summarise(energy, "month", "energy_monthly_summary.csv")
    summarise(energy, "year", "energy_yearly_summary.csv")
    summarise(energy, "season", "energy_seasonal_summary.csv")

    for name, required in REPORT_FILES.items():
        source = REPORTS / name
        if source.exists():
            pd.read_csv(source).to_csv(POWERBI / name, index=False)
        elif required:
            raise FileNotFoundError(f"Required file missing: {source}. Run the pipeline first.")
        else:
            print(f"Skipped (not found): {name}")

    final = load_json(FINAL_MODEL_FILE)
    if final:
        pd.DataFrame([{"final_model": final["final_model"], "test_rows": final["test_rows"],
                       **final["test_metrics"]}]).to_csv(POWERBI / "final_model_summary.csv", index=False)
    else:
        print("Skipped (not found): final_model.json - run compare_models.py first")

    yolo_path = YOLO_REPORTS / "test_metrics_summary.json"
    if yolo_path.exists():
        yolo = json.loads(yolo_path.read_text(encoding="utf-8"))
        pd.DataFrame([yolo]).to_csv(POWERBI / "yolo_test_metrics.csv", index=False)
    else:
        print("Skipped (not found): YOLO test_metrics_summary.json")

    print(f"Power BI tables exported to {POWERBI}")


if __name__ == "__main__":
    main()
