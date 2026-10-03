"""Shared paths, constants and business rules for the AI Smart Energy project.

Every script and the Streamlit dashboard import their thresholds, saving rates
and occupancy bands from THIS file. That keeps the code, the dashboard and the
project report consistent - change a number here and it changes everywhere.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
YOLO_REPORTS = ROOT / "data" / "yolo" / "reports"
YOLO_MODEL = MODELS / "yolo_person_detector.pt"
INTEGRATION_LOG = REPORTS / "integration" / "occupancy_energy_events.csv"

RANDOM_STATE = 42
TARGET = "target_next_day_kwh"

# Confirmed final / deployed forecasting model (2026-09-23). All scripts and the
# dashboard should treat this model's predictions as authoritative. Other models
# (BiLSTM, CNN-LSTM, CNN-BiLSTM-Attention, the tabular ML models) remain in the
# project only as a documented comparison in the report.
FINAL_MODEL = "LSTM"
FINAL_MODEL_CHECKPOINT = "lstm.pt"
FINAL_MODEL_CHECKPOINT_PATH = MODELS / FINAL_MODEL_CHECKPOINT
FINAL_MODEL_PREDICTIONS_PATH = REPORTS / "predictions_lstm.csv"
FINAL_MODEL_FILE = REPORTS / "final_model.json"

for directory in (PROCESSED, MODELS, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Season encoding (numeric code used by the deployed LSTM pipeline)
# ---------------------------------------------------------------------------
SEASON_NUMBER = {
    12: 1, 1: 1, 2: 1,
    3: 2, 4: 2, 5: 2,
    6: 3, 7: 3, 8: 3, 9: 3,
    10: 4, 11: 4,
}
SEASON_NAME = {1: "Winter", 2: "Summer", 3: "Monsoon", 4: "Post-monsoon"}


def season_number(month: int) -> int:
    return SEASON_NUMBER[int(month)]


def season_name(month: int) -> str:
    return SEASON_NAME[SEASON_NUMBER[int(month)]]


# ---------------------------------------------------------------------------
# Energy status and forecast-based saving estimate
# ---------------------------------------------------------------------------
STATUS_HIGH_RATIO = 1.20      # High   : predicted >= 120 % of the 7-day baseline
STATUS_LOW_RATIO = 0.85       # Low    : predicted <=  85 % of the 7-day baseline
HIGH_KWH_ALERT = 30.0         # absolute kWh level that triggers the "high use" tip
STATUS_SAVING_RATE = {"High": 0.10, "Normal": 0.05, "Low": 0.02}


def energy_status(predicted: float, baseline: float) -> str:
    """Classify a prediction against its 7-day baseline."""
    try:
        p, b = float(predicted), float(baseline)
    except (TypeError, ValueError):
        return "Normal"
    if math.isnan(p) or math.isnan(b) or b <= 0:
        return "Normal"
    if p >= b * STATUS_HIGH_RATIO:
        return "High"
    if p <= b * STATUS_LOW_RATIO:
        return "Low"
    return "Normal"


def forecast_saving_kwh(predicted: float, baseline: float, status: str) -> float:
    """Rule-based estimate: (excess over baseline) x assumed reduction rate.

    This is an assumption-based estimate, NOT a measured saving.
    """
    excess = max(0.0, float(predicted) - float(baseline))
    return excess * STATUS_SAVING_RATE.get(status, 0.0)


# ---------------------------------------------------------------------------
# YOLO occupancy layer (current human presence -> operational advice)
# ---------------------------------------------------------------------------
YOLO_CONFIDENCE = 0.25
PERSON_CLASS_ID = 0

# (max people inclusive, level, saving factor, advice)
OCCUPANCY_BANDS = (
    (0, "Empty room", 0.20,
     "No people detected. After a short safety delay, switch off non-essential "
     "lights, AC and standby appliances. Never switch off safety-critical devices."),
    (2, "Low occupancy", 0.10,
     "Low occupancy. Keep essential comfort systems active and switch off "
     "devices that are not being used."),
    (4, "Moderate occupancy", 0.05,
     "Moderate occupancy. Keep ventilation and comfort active, set the AC to "
     "24-26 C, use efficient lighting and avoid unnecessary high-load appliances."),
    (None, "High occupancy", 0.02,
     "High occupancy. Keep ventilation and safety systems active, but avoid "
     "unnecessary high-load equipment and optimise AC usage."),
)


def occupancy_profile(person_count: int) -> dict:
    """Return the occupancy level, saving factor and advice for a person count."""
    n = max(0, int(person_count))
    for limit, level, factor, advice in OCCUPANCY_BANDS:
        if limit is None or n <= limit:
            return {"level": level, "saving_factor": factor, "advice": advice}
    raise RuntimeError("OCCUPANCY_BANDS must end with an open-ended band")


def energy_advice(status: str, predicted_kwh: float) -> str:
    if status == "High":
        return (f"Predicted energy is high ({predicted_kwh:.2f} kWh), so prioritise "
                "load reduction without affecting safety or occupant comfort.")
    if status == "Low":
        return (f"Predicted energy is low ({predicted_kwh:.2f} kWh); maintain the "
                "current efficient usage pattern.")
    return (f"Predicted energy is normal ({predicted_kwh:.2f} kWh); continue "
            "normal monitoring and avoid idle loads.")


def occupancy_energy_impact(predicted_kwh: float, person_count: int) -> dict:
    """Occupancy-based scenario: potential saving = forecast x saving factor."""
    profile = occupancy_profile(person_count)
    saving = float(predicted_kwh) * profile["saving_factor"]
    return {
        "level": profile["level"],
        "saving_factor": profile["saving_factor"],
        "saving_percent": profile["saving_factor"] * 100,
        "potential_saving_kwh": saving,
        "optimized_kwh": max(0.0, float(predicted_kwh) - saving),
        "advice": profile["advice"],
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def normalise_id(series: pd.Series) -> pd.Series:
    """Make house IDs comparable across files (1, 1.0 and '1' all become '1')."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and (numeric == numeric.round()).all():
        return numeric.astype("int64").astype(str)
    return series.astype(str)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_json(path: Path) -> dict:
    """Read a JSON report; returns {} if the file does not exist yet."""
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))
