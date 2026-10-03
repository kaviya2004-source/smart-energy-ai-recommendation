"""Weighted deep-learning hybrid + fair comparison against simple baselines.

This script does NOT train a network. It blends the saved TEST/VALIDATION
predictions of LSTM, BiLSTM and CNN-LSTM:

  1. Blend weights are searched on the VALIDATION split only (0.05 steps).
  2. The chosen weights are applied once to the held-out TEST split.
  3. Every model, the hybrid and two naive baselines are scored on the SAME
     test rows, so the comparison is fair.

The FINAL model of the project is fixed in common.py (FINAL_MODEL = "LSTM").
This script records that decision together with the evidence in
reports/final_model.json, and WARNS if another compared model has a lower RMSE,
so the report never claims something the numbers do not support.

Outputs
    reports/final_model.json              (new: single source for dashboard + report)
    reports/predictions_hybrid_dl.csv
    reports/hybrid_metrics.json
    reports/dl_model_comparison.csv       (new: full comparison incl. baselines)
    models/hybrid_dl_config.json

reports/all_model_comparison.csv is NOT modified by this script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FINAL_MODEL, FINAL_MODEL_FILE, MODELS, PROCESSED, REPORTS, TARGET, normalise_id, write_json,
)

BLEND_MODELS = ["LSTM", "BiLSTM", "CNN-LSTM"]
HYBRID_NAME = "LSTM + BiLSTM + CNN-LSTM Hybrid"
WEIGHT_UNITS = 20                      # 1 / 20 = 0.05 weight step
KEY = ["timestamp", "house_id"]


def calculate_metrics(actual, prediction) -> dict:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    smape = 2 * np.abs(actual - prediction) / np.clip(np.abs(actual) + np.abs(prediction), 1e-6, None)
    return {
        "MAE": float(mean_absolute_error(actual, prediction)),
        "RMSE": float(mean_squared_error(actual, prediction) ** 0.5),
        "MAPE": float(np.mean(np.abs(actual - prediction) / np.clip(np.abs(actual), 0.1, None)) * 100),
        "sMAPE": float(np.mean(smape) * 100),
        "R2": float(r2_score(actual, prediction)),
    }


# ---------------------------------------------------------------------------
# Loading and aligning prediction files
# ---------------------------------------------------------------------------
def prediction_path(model_name: str, split: str) -> Path:
    stem = model_name.lower().replace(" ", "_")
    suffix = "_validation" if split == "validation" else ""
    for name in dict.fromkeys([stem, stem.replace("-", "_")]):
        path = REPORTS / f"predictions_{name}{suffix}.csv"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing {split} prediction file for {model_name} in {REPORTS}. Run train_dl.py first."
    )


def load_prediction(model_name: str, split: str) -> pd.DataFrame:
    path = prediction_path(model_name, split)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = [c for c in ("timestamp", "house_id", "actual", "prediction") if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    df = df[["timestamp", "house_id", "actual", "prediction"]].copy()
    df["house_id"] = normalise_id(df["house_id"])
    return df.drop_duplicates(KEY)


def prepare_models(model_names: list, split: str) -> pd.DataFrame:
    """Inner-join the predictions of all models on (timestamp, house_id)."""
    merged = None
    for name in model_names:
        frame = load_prediction(name, split).rename(
            columns={"prediction": name, "actual": f"actual__{name}"})
        merged = frame if merged is None else merged.merge(frame, on=KEY, how="inner")
    if merged.empty:
        raise ValueError(f"No common {split} rows across {model_names}.")

    first = f"actual__{model_names[0]}"
    for name in model_names[1:]:
        if not np.allclose(merged[first], merged[f"actual__{name}"], atol=1e-6):
            raise ValueError(f"'actual' values differ between {model_names[0]} and {name} ({split}).")

    result = merged[KEY + [first]].rename(columns={first: "actual"})
    for name in model_names:
        result[name] = merged[name].to_numpy()
    return result.sort_values(KEY).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Weight search (validation only)
# ---------------------------------------------------------------------------
def search_weights(validation: pd.DataFrame) -> dict:
    actual = validation["actual"].to_numpy()
    columns = [validation[m].to_numpy() for m in BLEND_MODELS]
    best = None
    for i in range(1, WEIGHT_UNITS - 1):
        for j in range(1, WEIGHT_UNITS - i):
            k = WEIGHT_UNITS - i - j
            weights = (i / WEIGHT_UNITS, j / WEIGHT_UNITS, k / WEIGHT_UNITS)
            blend = sum(w * col for w, col in zip(weights, columns))
            rmse = float(np.sqrt(np.mean((actual - blend) ** 2)))
            if best is None or rmse < best["RMSE"]:
                best = {"weights": weights, "RMSE": rmse}
    best["R2"] = float(r2_score(actual, sum(w * c for w, c in zip(best["weights"], columns))))
    return best


# ---------------------------------------------------------------------------
# Naive baselines (are the deep models really better than trivial rules?)
# ---------------------------------------------------------------------------
def load_baselines(test: pd.DataFrame) -> tuple:
    path = PROCESSED / "energy_model_data.csv"
    if not path.exists():
        return {}, "energy_model_data.csv not found"
    available = pd.read_csv(path, nrows=0).columns
    wanted = [c for c in ("timestamp", "house_id", "kwh", "kwh_rolling_7", TARGET) if c in available]
    if not {"timestamp", "house_id", TARGET}.issubset(wanted):
        return {}, f"timestamp/house_id/{TARGET} not all present"

    data = pd.read_csv(path, usecols=wanted, parse_dates=["timestamp"])
    data["house_id"] = normalise_id(data["house_id"])
    merged = test[KEY + ["actual"]].merge(data, on=KEY, how="left")

    agreement = float(np.isclose(merged["actual"], merged[TARGET], atol=1e-3).mean())
    if agreement < 0.99:
        return {}, f"test rows only {agreement:.1%} aligned with {TARGET}"

    baselines = {}
    if "kwh" in merged:
        baselines["Naive baseline (tomorrow = today)"] = merged["kwh"].to_numpy()
    if "kwh_rolling_7" in merged:
        baselines["7-day moving-average baseline"] = merged["kwh_rolling_7"].to_numpy()
    return baselines, ""


def score_row(name: str, kind: str, actual, prediction) -> dict:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    keep = np.isfinite(actual) & np.isfinite(prediction)
    return {"Model": name, "Type": kind, "Test_rows": int(keep.sum()),
            **calculate_metrics(actual[keep], prediction[keep])}


def write_final_model_info(comparison: pd.DataFrame, common_rows: int) -> None:
    """Save the final-model decision and its evidence to reports/final_model.json."""
    own = load_prediction(FINAL_MODEL, "test")             # the model's full test file
    metrics = calculate_metrics(own["actual"], own["prediction"])

    ranked = comparison.sort_values("RMSE").reset_index(drop=True)
    lowest = ranked.iloc[0]["Model"]
    rank = int(ranked.index[ranked["Model"] == FINAL_MODEL][0]) + 1 if (ranked["Model"] == FINAL_MODEL).any() else None

    info = {
        "final_model": FINAL_MODEL,
        "checkpoint": "models/lstm.pt",
        "task": "next-day household energy (kWh) forecasting",
        "test_rows": int(len(own)),
        "test_metrics": metrics,
        "deep_learning_comparison": {
            "rows_used": int(common_rows),
            "rank_by_rmse": rank,
            "models_compared": int(len(ranked)),
            "lowest_rmse_model": lowest,
        },
    }

    # Read-only look at the earlier ML-vs-DL table, if it exists.
    all_path = REPORTS / "all_model_comparison.csv"
    if all_path.exists():
        table = pd.read_csv(all_path)
        if {"Model", "RMSE"}.issubset(table.columns):
            table = table.sort_values("RMSE").reset_index(drop=True)
            match = table.index[table["Model"].astype(str).str.upper() == FINAL_MODEL.upper()]
            info["all_model_comparison"] = {
                "rank_by_rmse": int(match[0]) + 1 if len(match) else None,
                "models_compared": int(len(table)),
                "lowest_rmse_model": str(table.iloc[0]["Model"]),
            }

    write_json(FINAL_MODEL_FILE, info)

    print(f"\nFINAL MODEL: {FINAL_MODEL} | test RMSE {metrics['RMSE']:.4f} | R2 {metrics['R2']:.4f}")
    if lowest != FINAL_MODEL:
        print(f"WARNING: '{lowest}' has a LOWER test RMSE than {FINAL_MODEL} in this comparison. "
              f"Do not write '{FINAL_MODEL} has the lowest RMSE' in the report unless "
              "all_model_comparison.csv confirms it.")
    other = info.get("all_model_comparison")
    if other and other["lowest_rmse_model"].upper() != FINAL_MODEL.upper():
        print(f"WARNING: all_model_comparison.csv lists '{other['lowest_rmse_model']}' "
              f"with the lowest RMSE, not {FINAL_MODEL}.")


def main() -> None:
    print("=" * 70)
    print("DEEP-LEARNING HYBRID (weight blending) + BASELINE COMPARISON")
    print("=" * 70)

    # 1. weights from validation only
    validation = prepare_models(BLEND_MODELS, "validation")
    best = search_weights(validation)
    w_lstm, w_bilstm, w_cnn = best["weights"]
    print(f"\nValidation rows : {len(validation):,}")
    print(f"Best weights    : LSTM {w_lstm:.2f} | BiLSTM {w_bilstm:.2f} | CNN-LSTM {w_cnn:.2f}")
    print(f"Validation RMSE : {best['RMSE']:.6f}")

    # 2. one-shot evaluation on the held-out test split
    test = prepare_models(BLEND_MODELS, "test")
    hybrid = w_lstm * test["LSTM"] + w_bilstm * test["BiLSTM"] + w_cnn * test["CNN-LSTM"]
    hybrid_metrics = calculate_metrics(test["actual"], hybrid)

    pd.DataFrame({"timestamp": test["timestamp"], "house_id": test["house_id"],
                  "actual": test["actual"], "prediction": hybrid}
                 ).to_csv(REPORTS / "predictions_hybrid_dl.csv", index=False)

    write_json(REPORTS / "hybrid_metrics.json", {
        HYBRID_NAME: {
            **hybrid_metrics,
            "model_1": "LSTM", "model_2": "BiLSTM", "model_3": "CNN-LSTM",
            "weight_1": w_lstm, "weight_2": w_bilstm, "weight_3": w_cnn,
            "validation_RMSE": best["RMSE"],
        }
    })
    write_json(MODELS / "hybrid_dl_config.json", {
        "model_type": "Hybrid Deep Learning (weighted average of predictions)",
        "models": BLEND_MODELS,
        "weights": {"LSTM": w_lstm, "BiLSTM": w_bilstm, "CNN-LSTM": w_cnn},
        "selection_method": "Validation RMSE weight search, step 0.05",
        "test_rows": int(len(test)),
    })

    # 3. full comparison on identical test rows
    rows = [score_row(m, "Deep learning", test["actual"], test[m]) for m in BLEND_MODELS]
    rows.append(score_row(HYBRID_NAME, "Hybrid (DL ensemble)", test["actual"], hybrid))

    baselines, reason = load_baselines(test)
    for name, values in baselines.items():
        rows.append(score_row(name, "Baseline", test["actual"], values))
    if reason:
        print(f"\nBaselines skipped: {reason}")

    comparison = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    comparison.to_csv(REPORTS / "dl_model_comparison.csv", index=False)

    print("\nHeld-out TEST comparison (same rows for every model)")
    print(comparison[["Model", "Type", "MAE", "RMSE", "R2"]].round(4).to_string(index=False))
    print(f"\nLowest test RMSE: {comparison.iloc[0]['Model']}")
    write_final_model_info(comparison, len(test))
    print("\nSaved: final_model.json, predictions_hybrid_dl.csv, hybrid_metrics.json, "
          "dl_model_comparison.csv, hybrid_dl_config.json")
    print("all_model_comparison.csv was not modified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
