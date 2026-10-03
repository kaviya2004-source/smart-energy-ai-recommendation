"""XAI outputs (permutation importance + SHAP) and auditable recommendations.

Which model produces what
    recommendations.csv  ->  predictions of the DEPLOYED model (common.FINAL_MODEL,
                             currently LSTM). Read from reports/predictions_lstm.csv
                             and checked against the true target before use. If the
                             file is missing or cannot be aligned, the best tabular
                             ML model is used instead and 'prediction_model' says so.
    XAI outputs          ->  the best TABULAR ML model, because permutation
                             importance and SHAP TreeExplainer work on the
                             engineered feature table. They describe feature
                             drivers of that model, not the internal weights of
                             the LSTM. The report must say this explicitly.

Outputs (reports/)
    permutation_importance.csv, shap_global_importance.png / .csv (if shap works),
    xai_metadata.json, recommendations.csv, recommendations_metadata.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

sys.path.append(str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FINAL_MODEL, FINAL_MODEL_CHECKPOINT, HIGH_KWH_ALERT, MODELS, PROCESSED, RANDOM_STATE,
    REPORTS, STATUS_HIGH_RATIO, STATUS_LOW_RATIO, STATUS_SAVING_RATE, TARGET, energy_status,
    forecast_saving_kwh, normalise_id, write_json,
)

MODEL_FILES = {
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
}


# ---------------------------------------------------------------------------
# Recommendation rules
# ---------------------------------------------------------------------------
def build_recommendations(row: pd.Series) -> list[str]:
    """Rule-based tips. A rule fires only when the column really exists."""
    messages = []
    if float(row["prediction"]) >= HIGH_KWH_ALERT:
        messages.append("High predicted consumption: schedule non-essential appliance "
                        "use outside the peak tariff window.")

    occupancy = row.get("occupancy")
    if occupancy is not None and pd.notna(occupancy) and occupancy == 0:
        messages.append("No occupancy indicated: switch off standby appliances and "
                        "verify AC, heater and lighting schedules.")

    person_present = row.get("person_present")
    if person_present is not None and pd.notna(person_present) and person_present == 0:
        messages.append("YOLO reports no person: enable energy-saving mode for "
                        "non-critical loads after a safe delay.")

    temperature, ac = row.get("airtc_mean"), row.get("ac")
    if temperature is not None and ac is not None and temperature >= 30 and ac == 1:
        messages.append("Hot day with AC available: use a 24-26 C set point and close "
                        "blinds before peak afternoon heat.")

    lag1, rolling = row.get("kwh_lag_1"), row.get("kwh_rolling_7")
    if lag1 is not None and rolling is not None and rolling > 0 and lag1 > rolling * 1.2:
        messages.append("Recent use was above the 7-day baseline: check for a one-time "
                        "high-load appliance event.")

    if not messages:
        messages.append("Forecast is within the recent range: maintain existing appliance "
                        "scheduling and monitor the daily forecast.")
    return messages


# ---------------------------------------------------------------------------
# Prediction sources
# ---------------------------------------------------------------------------
def load_best_ml_model():
    ml_metrics = json.loads((REPORTS / "ml_metrics.json").read_text())
    winner = min(ml_metrics, key=lambda name: ml_metrics[name]["RMSE"])
    if winner not in MODEL_FILES:
        raise KeyError(f"Unknown model '{winner}' in ml_metrics.json. Known: {list(MODEL_FILES)}")
    path = MODELS / MODEL_FILES[winner]
    return winner, joblib.load(path)


def align_final_model_predictions(test: pd.DataFrame):
    """Return (Series of FINAL_MODEL predictions indexed like `test`, message)."""
    stem = FINAL_MODEL.lower().replace(" ", "_").replace("-", "_")
    path = REPORTS / f"predictions_{stem}.csv"
    if not path.exists():
        return None, f"predictions_{stem}.csv not found (train_dl.py must run first)"

    pred = pd.read_csv(path, parse_dates=["timestamp"])
    needed = {"timestamp", "house_id", "actual", "prediction"}
    if not needed.issubset(pred.columns):
        return None, f"{path.name} lacks columns {sorted(needed - set(pred.columns))}"

    pred["_hid"] = normalise_id(pred["house_id"])
    pred = pred.drop_duplicates(["timestamp", "_hid"])
    keys = test[["timestamp", "_hid", TARGET]].reset_index(names="row")

    # The prediction file may be stamped with the feature date (offset 0) or the
    # target date (offset 1). Accept an offset only if 'actual' matches the target.
    for offset in (0, 1):
        shifted = pred.assign(timestamp=pred["timestamp"] - pd.Timedelta(days=offset))
        merged = keys.merge(shifted[["timestamp", "_hid", "actual", "prediction"]],
                            on=["timestamp", "_hid"], how="left")
        matched = merged["prediction"].notna()
        coverage = float(matched.mean())
        if matched.sum() == 0:
            continue
        agreement = float(np.isclose(merged.loc[matched, "actual"],
                                     merged.loc[matched, TARGET], atol=1e-3).mean())
        if coverage >= 0.5 and agreement >= 0.99:
            series = pd.Series(merged["prediction"].to_numpy(), index=merged["row"])
            return series, f"aligned with offset {offset} day(s); coverage {coverage:.1%}"
    return None, f"{path.name} rows could not be verified against the true target"


# ---------------------------------------------------------------------------
# XAI
# ---------------------------------------------------------------------------
def write_permutation_importance(model, test: pd.DataFrame, features: list) -> None:
    result = permutation_importance(
        model, test[features], test[TARGET], scoring="neg_mean_absolute_error",
        n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1)
    (pd.DataFrame({"feature": features,
                   "importance_mean": result.importances_mean,
                   "importance_std": result.importances_std})
       .sort_values("importance_mean", ascending=False)
       .to_csv(REPORTS / "permutation_importance.csv", index=False))


def write_shap(model, test: pd.DataFrame, features: list) -> str:
    """Save SHAP bar chart + numeric table. Returns a status message."""
    status_file = REPORTS / "shap_status.txt"
    try:
        import shap

        sample = test[features].sample(min(500, len(test)), random_state=RANDOM_STATE)
        if hasattr(model, "named_steps") and {"preprocess", "model"} <= set(model.named_steps):
            transformed = model.named_steps["preprocess"].transform(sample)
            estimator = model.named_steps["model"]
            try:
                names = list(model.named_steps["preprocess"].get_feature_names_out())
            except Exception:
                names = list(features)
        else:
            transformed, estimator, names = sample, model, list(features)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        if len(names) != np.asarray(transformed).shape[1]:
            names = [f"feature_{i}" for i in range(np.asarray(transformed).shape[1])]

        explanation = shap.Explainer(estimator, transformed, feature_names=names)(transformed)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[:, :, 0]
        (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False)
           .to_csv(REPORTS / "shap_global_importance.csv", index=False))

        shap.plots.bar(explanation, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(REPORTS / "shap_global_importance.png", dpi=200, bbox_inches="tight")
        plt.close()
        status_file.unlink(missing_ok=True)
        return "SHAP created"
    except Exception as error:  # SHAP is optional; permutation importance remains.
        plt.close("all")
        message = (f"SHAP plot not created: {error}. "
                   "Use permutation_importance.csv as the XAI output.")
        status_file.write_text(message)
        return message


# ---------------------------------------------------------------------------
def main() -> None:
    df = pd.read_csv(PROCESSED / "energy_model_data.csv", parse_dates=["timestamp"])
    if "split" not in df.columns:
        raise KeyError("energy_model_data.csv needs a 'split' column (train/validation/test).")
    features = json.loads((MODELS / "tabular_features.json").read_text())
    test = df[df["split"] == "test"].copy().reset_index(drop=True)
    test["_hid"] = normalise_id(test["house_id"])

    ml_name, ml_model = load_best_ml_model()

    # ---- which model supplies the recommendation predictions? ----
    final_prediction, message = align_final_model_predictions(test)
    if final_prediction is not None:
        frame = test[test.index.isin(final_prediction.dropna().index)].copy()
        frame["prediction"] = final_prediction.loc[frame.index].to_numpy()
        source = f"{FINAL_MODEL} (deployed, {FINAL_MODEL_CHECKPOINT})"
    else:
        print(f"WARNING: {FINAL_MODEL} predictions not used ({message}). "
              f"Falling back to {ml_name}.")
        frame = test.copy()
        frame["prediction"] = ml_model.predict(frame[features])
        source = f"{ml_name} (fallback)"
    print(f"Recommendation predictions: {source} | {message}")

    # ---- recommendations ----
    output = frame[["timestamp", "house_id", "prediction", TARGET, "kwh_rolling_7"]].rename(
        columns={"kwh_rolling_7": "baseline_kwh"})
    output["energy_status"] = [energy_status(p, b) for p, b in
                               zip(output["prediction"], output["baseline_kwh"])]
    output["assumed_reduction_rate"] = output["energy_status"].map(STATUS_SAVING_RATE)
    output["estimated_saving_kwh"] = [
        round(forecast_saving_kwh(p, b, s), 3) for p, b, s in
        zip(output["prediction"], output["baseline_kwh"], output["energy_status"])]
    output["recommendations"] = frame.apply(
        lambda row: " | ".join(build_recommendations(row)), axis=1)
    output["prediction_model"] = source
    output = output.sort_values(["timestamp", "house_id"]).reset_index(drop=True)
    output.to_csv(REPORTS / "recommendations.csv", index=False)

    write_json(REPORTS / "recommendations_metadata.json", {
        "prediction_model": source,
        "alignment": message,
        "rows": int(len(output)),
        "status_high_ratio": STATUS_HIGH_RATIO,
        "status_low_ratio": STATUS_LOW_RATIO,
        "high_kwh_alert": HIGH_KWH_ALERT,
        "saving_rates": STATUS_SAVING_RATE,
        "saving_formula": "max(0, predicted - 7-day baseline) x assumed_reduction_rate",
        "note": "Saving is a rule-based assumption, not a measured field result.",
    })

    # ---- XAI on the tabular model ----
    write_permutation_importance(ml_model, test, features)
    shap_message = write_shap(ml_model, test, features)
    write_json(REPORTS / "xai_metadata.json", {
        "xai_model": ml_name,
        "xai_methods": ["permutation importance (neg. MAE, 8 repeats)", "SHAP"],
        "shap_status": shap_message,
        "deployed_prediction_model": source,
        "note": ("XAI explains the best tabular ML model built on the same engineered "
                 "features. It shows which inputs drive predictions; it is not a "
                 "causal explanation and does not open the LSTM itself."),
    })

    print(f"Done. {len(output):,} recommendation rows; XAI model: {ml_name}; {shap_message}")


if __name__ == "__main__":
    main()
