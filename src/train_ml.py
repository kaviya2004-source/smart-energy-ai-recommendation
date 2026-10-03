"""Train the three classical regressors with identical chronology-safe inputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parent))
from common import MODELS, PROCESSED, RANDOM_STATE, REPORTS, TARGET, write_json


def metrics(y, pred) -> dict:
    return {"MAE": float(mean_absolute_error(y, pred)), "RMSE": float(mean_squared_error(y, pred) ** .5),
            "MAPE": float((abs((y - pred) / y.clip(lower=.1))).mean() * 100), "R2": float(r2_score(y, pred))}


def main() -> None:
    df = pd.read_csv(PROCESSED / "energy_model_data.csv", parse_dates=["timestamp"])
    drop = [TARGET, "timestamp", "split", "kwh"]  # kwh is today's measured use; lags are the allowed history.
    features = [c for c in df.columns if c not in drop]
    cat = df[features].select_dtypes(include="object").columns.tolist()
    num = [c for c in features if c not in cat]
    prep = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), num),
                              ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
    estimators = {
        "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=500, max_depth=7, learning_rate=.04, subsample=.8,
                                colsample_bytree=.8, objective="reg:squarederror", n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=.04,
                                                          loss="huber", random_state=RANDOM_STATE),
    }
    train, validation, test = df[df.split == "train"], df[df.split == "validation"], df[df.split == "test"]
    results = {}
    for name, estimator in estimators.items():
        model = Pipeline([("preprocess", prep), ("model", estimator)])
        model.fit(train[features], train[TARGET])
        validation_prediction = model.predict(validation[features])
        prediction = model.predict(test[features])
        results[name] = metrics(test[TARGET], prediction)
        joblib.dump(model, MODELS / f"{name.lower().replace(' ', '_')}.joblib")
        pd.DataFrame({"actual": test[TARGET], "prediction": prediction, "timestamp": test.timestamp, "house_id": test.house_id}).to_csv(REPORTS / f"predictions_{name.lower().replace(' ', '_')}.csv", index=False)
        pd.DataFrame({"actual": validation[TARGET], "prediction": validation_prediction, "timestamp": validation.timestamp, "house_id": validation.house_id}).to_csv(REPORTS / f"predictions_{name.lower().replace(' ', '_')}_validation.csv", index=False)
        print(name, results[name])
    write_json(REPORTS / "ml_metrics.json", results)
    (MODELS / "tabular_features.json").write_text(json.dumps(features), encoding="utf-8")


if __name__ == "__main__":
    main()
