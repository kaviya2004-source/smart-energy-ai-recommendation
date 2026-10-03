from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

sys.path.append(str(Path(__file__).resolve().parent))

from common import REPORTS, MODELS


# =========================================================
# METRICS
# =========================================================
def calculate_metrics(actual, prediction):
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    return {
        "MAE": float(mean_absolute_error(actual, prediction)),
        "RMSE": float(
            mean_squared_error(actual, prediction) ** 0.5
        ),
        "MAPE": float(
            np.mean(
                np.abs(
                    (actual - prediction)
                    / np.clip(np.abs(actual), 0.1, None)
                )
            ) * 100
        ),
        "R2": float(r2_score(actual, prediction)),
    }


# =========================================================
# LOAD DL PREDICTION FILE
# =========================================================
def load_prediction(model_name, split):

    stem = model_name.lower().replace(" ", "_")

    filename = f"predictions_{stem}"

    if split == "validation":
        filename += "_validation"

    path = REPORTS / f"{filename}.csv"

    # Compatibility with the existing CNN-LSTM filename
    if not path.exists() and model_name == "CNN-LSTM":
        filename = "predictions_cnn-lstm"
        if split == "validation":
            filename += "_validation"

        path = REPORTS / f"{filename}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{path}\n\n"
            f"Run train_dl.py before train_hybrid.py."
        )

    df = pd.read_csv(
        path,
        parse_dates=["timestamp"]
    )

    required = [
        "timestamp",
        "house_id",
        "actual",
        "prediction",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {missing}"
        )

    return df[
        [
            "timestamp",
            "house_id",
            "actual",
            "prediction",
        ]
    ].copy()


# =========================================================
# CREATE COMMON DATASET
# =========================================================
def prepare_models(model_names, split):

    datasets = {}

    for name in model_names:

        df = load_prediction(
            name,
            split
        )

        df["key"] = (
            df["timestamp"].astype(str)
            + "_"
            + df["house_id"].astype(str)
        )

        datasets[name] = df.set_index("key")

    common_keys = None

    for df in datasets.values():

        keys = set(df.index)

        if common_keys is None:
            common_keys = keys
        else:
            common_keys = common_keys.intersection(keys)

    if not common_keys:
        raise ValueError(
            "No common timestamp + house_id records found "
            "between DL prediction files."
        )

    common_keys = sorted(common_keys)

    base = datasets[
        model_names[0]
    ].loc[common_keys].copy()

    result = pd.DataFrame(
        {
            "timestamp": base["timestamp"].values,
            "house_id": base["house_id"].values,
            "actual": base["actual"].values,
        },
        index=common_keys,
    )

    for name in model_names:

        result[name] = datasets[name].loc[
            common_keys,
            "prediction"
        ].values

    return result.reset_index(drop=True)


# =========================================================
# FIND BEST DL-ONLY HYBRID
# =========================================================
def find_best_hybrid_pair():

    # ONLY DEEP LEARNING MODELS
    models = [
        "LSTM",
        "BiLSTM",
        "CNN-LSTM",
    ]

    validation = prepare_models(
        models,
        "validation"
    )

    best = None

    # Fine weight search
    weights = np.arange(
        0.05,
        1.00,
        0.05
    )

    for i in range(len(models)):

        for j in range(
            i + 1,
            len(models)
        ):

            m1 = models[i]
            m2 = models[j]

            for w in weights:

                prediction = (
                    w * validation[m1].values
                    + (1.0 - w)
                    * validation[m2].values
                )

                metrics = calculate_metrics(
                    validation["actual"].values,
                    prediction
                )

                if (
                    best is None
                    or metrics["RMSE"]
                    < best["RMSE"]
                ):

                    best = {
                        "model_1": m1,
                        "model_2": m2,
                        "weight_1": float(w),
                        "weight_2": float(
                            1.0 - w
                        ),
                        "RMSE": float(
                            metrics["RMSE"]
                        ),
                    }

    return best


# =========================================================
# MAIN
# =========================================================
def main():

    print("=" * 65)
    print("DEEP LEARNING HYBRID TRAINING")
    print("=" * 65)

    print(
        "\nModels considered:"
    )

    print("1. LSTM")
    print("2. BiLSTM")
    print("3. CNN-LSTM")

    print(
        "\nHybrid combinations:"
    )

    print(
        "LSTM + BiLSTM"
    )

    print(
        "LSTM + CNN-LSTM"
    )

    print(
        "BiLSTM + CNN-LSTM"
    )

    # -----------------------------------------------------
    # STEP 1: BEST DL PAIR
    # -----------------------------------------------------

    best = find_best_hybrid_pair()

    m1 = best["model_1"]
    m2 = best["model_2"]

    w1 = best["weight_1"]
    w2 = best["weight_2"]

    print(
        "\nBest DL Hybrid Combination"
    )

    print("-" * 40)

    print(
        f"Model 1 : {m1}"
    )

    print(
        f"Model 2 : {m2}"
    )

    print(
        f"Weight  : {w1:.2f} + {w2:.2f}"
    )

    print(
        f"Validation RMSE : "
        f"{best['RMSE']:.6f}"
    )

    # -----------------------------------------------------
    # STEP 2: TEST
    # -----------------------------------------------------

    test = prepare_models(
        [m1, m2],
        "test"
    )

    hybrid_prediction = (
        w1 * test[m1].values
        + w2 * test[m2].values
    )

    metrics = calculate_metrics(
        test["actual"].values,
        hybrid_prediction
    )

    # -----------------------------------------------------
    # STEP 3: SAVE PREDICTIONS
    # -----------------------------------------------------

    output = pd.DataFrame(
        {
            "timestamp":
                test["timestamp"],

            "house_id":
                test["house_id"],

            "actual":
                test["actual"],

            "prediction":
                hybrid_prediction,
        }
    )

    output.to_csv(
        REPORTS / "predictions_hybrid_dl.csv",
        index=False
    )

    # -----------------------------------------------------
    # STEP 4: SAVE METRICS
    # -----------------------------------------------------

    hybrid_name = (
        f"{m1} + {m2} Hybrid"
    )

    final_result = {
        hybrid_name: {
            **metrics,
            "model_1": m1,
            "model_2": m2,
            "weight_1": w1,
            "weight_2": w2,
        }
    }

    (
        REPORTS / "hybrid_metrics.json"
    ).write_text(
        json.dumps(
            final_result,
            indent=2
        ),
        encoding="utf-8"
    )

    # -----------------------------------------------------
    # STEP 5: SAVE CONFIGURATION
    # -----------------------------------------------------

    config = {
        "model_type":
            "Hybrid Deep Learning",

        "model_1":
            m1,

        "model_2":
            m2,

        "weight_1":
            w1,

        "weight_2":
            w2,

        "selection_method":
            "Validation RMSE",

        "candidate_models": [
            "LSTM",
            "BiLSTM",
            "CNN-LSTM",
        ],

        "hybrid_type":
            "Deep Learning + Deep Learning",
    }

    (
        MODELS / "hybrid_dl_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2
        ),
        encoding="utf-8"
    )

    # -----------------------------------------------------
    # FINAL OUTPUT
    # -----------------------------------------------------

    print(
        "\n" + "=" * 65
    )

    print(
        "HYBRID DEEP LEARNING COMPLETED"
    )

    print(
        "=" * 65
    )

    print(
        f"Hybrid Model : "
        f"{hybrid_name}"
    )

    print(
        f"Weight       : "
        f"{w1:.2f} / {w2:.2f}"
    )

    print(
        f"MAE          : "
        f"{metrics['MAE']:.6f}"
    )

    print(
        f"RMSE         : "
        f"{metrics['RMSE']:.6f}"
    )

    print(
        f"MAPE         : "
        f"{metrics['MAPE']:.6f}"
    )

    print(
        f"R2           : "
        f"{metrics['R2']:.6f}"
    )

    print("\nSaved:")

    print(
        REPORTS /
        "predictions_hybrid_dl.csv"
    )

    print(
        REPORTS /
        "hybrid_metrics.json"
    )

    print(
        MODELS /
        "hybrid_dl_config.json"
    )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()