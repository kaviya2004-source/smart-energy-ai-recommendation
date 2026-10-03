"""
Deep Learning + True Hybrid Deep Learning forecasting.

Models:
1. LSTM
2. BiLSTM
3. CNN-LSTM
4. CNN-BiLSTM-Attention Hybrid

Chronology-safe train / validation / test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.preprocessing import StandardScaler

from torch import nn
from torch.utils.data import DataLoader, TensorDataset


sys.path.append(
    str(Path(__file__).resolve().parent)
)

from common import (
    MODELS,
    PROCESSED,
    RANDOM_STATE,
    REPORTS,
    TARGET,
    write_json,
)


# =========================================================
# FEATURES
# =========================================================

FEATURE_CANDIDATES = [
    "kwh_lag_1",
    "kwh_lag_7",
    "kwh_rolling_7",
    "occupancy",

    "area",
    "noof_windows",
    "noof_floors",
    "noof_rooms",

    "dishwasher",
    "washing_machine",
    "boiler",
    "electric_heater",
    "fan",
    "ac",
    "oven",
    "microwave",
    "eac",
    "fridge",
    "refrigratore",
    "tv",
    "play",
    "shofaz",

    "airtc_mean",
    "airtc_min",
    "airtc_max",

    "rh_mean",
    "rh_min",
    "rh_max",

    "bp_mbar_mean",
    "bp_mbar_min",
    "bp_mbar_max",

    "ws_ms_avg_mean",
    "ws_ms_avg_min",
    "ws_ms_avg_max",

    "ws_gust_max_mean",
    "ws_gust_max_min",
    "ws_gust_max_max",

    "winddir_mean",

    "slrkw_avg_mean",
    "slrkw_avg_max",
    "slrmj_tot_sum",

    "month",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "week_of_year",
    "quarter",
    "season",

    "month_sin",
    "month_cos",

    "day_of_week_sin",
    "day_of_week_cos",

    "day_of_year_sin",
    "day_of_year_cos",

    "temperature_range",
    "humidity_range",
    "pressure_range",
    "wind_speed_range",
    "wind_gust_range",
    "solar_variability",

    "area_per_room",
    "area_per_floor",

    "total_appliance_load",
    "active_appliance_count",

    "occupancy_area_interaction",
    "occupancy_appliance_interaction",

    "lag_energy_difference",
    "lag_energy_ratio",
]


# =========================================================
# BASIC MODEL
# =========================================================

class Net(nn.Module):

    def __init__(
        self,
        n_features,
        kind,
    ):

        super().__init__()

        self.kind = kind

        if kind == "LSTM":

            self.seq = nn.LSTM(
                n_features,
                96,
                num_layers=2,
                dropout=0.20,
                batch_first=True,
            )

            out = 96

        elif kind == "BiLSTM":

            self.seq = nn.LSTM(
                n_features,
                64,
                num_layers=2,
                dropout=0.20,
                batch_first=True,
                bidirectional=True,
            )

            out = 128

        elif kind == "CNN-LSTM":

            self.conv = nn.Sequential(
                nn.Conv1d(
                    n_features,
                    64,
                    kernel_size=3,
                    padding=1,
                ),

                nn.BatchNorm1d(64),
                nn.ReLU(),

                nn.Conv1d(
                    64,
                    64,
                    kernel_size=3,
                    padding=1,
                ),

                nn.ReLU(),
            )

            self.seq = nn.LSTM(
                64,
                64,
                num_layers=2,
                dropout=0.20,
                batch_first=True,
            )

            out = 64

        else:

            raise ValueError(kind)

        self.head = nn.Sequential(
            nn.LayerNorm(out),
            nn.Dropout(0.20),
            nn.Linear(out, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x):

        if self.kind == "CNN-LSTM":

            x = self.conv(
                x.transpose(1, 2)
            )

            x = x.transpose(1, 2)

        y = self.seq(x)[0][:, -1]

        return self.head(y).squeeze(1)


# =========================================================
# TRUE HYBRID:
# CNN + BiLSTM + ATTENTION
# =========================================================

class CNNBiLSTMAttention(nn.Module):

    def __init__(
        self,
        n_features,
    ):

        super().__init__()

        # CNN feature extraction
        self.cnn = nn.Sequential(

            nn.Conv1d(
                n_features,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(
                64,
                96,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm1d(96),
            nn.ReLU(),

            nn.Dropout(0.15),
        )

        # Bidirectional temporal learning
        self.bilstm = nn.LSTM(
            input_size=96,
            hidden_size=64,
            num_layers=2,
            dropout=0.20,
            batch_first=True,
            bidirectional=True,
        )

        # Attention
        self.attention = nn.Sequential(

            nn.Linear(128, 64),
            nn.Tanh(),

            nn.Linear(64, 1),
        )

        # Final prediction head
        self.head = nn.Sequential(

            nn.LayerNorm(128),

            nn.Dropout(0.20),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.10),

            nn.Linear(64, 1),
        )

    def forward(self, x):

        # [batch, sequence, features]
        # -> [batch, features, sequence]

        x = x.transpose(1, 2)

        x = self.cnn(x)

        # -> [batch, sequence, channels]

        x = x.transpose(1, 2)

        # BiLSTM

        x, _ = self.bilstm(x)

        # Attention scores

        scores = self.attention(x)

        weights = torch.softmax(
            scores,
            dim=1,
        )

        # Weighted temporal representation

        context = torch.sum(
            weights * x,
            dim=1,
        )

        return self.head(
            context
        ).squeeze(1)


# =========================================================
# METRICS
# =========================================================

def score(y, p):

    y = np.asarray(y)

    p = np.asarray(p)

    return {
        "MAE": float(
            mean_absolute_error(
                y,
                p,
            )
        ),

        "RMSE": float(
            mean_squared_error(
                y,
                p,
            ) ** 0.5
        ),

        "MAPE": float(
            (
                np.abs(
                    (y - p)
                    /
                    np.clip(
                        np.abs(y),
                        0.1,
                        None,
                    )
                )
            ).mean()
            * 100
        ),

        "R2": float(
            r2_score(
                y,
                p,
            )
        ),
    }


# =========================================================
# TRAIN ONE MODEL
# =========================================================

def train_model(
    model,
    loader,
    xva,
    yva,
    epochs,
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-4,
    )

    loss_fn = nn.HuberLoss(
        delta=1.0
    )

    best_loss = float("inf")

    best_state = None

    bad_epochs = 0

    for epoch in range(epochs):

        model.train()

        for xb, yb in loader:

            optimizer.zero_grad()

            prediction = model(xb)

            loss = loss_fn(
                prediction,
                yb,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

        # Validation

        model.eval()

        with torch.no_grad():

            val_prediction = model(
                torch.tensor(
                    xva,
                    dtype=torch.float32,
                )
            )

            val_loss = loss_fn(
                val_prediction,
                torch.tensor(
                    yva,
                    dtype=torch.float32,
                ),
            ).item()

        if val_loss < best_loss:

            best_loss = val_loss

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v
                in model.state_dict().items()
            }

            bad_epochs = 0

        else:

            bad_epochs += 1

        if bad_epochs >= 6:

            break

    model.load_state_dict(
        best_state
    )

    model.eval()

    return model


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
    )

    parser.add_argument(
        "--sequence-days",
        type=int,
        default=7,
    )

    args = parser.parse_args()

    torch.manual_seed(
        RANDOM_STATE
    )

    np.random.seed(
        RANDOM_STATE
    )

    print("=" * 75)
    print(
        "DEEP LEARNING + TRUE HYBRID TRAINING"
    )
    print("=" * 75)

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    df = pd.read_csv(
        PROCESSED / "energy_model_data.csv",
        parse_dates=["timestamp"],
    )

    features = [
        c
        for c in FEATURE_CANDIDATES
        if c in df.columns
    ]

    if len(features) < 10:

        raise ValueError(
            "Not enough DL features available."
        )

    print(
        f"\nDL features used : {len(features)}"
    )

    print(
        f"Rows             : {len(df):,}"
    )

    print(
        f"Occupancy used   : "
        f"{'occupancy' in features}"
    )

    # -----------------------------------------------------
    # SCALE USING TRAIN ONLY
    # -----------------------------------------------------

    train_df = df[
        df["split"] == "train"
    ]

    scaler = StandardScaler()

    scaler.fit(
        train_df[features]
    )

    scaled = df.copy()

    scaled[features] = scaler.transform(
        scaled[features]
    )

    # -----------------------------------------------------
    # CREATE SEQUENCES
    # -----------------------------------------------------

    rows = []

    for house, group in (
        scaled
        .sort_values(
            [
                "house_id",
                "timestamp",
            ]
        )
        .groupby("house_id")
    ):

        values = group[
            features
        ].to_numpy(
            dtype=np.float32
        )

        targets = group[
            TARGET
        ].to_numpy(
            dtype=np.float32
        )

        dates = group[
            "timestamp"
        ].to_numpy()

        for i in range(
            args.sequence_days - 1,
            len(group),
        ):

            rows.append(
                (
                    values[
                        i
                        - args.sequence_days
                        + 1
                        :
                        i + 1
                    ],

                    targets[i],

                    dates[i],

                    house,
                )
            )

    split_dates = (
        df.groupby("split")
        .timestamp
        .agg(["min", "max"])
    )

    train_end = pd.Timestamp(
        split_dates.loc[
            "train",
            "max",
        ]
    )

    valid_end = pd.Timestamp(
        split_dates.loc[
            "validation",
            "max",
        ]
    )

    def pack(function):

        selected = [
            row
            for row in rows
            if function(
                pd.Timestamp(row[2])
            )
        ]

        return (
            np.stack(
                [
                    row[0]
                    for row in selected
                ]
            ),

            np.array(
                [
                    row[1]
                    for row in selected
                ],
                dtype=np.float32,
            ),

            np.array(
                [
                    row[2]
                    for row in selected
                ]
            ),

            np.array(
                [
                    row[3]
                    for row in selected
                ]
            ),
        )

    xtr, ytr, _, _ = pack(
        lambda d:
            d <= train_end
    )

    xva, yva, dva, hva = pack(
        lambda d:
            train_end
            < d
            <= valid_end
    )

    xte, yte, dte, hte = pack(
        lambda d:
            d > valid_end
    )

    print(
        f"Train sequences      : {len(xtr):,}"
    )

    print(
        f"Validation sequences : {len(xva):,}"
    )

    print(
        f"Test sequences       : {len(xte):,}"
    )

    # -----------------------------------------------------
    # DATALOADER
    # -----------------------------------------------------

    loader = DataLoader(
        TensorDataset(
            torch.tensor(
                xtr,
                dtype=torch.float32,
            ),

            torch.tensor(
                ytr,
                dtype=torch.float32,
            ),
        ),

        batch_size=512,

        shuffle=True,
    )

    # -----------------------------------------------------
    # MODELS
    # -----------------------------------------------------

    models = {
        "LSTM":
            Net(
                len(features),
                "LSTM",
            ),

        "BiLSTM":
            Net(
                len(features),
                "BiLSTM",
            ),

        "CNN-LSTM":
            Net(
                len(features),
                "CNN-LSTM",
            ),

        "CNN-BiLSTM-Attention":
            CNNBiLSTMAttention(
                len(features)
            ),
    }

    results = {}

    # -----------------------------------------------------
    # TRAIN ALL MODELS
    # -----------------------------------------------------

    for kind, model in models.items():

        print(
            f"\nTraining {kind}..."
        )

        model = train_model(
            model,
            loader,
            xva,
            yva,
            args.epochs,
        )

        model.eval()

        with torch.no_grad():

            validation_prediction = (
                model(
                    torch.tensor(
                        xva,
                        dtype=torch.float32,
                    )
                )
                .cpu()
                .numpy()
            )

            test_prediction = (
                model(
                    torch.tensor(
                        xte,
                        dtype=torch.float32,
                    )
                )
                .cpu()
                .numpy()
            )

        metrics = score(
            yte,
            test_prediction,
        )

        results[kind] = metrics

        # -------------------------------------------------
        # FILE NAME
        # -------------------------------------------------

        stem = (
            kind
            .lower()
            .replace("-", "_")
        )

        # -------------------------------------------------
        # SAVE MODEL
        # -------------------------------------------------

        torch.save(
            {
                "state_dict":
                    model.state_dict(),

                "features":
                    features,

                "scaler":
                    scaler,

                "sequence_days":
                    args.sequence_days,

                "kind":
                    kind,
            },

            MODELS /
            f"{stem}.pt",
        )

        # -------------------------------------------------
        # TEST PREDICTIONS
        # -------------------------------------------------

        test_output = pd.DataFrame(
            {
                "actual":
                    yte,

                "prediction":
                    test_prediction,

                "timestamp":
                    dte,

                "house_id":
                    hte,
            }
        )

        test_output.to_csv(
            REPORTS /
            f"predictions_{stem}.csv",

            index=False,
        )

        # -------------------------------------------------
        # VALIDATION PREDICTIONS
        # -------------------------------------------------

        validation_output = pd.DataFrame(
            {
                "actual":
                    yva,

                "prediction":
                    validation_prediction,

                "timestamp":
                    dva,

                "house_id":
                    hva,
            }
        )

        validation_output.to_csv(
            REPORTS /
            f"predictions_{stem}_validation.csv",

            index=False,
        )

        print(
            f"{kind} {metrics}"
        )

    # -----------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------

    write_json(
        REPORTS / "dl_metrics.json",
        results,
    )

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    result_df = (
        pd.DataFrame(results)
        .T
        .sort_values(
            "RMSE"
        )
    )

    print("\n")
    print("=" * 75)
    print(
        "DEEP LEARNING MODEL COMPARISON"
    )
    print("=" * 75)

    print(
        result_df.to_string()
    )

    best_model = result_df.index[0]

    print("\n" + "=" * 75)

    print(
        f"BEST DL MODEL : {best_model}"
    )

    print(
        f"RMSE         : "
        f"{result_df.loc[best_model, 'RMSE']:.6f}"
    )

    print(
        f"R2           : "
        f"{result_df.loc[best_model, 'R2']:.6f}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()