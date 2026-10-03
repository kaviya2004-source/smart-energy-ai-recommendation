"""Shared LSTM code: architecture, checkpoint loader, input windows, lag convention.

Used by generate_future_forecasts.py and the Streamlit dashboard, so the model
definition exists in exactly one place. It reads the architecture details
(head size, extra dropout) directly from the saved checkpoint, so it works with
both the older and the newer models/lstm.pt files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from common import FINAL_MODEL_CHECKPOINT_PATH


class LSTMNet(nn.Module):
    """2-layer LSTM (hidden 96) + LayerNorm + small dense head."""

    def __init__(self, n_features: int, head_hidden: int = 32, extra_dropout: bool = False):
        super().__init__()
        self.seq = nn.LSTM(n_features, 96, num_layers=2, dropout=0.20, batch_first=True)
        layers = [
            nn.LayerNorm(96),
            nn.Dropout(0.15),
            nn.Linear(96, head_hidden),
            nn.ReLU(),
        ]
        if extra_dropout:
            layers.append(nn.Dropout(0.15))
        layers.append(nn.Linear(head_hidden, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(self.seq(x)[0][:, -1]).squeeze(1)


@dataclass
class LSTMBundle:
    model: nn.Module
    features: list
    scaler: object
    sequence_days: int
    head_hidden: int
    extra_dropout: bool

    def predict_batch(self, windows, batch_size: int = 2048) -> np.ndarray:
        """Predict next-day kWh for many windows at once.

        `windows` has shape (n, sequence_days, n_features) with RAW feature values
        in the training order. Returns an array of n predictions (never below 0).
        """
        windows = np.asarray(windows, dtype=float)
        if windows.ndim != 3 or windows.shape[1] != self.sequence_days \
                or windows.shape[2] != len(self.features):
            raise ValueError(
                f"Expected shape (n, {self.sequence_days}, {len(self.features)}), "
                f"got {windows.shape}."
            )
        if not np.isfinite(windows).all():
            raise ValueError("Non-finite values in the LSTM input windows.")

        n, steps, n_features = windows.shape
        flat = pd.DataFrame(windows.reshape(-1, n_features), columns=self.features)
        scaled = np.asarray(self.scaler.transform(flat), dtype=np.float32)
        scaled = scaled.reshape(n, steps, n_features)

        outputs = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = torch.tensor(scaled[start:start + batch_size], dtype=torch.float32)
                outputs.append(np.asarray(self.model(batch).numpy()).reshape(-1))
        return np.maximum(np.concatenate(outputs), 0.0)

    def predict(self, window) -> float:
        """Predict next-day kWh from ONE window of `sequence_days` rows."""
        if not isinstance(window, pd.DataFrame):
            window = pd.DataFrame(np.asarray(window, dtype=float), columns=self.features)
        values = window[self.features].astype(float).to_numpy()
        if len(values) != self.sequence_days:
            raise ValueError(
                f"LSTM needs exactly {self.sequence_days} rows, got {len(values)}."
            )
        if not np.isfinite(values).all():
            bad = [c for c, ok in zip(self.features, np.isfinite(values).all(axis=0)) if not ok]
            raise ValueError(f"Non-finite values in LSTM input columns: {bad}")
        return float(self.predict_batch(values[None, :, :])[0])


def load_lstm_checkpoint(path: Path | str | None = None) -> LSTMBundle:
    path = Path(path) if path else FINAL_MODEL_CHECKPOINT_PATH
    if not path.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "features", "scaler"):
        if key not in checkpoint:
            raise RuntimeError(f"Invalid {path.name}: '{key}' is missing.")

    state = checkpoint["state_dict"]
    head2 = state.get("head.2.weight")
    if head2 is None:
        raise RuntimeError(f"Invalid {path.name}: head.2.weight is missing.")

    head_hidden = int(head2.shape[0])
    extra_dropout = "head.5.weight" in state

    model = LSTMNet(len(checkpoint["features"]), head_hidden, extra_dropout)
    model.load_state_dict(state, strict=True)
    model.eval()

    return LSTMBundle(
        model=model,
        features=list(checkpoint["features"]),
        scaler=checkpoint["scaler"],
        sequence_days=int(checkpoint.get("sequence_days", 7)),
        head_hidden=head_hidden,
        extra_dropout=extra_dropout,
    )


def build_windows(df: pd.DataFrame, features: list, sequence_days: int, select=None):
    """One input window per selected row.

    Each window holds the `sequence_days` consecutive DAILY rows that end at the
    selected row (same house). Windows that cross a missing day, a house boundary
    or contain NaN are dropped.

    df      needs 'house_id' and 'timestamp' plus every column in `features`
    select  boolean Series (aligned with df) marking the rows to predict;
            default = every row
    Returns (windows, rows): windows has shape (n, sequence_days, n_features),
    rows is the matching slice of df (same order).
    """
    if select is None:
        select = pd.Series(True, index=df.index)
    select = select.reindex(df.index).fillna(False).to_numpy(dtype=bool)
    position = pd.Series(np.arange(len(df)), index=df.index)

    pieces, order = [], []
    for _, group in df.groupby("house_id", sort=False):
        group = group.sort_values("timestamp")
        n = len(group)
        if n < sequence_days:
            continue
        values = group[features].to_numpy(dtype=float)
        windows = sliding_window_view(values, sequence_days, axis=0).transpose(0, 2, 1)

        days = group["timestamp"].dt.floor("D").to_numpy("datetime64[D]")
        span = (days[sequence_days - 1:] - days[: n - sequence_days + 1]).astype(int)
        wanted = select[position.loc[group.index].to_numpy()][sequence_days - 1:]
        keep = (span == sequence_days - 1) & np.isfinite(windows).all(axis=(1, 2)) & wanted

        pieces.append(windows[keep])
        order.append(group.index[sequence_days - 1:][keep])

    if not pieces or sum(len(p) for p in pieces) == 0:
        raise RuntimeError(f"No complete {sequence_days}-day windows could be built.")
    return np.concatenate(pieces).copy(), df.loc[np.concatenate(order)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# How were kwh_lag_1 / kwh_lag_7 / kwh_rolling_7 built in the training data?
# ---------------------------------------------------------------------------
MATCH_THRESHOLD = 0.95


def detect_lag_convention(df: pd.DataFrame) -> dict:
    """Read the convention from the real data instead of guessing it.

    back = how many days BEFORE the row's own date the lag looks.
        kwh_lag_1 usually has back=1 (yesterday) and kwh_lag_7 back=7.
    rolling_inclusive = True if kwh_rolling_7 includes the row's own kWh.
    """
    conv = {"lag1_back": 1, "lag7_back": 7, "rolling_inclusive": True, "detected": {}}
    df = df.sort_values(["house_id", "timestamp"])
    grouped = df.groupby("house_id")["kwh"]
    # Skip each house's first rows: their windows are cut short by the data start.
    settled = df.groupby("house_id").cumcount() >= 8

    def match_rate(column: pd.Series, reference: pd.Series) -> float:
        both = column.notna() & reference.notna() & settled
        if both.sum() == 0:
            return 0.0
        return float(np.isclose(column[both], reference[both], atol=1e-6).mean())

    for name, candidates in (("kwh_lag_1", (1, 0)), ("kwh_lag_7", (7, 6))):
        if name not in df.columns:
            continue
        rates = {back: match_rate(df[name], grouped.shift(back)) for back in candidates}
        best = max(rates, key=rates.get)
        conv["detected"][name] = {str(k): round(v, 4) for k, v in rates.items()}
        if rates[best] >= MATCH_THRESHOLD:
            conv["lag1_back" if name == "kwh_lag_1" else "lag7_back"] = best
        else:
            print(f"WARNING: could not verify the convention of {name}; "
                  "using the standard shift convention.")

    if "kwh_rolling_7" in df.columns:
        inclusive = grouped.transform(lambda s: s.rolling(7, min_periods=1).mean())
        exclusive = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        rates = {True: match_rate(df["kwh_rolling_7"], inclusive),
                 False: match_rate(df["kwh_rolling_7"], exclusive)}
        best = max(rates, key=rates.get)
        conv["detected"]["kwh_rolling_7"] = {
            "inclusive": round(rates[True], 4), "exclusive": round(rates[False], 4)}
        if rates[best] >= MATCH_THRESHOLD:
            conv["rolling_inclusive"] = best
        else:
            print("WARNING: could not verify kwh_rolling_7; assuming it includes today.")
    return conv

