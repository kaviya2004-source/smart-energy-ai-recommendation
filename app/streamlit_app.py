"""AI Smart Energy Management Dashboard."""

from pathlib import Path
import sys
import json
from datetime import datetime

import numpy as np
import plotly.express as px
import pandas as pd
import streamlit as st
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from common import REPORTS  # noqa: E402

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="AI Smart Energy Management",
    page_icon="⚡",
    layout="wide",
)

# ---------- PREMIUM UI / LIGHTWEIGHT ANIMATION ----------
st.markdown("""
<style>
/* Hero background */
.hero-shell {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 34px 36px;
    margin-bottom: 22px;
    min-height: 205px;
    background:
        linear-gradient(115deg, rgba(10,28,52,.96), rgba(18,74,92,.90)),
        radial-gradient(circle at 85% 20%, rgba(91,220,180,.35), transparent 30%);
    box-shadow: 0 16px 40px rgba(8,30,45,.18);
}
.hero-shell:before {
    content: "";
    position: absolute;
    width: 280px; height: 280px;
    right: -70px; top: -110px;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,.18);
    box-shadow: 0 0 0 28px rgba(255,255,255,.04),
                0 0 0 58px rgba(255,255,255,.025);
    animation: pulseRing 5s ease-in-out infinite;
}
.hero-shell:after {
    content: "";
    position: absolute;
    left: -50px; bottom: -95px;
    width: 230px; height: 230px;
    border-radius: 50%;
    background: rgba(255,255,255,.045);
    animation: floatBlob 7s ease-in-out infinite;
}
.hero-content { position: relative; z-index: 2; }
.hero-title {
    color: #fff; font-size: 2.25rem; font-weight: 800;
    letter-spacing: -.8px; margin: 0 0 8px 0;
}
.hero-subtitle {
    color: rgba(255,255,255,.82); font-size: 1rem;
    margin: 0; max-width: 780px;
}
.hero-chip {
    display:inline-block; margin-top:18px; padding:7px 13px;
    border-radius:999px; background:rgba(255,255,255,.12);
    color:#fff; font-size:.82rem; border:1px solid rgba(255,255,255,.16);
}
.kpi-card {
    padding: 18px 18px; border-radius: 18px;
    background: rgba(255,255,255,.90);
    border: 1px solid rgba(40,70,90,.10);
    box-shadow: 0 8px 25px rgba(20,45,60,.08);
    transition: transform .22s ease, box-shadow .22s ease;
}
.kpi-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 14px 30px rgba(20,45,60,.14);
}
.section-card {
    padding: 20px; border-radius: 20px;
    background: rgba(255,255,255,.72);
    border: 1px solid rgba(40,70,90,.09);
    box-shadow: 0 7px 24px rgba(20,45,60,.06);
    animation: fadeUp .45s ease both;
}
.reco-card {
    padding: 18px 20px; border-radius: 16px; margin: 10px 0;
    background: linear-gradient(135deg, rgba(255,248,226,.95), rgba(255,255,255,.92));
    border-left: 5px solid #f2b84b;
    box-shadow: 0 6px 20px rgba(40,50,40,.07);
}
@keyframes fadeUp {
    from {opacity:0; transform:translateY(10px);}
    to {opacity:1; transform:translateY(0);}
}
@keyframes floatBlob {
    0%,100% {transform:translate(0,0);}
    50% {transform:translate(22px,-12px);}
}
@keyframes pulseRing {
    0%,100% {transform:scale(.96); opacity:.55;}
    50% {transform:scale(1.04); opacity:1;}
}
/* Softer Streamlit chrome */
div[data-testid="stMetric"] {
    border-radius: 16px;
    padding: 12px 14px;
    background: rgba(255,255,255,.72);
    border: 1px solid rgba(40,70,90,.08);
}
div[data-testid="stButton"] button {
    border-radius: 12px;
    transition: transform .18s ease, box-shadow .18s ease;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(20,60,80,.12);
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1400px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }
        [data-testid="stMetric"] {
            border-radius: 12px;
            padding: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------
@st.cache_data

def premium_energy_donut(values, labels, title="Energy Status Distribution"):
    fig = px.pie(
        values=values,
        names=labels,
        hole=0.58,
        title=title,
        color_discrete_sequence=["#2E86DE", "#27AE60", "#F39C12", "#E74C3C", "#8E44AD"]
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="%{label}<br>Records: %{value}<br>%{percent}<extra></extra>",
        marker=dict(line=dict(color="white", width=2))
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10,r=10,t=55,b=10),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13)
    )
    return fig

def load_csv(name: str) -> pd.DataFrame:
    path = REPORTS / name
    if not path.exists():
        return pd.DataFrame()

    parse_dates = None
    if "prediction" in name or name == "recommendations.csv":
        parse_dates = ["timestamp"]

    return pd.read_csv(path, parse_dates=parse_dates)


@st.cache_data
def load_energy_data() -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "data" / "processed" / "energy_model_data.csv",
        parse_dates=["timestamp"],
    )


@st.cache_resource
def load_yolo_model():
    from ultralytics import YOLO
    return YOLO(ROOT / "models" / "yolo_person_detector.pt")


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def occupancy_advice(person_count: int, energy_status: str):
    if person_count == 0:
        return (
            "Empty room",
            "No person detected. After a safe delay, switch non-critical "
            "lights and standby appliances to energy-saving mode. Do not "
            "switch off safety-critical devices.",
        )

    if person_count == 1:
        return (
            "Single occupancy",
            "One person detected. Maintain essential comfort; use task "
            "lighting and avoid unnecessary appliance use.",
        )

    if person_count <= 4:
        return (
            "Moderate occupancy",
            "Multiple people detected. Keep ventilation and comfort active; "
            "optimise AC set point to 24–26°C and avoid non-essential loads.",
        )

    if energy_status == "High":
        return (
            "High occupancy",
            "High person count and high predicted energy use. Keep "
            "ventilation and safety systems active, but postpone "
            "non-essential high-load appliances.",
        )

    return (
        "High occupancy",
        "High person count detected. Prioritise comfort, ventilation and "
        "safety; use efficient lighting and avoid unnecessary high-load appliances.",
    )


def safe_number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clear_yolo_recommendation(person_count: int, energy_status: str, predicted_kwh: float):
    if person_count == 0:
        occupancy_text = (
            "No people detected. After a short safety delay, switch off "
            "non-essential lights, AC and standby appliances."
        )
    elif person_count <= 2:
        occupancy_text = (
            "Low occupancy. Keep essential comfort systems active and "
            "switch off devices that are not being used."
        )
    elif person_count <= 4:
        occupancy_text = (
            "Moderate occupancy. Keep ventilation/comfort active, use "
            "efficient lighting and avoid unnecessary high-load appliances."
        )
    else:
        occupancy_text = (
            "High occupancy. Keep ventilation and safety systems active, "
            "but avoid unnecessary high-load equipment and optimise AC usage."
        )

    if energy_status == "High":
        energy_text = (
            f"Predicted energy is high ({predicted_kwh:.2f} kWh), so prioritise "
            "load reduction without affecting safety or occupant comfort."
        )
    elif energy_status == "Low":
        energy_text = (
            f"Predicted energy is low ({predicted_kwh:.2f} kWh); maintain the "
            "current efficient usage pattern."
        )
    else:
        energy_text = (
            f"Predicted energy is normal ({predicted_kwh:.2f} kWh); continue "
            "normal monitoring and avoid idle loads."
        )

    return occupancy_text + " " + energy_text



# ---------------------------------------------------------------------
# FINAL LSTM + NEW ELECTRICITY DATASET UPLOAD
# ---------------------------------------------------------------------

class LSTMNet(nn.Module):
    def __init__(self, n_features, head_hidden=32, extra_dropout=False):
        super().__init__()
        self.seq = nn.LSTM(
            n_features,
            96,
            num_layers=2,
            dropout=.20,
            batch_first=True,
        )
        layers = [
            nn.LayerNorm(96),
            nn.Dropout(.15),
            nn.Linear(96, head_hidden),
            nn.ReLU(),
        ]
        if extra_dropout:
            layers.append(nn.Dropout(.15))
        layers.append(nn.Linear(head_hidden, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(self.seq(x)[0][:, -1]).squeeze(1)


@st.cache_resource
def load_lstm_model():
    p = ROOT / "models" / "lstm.pt"
    if not p.exists():
        raise FileNotFoundError("models/lstm.pt not found")

    ck = torch.load(p, map_location="cpu", weights_only=False)
    state = ck["state_dict"]
    features = ck["features"]
    scaler = ck["scaler"]
    seq_days = int(ck.get("sequence_days", 7))

    # Detect the exact saved head instead of assuming a newer architecture.
    # Current train_dl.py uses head.2 = 32 and head.4 = final layer.
    # The existing checkpoint may use head.2 = 64 and head.5 = final layer.
    head2 = state.get("head.2.weight")
    has_head5 = "head.5.weight" in state

    if head2 is None:
        raise RuntimeError(
            "The saved LSTM checkpoint is missing head.2.weight. "
            "Please check models/lstm.pt."
        )

    head_hidden = int(head2.shape[0])
    model = LSTMNet(
        len(features),
        head_hidden=head_hidden,
        extra_dropout=has_head5,
    )

    model.load_state_dict(state, strict=True)
    model.eval()
    return model, features, scaler, seq_days


def read_uploaded_energy_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    if name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Please upload an electricity/energy CSV, XLSX or XLS file.")


def standardize_uploaded_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]

    # Household Power Consumption dataset:
    # Date + Time + Global_active_power (kW, one-minute interval).
    if "date" in df.columns and "time" in df.columns and "timestamp" not in df.columns:
        df["timestamp"] = (
            df["date"].astype(str).str.strip()
            + " "
            + df["time"].astype(str).str.strip()
        )

    aliases = {
        "datetime": "timestamp",
        "date_time": "timestamp",
        "energy": "kwh",
        "electricity": "kwh",
        "energy_consumption": "kwh",
        "energy_consumption_kwh": "kwh",
        "power_consumption": "kwh",
        "power": "kwh",
        "global_active_power": "global_active_power",
        "house": "house_id",
        "houseid": "house_id",
        "home_id": "house_id",
        "people": "occupancy",
        "persons": "occupancy",
        "occupants": "occupancy",
        "occupancy_count": "occupancy",
        "temperature": "airtc_mean",
        "temp": "airtc_mean",
        "humidity": "rh_mean",
        "relative_humidity": "rh_mean",
        "pressure": "bp_mbar_mean",
        "air_pressure": "bp_mbar_mean",
        "wind_speed": "ws_ms_avg_mean",
        "solar": "slrkw_avg_mean",
        "solar_radiation": "slrkw_avg_mean",
    }

    df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})

    # If the file has Global_active_power, convert kW sampled every minute
    # into per-record kWh. This avoids treating kW as kWh.
    if "kwh" not in df.columns and "global_active_power" in df.columns:
        df["kwh"] = pd.to_numeric(
            df["global_active_power"].replace("?", np.nan),
            errors="coerce",
        ) / 60.0

    return df


def prepare_uploaded_energy_data(df, training_df, model_features):
    df = standardize_uploaded_columns(df)

    if "timestamp" not in df.columns:
        raise ValueError(
            "This file needs a date/time column. "
            "Examples: timestamp, datetime, or Date + Time."
        )

    if "kwh" not in df.columns:
        raise ValueError(
            "This file needs an electricity/energy column. "
            "Examples: kWh, energy, electricity, power, or Global_active_power."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
    df = df.dropna(subset=["timestamp", "kwh"])

    if df.empty:
        raise ValueError("No valid electricity records were found in the uploaded file.")

    if "house_id" not in df.columns:
        df["house_id"] = 1

    df["house_id"] = pd.to_numeric(df["house_id"], errors="coerce").fillna(1)
    df = df.sort_values(["house_id", "timestamp"]).reset_index(drop=True)

    # If this is a minute-level Household Power Consumption file,
    # convert the minute kWh values into daily household energy.
    if "global_active_power" in df.columns:
        daily = (
            df.assign(day=df["timestamp"].dt.floor("D"))
              .groupby(["house_id", "day"], as_index=False)["kwh"]
              .sum()
              .rename(columns={"day": "timestamp"})
        )
        df = daily.sort_values(["house_id", "timestamp"]).reset_index(drop=True)

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["timestamp"].dt.quarter

    df["season"] = df["month"].map({
        12:1, 1:1, 2:1, 3:2, 4:2, 5:2,
        6:3, 7:3, 8:3, 9:3, 10:4, 11:4
    })

    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["day_of_week_sin"] = np.sin(2*np.pi*df["day_of_week"]/7)
    df["day_of_week_cos"] = np.cos(2*np.pi*df["day_of_week"]/7)

    df["kwh_lag_1"] = df.groupby("house_id")["kwh"].shift(1)
    df["kwh_lag_7"] = df.groupby("house_id")["kwh"].shift(7)
    df["kwh_rolling_7"] = (
        df.groupby("house_id")["kwh"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    for feature in model_features:
        if feature not in df.columns:
            if feature in training_df.columns:
                value = pd.to_numeric(training_df[feature], errors="coerce").median()
                if pd.isna(value):
                    mode = training_df[feature].mode()
                    value = mode.iloc[0] if len(mode) else 0
                df[feature] = value
            else:
                df[feature] = 0

        df[feature] = pd.to_numeric(df[feature], errors="coerce")

        if feature in training_df.columns:
            med = pd.to_numeric(training_df[feature], errors="coerce").median()
            med = 0 if pd.isna(med) else med
        else:
            med = 0

        df[feature] = df[feature].fillna(med)

    return df


def predict_uploaded_energy(df, model, model_features, scaler, sequence_days):
    rows = []

    for house_id, group in df.groupby("house_id"):
        group = group.sort_values("timestamp").copy()

        # The deployed LSTM uses a 7-step sequence. For small demo uploads
        # (such as a 5-row sample shown to a guide), pad the beginning by
        # repeating the earliest available row instead of rejecting the file.
        if len(group) < sequence_days:
            if len(group) == 0:
                continue
            pad_count = sequence_days - len(group)
            first_row = group.iloc[[0]].copy()
            padding = pd.concat([first_row] * pad_count, ignore_index=True)
            group_for_model = pd.concat([padding, group], ignore_index=True)
        else:
            group_for_model = group

        x = scaler.transform(group_for_model[model_features]).astype(np.float32)
        sequence = x[-sequence_days:]

        with torch.no_grad():
            prediction = max(
                0.0,
                float(model(torch.tensor(sequence).unsqueeze(0)).item())
            )

        latest = group.iloc[-1]

        rows.append({
            "house_id": house_id,
            "last_timestamp": latest["timestamp"],
            "latest_actual_kwh": float(latest["kwh"]),
            "predicted_next_day_kwh": prediction
        })

    if not rows:
        raise ValueError(
            "No valid electricity records were found for prediction."
        )

    return pd.DataFrame(rows)


def upload_energy_recommendation(result, training_df):
    mean_energy = float(training_df["kwh"].mean())

    def status(value):
        if value > mean_energy * 1.20:
            return "High"
        if value < mean_energy * 0.80:
            return "Low"
        return "Normal"

    result["energy_status"] = result["predicted_next_day_kwh"].apply(status)

    def recommendation(row):
        predicted = float(row["predicted_next_day_kwh"])
        if row["energy_status"] == "High":
            return (
                f"High energy expected ({predicted:.2f} kWh). "
                "Action: switch off unused lights/devices, reduce unnecessary appliance use, "
                "and optimise AC/high-load equipment. Recheck after the next reading."
            )
        if row["energy_status"] == "Low":
            return (
                f"Low energy expected ({predicted:.2f} kWh). "
                "Action: current usage is efficient; keep essential devices only "
                "and continue monitoring."
            )
        return (
            f"Normal energy expected ({predicted:.2f} kWh). "
            "Action: maintain normal usage, switch off idle devices, "
            "and monitor for sudden increases."
        )

    result["recommendation"] = result.apply(recommendation, axis=1)
    return result


def show_colored_bar_chart(series, title, ylabel):
    values = series.dropna()
    if values.empty:
        st.info("No chart data available.")
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(
        values.index.astype(str),
        values.values,
        color=(
            ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
            if len(values) <= 6
            else "#3498db"
        ),
    )
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def show_colorful_historical_charts(filtered_energy):
    if filtered_energy.empty:
        return

    t = filtered_energy.copy()
    t["year_label"] = t["timestamp"].dt.year.astype(str)
    t["month_label"] = t["timestamp"].dt.to_period("M").astype(str)
    t["season"] = t["timestamp"].dt.month.map({
        12:"Winter", 1:"Winter", 2:"Winter",
        3:"Summer", 4:"Summer", 5:"Summer",
        6:"Monsoon", 7:"Monsoon", 8:"Monsoon", 9:"Monsoon",
        10:"Post-monsoon", 11:"Post-monsoon"
    })

    daily = t.groupby("timestamp")["kwh"].mean()
    monthly = t.groupby("month_label")["kwh"].mean()
    yearly = t.groupby("year_label")["kwh"].mean()
    seasonal = (
        t.groupby("season")["kwh"].mean()
        .reindex(["Winter", "Summer", "Monsoon", "Post-monsoon"])
        .dropna()
    )

    a, b = st.columns(2)

    with a:
        st.subheader("Monthly Energy")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(monthly.index, monthly.values, color="#3498db", marker="o", linewidth=2)
        ax.tick_params(axis="x", rotation=60)
        ax.set_ylabel("Mean kWh")
        ax.grid(alpha=.25)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with b:
        st.subheader("Yearly Energy")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(yearly.index, yearly.values, color="#9b59b6")
        ax.set_ylabel("Mean kWh")
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    st.subheader("Seasonal Energy")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        seasonal.index,
        seasonal.values,
        color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    )
    ax.set_ylabel("Mean kWh")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.subheader("Daily Energy Trend")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily.values, color="#1abc9c", linewidth=1.8)
    ax.set_ylabel("Mean kWh")
    ax.grid(alpha=.25)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


# ---------------------------------------------------------------------
# REQUIRED REPORTS
# ---------------------------------------------------------------------
required = [
    "all_model_comparison.csv",
    "recommendations.csv",
]

missing = [name for name in required if not (REPORTS / name).exists()]

if missing:
    st.error(
        "Required energy reports are missing: "
        + ", ".join(missing)
        + ". Run the existing energy pipeline first."
    )
    st.stop()

# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------
metrics = load_csv("all_model_comparison.csv")
recommendations = load_csv("recommendations.csv")
energy = load_energy_data()

importance_path = REPORTS / "permutation_importance.csv"
importance = (
    pd.read_csv(importance_path)
    if importance_path.exists()
    else pd.DataFrame()
)

future_path = REPORTS / "future_energy_forecasts.csv"
future_forecasts = (
    pd.read_csv(future_path, parse_dates=["forecast_date"])
    if future_path.exists()
    else pd.DataFrame()
)

yolo_reports = ROOT / "data" / "yolo" / "reports"
yolo_metric_path = yolo_reports / "test_metrics_summary.json"
yolo_metrics = (
    json.loads(yolo_metric_path.read_text(encoding="utf-8"))
    if yolo_metric_path.exists()
    else None
)

metrics = metrics.sort_values("RMSE").reset_index(drop=True)
lstm_rows = metrics[metrics["Model"].astype(str).str.upper().eq("LSTM")]

# LSTM is the final deployed deep-learning model.
if not lstm_rows.empty:
    best = lstm_rows.iloc[0]
    LSTM_RMSE = safe_number(best["RMSE"])
else:
    best = metrics.iloc[0]
    LSTM_RMSE = None

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------

st.markdown("""
<div class="hero-shell">
  <div class="hero-content">
    <div class="hero-title">⚡ AI Smart Building Energy Management</div>
    <div class="hero-subtitle">
      Intelligent energy forecasting, explainable insights, human-presence detection
      and practical energy-saving recommendations in one dashboard.
    </div>
    <span class="hero-chip">● AI Forecasting &nbsp; • &nbsp; XAI &nbsp; • &nbsp; YOLOv8 &nbsp; • &nbsp; Smart Recommendations</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.title("⚡ AI Smart Energy Management")
st.caption(
    "Energy forecasting • Future projection • Explainable AI • "
    "YOLO human detection • Energy-saving recommendations"
)

st.info(
    f"Final deployed model: **{best.Model}** — selected using the "
    f"lowest chronological held-out test RMSE."
)

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
if "forecast_view" not in st.session_state:
    st.session_state["forecast_view"] = "All Data"

with st.sidebar:
    st.title("⚡ Energy AI")
    st.caption("Smart Building Energy Management")

    # 1) Module definitions FIRST
    st.divider()
    st.subheader("Module Guide")
    st.caption("Overview — quick summary of energy usage and prediction results.")
    st.caption("EDA — analyzes historical monthly, yearly and seasonal energy patterns.")
    st.caption("Future Forecast — predicts future energy consumption for selected periods.")
    st.caption("Model Comparison — compares ML and Deep Learning model performance.")
    st.caption("Explainable AI — identifies the features influencing energy prediction.")
    st.caption("Recommendations — gives clear actions to reduce unnecessary energy use.")
    st.caption("New Dataset Upload — uploads electricity data and generates an LSTM prediction.")
    st.caption("Live Energy + YOLO — combines energy prediction with human-presence detection.")
    st.caption("YOLO Human Detection — detects people and gives occupancy-aware energy advice.")

    # 2) Forecast/Data View filter
    st.divider()
    st.subheader("Forecast / Data View")

    if "forecast_view" not in st.session_state:
        st.session_state["forecast_view"] = "All Data"

    for option in ["All Data", "7-Day", "30-Day", "2026", "2027"]:
        if st.button(
            option,
            key=f"sidebar_view_{option.replace('-', '_').replace(' ', '_')}",
            use_container_width=True,
            type="primary" if st.session_state["forecast_view"] == option else "secondary",
        ):
            st.session_state["forecast_view"] = option
            st.rerun()

    # 3) Year filter
    st.divider()
    st.subheader("Year Filter")
    years = sorted(
        pd.to_datetime(energy["timestamp"], errors="coerce")
        .dt.year.dropna().astype(int).unique().tolist()
    )
    selected_years = st.multiselect(
        "Year",
        years,
        default=years,
        key="final_year_filter",
    )

    # 4) House filter
    st.subheader("House Filter")
    houses = sorted(
        pd.to_numeric(energy["house_id"], errors="coerce")
        .dropna().unique().tolist()
    )
    selected_houses = st.multiselect(
        "House",
        houses,
        default=[],
        key="final_house_filter",
    )

forecast_view = st.session_state["forecast_view"]

filtered_energy = energy.copy()
if selected_years:
    filtered_energy = filtered_energy[
        pd.to_datetime(filtered_energy["timestamp"], errors="coerce").dt.year.isin(selected_years)
    ].copy()
if selected_houses:
    filtered_energy = filtered_energy[
        filtered_energy["house_id"].isin(selected_houses)
    ].copy()

# ---------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------
(
    overview,
    eda,
    forecast,
    comparison,
    xai,
    actions,
    upload_tab,
    live,
    yolo,
) = st.tabs(
    [
        "Overview",
        "EDA",
        "Future Forecast",
        "Model Comparison",
        "Explainable AI",
        "Recommendations",
        "New Dataset Upload",
        "Live Energy + YOLO",
        "YOLO Human Detection",
    ]
)

# =====================================================================
# OVERVIEW
# =====================================================================
with overview:
    st.caption("Overview: quick view of energy usage, predictions and savings opportunity.")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Best Model", str(best.Model))
    c2.metric("Test RMSE", f"{safe_number(best.RMSE):.2f} kWh")
    c3.metric("Test R²", f"{safe_number(best.R2):.3f}")
    c4.metric(
        "High-forecast cases",
        f"{int((recommendations['energy_status'] == 'High').sum()):,}",
    )

    st.divider()

    st.subheader("Actual vs Predicted Next-Day Energy")

    daily = (
        recommendations.groupby("timestamp")[
            ["prediction", "target_next_day_kwh"]
        ]
        .mean()
        .rename(
            columns={
                "prediction": "Predicted kWh",
                "target_next_day_kwh": "Actual kWh",
            }
        )
    )

    st.line_chart(daily, height=350)

    a, b = st.columns(2)

    with a:
        st.subheader("Energy Status Distribution")
        show_colored_bar_chart(
            recommendations["energy_status"].value_counts().reindex(
                ["High", "Normal", "Low"]
            ).dropna(),
            "Energy Status Distribution",
            "Number of records",
        )

    with b:
        st.subheader("Estimated Saving Opportunity")
        saving = safe_number(
            recommendations["estimated_saving_kwh"].sum()
        )
        st.metric("Estimated test-period saving", f"{saving:.1f} kWh")
        st.caption(
            "Rule-based estimate; this is not field-measured savings."
        )

    st.divider()

    st.subheader("Project Pipeline")
    st.markdown(
        """
        **Historical data → Cleaning → Feature Engineering → ML/DL comparison
        → Best model → Future energy forecasting → YOLO human detection
        → Occupancy-aware recommendation**
        """
    )

# =====================================================================
# EDA
# =====================================================================
with eda:
    st.subheader("Exploratory Data Analysis")

    if filtered_energy.empty:
        st.warning("No records match the selected filters.")
    else:
        st.caption(
            f"Current filter: {len(filtered_energy):,} records | "
            f"{filtered_energy.house_id.nunique()} homes | "
            f"{filtered_energy.timestamp.min():%d-%b-%Y} to "
            f"{filtered_energy.timestamp.max():%d-%b-%Y}"
        )

        e1, e2 = st.columns(2)

        with e1:
            p = REPORTS / "eda_daily_energy_trend.png"
            if p.exists():
                st.image(str(p), caption="Mean daily household consumption")

            p = REPORTS / "eda_monthly_energy.png"
            if p.exists():
                st.image(str(p), caption="Monthly consumption distribution")

        with e2:
            p = REPORTS / "eda_correlation_heatmap.png"
            if p.exists():
                st.image(str(p), caption="Correlation of top variables with energy")

        st.divider()
        st.subheader("Historical Energy by Time Period")

        temp = filtered_energy.copy()
        temp["year_label"] = temp["timestamp"].dt.year.astype(str)
        temp["month_label"] = temp["timestamp"].dt.to_period("M").astype(str)

        temp["season"] = temp["timestamp"].dt.month.map(
            {
                12: "Winter",
                1: "Winter",
                2: "Winter",
                3: "Summer",
                4: "Summer",
                5: "Summer",
                6: "Monsoon",
                7: "Monsoon",
                8: "Monsoon",
                9: "Monsoon",
                10: "Post-monsoon",
                11: "Post-monsoon",
            }
        )

        monthly = (
            temp.groupby("month_label")["kwh"]
            .mean()
            .rename("Mean kWh")
        )

        yearly = (
            temp.groupby("year_label")["kwh"]
            .mean()
            .rename("Mean kWh")
        )

        seasonal = (
            temp.groupby("season")["kwh"]
            .mean()
            .reindex(
                ["Winter", "Summer", "Monsoon", "Post-monsoon"]
            )
            .dropna()
            .rename("Mean kWh")
        )

        m1, m2 = st.columns(2)

        with m1:
            st.subheader("Monthly Energy")
            st.line_chart(monthly, height=320)

        with m2:
            st.subheader("Yearly Energy")
            show_colored_bar_chart(yearly, "Yearly Mean Energy", "Mean kWh")

        st.subheader("Seasonal Energy")
        show_colored_bar_chart(
            seasonal,
            "Seasonal Mean Energy",
            "Mean kWh",
        )

        st.caption(
            "These monthly/yearly charts represent historical observed "
            "energy consumption, not future predictions."
        )

# =====================================================================
# FUTURE FORECAST
# =====================================================================
with forecast:
    st.subheader("🔮 AI Future Energy Forecast")

    if future_forecasts.empty:
        st.warning(
            "Future forecast file not found. Run "
            "generate_future_forecasts.py once."
        )
    else:
        ff = future_forecasts.copy()
        ff["forecast_date"] = pd.to_datetime(ff["forecast_date"])

        houses = sorted(ff["house_id"].dropna().unique().tolist())

        selected_forecast_house = st.selectbox(
            "Select house",
            houses,
            key="future_house",
        )

        house_rows = (
            ff[ff["house_id"] == selected_forecast_house]
            .sort_values("forecast_date")
            .copy()
        )

        if forecast_view == "All Data":
            house_future = house_rows.copy()
            horizon = len(house_future)
            forecast_title = "All Available Future Forecast Data"
        elif forecast_view in ("7-Day", "30-Day"):
            horizon = 7 if forecast_view == "7-Day" else 30
            house_future = house_rows.head(horizon).copy()
            forecast_title = f"{horizon}-Day Future Energy Forecast"
        else:
            selected_year = int(forecast_view)
            house_future = house_rows[
                house_rows["forecast_date"].dt.year == selected_year
            ].copy()
            horizon = len(house_future)
            forecast_title = f"{selected_year} Future Energy Forecast"

        if house_future.empty:
            st.warning(
                f"No forecast records are currently available for House "
                f"{selected_forecast_house} in the selected view **{forecast_view}**. "
                "Run the future-forecast generation pipeline for that period."
            )
        else:
            total_forecast = house_future["predicted_kwh"].sum()
            average_forecast = house_future["predicted_kwh"].mean()
            maximum_forecast = house_future["predicted_kwh"].max()

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Forecast Days",
                len(house_future),
            )
            k2.metric(
                "Average Daily Energy",
                f"{average_forecast:.2f} kWh",
            )
            k3.metric(
                "Total Forecast",
                f"{total_forecast:.2f} kWh",
            )
            k4.metric(
                "Peak Forecast",
                f"{maximum_forecast:.2f} kWh",
            )

            chart_data = house_future.set_index("forecast_date")[
                ["predicted_kwh"]
            ].rename(columns={"predicted_kwh": "Predicted kWh"})

            st.subheader(
                f"{forecast_title} — House {selected_forecast_house}"
            )
            st.line_chart(chart_data, height=400)

            st.dataframe(
                house_future,
                width="stretch",
                hide_index=True,
            )

            st.download_button(
                "Download selected-house forecast",
                house_future.to_csv(index=False).encode("utf-8"),
                f"house_{selected_forecast_house}_{horizon}day_forecast.csv",
                "text/csv",
            )

        st.divider()
        st.subheader("All-House Future Forecast Summary")

        summary = (
            ff.groupby("forecast_date")["predicted_kwh"]
            .mean()
            .rename("Average predicted kWh")
        )

        st.line_chart(summary, height=350)

        st.caption(
            "The generated multi-day forecasts are scenario projections. "
            "Future weather is assumed to remain at the latest available values."
        )

# =====================================================================
# MODEL COMPARISON
# =====================================================================
with comparison:
    st.caption("Model Comparison: compare ML and Deep Learning performance metrics.")
    st.subheader("ML / DL Model Comparison")

    image_path = REPORTS / "five_model_comparison.png"
    if image_path.exists():
        st.image(
            str(image_path),
            caption="Chronological held-out test comparison",
        )

    st.dataframe(
        metrics,
        width="stretch",
        hide_index=True,
    )

    st.success(
        "LSTM is the final selected Deep Learning model for the deployed prediction workflow. "
        "Other algorithms remain available for research comparison."
    )

# =====================================================================
# XAI
# =====================================================================
with xai:
    st.caption("Explainable AI: see which features influence energy predictions.")
    st.subheader("🔎 Explainable AI")

    shap_path = REPORTS / "shap_global_importance.png"

    if shap_path.exists():
        st.image(
            str(shap_path),
            caption="SHAP global feature importance for the selected model",
        )

    if importance.empty:
        st.info("Permutation importance data is not available.")
    else:
        xai_series = (
            importance.set_index("feature")["importance_mean"]
            .head(15)
            .sort_values()
        )
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.barh(
            xai_series.index.astype(str),
            xai_series.values,
            color="#8e44ad",
        )
        ax.set_title("Top Explainable AI Features", fontweight="bold")
        ax.set_xlabel("Mean importance")
        ax.grid(axis="x", alpha=0.22)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        st.dataframe(
            importance.head(15),
            width="stretch",
            hide_index=True,
        )

        st.warning(
            "Feature importance indicates predictive association, "
            "not causal proof."
        )

# =====================================================================
# RECOMMENDATIONS
# =====================================================================
with actions:
    st.caption("Recommendations: convert predicted energy status into saving actions.")
    st.subheader("🏠 Household-Specific Recommendations")

    houses = sorted(recommendations.house_id.dropna().unique().tolist())

    house = st.selectbox(
        "Select house",
        houses,
        key="recommendation_house",
    )

    view = (
        recommendations[
            recommendations.house_id == house
        ]
        .sort_values("timestamp", ascending=False)
        .copy()
    )

    latest = view.iloc[0]

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Predicted Next-Day Use",
        f"{safe_number(latest.prediction):.2f} kWh",
    )
    r2.metric(
        "Seven-Day Baseline",
        f"{safe_number(latest.baseline_kwh):.2f} kWh",
    )
    r3.metric(
        "Energy Status",
        str(latest.energy_status),
    )

    predicted_value = safe_number(latest.prediction)
    baseline_value = safe_number(latest.baseline_kwh)
    saving_value = safe_number(latest.get("estimated_saving_kwh", 0))

    if str(latest.energy_status) == "High":
        st.warning(
            f"🔴 HIGH ENERGY: predicted use is {predicted_value:.2f} kWh "
            f"against a 7-day baseline of {baseline_value:.2f} kWh. "
            "Action: switch off unused devices, reduce unnecessary lighting, "
            "and optimise AC/high-load appliances."
        )
    elif str(latest.energy_status) == "Low":
        st.success(
            f"🟢 LOW ENERGY: predicted use is {predicted_value:.2f} kWh. "
            "Action: current usage is efficient; keep essential devices active "
            "and continue monitoring."
        )
    else:
        st.info(
            f"🟡 NORMAL ENERGY: predicted use is {predicted_value:.2f} kWh "
            f"with a 7-day baseline of {baseline_value:.2f} kWh. "
            "Action: maintain normal usage and switch off idle loads."
        )

    if saving_value > 0:
        st.caption(
            f"Estimated saving opportunity: {saving_value:.2f} kWh. "
            "This is a rule-based estimate, not measured savings."
        )

    display_cols = [
        "timestamp",
        "prediction",
        "target_next_day_kwh",
        "baseline_kwh",
        "energy_status",
        "estimated_saving_kwh",
        "recommendations",
    ]

    display_cols = [
        c for c in display_cols if c in view.columns
    ]

    st.dataframe(
        view[display_cols].head(30),
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Download selected-house recommendations",
        view.to_csv(index=False).encode("utf-8"),
        f"house_{house}_recommendations.csv",
        "text/csv",
    )

    st.divider()
    st.subheader("Occupancy-Aware Recommendation Logic")

    st.info(
        "YOLO human presence is used as a current operational signal for "
        "recommendation and energy-impact interpretation. It is not silently "
        "inserted into the historical forecasting model unless the model "
        "was explicitly trained with occupancy as an input feature."
    )


# =====================================================================
# NEW ELECTRICITY DATASET UPLOAD
# =====================================================================
with upload_tab:
    st.caption("New Dataset Upload: upload electricity data and generate an LSTM prediction.")
    st.subheader("📤 Upload New Electricity Dataset")

    st.write(
        "Upload a new electricity/energy CSV or Excel file. "
        "The dashboard validates the file, processes the available "
        "features and generates a next-day prediction using LSTM."
    )

    st.caption(
        "Required: date/time + electricity/energy column. "
        "Date + Time and Global_active_power files are supported. "
        "House ID, occupancy and weather columns are optional."
    )
    st.caption(
        "Demo-friendly: small files are accepted; if fewer than 7 rows are uploaded, "
        "the LSTM sequence is safely padded using the earliest available row."
    )

    uploaded_file = st.file_uploader(
        "Choose electricity / energy dataset",
        type=["csv", "xlsx", "xls"],
        key="new_electricity_dataset"
    )

    if uploaded_file is None:
        st.info("👆 Upload an electricity/energy CSV or XLSX file to begin.")
    else:
        try:
            raw_upload = read_uploaded_energy_file(uploaded_file)

            st.success(f"Uploaded: **{uploaded_file.name}**")
            st.caption(
                f"{len(raw_upload):,} rows × {len(raw_upload.columns)} columns"
            )

            st.dataframe(
                raw_upload.head(10),
                width="stretch",
                hide_index=True
            )

            if st.button(
                "🚀 Process Dataset & Predict with LSTM",
                type="primary",
                use_container_width=True,
                key="upload_lstm_predict"
            ):
                try:
                    with st.spinner("Loading final LSTM model..."):
                        lstm_model, lstm_features, lstm_scaler, seq_days = load_lstm_model()

                    with st.spinner("Validating and processing dataset..."):
                        prepared_upload = prepare_uploaded_energy_data(
                            raw_upload, energy, lstm_features
                        )

                    with st.spinner("Generating LSTM prediction..."):
                        upload_result = predict_uploaded_energy(
                            prepared_upload,
                            lstm_model,
                            lstm_features,
                            lstm_scaler,
                            seq_days
                        )

                    upload_result = upload_energy_recommendation(
                        upload_result, energy
                    )

                    st.session_state["uploaded_lstm_result"] = upload_result
                    st.success("✅ Dataset processed and LSTM prediction completed.")

                except ValueError as exc:
                    st.error(f"❌ {exc}")

                except Exception as exc:
                    st.error("❌ The uploaded file could not be processed.")
                    st.exception(exc)

        except Exception as exc:
            st.error(
                "❌ Please upload a valid electricity/energy CSV, XLSX or XLS file."
            )
            st.exception(exc)

    if "uploaded_lstm_result" in st.session_state:
        result = st.session_state["uploaded_lstm_result"]

        st.divider()
        st.subheader("🔮 LSTM Prediction Result")

        latest = result.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("House", str(latest["house_id"]))
        c2.metric("Latest Actual", f"{latest['latest_actual_kwh']:.2f} kWh")
        c3.metric("Predicted Next-Day", f"{latest['predicted_next_day_kwh']:.2f} kWh")
        c4.metric("Energy Status", str(latest["energy_status"]))

        st.dataframe(result, width="stretch", hide_index=True)

        st.subheader("💡 Recommendation")

        for _, row in result.iterrows():
            msg = f"**House {row['house_id']}** — {row['recommendation']}"
            if row["energy_status"] == "High":
                st.warning(msg)
            elif row["energy_status"] == "Low":
                st.success(msg)
            else:
                st.info(msg)

        
        # Compact status distribution for the uploaded prediction result.
        status_counts = result["energy_status"].value_counts()
        if not status_counts.empty:
            pc1, pc2 = st.columns([1, 1.7])
            with pc1:
                st.plotly_chart(
                    premium_energy_donut(
                        status_counts.values.tolist(),
                        status_counts.index.tolist(),
                        "Prediction Status"
                    ),
                    use_container_width=True
                )
            with pc2:
                st.markdown(
                    '<div class="section-card"><h4>📌 Prediction Insight</h4>'
                    '<p>The donut chart summarizes the energy-status distribution '
                    'for the uploaded records. Use Actual vs Predicted below to '
                    'inspect the prediction behaviour.</p></div>',
                    unsafe_allow_html=True
                )

        st.subheader("📊 Actual vs Predicted")

        chart_df = result[
            ["house_id", "latest_actual_kwh", "predicted_next_day_kwh"]
        ].copy()

        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(chart_df))
        width = 0.36

        ax.bar(
            x - width/2,
            chart_df["latest_actual_kwh"],
            width,
            label="Actual kWh",
            color=["#2E86DE", "#5DADE2", "#85C1E9", "#AED6F1", "#1B4F72"]
        )
        ax.bar(
            x + width/2,
            chart_df["predicted_next_day_kwh"],
            width,
            label="Predicted Next-Day kWh",
            color=["#E74C3C", "#EC7063", "#F1948A", "#CD6155", "#922B21"]
        )

        ax.set_xticks(x)
        ax.set_xticklabels(chart_df["house_id"].astype(str))
        ax.set_xlabel("House ID")
        ax.set_ylabel("Energy (kWh)")
        ax.set_title(
            "Uploaded Dataset — Actual vs LSTM Prediction",
            fontweight="bold"
        )
        ax.legend()
        ax.grid(axis="y", alpha=.25)

        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        st.download_button(
            "⬇️ Download Prediction + Recommendation",
            result.to_csv(index=False).encode("utf-8"),
            "uploaded_energy_lstm_prediction.csv",
            "text/csv",
            use_container_width=True
        )


# =====================================================================
# LIVE ENERGY + YOLO
# =====================================================================
with live:
    st.subheader("🎥 Live Energy + YOLO")

    st.write(
        "Select a house, view its saved future forecast, and upload/capture "
        "a current room image. YOLO detects people and the dashboard combines "
        "the current human-presence result with the energy forecast for an "
        "operational recommendation."
    )

    if future_forecasts.empty:
        st.warning(
            "Run generate_future_forecasts.py first. "
            "The existing future forecast files are required."
        )
    else:
        live_houses = sorted(
            future_forecasts.house_id.dropna().unique().tolist()
        )

        current_house = st.selectbox(
            "House for current forecast",
            live_houses,
            key="live_house",
        )

        house_rows = (
            future_forecasts[
                future_forecasts.house_id == current_house
            ]
            .sort_values("forecast_date")
        )

        forecast = house_rows.iloc[0]

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "Forecast Date",
            forecast.forecast_date.strftime("%d-%b-%Y"),
        )
        p2.metric(
            "Predicted Energy",
            f"{safe_number(forecast.predicted_kwh):.2f} kWh",
        )
        p3.metric(
            "Energy Status",
            str(forecast.energy_status),
        )
        p4.metric(
            "Typical Forecast Error",
            f"± {safe_number(LSTM_RMSE):.2f} kWh" if LSTM_RMSE is not None else "N/A",
        )

        st.caption(
            "The uncertainty indicator is the selected model's held-out "
            "test RMSE. It is not a guaranteed prediction interval."
        )

        st.download_button(
            "Download all future forecasts",
            future_forecasts.to_csv(index=False).encode("utf-8"),
            "future_energy_forecasts.csv",
            "text/csv",
        )

        captured = st.camera_input(
            "Capture a current room image",
            key="camera_capture",
        )

        uploaded = st.file_uploader(
            "Or upload a CCTV / room image",
            type=["jpg", "jpeg", "png"],
            key="camera_image",
        )

        image_input = captured if captured is not None else uploaded

        if image_input is not None:
            image = Image.open(image_input).convert("RGB")

            st.image(
                image,
                caption="Current room image",
                width="stretch",
            )

            if st.button(
                "Detect Human + Generate Recommendation",
                type="primary",
                key="detect_and_recommend",
            ):
                try:
                    model = load_yolo_model()

                    result = model.predict(
                        image,
                        conf=0.25,
                        verbose=False,
                    )[0]

                    person_count = 0

                    if result.boxes is not None and len(result.boxes) > 0:
                        classes = result.boxes.cls
                        if classes is None:
                            person_count = len(result.boxes)
                        else:
                            person_class_ids = [
                                int(x) for x in classes.tolist()
                            ]
                            person_count = sum(
                                x == 0 for x in person_class_ids
                            )

                    person_present = int(person_count > 0)

                    occupancy_level, occupancy_message = occupancy_advice(
                        person_count,
                        str(forecast.energy_status),
                    )

                    annotated = result.plot()

                    st.image(
                        annotated,
                        caption=f"YOLO result: {person_count} person(s) detected",
                        width="stretch",
                    )

                    baseline = safe_number(forecast.predicted_kwh)

                    advice = []

                    if str(forecast.energy_status) == "High":
                        advice.append(
                            "Predicted energy use is high: schedule "
                            "non-essential loads outside peak periods."
                        )

                    advice.append(occupancy_message)

                    a1, a2, a3 = st.columns(3)

                    a1.metric("Person Count", person_count)
                    a2.metric("Room Status", occupancy_level)
                    a3.metric("Energy Status", str(forecast.energy_status))

                    message = clear_yolo_recommendation(
                        person_count,
                        str(forecast.energy_status),
                        baseline if "baseline" in locals() else safe_number(forecast.predicted_kwh),
                    )

                    st.subheader("💡 YOLO Human-Presence Recommendation")
                    st.info(
                        f"👤 Detected **{person_count} person(s)** | "
                        f"Room status: **{occupancy_level}** | "
                        f"Predicted energy: **{baseline:.2f} kWh** | "
                        f"Energy status: **{forecast.energy_status}**"
                    )

                    if person_count == 0:
                        st.warning(message)
                    elif person_count >= 5:
                        st.error(message)
                    else:
                        st.success(message)

                    # -------------------------------------------------
                    # Transparent energy-impact indicator
                    # -------------------------------------------------
                    if person_count == 0:
                        saving_factor = 0.20
                    elif person_count <= 2:
                        saving_factor = 0.10
                    elif person_count <= 4:
                        saving_factor = 0.05
                    else:
                        saving_factor = 0.02

                    potential_saving = baseline * saving_factor
                    optimized_estimate = max(
                        0.0,
                        baseline - potential_saving,
                    )

                    st.divider()
                    st.subheader("AI Energy Impact Scenario")

                    i1, i2, i3 = st.columns(3)

                    i1.metric(
                        "Baseline Forecast",
                        f"{baseline:.2f} kWh",
                    )
                    i2.metric(
                        "Scenario Optimized",
                        f"{optimized_estimate:.2f} kWh",
                    )
                    i3.metric(
                        "Potential Saving",
                        f"{potential_saving:.2f} kWh",
                    )

                    st.caption(
                        "The optimized value is a rule-based scenario estimate "
                        "derived from the detected human-presence state. It is "
                        "not an independently measured saving and is not claimed "
                        "as a direct output of the historical forecasting model."
                    )

                    # -------------------------------------------------
                    # Save integrated event
                    # -------------------------------------------------
                    log_dir = REPORTS / "integration"
                    log_dir.mkdir(exist_ok=True)

                    event = pd.DataFrame(
                        [
                            {
                                "event_time": datetime.now().isoformat(
                                    timespec="seconds"
                                ),
                                "house_id": current_house,
                                "forecast_date": forecast.forecast_date.date(),
                                "predicted_kwh": baseline,
                                "energy_status": forecast.energy_status,
                                "person_present": person_present,
                                "person_count": person_count,
                                "occupancy_level": occupancy_level,
                                "scenario_optimized_kwh": optimized_estimate,
                                "potential_saving_kwh": potential_saving,
                                "saving_percent": saving_factor * 100,
                                "source_image": getattr(
                                    image_input,
                                    "name",
                                    "camera_capture",
                                ),
                            }
                        ]
                    )

                    log_path = (
                        log_dir / "occupancy_energy_events.csv"
                    )

                    event.to_csv(
                        log_path,
                        mode="a",
                        header=not log_path.exists(),
                        index=False,
                    )

                    st.caption(
                        f"Combined event saved: {log_path}"
                    )

                except Exception as exc:
                    st.error(
                        "YOLO detection failed. Check the YOLO model path "
                        "and environment."
                    )
                    st.exception(exc)

        else:
            st.info(
                "Upload or capture an image to create a real-time "
                "occupancy + energy event."
            )

        event_path = (
            REPORTS / "integration" / "occupancy_energy_events.csv"
        )

        if event_path.exists():
            st.divider()
            st.subheader("Recent Occupancy + Energy Events")

            events = pd.read_csv(event_path)

            if "event_time" in events.columns:
                events = events.sort_values(
                    "event_time",
                    ascending=False,
                )

            st.dataframe(
                events.head(10),
                width="stretch",
                hide_index=True,
            )


# =====================================================================
# YOLO
# =====================================================================
with yolo:
    st.subheader("🎯 YOLOv8 Person Detection")

    if not yolo_metrics:
        st.info(
            "YOLO test metrics are not available yet."
        )
    else:
        y1, y2, y3, y4 = st.columns(4)

        y1.metric(
            "Test Precision",
            f"{safe_number(yolo_metrics.get('precision')):.3f}",
        )
        y2.metric(
            "Test Recall",
            f"{safe_number(yolo_metrics.get('recall')):.3f}",
        )
        y3.metric(
            "Test mAP50",
            f"{safe_number(yolo_metrics.get('mAP50')):.3f}",
        )
        y4.metric(
            "Test mAP50-95",
            f"{safe_number(yolo_metrics.get('mAP50_95')):.3f}",
        )

        st.caption(
            f"Test set: {yolo_metrics.get('test_images', 0)} images, "
            f"{yolo_metrics.get('test_person_instances', 0)} "
            "annotated person instances."
        )

        st.divider()
        st.subheader("💡 Human Detection → Energy Recommendation")
        st.write(
            "Upload a room image in **Live Energy + YOLO**. "
            "YOLO counts people, determines the occupancy level, and combines "
            "that result with the selected house energy forecast."
        )
        st.info(
            "The recommendation is generated from two signals: "
            "**human presence** + **predicted energy status**."
        )
        st.markdown(
            "- **0 people:** switch off non-essential lights/devices after a safe delay.\n"
            "- **1–2 people:** keep essential comfort and switch off idle loads.\n"
            "- **3–4 people:** optimise AC/lighting while maintaining comfort.\n"
            "- **5+ people:** keep ventilation/safety active and avoid unnecessary high-load equipment."
        )

        y5, y6 = st.columns(2)

        with y5:
            p = yolo_reports / "test_metrics" / "BoxPR_curve.png"
            if p.exists():
                st.image(
                    str(p),
                    caption="Precision–Recall Curve",
                )

            p = yolo_reports / "test_metrics" / "confusion_matrix_normalized.png"
            if p.exists():
                st.image(
                    str(p),
                    caption="Normalised Confusion Matrix",
                )

        with y6:
            p = yolo_reports / "test_metrics" / "val_batch0_labels.jpg"
            if p.exists():
                st.image(
                    str(p),
                    caption="Ground-Truth Person Labels",
                )

            p = yolo_reports / "test_metrics" / "val_batch0_pred.jpg"
            if p.exists():
                st.image(
                    str(p),
                    caption="YOLOv8 Predicted Person Boxes",
                )

        st.success(
            "YOLO human detection is evaluated separately on the held-out "
            "test set. Live detections are then used as a current occupancy "
            "signal for the operational energy-recommendation layer."
        )
