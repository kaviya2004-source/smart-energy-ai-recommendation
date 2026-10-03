"""AI Smart Building Energy Management Dashboard.

Single source of truth: every threshold, saving formula and occupancy band
used here comes from src/common.py. The forecasting model is loaded through
src/lstm_net.py, so the dashboard always uses the exact same architecture and
feature list as the training and reporting scripts.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from common import (  # noqa: E402
    FINAL_MODEL, FINAL_MODEL_CHECKPOINT, HIGH_KWH_ALERT, MODELS, PROCESSED, REPORTS,
    STATUS_HIGH_RATIO, STATUS_LOW_RATIO, STATUS_SAVING_RATE, YOLO_CONFIDENCE,
    YOLO_MODEL, energy_advice, energy_status, forecast_saving_kwh, load_json,
    normalise_id, occupancy_energy_impact, season_name,
)
from lstm_net import LSTMBundle, load_lstm_checkpoint  # noqa: E402
from prepare_data import clean_name  # noqa: E402
from feature_engineering import engineer  # noqa: E402

# ===========================================================================
# PAGE CONFIG + THEME
# ===========================================================================
st.set_page_config(page_title="Smart Energy AI | Sarah Tucker College", page_icon="🏢", layout="wide")

PALETTE = {
    "ink": "#0E2A2F",
    "surface": "#F4F8F7",
    "card": "#FFFFFF",
    "line": "#D8E3E1",
    "brand": "#0F6E5E",
    "brand_soft": "#E3F1EC",
    "amber": "#C77D2E",
    "amber_soft": "#FBEFDF",
    "red": "#B84C3F",
    "red_soft": "#FBEAE6",
    "blue": "#2C6E9E",
    "muted": "#5B7370",
}

st.markdown(f"""
<style>
:root {{
    --ink: {PALETTE['ink']}; --surface: {PALETTE['surface']}; --card: {PALETTE['card']};
    --line: {PALETTE['line']}; --brand: {PALETTE['brand']}; --brand-soft: {PALETTE['brand_soft']};
    --amber: {PALETTE['amber']}; --amber-soft: {PALETTE['amber_soft']};
    --red: {PALETTE['red']}; --red-soft: {PALETTE['red_soft']}; --blue: {PALETTE['blue']};
    --muted: {PALETTE['muted']};
}}
html, body, [class*="css"] {{ font-family: "Source Sans Pro", "Segoe UI", sans-serif; }}

/* Project-themed watermark: a faint smart-building silhouette with a lit
   window grid + an energy bolt, plus a faint circuit/node pattern. Both are
   self-contained SVG (no external image, nothing to go missing offline) and
   sit at very low opacity behind a near-opaque gradient, so they read as a
   deliberate "smart building energy" motif without ever competing with the
   content on top of them. */
.stApp {{
    background:
        linear-gradient(180deg, rgba(244,248,247,.95), rgba(244,248,247,.975) 420px, var(--surface) 900px),
        url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='620' height='620' viewBox='0 0 620 620'%3E%3Cg opacity='0.055'%3E%3Crect x='70' y='230' width='150' height='340' fill='%230E2A2F'/%3E%3Crect x='235' y='160' width='120' height='410' fill='%230E2A2F'/%3E%3Crect x='370' y='280' width='100' height='290' fill='%230E2A2F'/%3E%3Cg fill='%23F4F8F7'%3E%3Crect x='85' y='250' width='16' height='16'/%3E%3Crect x='112' y='250' width='16' height='16'/%3E%3Crect x='139' y='250' width='16' height='16'/%3E%3Crect x='166' y='250' width='16' height='16'/%3E%3Crect x='193' y='250' width='16' height='16'/%3E%3Crect x='85' y='280' width='16' height='16'/%3E%3Crect x='112' y='280' width='16' height='16'/%3E%3Crect x='139' y='280' width='16' height='16'/%3E%3Crect x='166' y='280' width='16' height='16'/%3E%3Crect x='193' y='280' width='16' height='16'/%3E%3Crect x='85' y='310' width='16' height='16'/%3E%3Crect x='112' y='310' width='16' height='16'/%3E%3Crect x='139' y='310' width='16' height='16'/%3E%3Crect x='166' y='310' width='16' height='16'/%3E%3Crect x='193' y='310' width='16' height='16'/%3E%3Crect x='250' y='185' width='16' height='16'/%3E%3Crect x='277' y='185' width='16' height='16'/%3E%3Crect x='304' y='185' width='16' height='16'/%3E%3Crect x='331' y='185' width='16' height='16'/%3E%3Crect x='250' y='215' width='16' height='16'/%3E%3Crect x='277' y='215' width='16' height='16'/%3E%3Crect x='304' y='215' width='16' height='16'/%3E%3Crect x='331' y='215' width='16' height='16'/%3E%3Crect x='250' y='245' width='16' height='16'/%3E%3Crect x='277' y='245' width='16' height='16'/%3E%3Crect x='304' y='245' width='16' height='16'/%3E%3Crect x='331' y='245' width='16' height='16'/%3E%3Crect x='385' y='300' width='16' height='16'/%3E%3Crect x='412' y='300' width='16' height='16'/%3E%3Crect x='439' y='300' width='16' height='16'/%3E%3Crect x='385' y='330' width='16' height='16'/%3E%3Crect x='412' y='330' width='16' height='16'/%3E%3Crect x='439' y='330' width='16' height='16'/%3E%3C/g%3E%3Cpath d='M305 60 L275 150 L305 150 L285 230 L345 120 L312 120 Z' fill='%23C77D2E'/%3E%3C/g%3E%3Cg fill='none' stroke='%230F6E5E' stroke-opacity='0.09' stroke-width='1.2'%3E%3Cpath d='M20 20h70v70h60V40h70M20 180h40v-40h50v60h40v-80h70M470 450h70v70h60v-50h20M500 560h60v-40h50'/%3E%3Ccircle cx='20' cy='20' r='3.5' fill='%230F6E5E' fill-opacity='.13' stroke='none'/%3E%3Ccircle cx='150' cy='40' r='3.5' fill='%230F6E5E' fill-opacity='.13' stroke='none'/%3E%3Ccircle cx='470' cy='450' r='3.5' fill='%230F6E5E' fill-opacity='.13' stroke='none'/%3E%3Ccircle cx='600' cy='470' r='3.5' fill='%230F6E5E' fill-opacity='.13' stroke='none'/%3E%3C/g%3E%3C/svg%3E") bottom right no-repeat,
        var(--surface);
    background-size: auto, min(640px, 60vw), auto;
    background-attachment: fixed, fixed, fixed;
}}
.block-container {{ max-width: 1360px; padding-top: 1.4rem; padding-bottom: 3rem; }}

.hero {{
    position: relative; overflow: hidden; border-radius: 20px;
    padding: 30px 34px; margin-bottom: 20px;
    background: linear-gradient(120deg, #0E2A2F 0%, #0F6E5E 100%);
    box-shadow: 0 14px 34px rgba(14,42,47,.20);
}}
.hero:before {{
    content: ""; position: absolute; right: -60px; top: -90px;
    width: 240px; height: 240px; border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,.10), transparent 65%);
}}
.hero-eyebrow {{ color: rgba(255,255,255,.72); font-size: .82rem; letter-spacing: .02em; margin: 0 0 6px 0; }}
.hero-title {{ color: #fff; font-size: 1.9rem; font-weight: 700; margin: 0 0 8px 0; line-height: 1.25; }}
.hero-sub {{ color: rgba(255,255,255,.86); font-size: .97rem; max-width: 760px; margin: 0; line-height: 1.55; }}
.hero-badges {{ margin-top: 16px; display: flex; gap: 10px; flex-wrap: wrap; }}
.hero-badge {{
    display: inline-flex; align-items: center; gap: 6px; padding: 6px 13px;
    border-radius: 999px; background: rgba(255,255,255,.14); color: #fff;
    font-size: .80rem; border: 1px solid rgba(255,255,255,.20);
}}

.card {{
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 18px 20px; box-shadow: 0 4px 14px rgba(14,42,47,.05);
}}
.stat-card {{
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 16px 18px;
}}
.stat-label {{ color: var(--muted); font-size: .82rem; margin-bottom: 4px; }}
.stat-value {{ color: var(--ink); font-size: 1.55rem; font-weight: 700; line-height: 1.1; }}
.stat-foot {{ color: var(--muted); font-size: .78rem; margin-top: 4px; }}

.note {{
    border-radius: 12px; padding: 12px 16px; font-size: .88rem; line-height: 1.5;
    border-left: 4px solid var(--blue); background: #EAF2F8; color: var(--ink);
}}
.note-warn {{ border-left-color: var(--amber); background: var(--amber-soft); }}
.note-bad {{ border-left-color: var(--red); background: var(--red-soft); }}

.badge {{
    display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .78rem; font-weight: 600;
}}
.badge-high {{ background: var(--red-soft); color: var(--red); }}
.badge-normal {{ background: var(--brand-soft); color: var(--brand); }}
.badge-low {{ background: #E7EEF6; color: var(--blue); }}

.reco-item {{
    padding: 11px 14px; border-radius: 10px; margin: 7px 0;
    background: var(--amber-soft); border-left: 4px solid var(--amber); font-size: .90rem; color: var(--ink);
}}

div[data-testid="stMetric"] {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 10px 14px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 10px 10px 0 0; padding: 8px 16px; }}
</style>
""", unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f"""
    <div class="hero"><div>
        <p class="hero-eyebrow">AI-Based Smart Building Energy Management System</p>
        <p class="hero-title">{title}</p>
        <p class="hero-sub">{subtitle}</p>
    </div></div>
    """, unsafe_allow_html=True)


def glass_pills(badges: list[str], height: int = 78) -> None:
    """Interactive glass-pill badge row.

    A thick, lens-like glass pill per badge: the background blurs and bends
    behind it, a highlight follows the pointer, and the pill tilts toward the
    cursor and presses down on click. The --mx/--my/--rx/--ry custom
    properties are registered with @property so the browser can smoothly
    interpolate them between pointer positions - a single `mousemove`
    listener per pill updates the target values and CSS does the animating,
    so there is no per-frame JS animation loop.
    """
    items = "".join(f'<div class="pill"><span>{b}</span></div>' for b in badges)
    html = f"""
    <style>
      @property --mx {{ syntax: '<percentage>'; inherits: false; initial-value: 50%; }}
      @property --my {{ syntax: '<percentage>'; inherits: false; initial-value: 50%; }}
      @property --rx {{ syntax: '<angle>'; inherits: false; initial-value: 0deg; }}
      @property --ry {{ syntax: '<angle>'; inherits: false; initial-value: 0deg; }}
      * {{ box-sizing: border-box; }}
      body {{ margin: 0; background: transparent; font-family: "Source Sans Pro","Segoe UI",sans-serif; }}
      .row {{ display: flex; flex-wrap: wrap; gap: 12px; padding: 4px 2px; perspective: 700px; }}
      .pill {{
          --mx: 50%; --my: 50%; --rx: 0deg; --ry: 0deg;
          position: relative; overflow: hidden; cursor: pointer;
          padding: 10px 20px; border-radius: 999px;
          color: #fff; font-size: .86rem; font-weight: 600; letter-spacing: .01em;
          background:
              radial-gradient(120px circle at var(--mx) var(--my), rgba(255,255,255,.55), transparent 62%),
              linear-gradient(135deg, rgba(255,255,255,.20), rgba(255,255,255,.04));
          backdrop-filter: blur(14px) saturate(140%);
          -webkit-backdrop-filter: blur(14px) saturate(140%);
          border: 1px solid rgba(255,255,255,.35);
          box-shadow:
              inset 0 1px 1px rgba(255,255,255,.55),
              inset 0 -6px 10px rgba(0,0,0,.18),
              0 8px 18px rgba(14,42,47,.28);
          transform: rotateX(var(--rx)) rotateY(var(--ry));
          transition: --mx .25s ease, --my .25s ease, --rx .18s ease, --ry .18s ease,
                      box-shadow .18s ease, transform .18s ease;
      }}
      .pill:active {{ transform: rotateX(var(--rx)) rotateY(var(--ry)) scale(.95) translateY(1px); }}
      .pill span {{ position: relative; z-index: 1; text-shadow: 0 1px 2px rgba(0,0,0,.25); }}
    </style>
    <div class="row" id="row">{items}</div>
    <script>
      document.querySelectorAll('.pill').forEach(function (pill) {{
        pill.addEventListener('mousemove', function (e) {{
          var r = pill.getBoundingClientRect();
          var x = (e.clientX - r.left) / r.width;
          var y = (e.clientY - r.top) / r.height;
          pill.style.setProperty('--mx', (x * 100).toFixed(1) + '%');
          pill.style.setProperty('--my', (y * 100).toFixed(1) + '%');
          pill.style.setProperty('--ry', ((x - 0.5) * 18).toFixed(1) + 'deg');
          pill.style.setProperty('--rx', ((0.5 - y) * 18).toFixed(1) + 'deg');
        }});
        pill.addEventListener('mouseleave', function () {{
          pill.style.setProperty('--mx', '50%'); pill.style.setProperty('--my', '50%');
          pill.style.setProperty('--rx', '0deg'); pill.style.setProperty('--ry', '0deg');
        }});
      }});
    </script>
    """
    components.html(html, height=height)


def stat_card(label: str, value: str, foot: str = "") -> None:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        <div class="stat-foot">{foot}</div>
    </div>
    """, unsafe_allow_html=True)


def note(text: str, kind: str = "info") -> None:
    cls = {"warn": "note-warn", "bad": "note-bad"}.get(kind, "")
    st.markdown(f'<div class="note {cls}">{text}</div>', unsafe_allow_html=True)


def status_badge(status: str) -> str:
    cls = {"High": "badge-high", "Normal": "badge-normal", "Low": "badge-low"}.get(status, "badge-normal")
    return f'<span class="badge {cls}">{status}</span>'


# ===========================================================================
# CACHED LOADERS  (all paths come from common.py - nothing hard-coded here)
#
# Every loader's cache key includes the file's last-modified time, so if you
# re-run a pipeline script and it rewrites a CSV/JSON/checkpoint, the very
# next rerun of the dashboard reads the new file automatically - no stale
# "previous output", and no need to restart Streamlit by hand. The sidebar's
# "Refresh data" button clears everything in one click as a manual fallback.
# ===========================================================================
def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@st.cache_data
def _read_csv_cached(path_str: str, mtime: float, parse_dates: tuple | None) -> pd.DataFrame:
    return pd.read_csv(path_str, parse_dates=list(parse_dates) if parse_dates else None)


def load_csv(name: str, parse_dates: list | None = None) -> pd.DataFrame:
    path = REPORTS / name
    if not path.exists():
        return pd.DataFrame()
    return _read_csv_cached(str(path), _mtime(path), tuple(parse_dates) if parse_dates else None)


@st.cache_data
def _read_energy_data_cached(path_str: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path_str, parse_dates=["timestamp"])
    df["house_id"] = normalise_id(df["house_id"])
    return df


def load_energy_data() -> pd.DataFrame:
    path = PROCESSED / "energy_model_data.csv"
    if not path.exists():
        return pd.DataFrame()
    return _read_energy_data_cached(str(path), _mtime(path))


@st.cache_data
def _read_json_cached(path_str: str, mtime: float) -> dict:
    return load_json(Path(path_str))


def load_json_cached(path: Path) -> dict:
    return _read_json_cached(str(path), _mtime(path))


@st.cache_resource
def _load_lstm_cached(checkpoint_mtime: float) -> LSTMBundle | None:
    try:
        return load_lstm_checkpoint()
    except Exception as error:
        st.session_state["_lstm_error"] = str(error)
        return None


def load_lstm() -> LSTMBundle | None:
    return _load_lstm_cached(_mtime(MODELS / FINAL_MODEL_CHECKPOINT))


@st.cache_resource
def _load_yolo_cached(model_mtime: float, model_path: str):
    try:
        from ultralytics import YOLO
        return YOLO(model_path)
    except Exception as error:
        st.session_state["_yolo_error"] = str(error)
        return None


def load_yolo():
    path = YOLO_MODEL if YOLO_MODEL.exists() else None
    if path is None:
        candidates = list((ROOT / "data" / "yolo" / "reports").rglob("best.pt"))
        path = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
    if path is None:
        st.session_state["_yolo_error"] = "no .pt file found under models/ or data/yolo/reports/"
        return None
    return _load_yolo_cached(_mtime(path), str(path))


def data_status() -> dict:
    """Which pipeline outputs exist, for the sidebar status list."""
    return {
        "Processed dataset": (PROCESSED / "energy_model_data.csv").exists(),
        "LSTM checkpoint": (MODELS / FINAL_MODEL_CHECKPOINT).exists(),
        "Model comparison": (REPORTS / "all_model_comparison.csv").exists(),
        "Recommendations": (REPORTS / "recommendations.csv").exists(),
        "Future forecast": (REPORTS / "future_energy_forecasts.csv").exists(),
        "XAI outputs": (REPORTS / "permutation_importance.csv").exists(),
        "YOLO model": YOLO_MODEL.exists(),
    }


# ===========================================================================
# PREDICTION HELPERS
# ===========================================================================
def build_prediction_window(raw: pd.DataFrame, bundle: LSTMBundle, house_id: str | None = None):
    """Turn a raw uploaded CSV into the most recent valid input window.

    Reuses the exact lag/temporal formulas from prepare_data.py and the exact
    engineer() function from feature_engineering.py, so a prediction made here
    is computed the same way as during training - nothing is approximated.
    Returns (window_df, meta_dict) or raises ValueError with a clear reason.
    """
    df = raw.rename(columns=clean_name).copy()
    for column in ("timestamp", "house_id", "kwh"):
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' not found after standardising headers.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["house_id"] = normalise_id(df["house_id"])
    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
    df = df.dropna(subset=["timestamp", "house_id", "kwh"])
    df = df.drop_duplicates(subset=["house_id", "timestamp"]).sort_values(["house_id", "timestamp"])

    if house_id is not None:
        df = df[df["house_id"] == str(house_id)]
    if df.empty:
        raise ValueError("No valid rows for the selected house after cleaning.")
    if df["house_id"].nunique() > 1 and house_id is None:
        raise ValueError("The file has more than one house_id; select a house to predict for.")

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day_of_month"] = df["timestamp"].dt.day
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["kwh_lag_1"] = df.groupby("house_id")["kwh"].shift(1)
    df["kwh_lag_7"] = df.groupby("house_id")["kwh"].shift(7)
    df["kwh_rolling_7"] = df.groupby("house_id")["kwh"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).mean())

    df, _ = engineer(df)
    df = df.dropna(subset=["kwh_lag_1", "kwh_lag_7", "kwh_rolling_7"])

    sequence_days = bundle.sequence_days
    if len(df) < sequence_days:
        raise ValueError(
            f"Only {len(df)} usable day(s) after preparing lag features; the model needs "
            f"{sequence_days} consecutive days (upload at least {sequence_days + 7} raw rows)."
        )

    missing = [f for f in bundle.features if f not in df.columns]
    if missing:
        raise ValueError(
            "The uploaded file is missing columns the model needs and none were invented: "
            + ", ".join(missing)
        )

    window = df.tail(sequence_days)
    gap_days = int((window["timestamp"].iloc[-1] - window["timestamp"].iloc[0]).days)
    meta = {
        "house_id": window["house_id"].iloc[-1],
        "window_start": window["timestamp"].iloc[0],
        "window_end": window["timestamp"].iloc[-1],
        "consecutive": gap_days == sequence_days - 1,
        "baseline_kwh": float(window["kwh"].tail(7).mean()) if len(window) >= 1 else None,
        "rolling_7": float(window["kwh_rolling_7"].iloc[-1]),
    }
    return window, meta


def render_recommendation_block(prediction: float, baseline: float, extra: list[str] | None = None) -> None:
    status = energy_status(prediction, baseline)
    saving = forecast_saving_kwh(prediction, baseline, status)
    c1, c2, c3 = st.columns(3)
    with c1:
        stat_card("Predicted next-day energy", f"{prediction:.2f} kWh")
    with c2:
        st.markdown(f"**Status:** {status_badge(status)}", unsafe_allow_html=True)
        stat_card("7-day baseline", f"{baseline:.2f} kWh")
    with c3:
        stat_card("Assumed saving opportunity", f"{saving:.2f} kWh",
                  f"{STATUS_SAVING_RATE[status]*100:.0f}% of the excess above baseline")
    st.markdown(f'<div class="reco-item">{energy_advice(status, prediction)}</div>', unsafe_allow_html=True)
    for line in (extra or []):
        st.markdown(f'<div class="reco-item">{line}</div>', unsafe_allow_html=True)
    note("This is a rule-based estimate for planning purposes, not a measured field result.")


# ===========================================================================
# SIDEBAR
# ===========================================================================
energy_df = load_energy_data()
final_model_info = load_json_cached(REPORTS / "final_model.json")
comparison_df = load_csv("all_model_comparison.csv")

with st.sidebar:
    st.markdown("### ⚡ Smart Energy AI")
    st.caption("MCA Final Project — Sarah Tucker College")
    if st.button("🔄 Refresh data", use_container_width=True,
                help="Data refreshes automatically when a file changes. Use this only if "
                    "you still see old numbers after re-running a pipeline script."):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.divider()

    st.markdown("**Deployed model**")
    if final_model_info:
        st.success(f"{final_model_info.get('final_model', FINAL_MODEL)}  ·  "
                  f"Test RMSE {final_model_info.get('test_metrics', {}).get('RMSE', float('nan')):.2f} kWh")
    else:
        st.info(f"{FINAL_MODEL} (run compare_models.py to record test metrics)")

    st.divider()
    st.markdown("**Pipeline status**")
    for label, ok in data_status().items():
        st.markdown(f"{'✅' if ok else '⬜'} {label}")

    st.divider()
    if not energy_df.empty:
        years = sorted(energy_df["timestamp"].dt.year.unique())
        houses = sorted(energy_df["house_id"].unique())
        selected_years = st.multiselect("Year filter", years, default=years)
        selected_houses = st.multiselect("House filter", houses, default=houses[: min(10, len(houses))])
    else:
        selected_years, selected_houses = [], []
        st.warning("energy_model_data.csv not found yet.")

filtered = pd.DataFrame()
if not energy_df.empty:
    filtered = energy_df[
        energy_df["timestamp"].dt.year.isin(selected_years) & energy_df["house_id"].isin(selected_houses)
    ]

# ===========================================================================
# HERO
# ===========================================================================
hero(
    "AI-Based Smart Building Energy Management",
    "Deep-learning next-day energy forecasting, explainable AI, and YOLOv8 occupancy "
    "detection combined into one operational dashboard.",
)
glass_pills([
    f"Model: {FINAL_MODEL}",
    f"{energy_df['house_id'].nunique() if not energy_df.empty else '—'} houses",
    f"{len(energy_df):,} rows" if not energy_df.empty else "no data loaded",
])

tab_overview, tab_eda, tab_forecast, tab_models, tab_xai, tab_reco, tab_upload, tab_live = st.tabs(
    ["Overview", "Data & EDA", "Future Forecast", "Model Comparison",
     "Explainable AI", "Recommendations", "Upload & Predict", "Live Occupancy"]
)

# ===========================================================================
# TAB 1 — OVERVIEW
# ===========================================================================
with tab_overview:
    st.subheader("Project overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Dataset rows", f"{len(energy_df):,}" if not energy_df.empty else "—")
    with c2:
        stat_card("Houses", f"{energy_df['house_id'].nunique()}" if not energy_df.empty else "—")
    with c3:
        span = (f"{energy_df['timestamp'].min():%b %Y} – {energy_df['timestamp'].max():%b %Y}"
               if not energy_df.empty else "—")
        stat_card("Date range", span)
    with c4:
        rmse = final_model_info.get("test_metrics", {}).get("RMSE") if final_model_info else None
        stat_card("Deployed model", FINAL_MODEL, f"Test RMSE {rmse:.2f} kWh" if rmse else "not yet evaluated")

    st.markdown("###")
    lstm_predictions = load_csv("predictions_lstm.csv", parse_dates=["timestamp"])
    left, right = st.columns([2, 1])
    with left:
        st.markdown("**Actual vs predicted — held-out test set**")
        if lstm_predictions.empty:
            note("reports/predictions_lstm.csv not found. Run train_dl.py.", "warn")
        else:
            sample = lstm_predictions.sort_values("timestamp").tail(200)
            fig = go.Figure()
            fig.add_scatter(x=sample["timestamp"], y=sample["actual"], name="Actual",
                            line=dict(color=PALETTE["ink"], width=2))
            fig.add_scatter(x=sample["timestamp"], y=sample["prediction"], name="Predicted",
                            line=dict(color=PALETTE["brand"], width=2, dash="dot"))
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                              legend=dict(orientation="h", y=1.1), paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            note("Shown on the model's own scale: this is the test split the model never trained on.")
    with right:
        st.markdown("**Energy status mix (recommendations)**")
        reco = load_csv("recommendations.csv", parse_dates=["timestamp"])
        if reco.empty or "energy_status" not in reco.columns:
            note("reports/recommendations.csv not found. Run explain_and_recommend.py.", "warn")
        else:
            counts = reco["energy_status"].value_counts()
            fig = px.pie(values=counts.values, names=counts.index, hole=0.6,
                        color=counts.index,
                        color_discrete_map={"High": PALETTE["red"], "Normal": PALETTE["brand"],
                                            "Low": PALETTE["blue"]})
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            avg_saving = reco["estimated_saving_kwh"].mean()
            stat_card("Avg. estimated saving / house-day", f"{avg_saving:.2f} kWh")

    st.markdown("###")
    st.markdown("**Pipeline**")
    steps = ["Data cleaning", "Feature engineering", "ML + DL model training",
            "Model comparison", "XAI (SHAP)", "YOLOv8 detection", "Recommendation engine", "Dashboard"]
    cols = st.columns(len(steps))
    for column, step, index in zip(cols, steps, range(1, len(steps) + 1)):
        with column:
            st.markdown(f"<div class='card' style='text-align:center;padding:14px 8px;'>"
                       f"<div style='color:{PALETTE['brand']};font-weight:700;'>{index}</div>"
                       f"<div style='font-size:.82rem;color:{PALETTE['ink']};'>{step}</div></div>",
                       unsafe_allow_html=True)

# ===========================================================================
# TAB 2 — DATA & EDA
# ===========================================================================
with tab_eda:
    st.subheader("Exploratory data analysis")
    quality = load_json_cached(REPORTS / "eda_data_quality_summary.json")
    if quality:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            stat_card("Zero-kWh records", f"{quality['zero_kwh_records']:,}", f"{quality['zero_kwh_percent']}%")
        with c2:
            stat_card("Duplicate records", f"{quality['duplicate_records']:,}")
        with c3:
            stat_card("Outlier records (IQR)", f"{quality['outlier_records']:,}", f"{quality['outlier_percent']}%")
        with c4:
            stat_card("Mean daily use", f"{quality['kwh_mean']:.2f} kWh", f"median {quality['kwh_median']:.2f}")
    else:
        note("Run eda.py to populate data-quality statistics.", "warn")

    if not filtered.empty:
        st.markdown("###")
        view = st.radio("Aggregate by", ["Daily", "Monthly", "Yearly", "Seasonal"], horizontal=True)
        work = filtered.copy()
        if view == "Daily":
            series = work.groupby("timestamp")["kwh"].mean().reset_index()
            fig = px.line(series, x="timestamp", y="kwh")
        elif view == "Monthly":
            work["period"] = work["timestamp"].dt.to_period("M").astype(str)
            series = work.groupby("period")["kwh"].mean().reset_index()
            fig = px.bar(series, x="period", y="kwh")
        elif view == "Yearly":
            work["year"] = work["timestamp"].dt.year
            series = work.groupby("year")["kwh"].mean().reset_index()
            fig = px.bar(series, x="year", y="kwh")
        else:
            work["season"] = work["timestamp"].dt.month.map(season_name)
            series = work.groupby("season")["kwh"].mean().reindex(
                ["Winter", "Summer", "Monsoon", "Post-monsoon"]).reset_index()
            fig = px.bar(series, x="season", y="kwh")
        if view == "Daily":
            fig.update_traces(line_color=PALETTE["brand"])
        else:
            fig.update_traces(marker_color=PALETTE["brand"])
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="kWh",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        note("No rows match the current sidebar filters.", "warn")

    st.markdown("###")
    figure_cols = st.columns(2)
    figures = ["eda_daily_energy_trend.png", "eda_monthly_energy.png",
              "eda_kwh_distribution.png", "eda_correlation_heatmap.png"]
    for index, name in enumerate(figures):
        path = REPORTS / name
        if path.exists():
            with figure_cols[index % 2]:
                st.image(str(path), use_container_width=True, caption=name.replace("_", " ").replace(".png", ""))

# ===========================================================================
# TAB 3 — FUTURE FORECAST
# ===========================================================================
with tab_forecast:
    st.subheader("Future energy forecast")
    future = load_csv("future_energy_forecasts.csv", parse_dates=["forecast_date"])
    future_meta = load_json_cached(REPORTS / "future_forecast_metadata.json")
    if future.empty:
        note("reports/future_energy_forecasts.csv not found. Run generate_future_forecasts.py.", "warn")
    else:
        note(future_meta.get("note", "Recursive multi-day forecasts are scenario projections; "
                                     "only the next-day forecast is validated on the test set."), "warn")
        future["house_id"] = normalise_id(future["house_id"])
        house_options = sorted(future["house_id"].unique())
        house_choice = st.selectbox("House", house_options)
        house_forecast = future[future["house_id"] == house_choice].sort_values("forecast_date")

        fig = go.Figure()
        fig.add_scatter(x=house_forecast["forecast_date"], y=house_forecast["predicted_kwh"],
                        name="Predicted kWh", line=dict(color=PALETTE["brand"], width=2))
        fig.add_scatter(x=house_forecast["forecast_date"], y=house_forecast["baseline_kwh"],
                        name="7-day baseline", line=dict(color=PALETTE["muted"], width=1, dash="dot"))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=1.1), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        if "horizon_day" in house_forecast.columns:
            far = house_forecast[house_forecast["horizon_day"] > 30]
            if not far.empty:
                note(f"{len(far)} of these rows are more than 30 days out (horizon_day > 30); "
                    "treat them as an illustrative trend, not a precise forecast.", "warn")

        st.markdown("**All-house summary**")
        summary = load_csv("future_forecast_summary.csv", parse_dates=["forecast_date"])
        if not summary.empty:
            fig2 = px.line(summary, x="forecast_date", y="predicted_kwh")
            fig2.update_traces(line_color=PALETTE["blue"])
            fig2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

# ===========================================================================
# TAB 4 — MODEL COMPARISON
# ===========================================================================
with tab_models:
    st.subheader("Model comparison")
    if comparison_df.empty:
        note("reports/all_model_comparison.csv not found. Run build_model_comparison.py "
            "after train_ml.py and train_dl.py.", "warn")
    else:
        type_colors = {"Machine Learning": PALETTE["blue"], "Deep Learning": PALETTE["brand"],
                      "Hybrid (weighted blend)": PALETTE["amber"]}
        metric = st.radio("Rank by", ["RMSE", "MAE", "R2", "MAPE"], horizontal=True,
                          help="RMSE and MAE: lower is better. R2: higher is better. MAPE: lower is better.")
        available_metrics = [m for m in ["RMSE", "MAE", "R2", "MAPE"] if m in comparison_df.columns]
        metric = metric if metric in available_metrics else available_metrics[0]
        ascending = metric != "R2"
        ranked = comparison_df.sort_values(metric, ascending=ascending).reset_index(drop=True)

        bar_colors = [type_colors.get(t, PALETTE["muted"]) for t in ranked["Type"]]
        line_colors = [PALETTE["ink"] if d else "rgba(0,0,0,0)" for d in ranked["Deployed"]]
        hover = [
            f"<b>{row.Model}</b><br>{row.Type}<br>"
            f"MAE {row.MAE:.3f} · RMSE {row.RMSE:.3f} · R² {row.R2:.3f}"
            + (f" · MAPE {row.MAPE:.2f}%" if "MAPE" in ranked.columns and pd.notna(row.MAPE) else "")
            + ("<br><b>★ Deployed model</b>" if row.Deployed else "")
            for row in ranked.itertuples()
        ]
        fig = go.Figure(go.Bar(
            x=ranked[metric], y=ranked["Model"], orientation="h",
            marker=dict(color=bar_colors, line=dict(color=line_colors, width=2.5)),
            text=[f"{v:.3f}" for v in ranked[metric]], textposition="outside",
            hovertext=hover, hoverinfo="text",
        ))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            height=max(320, 46 * len(ranked)), margin=dict(l=10, r=40, t=10, b=10),
            xaxis_title=f"Test {metric}" + (" (kWh)" if metric in ("RMSE", "MAE") else ""),
            yaxis_title="", showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        legend_bits = " · ".join(f'<span style="color:{c}">●</span> {t}' for t, c in type_colors.items())
        st.markdown(f'<div style="font-size:.82rem;color:{PALETTE["muted"]};margin:4px 0 10px;">'
                   f'{legend_bits} · dark outline = deployed model. Hover a bar for full metrics.</div>',
                   unsafe_allow_html=True)

        show = comparison_df.sort_values("RMSE").copy()
        show["Deployed"] = show["Deployed"].map({True: "✅", False: ""})
        st.dataframe(show[["Rank", "Model", "Type", "MAE", "RMSE", "R2", "Deployed"]],
                    use_container_width=True, hide_index=True)

        dl_info = (final_model_info or {}).get("deep_learning_comparison", {})
        lowest_in_dl = dl_info.get("lowest_rmse_model")
        if lowest_in_dl and lowest_in_dl != FINAL_MODEL:
            note(f"'{lowest_in_dl}' has a lower test RMSE than the deployed {FINAL_MODEL} in this run. "
                f"{FINAL_MODEL} remains deployed for practical reasons (sequence modelling of daily "
                "consumption) — state this trade-off explicitly in the report rather than claiming "
                f"{FINAL_MODEL} has the lowest error.", "bad")
        elif not comparison_df.empty:
            deployed_is_lowest = bool(comparison_df.sort_values("RMSE").iloc[0]["Deployed"])
            if not deployed_is_lowest:
                best = comparison_df.sort_values("RMSE").iloc[0]["Model"]
                note(f"'{best}' has the lowest test RMSE in all_model_comparison.csv, not the "
                    f"deployed {FINAL_MODEL}. Report this honestly.", "bad")

# ===========================================================================
# TAB 5 — EXPLAINABLE AI
# ===========================================================================
with tab_xai:
    st.subheader("Explainable AI")
    xai_meta = load_json_cached(REPORTS / "xai_metadata.json")
    if xai_meta:
        note(xai_meta.get("note", ""), "warn")
        st.caption(f"XAI computed on: **{xai_meta.get('xai_model', '—')}**  |  "
                  f"Deployed forecasting model: **{FINAL_MODEL}**")

    perm = load_csv("permutation_importance.csv")
    left, right = st.columns(2)
    with left:
        st.markdown("**Permutation importance**")
        if perm.empty:
            note("reports/permutation_importance.csv not found.", "warn")
        else:
            show_variability = st.checkbox(
                "Show variability across the 8 repeats", value=False,
                help="Each feature is shuffled 8 times; the whisker shows how much the "
                    "importance score moved across those repeats. A short whisker means "
                    "a stable, trustworthy ranking; a long one means treat the ranking "
                    "with caution for that feature.")
            top = perm.head(15).sort_values("importance_mean")
            fig = px.bar(top, x="importance_mean", y="feature", orientation="h",
                        error_x="importance_std" if show_variability else None)
            fig.update_traces(marker_color=PALETTE["brand"],
                              error_x=dict(color=PALETTE["muted"], thickness=1.3, width=3))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="Mean importance (drop in accuracy when shuffled)",
                              yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Longer bars = the model relies on that feature more. Permutation "
                      "importance measures the drop in accuracy when a feature's values "
                      "are randomly shuffled, so larger = more important.")
    with right:
        st.markdown("**SHAP global importance**")
        shap_png = REPORTS / "shap_global_importance.png"
        shap_csv = load_csv("shap_global_importance.csv")
        if shap_png.exists():
            st.image(str(shap_png), use_container_width=True)
        elif not shap_csv.empty:
            fig = px.bar(shap_csv.head(15).sort_values("mean_abs_shap"),
                        x="mean_abs_shap", y="feature", orientation="h")
            fig.update_traces(marker_color=PALETTE["amber"])
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            status_text = (REPORTS / "shap_status.txt")
            note(status_text.read_text() if status_text.exists()
                else "SHAP output not found. Run explain_and_recommend.py.", "warn")

# ===========================================================================
# TAB 6 — RECOMMENDATIONS
# ===========================================================================
with tab_reco:
    st.subheader("Household recommendations")
    reco = load_csv("recommendations.csv", parse_dates=["timestamp"])
    reco_meta = load_json_cached(REPORTS / "recommendations_metadata.json")
    if reco.empty:
        note("reports/recommendations.csv not found. Run explain_and_recommend.py.", "warn")
    else:
        if reco_meta:
            st.caption(f"Predictions from: **{reco_meta.get('prediction_model', '—')}**  "
                      f"({reco_meta.get('alignment', '')})")
        reco["house_id"] = normalise_id(reco["house_id"])
        house_choice = st.selectbox("Filter by house", ["All"] + sorted(reco["house_id"].unique()),
                                    key="reco_house")
        view = reco if house_choice == "All" else reco[reco["house_id"] == house_choice]

        c1, c2, c3 = st.columns(3)
        with c1:
            stat_card("Rows", f"{len(view):,}")
        with c2:
            stat_card("Avg. predicted kWh", f"{view['prediction'].mean():.2f}")
        with c3:
            stat_card("Total estimated saving", f"{view['estimated_saving_kwh'].sum():.1f} kWh")

        display = view.sort_values("timestamp", ascending=False).head(200).copy()
        display["timestamp"] = display["timestamp"].dt.strftime("%d-%b-%Y")
        st.dataframe(
            display[["timestamp", "house_id", "prediction", "baseline_kwh", "energy_status",
                    "estimated_saving_kwh", "recommendations"]],
            use_container_width=True, hide_index=True,
        )
        note("Saving = max(0, predicted − 7-day baseline) × assumed reduction rate "
            "(High 10%, Normal 5%, Low 2%). This is a planning estimate, not a measured result.")

    with st.expander("Occupancy-aware recommendation logic"):
        rows = []
        for limit, level, factor, advice in [
            (0, "Empty room", 0.20, "no people detected"),
            (2, "Low occupancy", 0.10, "1–2 people"),
            (4, "Moderate occupancy", 0.05, "3–4 people"),
            (None, "High occupancy", 0.02, "5+ people"),
        ]:
            rows.append({"People detected": advice, "Level": level, "Assumed saving": f"{factor*100:.0f}%"})
        st.table(pd.DataFrame(rows))
        note(f"Status bands: High ≥ {STATUS_HIGH_RATIO*100:.0f}% of baseline, "
            f"Low ≤ {STATUS_LOW_RATIO*100:.0f}% of baseline, else Normal. "
            f"High-use alert also fires at ≥ {HIGH_KWH_ALERT:.0f} kWh regardless of baseline.")

# ===========================================================================
# TAB 7 — UPLOAD & PREDICT
# ===========================================================================
with tab_upload:
    st.subheader("Upload a dataset and predict with the deployed LSTM")
    bundle = load_lstm()
    if bundle is None:
        note(f"Could not load the LSTM checkpoint ({st.session_state.get('_lstm_error', 'unknown error')}). "
            "Train the model first (train_dl.py).", "bad")
    else:
        st.caption(f"Model expects **{bundle.sequence_days} consecutive daily rows** and "
                  f"**{len(bundle.features)} features** per row (read from the checkpoint).")
        uploaded = st.file_uploader("CSV with at least timestamp, house_id, kwh "
                                    f"(and ideally {bundle.sequence_days + 7}+ rows per house)", type=["csv"])
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
                st.dataframe(raw.head(8), use_container_width=True)
                candidate_ids = None
                cleaned_preview = raw.rename(columns=clean_name)
                if "house_id" in cleaned_preview.columns:
                    candidate_ids = sorted(cleaned_preview["house_id"].dropna().astype(str).unique())
                house_pick = (st.selectbox("House to predict for", candidate_ids)
                             if candidate_ids and len(candidate_ids) > 1 else None)

                if st.button("Run prediction", type="primary"):
                    window, meta = build_prediction_window(raw, bundle, house_id=house_pick)
                    if not meta["consecutive"]:
                        note("The most recent rows used are not on consecutive calendar days; "
                            "the forecast may be less reliable.", "warn")
                    prediction = bundle.predict(window)
                    st.markdown(f"**Prediction window:** house `{meta['house_id']}`, "
                              f"{meta['window_start']:%d-%b-%Y} → {meta['window_end']:%d-%b-%Y}  "
                              f"→ forecast for **{meta['window_end'] + pd.Timedelta(days=1):%d-%b-%Y}**")
                    render_recommendation_block(prediction, meta["rolling_7"] or meta["baseline_kwh"])
            except ValueError as error:
                note(str(error), "bad")
            except Exception as error:
                note(f"Could not process this file: {error}", "bad")

# ===========================================================================
# TAB 8 — LIVE OCCUPANCY (YOLOv8 + energy impact)
# ===========================================================================
with tab_live:
    st.subheader("Live occupancy detection and energy impact")
    yolo_metrics = load_json_cached(ROOT / "data" / "yolo" / "reports" / "test_metrics_summary.json")
    if yolo_metrics:
        cols = st.columns(4)
        keys = [k for k in yolo_metrics if any(m in k.lower() for m in ("precision", "recall", "map50"))][:4]
        for column, key in zip(cols, keys):
            with column:
                stat_card(key.split("/")[-1].replace("(B)", ""), f"{yolo_metrics[key]:.3f}")
    else:
        note("YOLO test_metrics_summary.json not found yet.", "warn")

    model = load_yolo()
    reco_all = load_csv("recommendations.csv", parse_dates=["timestamp"])

    left, right = st.columns([1, 1])
    with left:
        input_mode = st.radio("Image source", ["📁 Upload image", "📷 Take photo"], horizontal=True)
        if input_mode == "📷 Take photo":
            image_file = st.camera_input("Take a photo of the room")
        else:
            image_file = st.file_uploader("Upload a room / CCTV image", type=["jpg", "jpeg", "png"])
        confidence = st.slider("Detection confidence threshold", 0.10, 0.90, YOLO_CONFIDENCE, 0.05,
                               help="Lower this if people in frame are being missed; raise it if "
                                   "furniture or shadows are being flagged as people.")
        predicted_kwh = st.number_input(
            "Predicted next-day energy for this house (kWh) — from the Upload & Predict "
            "or Recommendations tab", min_value=0.0, value=20.0, step=0.5)

    with right:
        if image_file is not None:
            # exif_transpose fixes sideways/upside-down phone and webcam captures
            # before detection, which otherwise silently tanks accuracy.
            image = ImageOps.exif_transpose(Image.open(image_file)).convert("RGB")
            st.image(image, caption="Captured image", use_container_width=True)
            if model is None:
                note(f"YOLO model not available ({st.session_state.get('_yolo_error', 'not found')}).", "bad")
            else:
                # Pass the PIL Image object itself, NOT np.array(image): Ultralytics
                # reads a raw numpy array as BGR (OpenCV convention). A PIL image is
                # RGB, so np.array(image) silently swaps the colour channels and the
                # model's confidence collapses - this was why real people in frame
                # were detected as 0 persons.
                results = model.predict(source=image, conf=confidence, verbose=False)
                person_count = int(len(results[0].boxes)) if results and results[0].boxes is not None else 0
                st.image(results[0].plot()[:, :, ::-1], caption=f"Detected: {person_count} person(s)",
                        use_container_width=True)

                if person_count == 0:
                    note("No person detected. Try: better lighting, the full person in frame, "
                        "or lowering the confidence slider above.", "warn")

                impact = occupancy_energy_impact(predicted_kwh, person_count)
                c1, c2, c3 = st.columns(3)
                with c1:
                    stat_card("People detected", str(person_count))
                with c2:
                    stat_card("Occupancy level", impact["level"])
                with c3:
                    stat_card("Potential saving", f"{impact['potential_saving_kwh']:.2f} kWh",
                              f"{impact['saving_percent']:.0f}% of forecast")
                st.markdown(f'<div class="reco-item">{impact["advice"]}</div>', unsafe_allow_html=True)
                note("This is a rule-based scenario (predicted kWh × occupancy-band saving factor), "
                    "not a measured saving.")

    if not reco_all.empty:
        with st.expander("Recent predictions available to reuse above"):
            st.dataframe(reco_all.sort_values("timestamp", ascending=False)
                        [["timestamp", "house_id", "prediction", "energy_status"]].head(20),
                        use_container_width=True, hide_index=True)

st.markdown("###")
st.caption(f"AI-Based Smart Building Energy Management System · MCA Final Project · "
          f"Sarah Tucker College · Guide: Dr. Jairuby · Generated {datetime.now():%d %b %Y}")
