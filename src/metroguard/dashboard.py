"""Streamlit research dashboard for MetroGuard."""

from __future__ import annotations

import json
from typing import Any, cast

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from metroguard.api import ScoringService, SensorReading
from metroguard.config import load_config
from metroguard.metrics import EVENTS

st.set_page_config(page_title="MetroGuard", page_icon="🚇", layout="wide")
config = load_config()


@st.cache_resource
def service() -> ScoringService:
    return ScoringService(config)


@st.cache_data
def metrics() -> dict[str, Any] | None:
    path = config.root / "reports" / "metrics.json"
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8"))) if path.exists() else None


@st.cache_data
def holdout_scores() -> pd.DataFrame | None:
    path = config.root / "reports" / "holdout_scores.parquet"
    return pd.read_parquet(path) if path.exists() else None


def banner() -> None:
    st.title("🚇 MetroGuard")
    st.caption("Explainable early-warning anomaly detection for a metro air compressor")
    st.warning(
        "Research demo — one APU and four consolidated air-leak events. Not validated for "
        "safety-critical or production use, and it does not estimate remaining useful life."
    )


def overview(report: dict[str, Any] | None) -> None:
    st.subheader("Pre-registered study design")
    st.markdown(
        "MetroGuard learns only from the February no-reported-failure reference period. "
        "All scaling and thresholds are locked before the March-September holdout."
    )
    if report is None:
        st.info("Run `metroguard reproduce` to generate the benchmark report.")
        return
    model_name = str(report["primary_model"])
    primary = report["models"][model_name]
    upper = st.columns(2)
    lower = st.columns(2)
    upper[0].metric("Early event recall", str(primary["early_event_recall"]))
    upper[1].metric("False alarms/day", f"{float(primary['false_alarms_per_day']):.3f}")
    lead = primary["lead_time_median_hours"]
    lower[0].metric("Median lead time", "—" if lead is None else f"{float(lead):.1f} h")
    lower[1].metric("Failure-window PR-AUC", f"{float(primary['pr_auc_official_failure_intervals']):.3f}")
    st.image(str(config.root / "reports" / "figures" / "model_comparison.png"), use_container_width=True)


def benchmark(report: dict[str, Any] | None) -> None:
    st.subheader("Locked holdout benchmark")
    if report is None:
        st.info("No metrics artifact is available yet.")
        return
    rows = []
    for name, values in report["models"].items():
        rows.append(
            {
                "Model": name,
                "Early recall": values["early_event_recall"],
                "False alarms/day": values["false_alarms_per_day"],
                "PR-AUC": values["pr_auc_official_failure_intervals"],
                "Time in alert (%)": values["time_in_alert_percent"],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.image(str(config.root / "reports" / "figures" / "holdout_timeline.png"), use_container_width=True)
    st.caption("The official incident bands assess detection; the blue bands show the 24-2 h early-warning windows.")


def exploratory_data() -> None:
    st.subheader("Sensor and incident explorer")
    upload = st.file_uploader("Upload MetroPT-3-schema CSV", type="csv")
    demo_path = config.root / "data" / "demo" / "replay.csv"
    frame: pd.DataFrame | None = None
    if upload is not None:
        frame = pd.read_csv(upload)
    elif demo_path.exists():
        frame = pd.read_csv(demo_path)
        st.caption("Showing the attributed CC BY 4.0 replay extract around the second incident.")
    if frame is None:
        st.info("Run the reproduction pipeline or upload a same-schema CSV.")
        return
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    choices = [column for column in ["tp2", "tp3", "reservoirs", "oil_temperature", "motor_current"] if column in frame]
    selected = st.multiselect("Signals", choices, default=choices[:3])
    if selected:
        long = frame.melt(id_vars="timestamp", value_vars=selected, var_name="sensor", value_name="value")
        chart = px.line(long, x="timestamp", y="value", color="sensor", title="Historical sensor replay")
        for event in EVENTS:
            chart.add_vrect(x0=event.start, x1=event.end, fillcolor="#ef8a62", opacity=0.2, line_width=0)
        st.plotly_chart(chart, use_container_width=True)


def replay() -> None:
    st.subheader("Historical alert replay")
    speed = st.slider("Replay speed", min_value=1, max_value=20, value=5)
    st.caption(f"Playback setting: {speed}x (visual control only)")
    scores = holdout_scores()
    if scores is None:
        st.info("Run `metroguard reproduce` to generate historical scores.")
        return
    scores["timestamp"] = pd.to_datetime(scores["timestamp"])
    options = list(scores["timestamp"].iloc[:: max(1, len(scores) // 500)])
    start, end = st.select_slider(
        "Time range",
        options=options,
        value=(options[0], options[-1]),
    )
    subset = scores.loc[scores["timestamp"].between(start, end)]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=subset["timestamp"], y=subset["tcn_autoencoder__smoothed_score"], name="Score"))
    figure.add_trace(go.Scatter(x=subset["timestamp"], y=subset["tcn_autoencoder__threshold"], name="Threshold", line={"dash": "dash"}))
    figure.add_trace(go.Scatter(x=subset["timestamp"], y=subset["tcn_autoencoder__alert"].astype(int), name="Alert", yaxis="y2", fill="tozeroy"))
    figure.update_layout(yaxis2={"overlaying": "y", "side": "right", "range": [0, 1.2]}, height=480)
    st.plotly_chart(figure, use_container_width=True)


def score_uploaded() -> None:
    st.subheader("Score a bounded historical window")
    upload = st.file_uploader("Upload at least 75 minutes of canonical sensor readings", type="csv", key="score")
    if upload is None:
        return
    frame = pd.read_csv(upload)
    try:
        readings = [SensorReading.model_validate(record) for record in frame.to_dict("records")]
        result = service().score(readings)
    except Exception as error:  # Streamlit must surface validation errors without crashing
        st.error(str(error))
        return
    st.metric("Anomaly score", f"{result.score:.3f}", delta=f"threshold {result.threshold:.3f}")
    st.error("ALERT") if result.alert else st.success("No sustained alert")
    st.dataframe(pd.DataFrame([item.model_dump() for item in result.top_contributors]), hide_index=True)


banner()
report = metrics()
tabs = st.tabs(["Overview", "EDA", "Benchmark", "Replay", "Score CSV", "Limitations"])
with tabs[0]:
    overview(report)
with tabs[1]:
    exploratory_data()
with tabs[2]:
    benchmark(report)
with tabs[3]:
    replay()
with tabs[4]:
    score_uploaded()
with tabs[5]:
    st.markdown("""
### What this project does not claim

- The four consolidated incidents cannot establish broad operational generalization.
- February has no reported failure; it is not guaranteed to be perfectly healthy.
- High anomaly contribution is not proof of root cause or causality.
- The model does not estimate RUL and must not drive safety-critical maintenance.
""")
