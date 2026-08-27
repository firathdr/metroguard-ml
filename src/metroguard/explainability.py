"""SHAP-based explanations for MetroGuard's tree-model comparator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metroguard.features import sensor_from_feature


@dataclass(frozen=True)
class IsolationForestShapResult:
    """Global and local SHAP outputs for sampled scaled windows.

    TreeSHAP returns contributions for ``score_samples`` where a lower value is
    more anomalous. MetroGuard therefore negates the values so that positive
    contributions consistently mean "more anomalous" in the portfolio plots.
    """

    anomaly_shap_values: np.ndarray
    sampled_indices: np.ndarray
    sensor_importance: pd.DataFrame
    sensor_time_importance: pd.DataFrame
    local_sensor_contributions: pd.DataFrame
    local_sample_position: int


def explain_isolation_forest(
    model: Any,
    scaled_windows: np.ndarray,
    feature_names: list[str],
    sequence_bins: int,
    *,
    sample_size: int = 256,
) -> IsolationForestShapResult:
    """Compute sampled TreeSHAP explanations and aggregate them by sensor.

    The model is explained on the same flattened, robust-scaled windows used at
    inference time. Aggregating engineered statistics back to their originating
    sensor keeps the result readable without losing the faithful SHAP values.
    """
    if scaled_windows.ndim != 3 or len(scaled_windows) == 0:
        raise ValueError("scaled_windows must be a non-empty 3D array")
    if len(feature_names) != scaled_windows.shape[2]:
        raise ValueError("feature_names does not match the window feature count")
    if sequence_bins != scaled_windows.shape[1]:
        raise ValueError("sequence_bins does not match the window length")
    if sample_size < 1:
        raise ValueError("sample_size must be positive")

    import shap

    stride = max(1, len(scaled_windows) // sample_size)
    sampled_indices = np.arange(0, len(scaled_windows), stride)[:sample_size]
    sampled = scaled_windows[sampled_indices]
    flat = sampled.reshape(len(sampled), -1)
    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(flat)
    if isinstance(raw_values, list):
        raw_values = raw_values[0]
    shap_values = np.asarray(raw_values, dtype=float)
    if shap_values.shape != flat.shape:
        raise ValueError(
            f"Unexpected SHAP shape {shap_values.shape}; expected {flat.shape}"
        )

    # IsolationForest's score_samples is lower for anomalies. Flip direction
    # so every report can use the intuitive "positive = more anomalous" rule.
    anomaly_values = -shap_values.reshape(len(sampled), sequence_bins, len(feature_names))
    sensors = [sensor_from_feature(feature) for feature in feature_names]
    unique_sensors = list(dict.fromkeys(sensors))

    sensor_abs: dict[str, float] = {sensor: 0.0 for sensor in unique_sensors}
    sensor_time: dict[str, np.ndarray] = {
        sensor: np.zeros(sequence_bins, dtype=float) for sensor in unique_sensors
    }
    for feature_index, sensor in enumerate(sensors):
        feature_values = anomaly_values[:, :, feature_index]
        sensor_abs[sensor] += float(np.abs(feature_values).mean())
        sensor_time[sensor] += np.abs(feature_values).mean(axis=0)

    importance = pd.DataFrame(
        {
            "sensor": list(sensor_abs),
            "mean_absolute_shap": list(sensor_abs.values()),
        }
    ).sort_values("mean_absolute_shap", ascending=False, ignore_index=True)
    total = float(importance["mean_absolute_shap"].sum()) or 1.0
    importance["normalized_contribution"] = importance["mean_absolute_shap"] / total

    ordered_sensors = importance["sensor"].tolist()
    time_importance = pd.DataFrame(
        {sensor: sensor_time[sensor] for sensor in ordered_sensors},
        index=[f"t-{sequence_bins - index - 1}" for index in range(sequence_bins)],
    )
    time_importance.index.name = "relative_bin"

    anomaly_scores = -np.asarray(model.score_samples(flat), dtype=float)
    local_position = int(np.argmax(anomaly_scores))
    local_values: dict[str, float] = {sensor: 0.0 for sensor in unique_sensors}
    for feature_index, sensor in enumerate(sensors):
        local_values[sensor] += float(anomaly_values[local_position, :, feature_index].sum())
    local = pd.DataFrame(
        {
            "sensor": list(local_values),
            "shap_value": list(local_values.values()),
        }
    ).sort_values("shap_value", ascending=False, ignore_index=True)

    return IsolationForestShapResult(
        anomaly_shap_values=anomaly_values,
        sampled_indices=sampled_indices,
        sensor_importance=importance,
        sensor_time_importance=time_importance,
        local_sensor_contributions=local,
        local_sample_position=local_position,
    )


def plot_global_sensor_importance(result: pd.DataFrame, path: Path) -> None:
    """Write the global mean-|SHAP| sensor ranking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    top = result.head(12).sort_values("normalized_contribution")
    axis.barh(top["sensor"], top["normalized_contribution"], color="#2166ac")
    axis.set_xlabel("Normalized mean |SHAP value|")
    axis.set_title("Isolation Forest: global sensor importance")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_local_sensor_contributions(result: pd.DataFrame, path: Path) -> None:
    """Write a signed explanation for the most anomalous sampled window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    local = result.sort_values("shap_value")
    colors = ["#b2182b" if value > 0 else "#2166ac" for value in local["shap_value"]]
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.barh(local["sensor"], local["shap_value"], color=colors)
    axis.axvline(0.0, color="#333333", linewidth=0.8)
    axis.set_xlabel("SHAP contribution to anomaly score")
    axis.set_title("One anomalous window: positive values increase anomaly score")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_sensor_time_heatmap(result: pd.DataFrame, path: Path) -> None:
    """Write the mean absolute SHAP intensity by relative time bin and sensor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = result.sensor_time_importance
    figure, axis = plt.subplots(figsize=(11, 5.2))
    image = axis.imshow(matrix.to_numpy().T, aspect="auto", cmap="YlOrRd")
    axis.set_yticks(np.arange(len(matrix.columns)), matrix.columns)
    axis.set_xticks(np.arange(len(matrix.index)), matrix.index)
    axis.set_xlabel("Relative 5-minute bin (t-0 = latest)")
    axis.set_title("Where the global SHAP signal appears in the 60-minute window")
    figure.colorbar(image, ax=axis, label="Mean absolute SHAP")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
