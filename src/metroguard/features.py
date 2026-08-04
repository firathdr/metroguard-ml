"""Causal feature engineering for irregular MetroPT-3 signals."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import pandas as pd

from metroguard.schema import ANALOG_COLUMNS, DIGITAL_COLUMNS


def _transitions(series: pd.Series) -> float:
    values = series.dropna().to_numpy()
    return float(np.count_nonzero(values[1:] != values[:-1])) if len(values) > 1 else 0.0


def aggregate_causal_bins(
    frame: pd.DataFrame,
    *,
    bin_minutes: int = 5,
    expected_samples: int = 30,
) -> pd.DataFrame:
    """Aggregate raw samples without interpolation or future-looking operations."""
    indexed = frame.set_index("timestamp").sort_index().copy()
    indexed["tp3_minus_reservoirs"] = indexed["tp3"] - indexed["reservoirs"]
    indexed["tp2_minus_tp3"] = indexed["tp2"] - indexed["tp3"]
    rule = f"{bin_minutes}min"
    grouped = indexed.resample(rule, label="right", closed="right")

    parts: list[pd.DataFrame] = []
    analog = grouped[ANALOG_COLUMNS].agg(["mean", "std", "min", "max", "last"])
    analog_pairs = cast(list[tuple[str, str]], list(analog.columns))
    analog.columns = [f"{sensor}__{stat}" for sensor, stat in analog_pairs]
    parts.append(analog)

    digital_frames: list[pd.DataFrame] = []
    for sensor in DIGITAL_COLUMNS:
        sensor_group = grouped[sensor]
        digital_frames.append(
            pd.DataFrame(
                {
                    f"{sensor}__active_ratio": sensor_group.mean(),
                    f"{sensor}__transitions": sensor_group.apply(_transitions),
                    f"{sensor}__last": sensor_group.last(),
                }
            )
        )
    parts.extend(digital_frames)

    for delta in ["tp3_minus_reservoirs", "tp2_minus_tp3"]:
        delta_features = grouped[delta].agg(["mean", "std"])
        delta_features.columns = [f"{delta}__{stat}" for stat in delta_features.columns]
        parts.append(delta_features)

    coverage = grouped.size().astype(float).div(expected_samples).clip(upper=1.0)
    features = pd.concat(parts, axis=1)
    features["coverage_ratio"] = coverage
    features.index.name = "timestamp"
    return features.replace([np.inf, -np.inf], np.nan)


def model_feature_columns(columns: Iterable[str]) -> list[str]:
    """Return model inputs; coverage is a quality gate, not a predictive feature."""
    return [column for column in columns if column != "coverage_ratio"]


def sensor_from_feature(feature_name: str) -> str:
    """Map engineered feature names back to a human-readable sensor group."""
    if feature_name.startswith("tp3_minus_reservoirs"):
        return "TP3 - Reservoirs"
    if feature_name.startswith("tp2_minus_tp3"):
        return "TP2 - TP3"
    return feature_name.split("__", maxsplit=1)[0]
