"""Leakage-safe temporal window construction and split masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metroguard.config import AppConfig
from metroguard.features import model_feature_columns


@dataclass(frozen=True)
class WindowSet:
    values: np.ndarray
    start_times: pd.DatetimeIndex
    end_times: pd.DatetimeIndex
    feature_names: list[str]

    def subset(self, mask: np.ndarray) -> WindowSet:
        return WindowSet(
            values=self.values[mask],
            start_times=self.start_times[mask],
            end_times=self.end_times[mask],
            feature_names=self.feature_names,
        )


def build_windows(
    features: pd.DataFrame,
    *,
    sequence_bins: int,
    bin_minutes: int,
    minimum_coverage: float,
) -> WindowSet:
    """Build consecutive, fully causal windows and reject gaps or low coverage."""
    columns = model_feature_columns(features.columns)
    expected_delta = pd.Timedelta(minutes=bin_minutes)
    values: list[np.ndarray] = []
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for end_index in range(sequence_bins - 1, len(features)):
        start_index = end_index - sequence_bins + 1
        chunk = features.iloc[start_index : end_index + 1]
        deltas = chunk.index.to_series().diff().dropna()
        if not deltas.eq(expected_delta).all():
            continue
        if chunk["coverage_ratio"].lt(minimum_coverage).any():
            continue
        matrix = chunk[columns].to_numpy(dtype=np.float32)
        if not np.isfinite(matrix).all():
            continue
        values.append(matrix)
        starts.append(pd.Timestamp(chunk.index[0]))
        ends.append(pd.Timestamp(chunk.index[-1]))
    array = np.stack(values) if values else np.empty((0, sequence_bins, len(columns)), np.float32)
    return WindowSet(
        values=array,
        start_times=pd.DatetimeIndex(starts),
        end_times=pd.DatetimeIndex(ends),
        feature_names=columns,
    )


def split_windows(windows: WindowSet, config: AppConfig) -> dict[str, WindowSet]:
    """Apply the pre-registered chronological split with purged boundaries."""
    split = config.split

    def between(start: str, end: str) -> np.ndarray:
        return np.asarray(
            (windows.start_times >= pd.Timestamp(start)) & (windows.end_times <= pd.Timestamp(end))
        )

    return {
        "train": windows.subset(between(split.train_start, split.train_end)),
        "calibration": windows.subset(
            between(split.calibration_start, split.calibration_end)
        ),
        "test": windows.subset(between(split.test_start, split.test_end)),
    }

