from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metroguard.features import aggregate_causal_bins, model_feature_columns, sensor_from_feature
from metroguard.schema import normalize_raw_frame, validate_digital_ranges


def test_normalize_maps_official_misspelling_and_drops_index(raw_frame: pd.DataFrame) -> None:
    source = raw_frame.rename(columns={"dv_electric": "DV_eletric"}).copy()
    source.insert(0, "Unnamed: 0", range(len(source)))
    source = source.iloc[::-1]
    result = normalize_raw_frame(source)
    assert "dv_electric" in result
    assert "Unnamed: 0" not in result
    assert result["timestamp"].is_monotonic_increasing


def test_normalize_rejects_missing_sensor(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Missing required"):
        normalize_raw_frame(raw_frame.drop(columns="tp2"))


def test_digital_range_warning(raw_frame: pd.DataFrame) -> None:
    changed = raw_frame.copy()
    changed.loc[0, "comp"] = 2
    assert validate_digital_ranges(changed) == ["comp contains non-binary values"]


def test_causal_aggregation_has_expected_features_and_coverage(raw_frame: pd.DataFrame) -> None:
    features = aggregate_causal_bins(raw_frame, bin_minutes=5, expected_samples=30)
    assert "tp2__mean" in features
    assert "comp__transitions" in features
    assert "tp3_minus_reservoirs__std" in features
    assert features["coverage_ratio"].between(0, 1).all()
    assert "coverage_ratio" not in model_feature_columns(features.columns)
    assert sensor_from_feature("tp2__mean") == "tp2"
    assert sensor_from_feature("tp3_minus_reservoirs__mean") == "TP3 - Reservoirs"


def test_aggregation_does_not_backfill_future_values(raw_frame: pd.DataFrame) -> None:
    truncated = raw_frame.copy()
    truncated.loc[truncated["timestamp"] < pd.Timestamp("2020-02-01 00:05:00"), "tp2"] = np.nan
    features = aggregate_causal_bins(truncated, bin_minutes=5, expected_samples=30)
    assert pd.isna(features.iloc[0]["tp2__mean"])
