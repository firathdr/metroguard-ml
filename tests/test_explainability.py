from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from metroguard.explainability import (
    explain_isolation_forest,
    plot_global_sensor_importance,
    plot_local_sensor_contributions,
    plot_sensor_time_heatmap,
)


def test_isolation_forest_shap_aggregates_sensor_and_time_importance(tmp_path) -> None:
    rng = np.random.default_rng(42)
    windows = rng.normal(size=(24, 4, 3)).astype(np.float32)
    feature_names = ["tp2__mean", "tp3__mean", "tp3_minus_reservoirs__mean"]
    model = IsolationForest(n_estimators=12, random_state=42).fit(
        windows.reshape(len(windows), -1)
    )

    result = explain_isolation_forest(model, windows, feature_names, 4, sample_size=12)

    assert result.anomaly_shap_values.shape == (12, 4, 3)
    assert set(result.sensor_importance["sensor"]) == {"tp2", "tp3", "TP3 - Reservoirs"}
    assert np.isclose(result.sensor_importance["normalized_contribution"].sum(), 1.0)
    assert result.sensor_time_importance.shape == (4, 3)
    assert len(result.local_sensor_contributions) == 3

    plot_global_sensor_importance(
        result.sensor_importance, tmp_path / "global.png"
    )
    plot_local_sensor_contributions(
        result.local_sensor_contributions, tmp_path / "local.png"
    )
    plot_sensor_time_heatmap(result, tmp_path / "heatmap.png")
    assert all((tmp_path / name).exists() for name in ("global.png", "local.png", "heatmap.png"))
