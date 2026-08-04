from __future__ import annotations

import numpy as np
import pandas as pd

from metroguard.alerts import apply_alert_policy, calibration_threshold, causal_ewma
from metroguard.config import load_config
from metroguard.metrics import evaluate_alerts, horizon_sensitivity, point_labels


def test_alert_policy_persistence_merge_and_cooldown() -> None:
    config = load_config().alert
    timestamps = pd.date_range("2020-04-16", periods=30, freq="5min")
    scores = np.zeros(30)
    scores[5:10] = 20
    scores[15:20] = 20
    result = apply_alert_policy(timestamps, scores, threshold=2, config=config)
    assert result.timeline["alert"].any()
    assert len(result.episodes) == 1
    assert causal_ewma(scores, 0.2).shape == scores.shape
    assert calibration_threshold(np.arange(100, dtype=float), config) > 90


def test_event_metrics_count_early_and_false_alerts() -> None:
    timestamps = pd.date_range("2020-03-01", "2020-08-01", freq="5min")
    timeline = pd.DataFrame(
        {
            "timestamp": timestamps,
            "smoothed_score": np.linspace(0, 1, len(timestamps)),
            "alert": False,
        }
    )
    episodes = pd.DataFrame(
        {
            "start": [pd.Timestamp("2020-04-17 03:00"), pd.Timestamp("2020-03-10 01:00")],
            "end": [pd.Timestamp("2020-04-17 04:00"), pd.Timestamp("2020-03-10 02:00")],
        }
    )
    summary, events = evaluate_alerts(timeline, episodes)
    assert summary["early_event_recall"] == "1/4"
    assert summary["false_alarm_count"] == 1
    assert bool(events.iloc[0]["early_detected"])
    assert len(horizon_sensitivity(timeline, episodes)) == 4
    assert point_labels(timestamps).sum() > 0

