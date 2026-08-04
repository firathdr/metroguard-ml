"""Event-aware metrics for the four canonical MetroPT-3 incidents."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class FailureEvent:
    event_id: str
    start: pd.Timestamp
    end: pd.Timestamp


EVENTS: tuple[FailureEvent, ...] = (
    FailureEvent("event_1", pd.Timestamp("2020-04-18 00:00:00"), pd.Timestamp("2020-04-18 23:59:59")),
    FailureEvent("event_2", pd.Timestamp("2020-05-29 23:30:00"), pd.Timestamp("2020-05-30 06:00:00")),
    FailureEvent("event_3", pd.Timestamp("2020-06-05 10:00:00"), pd.Timestamp("2020-06-07 14:30:00")),
    FailureEvent("event_4", pd.Timestamp("2020-07-15 14:30:00"), pd.Timestamp("2020-07-15 19:00:00")),
)


def _overlaps(start: pd.Timestamp, end: pd.Timestamp, left: pd.Timestamp, right: pd.Timestamp) -> bool:
    return start <= right and end >= left


def point_labels(timestamps: pd.DatetimeIndex) -> np.ndarray:
    labels = np.zeros(len(timestamps), dtype=int)
    for event in EVENTS:
        labels |= np.asarray((timestamps >= event.start) & (timestamps <= event.end), dtype=int)
    return labels


def evaluate_alerts(
    timeline: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    early_warning_hours: int = 24,
    late_boundary_hours: int = 2,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate early alerts, late detections, false episodes, and pointwise ranking."""
    timestamps = pd.DatetimeIndex(timeline["timestamp"])
    episode_records = episodes.to_dict("records")
    per_event: list[dict[str, object]] = []
    associated_episode_indices: set[int] = set()
    lead_times: list[float] = []

    for event in EVENTS:
        early_start = event.start - pd.Timedelta(hours=early_warning_hours)
        early_end = event.start - pd.Timedelta(hours=late_boundary_hours)
        early_hits: list[tuple[int, pd.Timestamp]] = []
        late_hit = False
        for index, episode in enumerate(episode_records):
            start = pd.Timestamp(episode["start"])
            end = pd.Timestamp(episode["end"])
            if _overlaps(start, end, early_start, early_end):
                early_hits.append((index, max(start, early_start)))
                associated_episode_indices.add(index)
            elif _overlaps(start, end, early_end, event.end):
                late_hit = True
                associated_episode_indices.add(index)
        early_detected = bool(early_hits)
        lead_time: float | None = None
        if early_detected:
            first_hit = min(hit for _, hit in early_hits)
            lead_time = (event.start - first_hit).total_seconds() / 3600.0
            lead_times.append(lead_time)
        event_mask = (timestamps >= event.start) & (timestamps <= event.end)
        event_scores = timeline.loc[event_mask, "smoothed_score"]
        per_event.append(
            {
                **asdict(event),
                "early_detected": early_detected,
                "late_detected": bool(late_hit and not early_detected),
                "lead_time_hours": lead_time,
                "max_score": float(event_scores.max()) if not event_scores.empty else None,
                "median_score": float(event_scores.median()) if not event_scores.empty else None,
            }
        )

    false_alarms = len(episode_records) - len(associated_episode_indices)
    exposure_start = timestamps.min()
    exposure_end = timestamps.max()
    excluded_hours = sum(
        (event.end - (event.start - pd.Timedelta(hours=early_warning_hours))).total_seconds()
        / 3600.0
        for event in EVENTS
    )
    normal_days = max(
        ((exposure_end - exposure_start).total_seconds() / 3600.0 - excluded_hours) / 24.0,
        1e-9,
    )
    labels = point_labels(timestamps)
    scores = timeline["smoothed_score"].to_numpy(dtype=float)
    associated_count = len(associated_episode_indices)
    event_frame = pd.DataFrame(per_event)
    summary: dict[str, object] = {
        "early_event_recall": f"{int(event_frame['early_detected'].sum())}/{len(EVENTS)}",
        "early_events_detected": int(event_frame["early_detected"].sum()),
        "late_events_detected": int(event_frame["late_detected"].sum()),
        "false_alarm_count": int(false_alarms),
        "false_alarms_per_day": float(false_alarms / normal_days),
        "lead_time_median_hours": float(np.median(lead_times)) if lead_times else None,
        "lead_time_range_hours": [float(min(lead_times)), float(max(lead_times))]
        if lead_times
        else None,
        "alarm_precision": float(associated_count / len(episode_records))
        if episode_records
        else None,
        "time_in_alert_percent": float(timeline["alert"].mean() * 100.0),
        "pr_auc_official_failure_intervals": float(average_precision_score(labels, scores)),
        "roc_auc_official_failure_intervals": float(roc_auc_score(labels, scores)),
        "normal_exposure_days": float(normal_days),
        "alert_episodes": len(episode_records),
    }
    return summary, event_frame


def horizon_sensitivity(
    timeline: pd.DataFrame, episodes: pd.DataFrame, horizons: tuple[int, ...] = (6, 12, 24, 48)
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        summary, _ = evaluate_alerts(
            timeline,
            episodes,
            early_warning_hours=horizon,
            late_boundary_hours=2,
        )
        rows.append(
            {
                "horizon_hours": horizon,
                "early_event_recall": summary["early_event_recall"],
                "false_alarms_per_day": summary["false_alarms_per_day"],
                "lead_time_median_hours": summary["lead_time_median_hours"],
            }
        )
    return pd.DataFrame(rows)
