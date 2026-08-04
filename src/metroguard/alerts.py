"""Fixed score normalization and operational alert policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metroguard.config import AlertConfig


@dataclass(frozen=True)
class AlertPolicyResult:
    timeline: pd.DataFrame
    episodes: pd.DataFrame


def causal_ewma(scores: np.ndarray, alpha: float) -> np.ndarray:
    return pd.Series(scores, dtype=float).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def calibration_threshold(scores: np.ndarray, config: AlertConfig) -> float:
    smoothed = causal_ewma(scores, config.ewma_alpha)
    return float(np.quantile(smoothed, config.threshold_quantile))


def _candidate_episodes(timestamps: pd.DatetimeIndex, active: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    points = timestamps[active]
    if len(points) == 0:
        return []
    episodes: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = pd.Timestamp(points[0])
    end = start
    for point in points[1:]:
        current = pd.Timestamp(point)
        if current - end > pd.Timedelta(minutes=5):
            episodes.append((start, end))
            start = current
        end = current
    episodes.append((start, end))
    return episodes


def _merge_and_cooldown(
    candidates: list[tuple[pd.Timestamp, pd.Timestamp]], config: AlertConfig
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not candidates:
        return []
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = [candidates[0]]
    merge_delta = pd.Timedelta(minutes=config.merge_minutes)
    for start, end in candidates[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end <= merge_delta:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))

    cooldown = pd.Timedelta(hours=config.cooldown_hours)
    accepted: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    blocked_until = pd.Timestamp.min
    for start, end in merged:
        if start < blocked_until:
            continue
        accepted.append((start, end))
        blocked_until = end + cooldown
    return accepted


def apply_alert_policy(
    timestamps: pd.DatetimeIndex,
    normalized_scores: np.ndarray,
    threshold: float,
    config: AlertConfig,
) -> AlertPolicyResult:
    """Apply EWMA, 3-of-4 persistence, episode merging, and cooldown."""
    if len(timestamps) != len(normalized_scores):
        raise ValueError("timestamps and scores must have the same length")
    smoothed = causal_ewma(normalized_scores, config.ewma_alpha)
    exceedance = smoothed > threshold
    persistent = (
        pd.Series(exceedance.astype(int))
        .rolling(config.persistence_window, min_periods=config.persistence_window)
        .sum()
        .ge(config.persistence_hits)
        .to_numpy()
    )
    episodes = _merge_and_cooldown(_candidate_episodes(timestamps, persistent), config)
    active = np.zeros(len(timestamps), dtype=bool)
    for start, end in episodes:
        active |= np.asarray((timestamps >= start) & (timestamps <= end))
    timeline = pd.DataFrame(
        {
            "timestamp": timestamps,
            "score": normalized_scores,
            "smoothed_score": smoothed,
            "threshold": threshold,
            "exceedance": exceedance,
            "alert": active,
        }
    )
    episode_frame = pd.DataFrame(episodes, columns=["start", "end"])
    if not episode_frame.empty:
        episode_frame["duration_minutes"] = (
            episode_frame["end"] - episode_frame["start"]
        ).dt.total_seconds() / 60.0
    return AlertPolicyResult(timeline=timeline, episodes=episode_frame)

