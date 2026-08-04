"""End-to-end training, evaluation, reporting, and demo artifact pipeline."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from metroguard.alerts import apply_alert_policy, calibration_threshold
from metroguard.artifacts import ModelBundle, git_revision, save_bundle, write_json
from metroguard.config import AppConfig
from metroguard.data import load_features, load_raw_csv, sha256_file
from metroguard.features import sensor_from_feature
from metroguard.metrics import EVENTS, evaluate_alerts, horizon_sensitivity
from metroguard.models import (
    IsolationForestModel,
    PCAModel,
    RobustZModel,
    ScoreNormalizer,
    TemporalConvAutoencoder,
    WindowScaler,
    autoencoder_score,
    fit_autoencoder,
)
from metroguard.windows import build_windows, split_windows

MODEL_NAMES = ("robust_z", "pca", "isolation_forest", "tcn_autoencoder")


def _mpl_date(timestamp: pd.Timestamp) -> float:
    return float(mdates.date2num(timestamp.to_pydatetime()))  # type: ignore[no-untyped-call]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _aggregate_contributions(contributions: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for feature, contribution in zip(feature_names, contributions, strict=True):
        sensor = sensor_from_feature(feature)
        grouped[sensor] = grouped.get(sensor, 0.0) + float(contribution)
    total = sum(grouped.values()) or 1.0
    return {sensor: value / total for sensor, value in sorted(grouped.items())}


def _make_figures(
    root: Path,
    summaries: dict[str, dict[str, object]],
    primary_timeline: pd.DataFrame,
) -> None:
    figure_dir = root / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    models = list(summaries)
    detected = [cast(int, summaries[name]["early_events_detected"]) for name in models]
    false_rates = [cast(float, summaries[name]["false_alarms_per_day"]) for name in models]
    fig, left = plt.subplots(figsize=(9, 4.8))
    positions = np.arange(len(models))
    left.bar(positions - 0.18, detected, width=0.36, color="#2166ac", label="Early events")
    left.set_ylabel("Early events detected (of 4)")
    left.set_ylim(0, 4.5)
    right = left.twinx()
    right.bar(positions + 0.18, false_rates, width=0.36, color="#ef8a62", label="False alarms/day")
    right.set_ylabel("False alarm episodes/day")
    left.set_xticks(positions, [name.replace("_", "\n") for name in models])
    left.set_title("Pre-registered holdout benchmark")
    left.grid(axis="y", alpha=0.2)
    handles_1, labels_1 = left.get_legend_handles_labels()
    handles_2, labels_2 = right.get_legend_handles_labels()
    left.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_dir / "model_comparison.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 4.8))
    axis.plot(
        primary_timeline["timestamp"],
        primary_timeline["smoothed_score"],
        linewidth=0.7,
        color="#2166ac",
        label="TCN anomaly score",
    )
    axis.plot(
        primary_timeline["timestamp"],
        primary_timeline["threshold"],
        linestyle="--",
        color="#b2182b",
        label="Fixed threshold",
    )
    for index, event in enumerate(EVENTS):
        axis.axvspan(
            _mpl_date(event.start),
            _mpl_date(event.end),
            color="#ef8a62",
            alpha=0.25,
            label="Official event" if index == 0 else None,
        )
        axis.axvspan(
            _mpl_date(event.start - pd.Timedelta(hours=24)),
            _mpl_date(event.start - pd.Timedelta(hours=2)),
            color="#67a9cf",
            alpha=0.12,
            label="24-2 h early window" if index == 0 else None,
        )
    axis.set_title("TCN autoencoder on the locked March-September holdout")
    axis.set_ylabel("Normalized anomaly score")
    axis.legend(loc="upper right", ncol=3)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "holdout_timeline.png", dpi=160)
    plt.close(fig)


def _make_demo_extract(config: AppConfig) -> None:
    if not config.data.raw_csv.exists():
        return
    frame = load_raw_csv(config.data.raw_csv)
    demo = frame.loc[
        frame["timestamp"].between("2020-05-29 12:00:00", "2020-05-30 08:00:00")
    ]
    destination = config.root / "data" / "demo" / "replay.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(destination, index=False)


def _block_stats(
    episodes: pd.DataFrame,
    event: Any,
    block_start: pd.Timestamp,
    block_end: pd.Timestamp,
) -> dict[str, float | bool | None]:
    early_start = event.start - pd.Timedelta(hours=24)
    early_end = event.start - pd.Timedelta(hours=2)
    early_hits: list[pd.Timestamp] = []
    late = False
    false_count = 0
    for record in episodes.to_dict("records"):
        start = pd.Timestamp(record["start"])
        end = pd.Timestamp(record["end"])
        if start <= early_end and end >= early_start:
            early_hits.append(max(start, early_start))
        elif start <= event.end and end >= early_end:
            late = True
        else:
            false_count += 1
    excluded_hours = (event.end - early_start).total_seconds() / 3600.0
    normal_days = max(
        ((block_end - block_start).total_seconds() / 3600.0 - excluded_hours) / 24.0,
        1e-9,
    )
    first_hit = min(early_hits) if early_hits else None
    return {
        "early_detected": bool(early_hits),
        "late_detected": bool(late and not early_hits),
        "lead_time_hours": (event.start - first_hit).total_seconds() / 3600.0
        if first_hit
        else None,
        "false_alarms_per_day": false_count / normal_days,
    }


def write_cross_event_analysis(config: AppConfig) -> pd.DataFrame:
    """Run an explicitly exploratory leave-one-event-block-out threshold analysis."""
    scores = pd.read_parquet(config.root / "reports" / "holdout_scores.parquet")
    scores["timestamp"] = pd.to_datetime(scores["timestamp"])
    blocks = (
        ("B1", pd.Timestamp("2020-03-01 01:00"), pd.Timestamp("2020-04-30 23:59"), EVENTS[0]),
        ("B2", pd.Timestamp("2020-05-01 00:00"), pd.Timestamp("2020-06-02 23:59"), EVENTS[1]),
        ("B3", pd.Timestamp("2020-06-03 00:00"), pd.Timestamp("2020-06-30 23:59"), EVENTS[2]),
        ("B4", pd.Timestamp("2020-07-01 00:00"), pd.Timestamp("2020-09-01 03:59"), EVENTS[3]),
    )
    rows: list[dict[str, object]] = []
    for model_name in MODEL_NAMES:
        score_column = f"{model_name}__score"
        for held_index, (block_name, block_start, block_end, event) in enumerate(blocks):
            tuning_blocks = [block for index, block in enumerate(blocks) if index != held_index]
            tuning_scores = pd.concat(
                [
                    scores.loc[scores["timestamp"].between(start, end), score_column]
                    for _, start, end, _ in tuning_blocks
                ],
                ignore_index=True,
            ).to_numpy(dtype=float)
            candidates: list[tuple[float, float, float]] = []
            for quantile in (0.99, 0.995, 0.999):
                threshold = float(np.quantile(tuning_scores, quantile))
                detected = 0
                false_rates: list[float] = []
                for _, start, end, tuning_event in tuning_blocks:
                    block = scores.loc[scores["timestamp"].between(start, end)]
                    result = apply_alert_policy(
                        pd.DatetimeIndex(block["timestamp"]),
                        block[score_column].to_numpy(dtype=float),
                        threshold,
                        config.alert,
                    )
                    stats = _block_stats(result.episodes, tuning_event, start, end)
                    detected += int(bool(stats["early_detected"]))
                    false_rates.append(float(stats["false_alarms_per_day"] or 0.0))
                objective = detected * 10.0 - float(np.mean(false_rates))
                candidates.append((objective, quantile, threshold))
            _, selected_quantile, selected_threshold = max(candidates)
            held = scores.loc[scores["timestamp"].between(block_start, block_end)]
            held_result = apply_alert_policy(
                pd.DatetimeIndex(held["timestamp"]),
                held[score_column].to_numpy(dtype=float),
                selected_threshold,
                config.alert,
            )
            stats = _block_stats(held_result.episodes, event, block_start, block_end)
            rows.append(
                {
                    "model": model_name,
                    "held_out_block": block_name,
                    "selected_quantile": selected_quantile,
                    "selected_threshold": selected_threshold,
                    **stats,
                    "status": "exploratory; threshold selected on the other three event blocks",
                }
            )
    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(config.root / "reports" / "cross_event_analysis.csv", index=False)
    return result_frame


def write_isolation_shap(config: AppConfig, sample_size: int = 256) -> pd.DataFrame:
    """Create sampled TreeSHAP signal contributions for the Isolation Forest comparator."""
    import shap

    from metroguard.artifacts import load_bundle

    bundle = load_bundle(config.root / "artifacts" / "release" / "model_bundle.joblib")
    windows = split_windows(
        build_windows(
            load_features(config.data.features_file),
            sequence_bins=config.data.sequence_bins,
            bin_minutes=config.data.bin_minutes,
            minimum_coverage=config.data.minimum_coverage,
        ),
        config,
    )["test"]
    stride = max(1, len(windows.values) // sample_size)
    sampled = bundle.scaler.transform(windows.values[::stride][:sample_size])
    flat = sampled.reshape(len(sampled), -1)
    explainer = shap.TreeExplainer(bundle.isolation_forest.model)
    shap_values = np.asarray(explainer.shap_values(flat))
    feature_values = cast(
        np.ndarray,
        np.abs(shap_values)
        .reshape(len(sampled), bundle.sequence_bins, len(bundle.feature_names))
        .mean(axis=(0, 1)),
    )
    grouped: dict[str, float] = {}
    for feature, value in zip(bundle.feature_names, feature_values, strict=True):
        sensor = sensor_from_feature(feature)
        grouped[sensor] = grouped.get(sensor, 0.0) + float(value)
    grouped_items: list[tuple[str, float]] = list(grouped.items())
    grouped_items.sort(key=lambda item: item[1], reverse=True)
    result = pd.DataFrame(
        grouped_items,
        columns=["sensor", "mean_absolute_shap"],
    )
    result["normalized_contribution"] = result["mean_absolute_shap"] / result[
        "mean_absolute_shap"
    ].sum()
    result.to_csv(config.root / "reports" / "isolation_forest_shap.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    top = result.head(10).sort_values("normalized_contribution")
    axis.barh(top["sensor"], top["normalized_contribution"], color="#2166ac")
    axis.set_xlabel("Normalized mean |SHAP value|")
    axis.set_title("Isolation Forest: sampled holdout signal contributions")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(config.root / "reports" / "figures" / "isolation_forest_shap.png", dpi=160)
    plt.close(figure)
    return result


def train_all(config: AppConfig) -> dict[str, Any]:
    """Fit the four fixed models, lock calibration thresholds, and evaluate once."""
    _set_seed(config.seed)
    features = load_features(config.data.features_file)
    windows = build_windows(
        features,
        sequence_bins=config.data.sequence_bins,
        bin_minutes=config.data.bin_minutes,
        minimum_coverage=config.data.minimum_coverage,
    )
    splits = split_windows(windows, config)
    train = splits["train"]
    calibration = splits["calibration"]
    test = splits["test"]
    if min(len(train.values), len(calibration.values), len(test.values)) == 0:
        raise ValueError("At least one chronological split is empty")

    scaler = WindowScaler().fit(train.values)
    train_scaled = scaler.transform(train.values)
    calibration_scaled = scaler.transform(calibration.values)
    test_scaled = scaler.transform(test.values)

    robust_z = RobustZModel().fit(train_scaled)
    pca = PCAModel(config.model.pca_variance).fit(train_scaled)
    isolation_forest = IsolationForestModel(
        config.model.isolation_forest_trees, config.seed
    ).fit(train_scaled)
    tcn = TemporalConvAutoencoder(len(train.feature_names))
    history = fit_autoencoder(
        tcn,
        train_scaled,
        calibration_scaled,
        batch_size=config.model.batch_size,
        max_epochs=config.model.max_epochs,
        patience=config.model.patience,
        learning_rate=config.model.learning_rate,
        seed=config.seed,
    )

    scoring_functions: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "robust_z": robust_z.score,
        "pca": pca.score,
        "isolation_forest": isolation_forest.score,
    }
    calibration_raw = {name: function(calibration_scaled) for name, function in scoring_functions.items()}
    test_raw = {name: function(test_scaled) for name, function in scoring_functions.items()}
    calibration_tcn, _ = autoencoder_score(tcn, calibration_scaled)
    test_tcn, tcn_contributions = autoencoder_score(tcn, test_scaled)
    calibration_raw["tcn_autoencoder"] = calibration_tcn
    test_raw["tcn_autoencoder"] = test_tcn

    normalizers: dict[str, ScoreNormalizer] = {}
    thresholds: dict[str, float] = {}
    summaries: dict[str, dict[str, object]] = {}
    event_frames: list[pd.DataFrame] = []
    sensitivity_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []
    primary_timeline: pd.DataFrame | None = None
    primary_episodes: pd.DataFrame | None = None

    for name in MODEL_NAMES:
        normalizer = ScoreNormalizer.fit(calibration_raw[name])
        normalizers[name] = normalizer
        normalized_calibration = normalizer.transform(calibration_raw[name])
        threshold = calibration_threshold(normalized_calibration, config.alert)
        thresholds[name] = threshold
        normalized_test = normalizer.transform(test_raw[name])
        alert_result = apply_alert_policy(test.end_times, normalized_test, threshold, config.alert)
        summary, event_frame = evaluate_alerts(
            alert_result.timeline,
            alert_result.episodes,
            early_warning_hours=config.alert.early_warning_hours,
            late_boundary_hours=config.alert.late_boundary_hours,
        )
        summary["threshold"] = threshold
        summary["score_calibration_median"] = normalizer.median
        summary["score_calibration_mad"] = normalizer.mad
        summaries[name] = summary
        event_frame.insert(0, "model", name)
        event_frames.append(event_frame)
        sensitivity = horizon_sensitivity(alert_result.timeline, alert_result.episodes)
        sensitivity.insert(0, "model", name)
        sensitivity_frames.append(sensitivity)
        timeline = alert_result.timeline.rename(
            columns={
                "score": f"{name}__score",
                "smoothed_score": f"{name}__smoothed_score",
                "threshold": f"{name}__threshold",
                "exceedance": f"{name}__exceedance",
                "alert": f"{name}__alert",
            }
        )
        score_frames.append(timeline.set_index("timestamp"))
        if name == config.model.primary:
            primary_timeline = alert_result.timeline
            primary_episodes = alert_result.episodes

    if primary_timeline is None or primary_episodes is None:
        raise ValueError(f"Unknown primary model: {config.model.primary}")

    artifact_dir = config.root / "artifacts" / "release"
    bundle = ModelBundle(
        version=config.version,
        feature_names=train.feature_names,
        sequence_bins=config.data.sequence_bins,
        bin_minutes=config.data.bin_minutes,
        scaler=scaler,
        robust_z=robust_z,
        pca=pca,
        isolation_forest=isolation_forest,
        normalizers=normalizers,
        thresholds=thresholds,
        metadata={
            "primary_model": config.model.primary,
            "train_window": [config.split.train_start, config.split.train_end],
            "calibration_window": [config.split.calibration_start, config.split.calibration_end],
            "test_window": [config.split.test_start, config.split.test_end],
            "seed": config.seed,
            "git_revision": git_revision(config.root),
        },
    )
    save_bundle(bundle, artifact_dir / "model_bundle.joblib")
    torch.save(
        {
            "state_dict": tcn.state_dict(),
            "feature_count": len(train.feature_names),
            "version": config.version,
        },
        artifact_dir / "tcn_autoencoder.pt",
    )
    joblib.dump(
        {
            "test_end_times": test.end_times,
            "tcn_feature_contributions": tcn_contributions,
        },
        artifact_dir / "explanations.joblib",
        compress=3,
    )

    reports = config.root / "reports"
    pd.concat(event_frames, ignore_index=True).to_csv(reports / "event_results.csv", index=False)
    pd.concat(sensitivity_frames, ignore_index=True).to_csv(
        reports / "horizon_sensitivity.csv", index=False
    )
    pd.concat(score_frames, axis=1).reset_index().to_parquet(reports / "holdout_scores.parquet")
    primary_contributions = np.mean(tcn_contributions[primary_timeline["alert"].to_numpy()], axis=0)
    if not np.isfinite(primary_contributions).all():
        primary_contributions = np.mean(tcn_contributions, axis=0)
    explanation_summary = _aggregate_contributions(primary_contributions, train.feature_names)
    dataset_hash = sha256_file(config.data.raw_zip) if config.data.raw_zip.exists() else None
    manifest: dict[str, Any] = {
        "project": config.name,
        "version": config.version,
        "seed": config.seed,
        "git_revision": git_revision(config.root),
        "dataset": {"doi": config.data.doi, "sha256": dataset_hash},
        "splits": {
            "train_windows": len(train.values),
            "calibration_windows": len(calibration.values),
            "test_windows": len(test.values),
        },
        "training": {"tcn": asdict(history)},
        "models": summaries,
        "primary_model": config.model.primary,
        "primary_alert_sensor_contributions": explanation_summary,
        "limitations": [
            "One APU and four consolidated air-leak events.",
            "February is a no-reported-failure reference period, not proven healthy operation.",
            "Research demonstration only; not validated for safety-critical or production use.",
            "The model does not estimate remaining useful life or establish root cause.",
        ],
    }
    write_json(reports / "metrics.json", manifest)
    write_json(artifact_dir / "metadata.json", manifest)
    _make_figures(config.root, summaries, primary_timeline)
    _make_demo_extract(config)
    write_cross_event_analysis(config)
    write_isolation_shap(config)
    return manifest


def read_metrics(config: AppConfig) -> dict[str, Any]:
    path = config.root / "reports" / "metrics.json"
    if not path.exists():
        raise FileNotFoundError("No evaluation report found; run `metroguard train --all` first")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
