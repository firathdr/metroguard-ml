"""Typed project configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(frozen=True)
class DataConfig:
    source_url: str
    doi: str
    expected_sha256: str | None
    raw_zip: Path
    raw_csv: Path
    features_file: Path
    bin_minutes: int
    expected_samples_per_bin: int
    minimum_coverage: float
    sequence_bins: int


@dataclass(frozen=True)
class SplitConfig:
    train_start: str
    train_end: str
    calibration_start: str
    calibration_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class ModelConfig:
    primary: str
    pca_variance: float
    isolation_forest_trees: int
    batch_size: int
    max_epochs: int
    patience: int
    learning_rate: float


@dataclass(frozen=True)
class AlertConfig:
    ewma_alpha: float
    threshold_quantile: float
    persistence_hits: int
    persistence_window: int
    merge_minutes: int
    cooldown_hours: int
    early_warning_hours: int
    late_boundary_hours: int


@dataclass(frozen=True)
class AppConfig:
    name: str
    version: str
    seed: int
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    alert: AlertConfig
    root: Path


def _path(root: Path, value: Any) -> Path:
    return root / Path(str(value))


def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    """Load the project YAML and resolve every configured path from the repo root."""
    config_path = Path(path).resolve()
    root = config_path.parent.parent
    with config_path.open(encoding="utf-8") as handle:
        raw = cast(dict[str, Any], yaml.safe_load(handle))

    project = cast(dict[str, Any], raw["project"])
    data = cast(dict[str, Any], raw["data"])
    split = cast(dict[str, Any], raw["split"])
    model = cast(dict[str, Any], raw["model"])
    alert = cast(dict[str, Any], raw["alert"])
    return AppConfig(
        name=str(project["name"]),
        version=str(project["version"]),
        seed=int(project["seed"]),
        data=DataConfig(
            source_url=str(data["source_url"]),
            doi=str(data["doi"]),
            expected_sha256=(
                str(data["expected_sha256"]) if data.get("expected_sha256") else None
            ),
            raw_zip=_path(root, data["raw_zip"]),
            raw_csv=_path(root, data["raw_csv"]),
            features_file=_path(root, data["features_file"]),
            bin_minutes=int(data["bin_minutes"]),
            expected_samples_per_bin=int(data["expected_samples_per_bin"]),
            minimum_coverage=float(data["minimum_coverage"]),
            sequence_bins=int(data["sequence_bins"]),
        ),
        split=SplitConfig(**{key: str(value) for key, value in split.items()}),
        model=ModelConfig(
            primary=str(model["primary"]),
            pca_variance=float(model["pca_variance"]),
            isolation_forest_trees=int(model["isolation_forest_trees"]),
            batch_size=int(model["batch_size"]),
            max_epochs=int(model["max_epochs"]),
            patience=int(model["patience"]),
            learning_rate=float(model["learning_rate"]),
        ),
        alert=AlertConfig(
            ewma_alpha=float(alert["ewma_alpha"]),
            threshold_quantile=float(alert["threshold_quantile"]),
            persistence_hits=int(alert["persistence_hits"]),
            persistence_window=int(alert["persistence_window"]),
            merge_minutes=int(alert["merge_minutes"]),
            cooldown_hours=int(alert["cooldown_hours"]),
            early_warning_hours=int(alert["early_warning_hours"]),
            late_boundary_hours=int(alert["late_boundary_hours"]),
        ),
        root=root,
    )

