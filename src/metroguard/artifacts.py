"""Serializable model bundle and provenance helpers."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from metroguard.models import (
    IsolationForestModel,
    PCAModel,
    RobustZModel,
    ScoreNormalizer,
    WindowScaler,
)


@dataclass
class ModelBundle:
    version: str
    feature_names: list[str]
    sequence_bins: int
    bin_minutes: int
    scaler: WindowScaler
    robust_z: RobustZModel
    pca: PCAModel
    isolation_forest: IsolationForestModel
    normalizers: dict[str, ScoreNormalizer]
    thresholds: dict[str, float]
    metadata: dict[str, Any]


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: Path) -> ModelBundle:
    loaded = joblib.load(path)
    if not isinstance(loaded, ModelBundle):
        raise TypeError("Artifact is not a MetroGuard ModelBundle")
    return loaded


def git_revision(root: Path) -> str:
    override = os.environ.get("METROGUARD_GIT_REVISION")
    if override:
        return override
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
