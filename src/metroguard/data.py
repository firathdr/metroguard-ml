"""Download, verify, load, and prepare MetroPT-3."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from metroguard.config import AppConfig
from metroguard.features import aggregate_causal_bins
from metroguard.schema import normalize_raw_frame, validate_digital_ranges


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(config: AppConfig, *, force: bool = False) -> dict[str, Any]:
    """Download the official UCI archive and verify the pinned checksum."""
    destination = config.data.raw_zip
    destination.parent.mkdir(parents=True, exist_ok=True)
    if force and destination.exists():
        destination.unlink()
    if not destination.exists():
        request = urllib.request.Request(
            config.data.source_url,
            headers={"User-Agent": "MetroGuard/1.0 (research reproducibility)"},
        )
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            destination.open("wb") as target,
        ):
            shutil.copyfileobj(response, target)

    checksum = sha256_file(destination)
    if config.data.expected_sha256 and checksum != config.data.expected_sha256:
        raise ValueError(
            "Dataset checksum mismatch: "
            f"expected {config.data.expected_sha256}, received {checksum}"
        )
    with zipfile.ZipFile(destination) as archive:
        member = next(
            (name for name in archive.namelist() if name.lower().endswith(".csv")),
            None,
        )
        if member is None:
            raise ValueError("The UCI archive does not contain a CSV file")
        config.data.raw_csv.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, config.data.raw_csv.open("wb") as target:
            shutil.copyfileobj(source, target)

    metadata = {
        "source_url": config.data.source_url,
        "doi": config.data.doi,
        "sha256": checksum,
        "archive_bytes": destination.stat().st_size,
        "csv_path": str(config.data.raw_csv.relative_to(config.root)),
    }
    metadata_path = destination.parent / "download_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_raw_csv(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Read the official CSV with low-memory parsing disabled for stable dtypes."""
    return normalize_raw_frame(pd.read_csv(path, nrows=nrows, low_memory=False))


def prepare_features(config: AppConfig, *, frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """Normalize raw data, create causal features, and persist Parquet plus metadata."""
    raw = frame if frame is not None else load_raw_csv(config.data.raw_csv)
    normalized = normalize_raw_frame(raw) if frame is not None else raw
    warnings = validate_digital_ranges(normalized)
    features = aggregate_causal_bins(
        normalized,
        bin_minutes=config.data.bin_minutes,
        expected_samples=config.data.expected_samples_per_bin,
    )
    config.data.features_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(config.data.features_file)
    cadence = pd.to_datetime(normalized["timestamp"]).diff().dropna().map(lambda value: value.total_seconds())
    metadata = {
        "raw_rows": len(normalized),
        "feature_rows": len(features),
        "feature_count": int(len(features.columns) - 1),
        "start": normalized["timestamp"].min().isoformat(),
        "end": normalized["timestamp"].max().isoformat(),
        "median_cadence_seconds": float(cadence.median()),
        "low_coverage_bins": int(
            features["coverage_ratio"].lt(config.data.minimum_coverage).sum()
        ),
        "warnings": warnings,
    }
    metadata_path = config.data.features_file.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_features(path: Path) -> pd.DataFrame:
    features = pd.read_parquet(path)
    features.index = pd.DatetimeIndex(features.index, name="timestamp")
    return features.sort_index()
