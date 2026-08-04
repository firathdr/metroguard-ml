from __future__ import annotations

import hashlib
import zipfile
from dataclasses import replace
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from metroguard.api import ScoringService, create_app
from metroguard.config import load_config
from metroguard.data import download_dataset, load_raw_csv, prepare_features, sha256_file


def test_download_verifies_local_archive(tmp_path: Path, raw_frame: pd.DataFrame) -> None:
    csv_path = tmp_path / "source.csv"
    raw_frame.to_csv(csv_path, index=False)
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(csv_path, "MetroPT3(AirCompressor).csv")
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    base = load_config()
    data = replace(
        base.data,
        source_url=archive.as_uri(),
        expected_sha256=checksum,
        raw_zip=tmp_path / "download.zip",
        raw_csv=tmp_path / "raw.csv",
        features_file=tmp_path / "features.parquet",
    )
    config = replace(base, data=data, root=tmp_path)
    metadata = download_dataset(config)
    assert metadata["sha256"] == checksum
    assert len(load_raw_csv(config.data.raw_csv)) == len(raw_frame)
    prepared = prepare_features(config)
    assert prepared["raw_rows"] == len(raw_frame)
    assert sha256_file(config.data.raw_zip) == checksum


def test_health_reports_missing_artifacts(tmp_path: Path) -> None:
    service = ScoringService(replace(load_config(), root=tmp_path))
    client = TestClient(create_app(service))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False
    score = client.post("/v1/score", json={"readings": []})
    assert score.status_code == 422
