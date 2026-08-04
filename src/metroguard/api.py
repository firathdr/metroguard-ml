"""FastAPI service for bounded historical/research scoring."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from metroguard import __version__
from metroguard.alerts import apply_alert_policy
from metroguard.artifacts import ModelBundle, load_bundle
from metroguard.config import AppConfig, load_config
from metroguard.features import aggregate_causal_bins, sensor_from_feature
from metroguard.models import TemporalConvAutoencoder, autoencoder_score
from metroguard.pipeline import read_metrics
from metroguard.schema import normalize_raw_frame
from metroguard.windows import build_windows


class SensorReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    tp2: float
    tp3: float
    h1: float
    dv_pressure: float
    reservoirs: float
    oil_temperature: float
    motor_current: float
    comp: float
    dv_electric: float
    towers: float
    mpg: float
    lps: float
    pressure_switch: float
    oil_level: float
    caudal_impulses: float


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    readings: list[SensorReading] = Field(min_length=1, max_length=20_000)


class Contributor(BaseModel):
    sensor: str
    contribution: float
    observed: float | None
    train_median: float | None
    train_iqr: float | None


class ScoreResponse(BaseModel):
    model_version: str
    window_end: datetime
    score: float
    threshold: float
    alert: bool
    data_quality: dict[str, Any]
    top_contributors: list[Contributor]
    warning: str


class ScoringService:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        artifact_dir = self.config.root / "artifacts" / "release"
        bundle_path = artifact_dir / "model_bundle.joblib"
        weights_path = artifact_dir / "tcn_autoencoder.pt"
        self.bundle: ModelBundle | None = None
        self.model: TemporalConvAutoencoder | None = None
        if bundle_path.exists() and weights_path.exists():
            self.bundle = load_bundle(bundle_path)
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.model = TemporalConvAutoencoder(int(checkpoint["feature_count"]))
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()

    @property
    def loaded(self) -> bool:
        return self.bundle is not None and self.model is not None

    def _contributors(
        self,
        contribution_row: np.ndarray,
        latest_raw: pd.Series,
    ) -> list[Contributor]:
        if self.bundle is None:
            return []
        grouped: dict[str, float] = {}
        for feature, value in zip(self.bundle.feature_names, contribution_row, strict=True):
            sensor = sensor_from_feature(feature)
            grouped[sensor] = grouped.get(sensor, 0.0) + float(value)
        total = sum(grouped.values()) or 1.0
        ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)[:5]
        center = np.asarray(self.bundle.scaler.scaler.center_)
        scale = np.asarray(self.bundle.scaler.scaler.scale_)
        contributors: list[Contributor] = []
        for sensor, value in ranked:
            matching = [
                index
                for index, feature in enumerate(self.bundle.feature_names)
                if sensor_from_feature(feature) == sensor and feature.endswith("__last")
            ]
            raw_key = sensor if sensor in latest_raw.index else None
            contributors.append(
                Contributor(
                    sensor=sensor,
                    contribution=value / total,
                    observed=float(latest_raw[raw_key]) if raw_key else None,
                    train_median=float(center[matching[0]]) if matching else None,
                    train_iqr=float(scale[matching[0]]) if matching else None,
                )
            )
        return contributors

    def score(self, readings: Sequence[SensorReading]) -> ScoreResponse:
        if not self.loaded or self.bundle is None or self.model is None:
            raise RuntimeError("Release model artifacts are not available")
        raw = normalize_raw_frame(pd.DataFrame([reading.model_dump() for reading in readings]))
        duration = raw["timestamp"].max() - raw["timestamp"].min()
        if duration < pd.Timedelta(minutes=75):
            raise ValueError("At least 75 consecutive minutes of readings are required")
        features = aggregate_causal_bins(
            raw,
            bin_minutes=self.bundle.bin_minutes,
            expected_samples=self.config.data.expected_samples_per_bin,
        )
        windows = build_windows(
            features,
            sequence_bins=self.bundle.sequence_bins,
            bin_minutes=self.bundle.bin_minutes,
            minimum_coverage=self.config.data.minimum_coverage,
        )
        if len(windows.values) < self.config.alert.persistence_window:
            raise ValueError("Not enough complete high-coverage windows to determine an alert")
        if windows.feature_names != self.bundle.feature_names:
            raise ValueError("Feature schema does not match the release model")
        scaled = self.bundle.scaler.transform(windows.values)
        raw_scores, contributions = autoencoder_score(self.model, scaled)
        normalized = self.bundle.normalizers["tcn_autoencoder"].transform(raw_scores)
        threshold = self.bundle.thresholds["tcn_autoencoder"]
        result = apply_alert_policy(windows.end_times, normalized, threshold, self.config.alert)
        latest = raw.iloc[-1]
        valid_bins = int(features["coverage_ratio"].ge(self.config.data.minimum_coverage).sum())
        return ScoreResponse(
            model_version=self.bundle.version,
            window_end=pd.Timestamp(windows.end_times[-1]).to_pydatetime(),
            score=float(result.timeline["smoothed_score"].iloc[-1]),
            threshold=float(threshold),
            alert=bool(result.timeline["alert"].iloc[-1]),
            data_quality={
                "duration_minutes": duration.total_seconds() / 60.0,
                "raw_readings": len(raw),
                "complete_windows": len(windows.values),
                "valid_bins": valid_bins,
                "minimum_coverage": self.config.data.minimum_coverage,
            },
            top_contributors=self._contributors(contributions[-1], latest),
            warning=(
                "Research demonstration only. Contributions identify signals driving the "
                "anomaly score; they do not establish root cause."
            ),
        )


def create_app(service: ScoringService | None = None) -> FastAPI:
    scoring_service = service or ScoringService()
    api = FastAPI(
        title="MetroGuard API",
        version=__version__,
        description="Historical research scoring for MetroPT-3; not safety-critical advice.",
    )

    @api.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok" if scoring_service.loaded else "degraded",
            "version": __version__,
            "model_loaded": scoring_service.loaded,
        }

    @api.get("/v1/model-card")
    def model_card() -> dict[str, Any]:
        try:
            return read_metrics(scoring_service.config)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @api.post("/v1/score", response_model=ScoreResponse)
    def score(request: ScoreRequest) -> ScoreResponse:
        try:
            return scoring_service.score(request.readings)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return api


app = create_app()
