"""Fixed anomaly-detection benchmark models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EPSILON = 1e-9


@dataclass
class ScoreNormalizer:
    median: float
    mad: float

    @classmethod
    def fit(cls, scores: np.ndarray) -> ScoreNormalizer:
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)))
        return cls(median=median, mad=max(mad, EPSILON))

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return (scores - self.median) / (1.4826 * self.mad)


class WindowScaler:
    """Fit each engineered feature on train rows only, preserving sequence shape."""

    def __init__(self) -> None:
        self.scaler = RobustScaler(quantile_range=(25.0, 75.0))

    def fit(self, values: np.ndarray) -> WindowScaler:
        self.scaler.fit(values.reshape(-1, values.shape[-1]))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        shape = values.shape
        transformed = self.scaler.transform(values.reshape(-1, shape[-1]))
        return cast(np.ndarray, transformed.reshape(shape).astype(np.float32))


class RobustZModel:
    """Transparent top-three robust deviation baseline."""

    def __init__(self) -> None:
        self.median: np.ndarray | None = None
        self.mad: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> RobustZModel:
        flat = values.reshape(len(values), -1)
        self.median = np.median(flat, axis=0)
        self.mad = np.maximum(np.median(np.abs(flat - self.median), axis=0), EPSILON)
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        if self.median is None or self.mad is None:
            raise RuntimeError("RobustZModel is not fitted")
        deviations = np.abs(values.reshape(len(values), -1) - self.median) / (
            1.4826 * self.mad
        )
        top = np.partition(deviations, -3, axis=1)[:, -3:]
        return cast(np.ndarray, top.mean(axis=1))


class PCAModel:
    def __init__(self, variance: float) -> None:
        self.model = PCA(n_components=variance, svd_solver="full")

    def fit(self, values: np.ndarray) -> PCAModel:
        self.model.fit(values.reshape(len(values), -1))
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        flat = values.reshape(len(values), -1)
        reconstructed = self.model.inverse_transform(self.model.transform(flat))
        return cast(np.ndarray, np.mean(np.square(flat - reconstructed), axis=1))


class IsolationForestModel:
    def __init__(self, trees: int, seed: int) -> None:
        self.model = IsolationForest(
            n_estimators=trees,
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, values: np.ndarray) -> IsolationForestModel:
        self.model.fit(values.reshape(len(values), -1))
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, -self.model.score_samples(values.reshape(len(values), -1)))


class TemporalConvAutoencoder(nn.Module):
    """Small temporal convolutional autoencoder with an 8-channel bottleneck."""

    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.feature_count = feature_count
        self.network = nn.Sequential(
            nn.Conv1d(feature_count, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, feature_count, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channels_first = inputs.transpose(1, 2)
        return cast(torch.Tensor, self.network(channels_first).transpose(1, 2))


@dataclass(frozen=True)
class TrainingHistory:
    train_loss: list[float]
    calibration_loss: list[float]
    best_epoch: int


def fit_autoencoder(
    model: TemporalConvAutoencoder,
    train_values: np.ndarray,
    calibration_values: np.ndarray,
    *,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    seed: int,
) -> TrainingHistory:
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_values)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    calibration_tensor = torch.from_numpy(calibration_values)
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    train_losses: list[float] = []
    calibration_losses: list[float] = []

    for epoch in range(max_epochs):
        model.train()
        losses: list[float] = []
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch), batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        train_losses.append(float(np.mean(losses)))
        model.eval()
        with torch.no_grad():
            calibration_loss = float(loss_function(model(calibration_tensor), calibration_tensor))
        calibration_losses.append(calibration_loss)
        if calibration_loss < best_loss - 1e-7:
            best_loss = calibration_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainingHistory(train_losses, calibration_losses, best_epoch)


def autoencoder_score(
    model: TemporalConvAutoencoder,
    values: np.ndarray,
    *,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    contributions: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size)
    with torch.no_grad():
        for (batch,) in loader:
            squared_error = torch.square(model(batch) - batch)
            scores.append(squared_error.mean(dim=(1, 2)).cpu().numpy())
            contributions.append(squared_error.mean(dim=1).cpu().numpy())
    return np.concatenate(scores), np.concatenate(contributions)
