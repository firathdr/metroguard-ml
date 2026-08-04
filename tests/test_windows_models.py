from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from metroguard.config import load_config
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
from metroguard.windows import WindowSet, build_windows, split_windows


def test_windows_reject_low_coverage_and_time_gaps() -> None:
    index = pd.date_range("2020-02-01", periods=16, freq="5min").delete(8)
    features = pd.DataFrame(
        {"feature": np.arange(len(index)), "coverage_ratio": 1.0}, index=index
    )
    features.loc[index[3], "coverage_ratio"] = 0.5
    result = build_windows(features, sequence_bins=4, bin_minutes=5, minimum_coverage=0.8)
    assert len(result.values) < len(features) - 3
    assert all(end - start == pd.Timedelta(minutes=15) for start, end in zip(result.start_times, result.end_times, strict=True))


def test_split_windows_respects_purged_boundaries() -> None:
    config = load_config()
    ends = pd.DatetimeIndex(
        ["2020-02-10 01:00", "2020-02-22 00:30", "2020-02-22 02:00", "2020-03-01 02:00"]
    )
    starts = ends - pd.Timedelta(minutes=55)
    windows = WindowSet(np.zeros((4, 12, 2), np.float32), starts, ends, ["a", "b"])
    result = split_windows(windows, config)
    assert len(result["train"].values) == 1
    assert len(result["calibration"].values) == 1
    assert len(result["test"].values) == 1


def test_fixed_models_produce_deterministic_scores() -> None:
    rng = np.random.default_rng(42)
    train = rng.normal(size=(40, 12, 4)).astype(np.float32)
    shifted = train[:5] + 3
    scaler = WindowScaler().fit(train)
    scaled = scaler.transform(train)
    shifted_scaled = scaler.transform(shifted)
    robust = RobustZModel().fit(scaled)
    pca = PCAModel(0.95).fit(scaled)
    forest = IsolationForestModel(10, 42).fit(scaled)
    assert robust.score(shifted_scaled).mean() > robust.score(scaled[:5]).mean()
    assert pca.score(shifted_scaled).shape == (5,)
    assert np.array_equal(forest.score(scaled[:5]), forest.score(scaled[:5]))
    normalizer = ScoreNormalizer.fit(robust.score(scaled))
    assert np.isfinite(normalizer.transform(robust.score(shifted_scaled))).all()


def test_tcn_shape_training_and_contributions() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(20, 12, 3)).astype(np.float32)
    calibration = rng.normal(size=(8, 12, 3)).astype(np.float32)
    model = TemporalConvAutoencoder(3)
    history = fit_autoencoder(
        model,
        train,
        calibration,
        batch_size=8,
        max_epochs=2,
        patience=2,
        learning_rate=0.001,
        seed=42,
    )
    scores, contributions = autoencoder_score(model, calibration)
    assert len(history.train_loss) == 2
    assert scores.shape == (8,)
    assert contributions.shape == (8, 3)
    with torch.no_grad():
        assert model(torch.from_numpy(calibration)).shape == torch.Size([8, 12, 3])
