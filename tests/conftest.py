from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metroguard.schema import ANALOG_COLUMNS, DIGITAL_COLUMNS


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2020-02-01", periods=721, freq="10s")
    phase = np.linspace(0, 8 * np.pi, len(timestamps))
    data: dict[str, object] = {"timestamp": timestamps}
    for index, column in enumerate(ANALOG_COLUMNS):
        data[column] = 5.0 + index + np.sin(phase + index) * 0.2
    for index, column in enumerate(DIGITAL_COLUMNS):
        data[column] = ((np.arange(len(timestamps)) // (20 + index)) % 2).astype(float)
    return pd.DataFrame(data)

