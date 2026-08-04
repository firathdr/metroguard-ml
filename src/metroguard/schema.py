"""Canonical MetroPT-3 schema and validation."""

from __future__ import annotations

from typing import Final

import pandas as pd

ANALOG_COLUMNS: Final[list[str]] = [
    "tp2",
    "tp3",
    "h1",
    "dv_pressure",
    "reservoirs",
    "oil_temperature",
    "motor_current",
]
DIGITAL_COLUMNS: Final[list[str]] = [
    "comp",
    "dv_electric",
    "towers",
    "mpg",
    "lps",
    "pressure_switch",
    "oil_level",
    "caudal_impulses",
]
SENSOR_COLUMNS: Final[list[str]] = ANALOG_COLUMNS + DIGITAL_COLUMNS
REQUIRED_COLUMNS: Final[list[str]] = ["timestamp", *SENSOR_COLUMNS]

COLUMN_ALIASES: Final[dict[str, str]] = {
    "timestamp": "timestamp",
    "tp2": "tp2",
    "tp3": "tp3",
    "h1": "h1",
    "dv_pressure": "dv_pressure",
    "reservoirs": "reservoirs",
    "oil_temperature": "oil_temperature",
    "motor_current": "motor_current",
    "comp": "comp",
    "dv_eletric": "dv_electric",
    "dv_electric": "dv_electric",
    "towers": "towers",
    "mpg": "mpg",
    "lps": "lps",
    "pressure_switch": "pressure_switch",
    "oil_level": "oil_level",
    "caudal_impulses": "caudal_impulses",
}


def _canonical_name(name: object) -> str:
    value = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    return COLUMN_ALIASES.get(value, value)


def normalize_raw_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop file indices, normalize names, parse time, and enforce numeric signals."""
    renamed = frame.rename(columns={column: _canonical_name(column) for column in frame.columns})
    drop_columns = [
        column
        for column in renamed.columns
        if column.startswith("unnamed") or column in {"index", "level_0"}
    ]
    normalized = renamed.drop(columns=drop_columns, errors="ignore").copy()
    missing = sorted(set(REQUIRED_COLUMNS).difference(normalized.columns))
    if missing:
        raise ValueError(f"Missing required MetroPT-3 columns: {', '.join(missing)}")

    normalized = normalized[REQUIRED_COLUMNS]
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="raise")
    for column in SENSOR_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")
    normalized = normalized.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if normalized.empty:
        raise ValueError("MetroPT-3 frame is empty after normalization")
    if normalized["timestamp"].isna().any():
        raise ValueError("Timestamps contain missing values")
    return normalized.reset_index(drop=True)


def validate_digital_ranges(frame: pd.DataFrame) -> list[str]:
    """Return warnings instead of coercing unusual digital sensor values."""
    warnings: list[str] = []
    for column in DIGITAL_COLUMNS:
        values = set(frame[column].dropna().unique().tolist())
        if not values.issubset({0, 1}):
            warnings.append(f"{column} contains non-binary values")
    return warnings
