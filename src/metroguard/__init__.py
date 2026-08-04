"""MetroGuard: leakage-safe anomaly detection for MetroPT-3."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("metroguard-ml")
except PackageNotFoundError:  # pragma: no cover - editable source fallback
    __version__ = "1.0.0"

__all__ = ["__version__"]

