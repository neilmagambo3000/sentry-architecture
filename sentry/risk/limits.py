"""Pre-registered hard risk limits from config/risk.yaml."""
from __future__ import annotations

from pathlib import Path

from sentry.config import RiskLimits, _build, _load

__all__ = ["RiskLimits", "load_limits"]


def load_limits(config_dir: Path | None = None) -> RiskLimits:
    return _build(RiskLimits, _load("risk.yaml", config_dir))
