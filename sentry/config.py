"""Typed settings loaded from config/*.yaml.

Secrets never live here: broker / LLM keys are read from the environment at the
point of use (execution/alpaca_paper.py, llm/client.py), never logged.
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, fields
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("SENTRY_CONFIG_DIR", ROOT / "config"))


def config_path(name: str, config_dir: Path | None = None) -> Path:
    return Path(config_dir or CONFIG_DIR) / name


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(name: str, config_dir: Path | None = None) -> dict:
    return yaml.safe_load(config_path(name, config_dir).read_text()) or {}


def _build(cls, data: Mapping):
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**{k: v for k, v in data.items() if k in names})


def _num(cls: str, **vals) -> None:
    for k, v in vals.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{cls}.{k} must be a real number, got {v!r}")


def _int(cls: str, **vals) -> None:
    for k, v in vals.items():
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{cls}.{k} must be an int, got {v!r}")


def _check(cls: str, ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(f"{cls}: {msg}")


def _date_or_none(cls: str, k: str, v) -> None:
    if v is not None and not isinstance(v, date):
        raise ValueError(f"{cls}.{k} must be null or a YAML date (YYYY-MM-DD), got {v!r}")


@dataclass(frozen=True)
class ModelsConfig:
    model_id: str
    cutoff: Optional[date]
    temperature: float
    effort: str
    max_tokens: int
    escalate_model_id: str
    price_per_mtok: Mapping[str, float]

    def __post_init__(self):
        n = "ModelsConfig"
        for k in ("model_id", "escalate_model_id"):
            v = getattr(self, k)
            _check(n, isinstance(v, str) and bool(v.strip()), f"{k} must be a non-empty string, got {v!r}")
        _check(n, self.effort in {"low", "medium", "high"}, f"effort must be low, medium or high, got {self.effort!r}")
        _int(n, max_tokens=self.max_tokens)
        _check(n, self.max_tokens > 0, "max_tokens must be > 0")
        _num(n, temperature=self.temperature)
        _date_or_none(n, "cutoff", self.cutoff)
        _check(n, isinstance(self.price_per_mtok, Mapping), "price_per_mtok must be a mapping")
        for k, v in self.price_per_mtok.items():
            _num(n, **{f"price_per_mtok.{k}": v})

    @property
    def cutoff_dt(self) -> Optional[datetime]:
        if self.cutoff is None:
            return None
        return datetime(self.cutoff.year, self.cutoff.month, self.cutoff.day, tzinfo=timezone.utc)


@dataclass(frozen=True)
class UniverseConfig:
    as_of: Optional[date]
    selection_rule: str
    symbols: tuple[str, ...]

    def __post_init__(self):
        n = "UniverseConfig"
        _date_or_none(n, "as_of", self.as_of)
        _check(n, isinstance(self.selection_rule, str), "selection_rule must be a string")
        bad = [x for x in self.symbols if not (isinstance(x, str) and x and x == x.upper() and x == x.strip())]
        _check(n, not bad, f"symbols must be non-empty uppercase strings, bad: {bad!r}")
        _check(n, len(set(self.symbols)) == len(self.symbols), "symbols contains duplicates")


@dataclass(frozen=True)
class CostModel:
    bps_per_side: float
    spread_proxy_bps: float

    def __post_init__(self):
        _num("CostModel", bps_per_side=self.bps_per_side, spread_proxy_bps=self.spread_proxy_bps)
        _check("CostModel", self.bps_per_side >= 0 and self.spread_proxy_bps >= 0, "costs must be >= 0")

    @property
    def per_side(self) -> float:
        """Fractional cost charged on traded notional, per side."""
        return (self.bps_per_side + self.spread_proxy_bps) / 1e4


@dataclass(frozen=True)
class RiskLimits:
    max_weight: float
    max_gross: float
    daily_dd_halt: float
    total_dd_halt: float
    earnings_blackout_days: int
    llm_daily_usd: float
    stale_after_hours: float
    long_only: bool  # reduce negative targets to 0 (Alpaca: no fractional shorts)

    def __post_init__(self):
        n = "RiskLimits"
        _num(n, max_weight=self.max_weight, max_gross=self.max_gross, daily_dd_halt=self.daily_dd_halt,
             total_dd_halt=self.total_dd_halt, llm_daily_usd=self.llm_daily_usd,
             stale_after_hours=self.stale_after_hours)
        _int(n, earnings_blackout_days=self.earnings_blackout_days)
        _check(n, isinstance(self.long_only, bool), f"long_only must be a bool, got {self.long_only!r}")
        _check(n, 0 < self.max_weight <= 1, "max_weight must be in (0, 1]")
        _check(n, 0 < self.max_gross <= 1, "max_gross must be in (0, 1]")
        _check(n, self.max_weight <= self.max_gross, "max_weight must be <= max_gross")
        _check(n, 0 < self.daily_dd_halt < self.total_dd_halt < 1, "need 0 < daily_dd_halt < total_dd_halt < 1")
        _check(n, self.earnings_blackout_days >= 0, "earnings_blackout_days must be >= 0")
        _check(n, self.llm_daily_usd > 0, "llm_daily_usd must be > 0")
        _check(n, self.stale_after_hours > 0, "stale_after_hours must be > 0")


@dataclass(frozen=True)
class GateConfig:
    """Point-in-time news visibility: an article is visible only news_lag_minutes after it first arrived."""

    news_lag_minutes: int
    news_window_days: int

    def __post_init__(self):
        n = "GateConfig"
        _int(n, news_lag_minutes=self.news_lag_minutes, news_window_days=self.news_window_days)
        _check(n, min(self.news_lag_minutes, self.news_window_days) >= 0, "minute/day fields must be >= 0")


@dataclass(frozen=True)
class Settings:
    models: ModelsConfig
    universe: UniverseConfig
    costs: CostModel
    risk: RiskLimits
    gate: GateConfig
    ablation: Mapping[str, str]
    config_dir: Path


ABLATION_ARMS = {"debate_rounds": {"0", "1", "2"}, "escalate": {"off", "on"}, "trader": {"off", "on"}}


def _ablation(raw: Mapping) -> dict:
    out = {k: str(v) for k, v in raw.items()}
    for k, v in out.items():
        if k not in ABLATION_ARMS:
            raise ValueError(f"ablation: unknown arm {k!r} (known: {sorted(ABLATION_ARMS)})")
        if v not in ABLATION_ARMS[k]:
            raise ValueError(f"ablation.{k}: {v!r} not in {sorted(ABLATION_ARMS[k])}")
    return out


def load_settings(config_dir: Path | None = None) -> Settings:
    d = Path(config_dir or CONFIG_DIR)
    uni = _load("universe.yaml", d)
    uni["symbols"] = tuple(uni.get("symbols") or ())
    return Settings(
        models=_build(ModelsConfig, _load("models.yaml", d)),
        universe=_build(UniverseConfig, uni),
        costs=_build(CostModel, _load("costs.yaml", d)),
        risk=_build(RiskLimits, _load("risk.yaml", d)),
        gate=_build(GateConfig, _load("gate.yaml", d)),
        ablation=_ablation(_load("ablation.yaml", d)),
        config_dir=d,
    )


def env_flag(name: str, default: bool = False) -> bool:
    """Only an explicit token flips the flag; unset, blank or unrecognised values keep the default, so a
    blank SENTRY_DRY_RUN (an empty compose interpolation) or a typo can never turn dry run off."""
    v = (os.environ.get(name) or "").strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return default
