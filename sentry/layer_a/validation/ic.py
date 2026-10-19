"""Forward information coefficient, grouped by strategy_version.

Signals: disc = disc_direction * disc_confidence (rows without a view are dropped), plus the baselines
random (seeded) and mom_12_1. Forward return for a decision dated d at horizon h is P[d + h] / P[d] - 1
on the price index, i.e. entry at the close of d, the first close strictly after the 08:00 as_of.

Per date: Spearman rank correlation across symbols (needs >= 3 pairs). Incremental IC: the signal is
first residualised on the controls by per-date OLS with an intercept, then ranked against the return.
Mean, t = mean / (std / sqrt(N)) and N are taken over NON-OVERLAPPING dates (every h-th decision date),
so 5- and 20-day windows do not inflate t by reusing the same returns.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sentry.layer_a.validation.baselines import momentum_12_1, random_signal

COLUMNS = ["strategy_version", "signal", "horizon", "mean_ic", "t_stat", "n_dates", "incr_ic", "incr_t", "incr_n"]


def _rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def _residual(x: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """OLS residual of x on [1, controls]. All zeros when the controls explain x exactly (a constant
    signal included), so float noise left by lstsq can never be ranked into a spurious IC."""
    a = np.column_stack([np.ones(len(x)), controls])
    beta, *_ = np.linalg.lstsq(a, x, rcond=None)
    r = x - a @ beta
    return np.zeros_like(r) if np.ptp(r) <= 1e-9 * max(1.0, float(np.ptp(x))) else r


def _stats(ics: list[float]) -> tuple[float, float, int]:
    v = np.array([i for i in ics if np.isfinite(i)])
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan), np.nan, len(v)
    sd = v.std(ddof=1)
    return float(v.mean()), (float(v.mean() / (sd / np.sqrt(len(v)))) if sd > 0 else np.nan), len(v)


def ic_table(features: pd.DataFrame, prices: pd.DataFrame, horizons: Sequence[int] = (1, 5, 20),
             controls: Sequence[str] = ("control_1", "control_2"), seed: int = 0) -> pd.DataFrame:
    """features: columns symbol, as_of (tz-aware), strategy_version, disc_direction, disc_confidence and the
    controls. prices: adjusted closes, index trading dates ascending, one column per symbol."""
    if features.empty:
        return pd.DataFrame(columns=COLUMNS)
    f = features.copy()
    f["date"] = pd.to_datetime(f["as_of"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    num = lambda c: pd.to_numeric(f[c], errors="coerce")  # noqa: E731
    f["disc"] = num("disc_direction") * num("disc_confidence")
    f["random"] = random_signal(f, seed)
    f["mom_12_1"] = momentum_12_1(f, prices)
    for c in controls:
        f[c] = num(c)
    px = prices.copy()
    px.index = pd.DatetimeIndex(px.index)
    out = []
    for version, g in f.groupby("strategy_version", sort=True):
        dates = sorted(g["date"].unique())
        for h in horizons:
            fwd = px.shift(-h) / px - 1
            for sig in ("disc", "random", "mom_12_1"):
                ics, incr = [], []
                for d in dates[::h]:
                    if d not in fwd.index:
                        continue
                    day = g[g["date"] == d]
                    y = fwd.loc[d].reindex(day["symbol"]).to_numpy(dtype=float)
                    x = day[sig].to_numpy(dtype=float)
                    ok = np.isfinite(x) & np.isfinite(y)
                    ics.append(_rank_ic(x[ok], y[ok]))
                    ctl = day[list(controls)].to_numpy(dtype=float)
                    ok &= np.isfinite(ctl).all(axis=1)
                    if ok.sum() > len(controls) + 2:
                        incr.append(_rank_ic(_residual(x[ok], ctl[ok]), y[ok]))
                m, t, n = _stats(ics)
                mi, ti, ni = _stats(incr)
                out.append([version, sig, h, m, t, n, mi, ti, ni])
    return pd.DataFrame(out, columns=COLUMNS)
