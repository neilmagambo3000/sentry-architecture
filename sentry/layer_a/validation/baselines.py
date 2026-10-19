"""Baseline signals for the forward IC table: seeded random and 12-1 momentum.

Both are pure functions of the feature frame and prices visible before each decision date.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def random_signal(features: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Uniform(-0.5, 0.5) per (date, symbol), hashed from the seed so row order never matters."""
    vals = [int(hashlib.sha256(f"{seed}|{d}|{s}".encode()).hexdigest()[:8], 16) / 2**32 - 0.5
            for d, s in zip(features["date"], features["symbol"], strict=True)]
    return pd.Series(vals, index=features.index, dtype=float)


def momentum_12_1(features: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """P[t-21] / P[t-252] - 1, where t is the last price date strictly before the decision date
    (closes visible at the 08:00 as_of). NaN without enough history."""
    idx = pd.DatetimeIndex(prices.index)
    out = np.full(len(features), np.nan)
    for i, (d, s) in enumerate(zip(features["date"], features["symbol"], strict=True)):
        t = idx.searchsorted(pd.Timestamp(d), side="left") - 1
        if s in prices and t - 252 >= 0:
            a, b = prices[s].iloc[t - 21], prices[s].iloc[t - 252]
            out[i] = a / b - 1 if b else np.nan
    return pd.Series(out, index=features.index)
