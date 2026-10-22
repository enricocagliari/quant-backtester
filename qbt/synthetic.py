"""Synthetic price generators with known structure, for tests and null experiments.

``random_walk`` has no exploitable structure by construction: any strategy's
expected Sharpe on it is zero, which makes it the right null for checking
that the research pipeline does not manufacture performance. The other
generators inject a specific, documented effect so that a strategy built for
that effect should recover it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _index(n_days: int, start: str = '2015-01-01') -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n_days)


def _frame(log_returns: np.ndarray, names: list[str] | None, start: str, p0: float = 100.0) -> pd.DataFrame:
    n_days, n_assets = log_returns.shape
    names = names or [f'A{i:02d}' for i in range(n_assets)]
    prices = p0 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(prices, index=_index(n_days, start), columns=names)


def random_walk(n_days: int = 2520, n_assets: int = 8, ann_vol: float = 0.20, market_beta: float = 0.6,
                regime_switching: bool = True, seed: int = 0, start: str = '2015-01-01',
                names: list[str] | None = None) -> pd.DataFrame:
    """Factor-correlated geometric random walks with optional two-state volatility regimes.

    Returns are independent across time, so there is no momentum or mean
    reversion to find. Cross-sectional correlation comes from a common
    market factor; regimes make volatility clustered, which is enough to
    give naive estimators realistic fat tails.
    """
    rng = np.random.default_rng(seed)
    daily = ann_vol / np.sqrt(252)
    if regime_switching:
        state = np.zeros(n_days, dtype=int)
        for t in range(1, n_days):
            stay = 0.98 if state[t - 1] == 0 else 0.95
            state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
        vol_mult = np.where(state == 0, 0.8, 1.8)[:, None]
    else:
        vol_mult = np.ones((n_days, 1))
    market = rng.standard_normal((n_days, 1)) * daily * market_beta
    idio = rng.standard_normal((n_days, n_assets)) * daily * np.sqrt(max(1 - market_beta ** 2, 0.0))
    log_returns = (market + idio) * vol_mult - 0.5 * (daily * vol_mult) ** 2
    return _frame(log_returns, names, start)


def trending(n_days: int = 2520, n_assets: int = 8, ann_vol: float = 0.20, drift_vol: float = 0.0008,
             drift_persistence: float = 0.995, seed: int = 0, start: str = '2015-01-01',
             names: list[str] | None = None) -> pd.DataFrame:
    """Random walks whose drift follows a slow AR(1): a tractable momentum effect."""
    rng = np.random.default_rng(seed)
    daily = ann_vol / np.sqrt(252)
    drift = np.zeros((n_days, n_assets))
    shocks = rng.standard_normal((n_days, n_assets)) * drift_vol * np.sqrt(1 - drift_persistence ** 2)
    for t in range(1, n_days):
        drift[t] = drift_persistence * drift[t - 1] + shocks[t]
    log_returns = drift + rng.standard_normal((n_days, n_assets)) * daily
    return _frame(log_returns, names, start)


def cointegrated_pair(n_days: int = 2520, ann_vol: float = 0.20, beta: float = 1.0, spread_vol: float = 0.02,
                      spread_half_life: float = 10.0, seed: int = 0, start: str = '2015-01-01',
                      names: tuple[str, str] = ('X', 'Y')) -> pd.DataFrame:
    """Two assets sharing a random walk with an Ornstein-Uhlenbeck spread between them."""
    rng = np.random.default_rng(seed)
    daily = ann_vol / np.sqrt(252)
    common = np.cumsum(rng.standard_normal(n_days) * daily)
    phi = 0.5 ** (1 / spread_half_life)
    spread = np.zeros(n_days)
    noise = rng.standard_normal(n_days) * spread_vol * np.sqrt(1 - phi ** 2)
    for t in range(1, n_days):
        spread[t] = phi * spread[t - 1] + noise[t]
    log_x = common
    log_y = beta * common + spread
    prices = 100.0 * np.exp(np.column_stack([log_x, log_y]))
    return pd.DataFrame(prices, index=_index(n_days, start), columns=list(names))


GENERATORS = {'random_walk': random_walk, 'trending': trending, 'cointegrated_pair': cointegrated_pair}
