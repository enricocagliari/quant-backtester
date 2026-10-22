"""Signal functions: prices in, target weights out.

Every function here obeys one contract, which the test suite checks by
truncation: the weight it emits for day t depends only on prices up to and
including day t. The engine then delays execution by at least one bar, so no
strategy can trade on the close that produced its signal.

Weights are portfolio fractions. Positive is long, negative is short. NaN
during a lookback warmup means flat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _check(prices: pd.DataFrame) -> None:
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError('prices must be indexed by DatetimeIndex')


def trailing_return(prices: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """Return from close t-lookback to close t-skip. ``skip`` drops the most recent bars."""
    if skip >= lookback:
        raise ValueError('skip must be shorter than lookback')
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def time_series_momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21,
                         vol_window: int = 63, vol_scale: bool = True) -> pd.DataFrame:
    """Long assets with positive trailing return, short the rest; optional inverse-vol sizing.

    Each asset gets sign(trailing return) scaled so that gross exposure is one.
    With ``vol_scale`` the sign is divided by trailing realized volatility
    before normalization, which equalizes risk contribution across assets.
    """
    _check(prices)
    signal = np.sign(trailing_return(prices, lookback, skip))
    if vol_scale:
        vol = prices.pct_change().rolling(vol_window).std()
        signal = signal / vol.replace(0.0, np.nan)
    gross = signal.abs().sum(axis=1).replace(0.0, np.nan)
    return signal.div(gross, axis=0)


def cross_sectional_momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21,
                             top: int = 2) -> pd.DataFrame:
    """Dollar-neutral: long the ``top`` best trailing returns, short the ``top`` worst."""
    _check(prices)
    if 2 * top > prices.shape[1]:
        raise ValueError('top is too large for the number of assets')
    ret = trailing_return(prices, lookback, skip)
    rank = ret.rank(axis=1, method='first')
    n = ret.notna().sum(axis=1)
    longs = rank.gt(n - top, axis=0) & ret.notna()
    shorts = rank.le(top, axis=0) & ret.notna()
    weights = longs.astype(float) * (0.5 / top) - shorts.astype(float) * (0.5 / top)
    return weights.where(n >= 2 * top)


def mean_reversion(prices: pd.DataFrame, window: int = 20, z_cap: float = 2.0) -> pd.DataFrame:
    """Fade the z-score of price against its rolling mean; gross exposure capped at one."""
    _check(prices)
    rolling = prices.rolling(window)
    std = rolling.std()
    z = ((prices - rolling.mean()) / std.replace(0.0, np.nan)).mask(std == 0.0, 0.0)
    raw = -z.clip(-z_cap, z_cap) / z_cap
    gross = raw.abs().sum(axis=1)
    return raw.div(gross.clip(lower=1.0), axis=0)


def rolling_ols_zscore(y: pd.Series, x: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Rolling OLS of y on x with intercept; returns (hedge ratio, z-score of today's residual).

    For each day t the regression uses observations t-window+1 .. t only. The
    z-score divides today's residual by the standard deviation of the
    residuals in that same window, so both numerator and denominator are
    computed from the same hedge ratio.
    """
    yv, xv = y.to_numpy(dtype=float), x.to_numpy(dtype=float)
    n = len(yv)
    beta = np.full(n, np.nan)
    z = np.full(n, np.nan)
    for t in range(window - 1, n):
        xs, ys = xv[t - window + 1:t + 1], yv[t - window + 1:t + 1]
        if np.isnan(xs).any() or np.isnan(ys).any():
            continue
        xm, ym = xs.mean(), ys.mean()
        var = ((xs - xm) ** 2).sum()
        if var <= 0:
            continue
        b = ((xs - xm) * (ys - ym)).sum() / var
        resid = ys - ym - b * (xs - xm)
        sd = resid.std(ddof=2)
        if sd > 0:
            beta[t] = b
            z[t] = resid[-1] / sd
    return pd.Series(beta, index=y.index), pd.Series(z, index=y.index)


def pairs_trading(prices: pd.DataFrame, left: str, right: str, window: int = 60,
                  entry: float = 2.0, exit: float = 0.5) -> pd.DataFrame:
    """Trade the residual of log(left) regressed on log(right) over a rolling window.

    Enter short the spread (short ``left``, long ``right``) when the residual
    z-score exceeds ``entry``, long below ``-entry``, and flatten when the
    absolute z-score falls under ``exit``. Legs are hedge-ratio weighted and
    sum to gross exposure of one while a position is open.
    """
    _check(prices)
    if exit >= entry:
        raise ValueError('exit threshold must be below entry threshold')
    beta, z = rolling_ols_zscore(np.log(prices[left]), np.log(prices[right]), window)
    state = np.zeros(len(z))
    zv = z.to_numpy()
    for t in range(1, len(zv)):
        prev = state[t - 1]
        if np.isnan(zv[t]):
            state[t] = 0.0
        elif prev == 0.0:
            state[t] = -1.0 if zv[t] > entry else 1.0 if zv[t] < -entry else 0.0
        else:
            state[t] = 0.0 if abs(zv[t]) < exit else prev
    denom = (1.0 + beta.abs()).to_numpy()
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights[left] = state / denom
    weights[right] = -state * beta.to_numpy() / denom
    return weights.fillna(0.0)


def buy_and_hold(prices: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Constant target weights, equal weight by default. Useful as a benchmark and a test."""
    _check(prices)
    if weights is None:
        weights = {c: 1.0 / prices.shape[1] for c in prices.columns}
    return pd.DataFrame([[weights.get(c, 0.0) for c in prices.columns]] * len(prices),
                        index=prices.index, columns=prices.columns)


REGISTRY = {
    'time_series_momentum': time_series_momentum,
    'cross_sectional_momentum': cross_sectional_momentum,
    'mean_reversion': mean_reversion,
    'pairs_trading': pairs_trading,
    'buy_and_hold': buy_and_hold,
}
