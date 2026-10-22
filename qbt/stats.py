"""Performance statistics with standard errors and multiple-testing corrections.

Point estimates of Sharpe are cheap; the interesting questions are how wide
the confidence interval is and how much of the result is explained by having
tried many configurations. This module implements:

* annualized return, volatility, Sharpe, Sortino, drawdown statistics;
* the Lo (2002) iid standard error of the Sharpe ratio;
* the probabilistic Sharpe ratio and deflated Sharpe ratio of Bailey and
  Lopez de Prado (2014), which adjust for non-normal returns and for the
  number of trials that produced the reported strategy;
* a circular block bootstrap confidence interval for the Sharpe ratio.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, refined by one Newton step)."""
    if not 0.0 < p < 1.0:
        raise ValueError('p must be in (0, 1)')
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    low, high = 0.02425, 1 - 0.02425
    if p < low or p > high:
        q = math.sqrt(-2 * math.log(p if p < low else 1 - p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        x = x if p < low else -x
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def _clean(returns: pd.Series) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return r[~np.isnan(r)]


def sharpe_ratio(returns: pd.Series, periods: int = TRADING_DAYS, rf: float = 0.0) -> float:
    r = _clean(returns) - rf / periods
    if len(r) < 2:
        return float('nan')
    sd = r.std(ddof=1)
    if sd <= 1e-12 * max(abs(r.mean()), 1e-300):
        return float('nan')
    return float(r.mean() / sd * math.sqrt(periods))


def sharpe_standard_error(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Lo (2002) standard error of the annualized Sharpe ratio under iid returns."""
    r = _clean(returns)
    sr = sharpe_ratio(r, periods) / math.sqrt(periods)
    return float(math.sqrt((1 + 0.5 * sr ** 2) / len(r)) * math.sqrt(periods))


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest run of periods spent below the previous high-water mark."""
    under = (equity < equity.cummax()).to_numpy()
    longest = run = 0
    for flag in under:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return int(longest)


def probabilistic_sharpe_ratio(sr: float, benchmark_sr: float, n: int, skew: float = 0.0,
                               kurtosis: float = 3.0) -> float:
    """P(true SR > benchmark) given an observed per-period SR over n observations.

    Bailey and Lopez de Prado (2012). ``kurtosis`` is the raw fourth moment
    ratio, 3 for a normal distribution. All Sharpe ratios are per period,
    not annualized.
    """
    if n < 2:
        return float('nan')
    denom = math.sqrt(max(1 - skew * sr + (kurtosis - 1) / 4 * sr ** 2, 1e-12))
    return norm_cdf((sr - benchmark_sr) * math.sqrt(n - 1) / denom)


def expected_max_sharpe(n_trials: int, trial_sr_variance: float) -> float:
    """Expected maximum of n_trials iid trial Sharpe ratios drawn with zero mean."""
    if n_trials < 2:
        return 0.0
    return math.sqrt(trial_sr_variance) * ((1 - EULER_GAMMA) * norm_ppf(1 - 1 / n_trials)
                                           + EULER_GAMMA * norm_ppf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe_ratio(sr: float, n: int, n_trials: int, trial_sr_variance: float,
                          skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """Probability that the best of ``n_trials`` configurations has true SR above zero.

    The benchmark is the Sharpe ratio one expects from the luckiest of
    n_trials unskilled strategies with Sharpe variance ``trial_sr_variance``
    (Bailey and Lopez de Prado, 2014). Per-period units throughout.
    """
    return probabilistic_sharpe_ratio(sr, expected_max_sharpe(n_trials, trial_sr_variance), n, skew, kurtosis)


def bootstrap_sharpe_ci(returns: pd.Series, n_boot: int = 2000, block: int = 20, level: float = 0.95,
                        periods: int = TRADING_DAYS, seed: int = 0) -> tuple[float, float]:
    """Circular block bootstrap percentile interval for the annualized Sharpe ratio."""
    r = _clean(returns)
    n = len(r)
    if n < block + 1:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    idx = ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(n_boot, -1)[:, :n]
    samples = r[idx]
    sds = samples.std(axis=1, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        srs = samples.mean(axis=1) / sds * math.sqrt(periods)
    srs = srs[np.isfinite(srs)]
    alpha = (1 - level) / 2
    return (float(np.quantile(srs, alpha)), float(np.quantile(srs, 1 - alpha)))


def summarize(returns: pd.Series, turnover: pd.Series | None = None, periods: int = TRADING_DAYS,
              n_trials: int = 1, trial_sr_variance: float | None = None) -> dict[str, float]:
    """Headline statistics for a return series. Annualized unless the key says otherwise."""
    r = _clean(returns)
    n = len(r)
    equity = pd.Series(np.cumprod(1 + r))
    years = n / periods
    total = float(equity.iloc[-1]) if n else float('nan')
    ann_return = total ** (1 / years) - 1 if n and years > 0 and total > 0 else float('nan')
    vol = float(r.std(ddof=1) * math.sqrt(periods)) if n > 1 else float('nan')
    sr = sharpe_ratio(r, periods)
    downside = r[r < 0]
    sortino = float(r.mean() / downside.std(ddof=1) * math.sqrt(periods)) if len(downside) > 1 else float('nan')
    dd = drawdown(equity)
    skew = float(pd.Series(r).skew()) if n > 2 else 0.0
    kurt = float(pd.Series(r).kurt() + 3.0) if n > 3 else 3.0
    sr_period = sr / math.sqrt(periods) if not math.isnan(sr) else float('nan')
    stats = {
        'periods': n, 'years': years, 'ann_return': ann_return, 'ann_vol': vol, 'sharpe': sr,
        'sharpe_se': sharpe_standard_error(r, periods) if n > 1 else float('nan'),
        't_stat': sr * math.sqrt(years) if not math.isnan(sr) else float('nan'),
        'sortino': sortino, 'max_drawdown': float(dd.min()) if n else float('nan'),
        'max_drawdown_days': max_drawdown_duration(equity),
        'calmar': ann_return / -float(dd.min()) if n and dd.min() < 0 else float('nan'),
        'skew': skew, 'kurtosis': kurt,
        'hit_rate': float((r > 0).sum() / max((r != 0).sum(), 1)) if n else float('nan'),
        'psr_vs_zero': probabilistic_sharpe_ratio(sr_period, 0.0, n, skew, kurt) if n > 1 else float('nan'),
    }
    if n_trials > 1 and trial_sr_variance is not None:
        stats['n_trials'] = n_trials
        stats['deflated_sharpe'] = deflated_sharpe_ratio(sr_period, n, n_trials, trial_sr_variance, skew, kurt)
    if turnover is not None:
        stats['ann_turnover'] = float(_clean(turnover).mean() * periods)
    return stats
