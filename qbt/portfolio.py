"""Portfolio construction overlays applied to raw signal weights.

All overlays preserve the no-lookahead contract: scaling applied on day t uses
only returns up to day t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qbt.engine import TRADING_DAYS


def cap_weights(weights: pd.DataFrame, max_abs: float) -> pd.DataFrame:
    return weights.clip(-max_abs, max_abs)


def limit_gross(weights: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """Scale each day down so that the sum of absolute weights is at most ``gross``."""
    total = weights.abs().sum(axis=1)
    scale = (gross / total).clip(upper=1.0).fillna(1.0)
    return weights.mul(scale, axis=0)


def target_volatility(weights: pd.DataFrame, prices: pd.DataFrame, target_vol: float = 0.10,
                      window: int = 63, max_leverage: float = 3.0, min_periods: int | None = None) -> pd.DataFrame:
    """Scale weights so trailing realized volatility of the strategy matches ``target_vol``.

    The strategy's own realized return proxy is yesterday's weights times
    today's asset returns. Its rolling standard deviation up to day t sets the
    scale applied to day t's weights, so the overlay never sees the future.
    """
    asset_returns = prices.pct_change()
    proxy = (weights.shift(1) * asset_returns).sum(axis=1, min_count=1)
    realized = proxy.rolling(window, min_periods=min_periods or window).std() * np.sqrt(TRADING_DAYS)
    scale = (target_vol / realized.replace(0.0, np.nan)).clip(upper=max_leverage)
    return weights.mul(scale, axis=0)
