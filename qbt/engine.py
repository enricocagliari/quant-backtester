"""Daily portfolio simulation with explicit timing and transaction costs.

Timing convention
-----------------
A target weight computed from information available at the close of day t is
executed at the close of day t + ``execution_lag`` (default 1). The engine
shifts the target frame itself, so a strategy returns weights indexed by the
day on which they were *decided*; the engine never lets a decision trade on
the same bar that produced it.

Accounting
----------
State is cash plus a vector of (fractional) share holdings. Each day:

1. financing accrues on the previous close: cash earns ``cash_rate``, short
   market value pays ``borrow_rate``;
2. the portfolio is marked at today's close to get pre-trade value V;
3. the target holdings are ``w * V / p``; the engine trades the difference,
   charging ``commission_bps + half_spread_bps`` on traded notional, plus an
   optional square-root market impact term;
4. equity is cash plus holdings at today's close.

Daily return is the change in post-trade equity, so costs reduce returns on
the day they are paid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Costs:
    """All rates in basis points of traded notional unless stated otherwise."""
    commission_bps: float = 1.0
    half_spread_bps: float = 2.5
    impact_bps_per_sqrt_pct_adv: float = 0.0  # cost = k * sqrt(notional / ADV), needs volume
    borrow_rate_annual: float = 0.0  # paid on short market value
    cash_rate_annual: float = 0.0  # earned on positive cash, paid on negative cash

    @property
    def linear_rate(self) -> float:
        return (self.commission_bps + self.half_spread_bps) / 1e4


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame  # held weights at each close, after trading
    target_weights: pd.DataFrame  # the executed targets (already lagged)
    turnover: pd.Series  # traded notional / pre-trade value
    costs: pd.Series  # currency paid in transaction costs each day
    financing: pd.Series  # currency earned (+) or paid (-) in financing each day
    shares: pd.DataFrame
    cash: pd.Series
    config: dict = field(default_factory=dict)

    @property
    def annual_cost_drag(self) -> float:
        """Transaction costs as a fraction of portfolio value, annualized."""
        value_before = self.equity.shift(1).fillna(self.config.get('initial_capital', np.nan))
        return float((self.costs / value_before).sum() / len(self.costs) * TRADING_DAYS)


def run_backtest(prices: pd.DataFrame, target_weights: pd.DataFrame, costs: Costs = Costs(),
                 initial_capital: float = 1_000_000.0, execution_lag: int = 1,
                 rebalance_threshold: float = 0.0, volume: pd.DataFrame | None = None) -> BacktestResult:
    """Simulate holding ``target_weights`` (decided at close t, executed at close t + lag).

    ``prices`` must be a complete, monotonically increasing DatetimeIndex frame
    of tradeable closes. Weights outside ``target_weights``' index or NaN are
    treated as zero (flat). ``rebalance_threshold`` skips trades in an asset
    whose absolute weight gap is at or below the threshold, which models the
    common practice of tolerating drift to save costs.
    """
    if execution_lag < 1:
        raise ValueError('execution_lag must be at least 1; a decision cannot trade on its own bar')
    if prices.isna().any().any():
        raise ValueError('prices contain NaN; align or forward-fill before backtesting')
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError('prices index must be strictly increasing')
    if (prices <= 0).any().any():
        raise ValueError('prices must be positive')
    targets = target_weights.reindex(index=prices.index, columns=prices.columns).astype(float)
    executed = targets.shift(execution_lag).fillna(0.0)
    P = prices.to_numpy(dtype=float)
    W = executed.to_numpy(dtype=float)
    V = None if volume is None else volume.reindex_like(prices).to_numpy(dtype=float)
    n_days, n_assets = P.shape
    shares = np.zeros(n_assets)
    cash = float(initial_capital)
    out_equity = np.empty(n_days)
    out_turnover = np.zeros(n_days)
    out_costs = np.zeros(n_days)
    out_fin = np.zeros(n_days)
    out_shares = np.empty((n_days, n_assets))
    out_cash = np.empty(n_days)
    out_weights = np.empty((n_days, n_assets))
    daily_cash_rate = costs.cash_rate_annual / TRADING_DAYS
    daily_borrow_rate = costs.borrow_rate_annual / TRADING_DAYS
    for t in range(n_days):
        p = P[t]
        if t > 0:
            short_mv = np.minimum(shares * P[t - 1], 0.0).sum()
            financing = cash * daily_cash_rate + short_mv * daily_borrow_rate
            cash += financing
            out_fin[t] = financing
        value = cash + shares @ p
        if value <= 0:
            raise RuntimeError(f'portfolio value {value:.2f} at {prices.index[t].date()}; bankrupt')
        target_shares = W[t] * value / p
        delta = target_shares - shares
        if rebalance_threshold > 0:
            current_w = shares * p / value
            delta[np.abs(W[t] - current_w) <= rebalance_threshold] = 0.0
        traded = np.abs(delta) * p
        notional = traded.sum()
        cost = notional * costs.linear_rate
        if V is not None and costs.impact_bps_per_sqrt_pct_adv > 0:
            adv = V[t] * p
            with np.errstate(divide='ignore', invalid='ignore'):
                participation = np.where(adv > 0, traded / adv, 0.0)
            cost += (traded * costs.impact_bps_per_sqrt_pct_adv / 1e4 * np.sqrt(participation * 100)).sum()
        cash -= delta @ p + cost
        shares = shares + delta
        equity = cash + shares @ p
        out_equity[t] = equity
        out_turnover[t] = notional / value
        out_costs[t] = cost
        out_shares[t] = shares
        out_cash[t] = cash
        out_weights[t] = shares * p / equity
    idx = prices.index
    equity = pd.Series(out_equity, index=idx, name='equity')
    returns = equity.pct_change()
    returns.iloc[0] = out_equity[0] / initial_capital - 1.0
    return BacktestResult(
        equity=equity, returns=returns.rename('return'),
        weights=pd.DataFrame(out_weights, index=idx, columns=prices.columns),
        target_weights=executed, turnover=pd.Series(out_turnover, index=idx, name='turnover'),
        costs=pd.Series(out_costs, index=idx, name='costs'),
        financing=pd.Series(out_fin, index=idx, name='financing'),
        shares=pd.DataFrame(out_shares, index=idx, columns=prices.columns),
        cash=pd.Series(out_cash, index=idx, name='cash'),
        config={'initial_capital': initial_capital, 'execution_lag': execution_lag,
                'rebalance_threshold': rebalance_threshold, 'costs': costs.__dict__})
