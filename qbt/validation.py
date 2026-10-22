"""Parameter search and walk-forward validation.

``grid_search`` is the thing that overfits: it backtests every parameter
combination on one window and reports the best. ``walk_forward`` is the
antidote: it repeats the search inside each training window, applies the
winner to the following unseen window, and stitches the out-of-sample
returns together. Comparing the two is the most informative chart in a
strategy research report, and ``in_sample_vs_out_of_sample`` builds it.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from qbt.engine import Costs, run_backtest
from qbt.stats import TRADING_DAYS, deflated_sharpe_ratio, sharpe_ratio

Strategy = Callable[[pd.DataFrame], pd.DataFrame]
Factory = Callable[..., Strategy]


def expand_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


@dataclass
class GridResult:
    table: pd.DataFrame  # one row per parameter set with its in-sample statistics
    best: dict
    best_sharpe: float
    best_returns: pd.Series

    @property
    def n_trials(self) -> int:
        return int(self.table.shape[0])

    @property
    def sharpe_variance(self) -> float:
        """Variance of per-period Sharpe across trials, the input the deflated Sharpe needs."""
        periodic = self.table['sharpe'].dropna() / math.sqrt(TRADING_DAYS)
        return float(periodic.var(ddof=1)) if len(periodic) > 1 else 0.0

    @property
    def deflated_sharpe(self) -> float:
        """P(true Sharpe of the selected configuration > 0), corrected for having tried n_trials.

        This is the statistic the deflated Sharpe ratio is designed for: the
        best of many in-sample trials. Out-of-sample returns were not selected,
        so they should be judged with the plain probabilistic Sharpe instead.
        """
        r = self.best_returns.dropna()
        if len(r) < 4 or math.isnan(self.best_sharpe):
            return float('nan')
        return deflated_sharpe_ratio(self.best_sharpe / math.sqrt(TRADING_DAYS), len(r), self.n_trials,
                                     self.sharpe_variance, float(r.skew()), float(r.kurt() + 3.0))


def grid_search(prices: pd.DataFrame, factory: Factory, grid: dict[str, list], costs: Costs = Costs(),
                objective: Callable[[pd.Series], float] = sharpe_ratio, **backtest_kwargs) -> GridResult:
    rows, returns = [], []
    for params in expand_grid(grid):
        weights = factory(**params)(prices)
        result = run_backtest(prices, weights, costs, **backtest_kwargs)
        score = objective(result.returns)
        rows.append({**params, 'sharpe': sharpe_ratio(result.returns), 'objective': score,
                     'ann_turnover': float(result.turnover.mean() * TRADING_DAYS)})
        returns.append(result.returns)
    table = pd.DataFrame(rows)
    best_row = int(table['objective'].fillna(-np.inf).idxmax())
    best = {k: table.loc[best_row, k] for k in grid}
    best = {k: (v.item() if hasattr(v, 'item') else v) for k, v in best.items()}
    return GridResult(table, best, float(table.loc[best_row, 'sharpe']), returns[best_row])


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: dict
    in_sample_sharpe: float
    out_of_sample_sharpe: float


@dataclass
class WalkForwardResult:
    returns: pd.Series  # stitched out-of-sample daily returns
    folds: list[Fold] = field(default_factory=list)
    n_trials: int = 1

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{'train': f'{f.train_start.date()} to {f.train_end.date()}',
                              'test': f'{f.test_start.date()} to {f.test_end.date()}',
                              **f.params, 'is_sharpe': f.in_sample_sharpe, 'oos_sharpe': f.out_of_sample_sharpe}
                             for f in self.folds])


def walk_forward(prices: pd.DataFrame, factory: Factory, grid: dict[str, list], train: int = 756,
                 test: int = 252, costs: Costs = Costs(), anchored: bool = False,
                 **backtest_kwargs) -> WalkForwardResult:
    """Rolling (or anchored) train/test splits with the parameter search redone in each fold.

    The strategy is evaluated on the test window using prices up to the end
    of that window, which the no-lookahead contract makes safe: signals for
    test day t depend only on data up to t. Parameters, however, come only
    from the training window, so the test returns are genuinely out of
    sample with respect to the selection.

    Each fold starts flat with fresh capital. The simulation begins
    ``execution_lag`` bars before the test window so that the weights decided
    on the last training days are executed on the first test day; the
    returns of those warm-up bars are discarded.
    """
    n = len(prices)
    if train + test > n:
        raise ValueError('not enough data for one train/test split')
    lag = backtest_kwargs.get('execution_lag', 1)
    folds, pieces = [], []
    for test_start in range(train, n - test + 1, test):
        train_start = 0 if anchored else test_start - train
        train_prices = prices.iloc[train_start:test_start]
        search = grid_search(train_prices, factory, grid, costs, **backtest_kwargs)
        strategy = factory(**search.best)
        history = prices.iloc[:test_start + test]
        weights = strategy(history)
        result = run_backtest(history.iloc[test_start - lag:], weights.iloc[test_start - lag:], costs,
                              **backtest_kwargs)
        oos = result.returns.iloc[lag:]
        pieces.append(oos)
        folds.append(Fold(train_prices.index[0], train_prices.index[-1], oos.index[0], oos.index[-1],
                          search.best, search.best_sharpe, sharpe_ratio(oos)))
    returns = pd.concat(pieces).rename('oos_return')
    return WalkForwardResult(returns, folds, len(expand_grid(grid)))


def in_sample_vs_out_of_sample(prices: pd.DataFrame, factory: Factory, grid: dict[str, list],
                               costs: Costs = Costs(), **wf_kwargs) -> dict:
    """Headline comparison: best in-sample Sharpe on the full history versus walk-forward Sharpe.

    The in-sample Sharpe is the maximum over the grid, so it is reported with
    its deflated probability. The walk-forward Sharpe was never selected, so
    it is judged by the plain probabilistic Sharpe in ``summarize``.
    """
    backtest_kwargs = {k: v for k, v in wf_kwargs.items() if k in ('initial_capital', 'execution_lag',
                                                                    'rebalance_threshold')}
    full = grid_search(prices, factory, grid, costs, **backtest_kwargs)
    wf = walk_forward(prices, factory, grid, costs=costs, **wf_kwargs)
    return {'in_sample_best': full.best, 'in_sample_sharpe': full.best_sharpe,
            'in_sample_deflated_sharpe': full.deflated_sharpe, 'n_trials': full.n_trials,
            'trial_sharpe_variance': full.sharpe_variance,
            'out_of_sample_sharpe': sharpe_ratio(wf.returns), 'walk_forward': wf}
