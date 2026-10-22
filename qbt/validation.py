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
from qbt.stats import sharpe_ratio

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

    @property
    def sharpe_variance(self) -> float:
        """Variance of per-period Sharpe across trials, the input the deflated Sharpe needs."""
        periodic = self.table['sharpe'] / math.sqrt(252)
        return float(periodic.var(ddof=1)) if len(periodic) > 1 else 0.0


def grid_search(prices: pd.DataFrame, factory: Factory, grid: dict[str, list], costs: Costs = Costs(),
                objective: Callable[[pd.Series], float] = sharpe_ratio, **backtest_kwargs) -> GridResult:
    rows = []
    for params in expand_grid(grid):
        weights = factory(**params)(prices)
        result = run_backtest(prices, weights, costs, **backtest_kwargs)
        score = objective(result.returns)
        rows.append({**params, 'sharpe': sharpe_ratio(result.returns), 'objective': score,
                     'ann_turnover': float(result.turnover.mean() * 252)})
    table = pd.DataFrame(rows)
    best_row = table['objective'].fillna(-np.inf).idxmax()
    best = {k: table.loc[best_row, k] for k in grid}
    best = {k: (v.item() if hasattr(v, 'item') else v) for k, v in best.items()}
    return GridResult(table, best, float(table.loc[best_row, 'sharpe']))


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
    trial_sharpe_variance: float = 0.0

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
    """
    n = len(prices)
    if train + test > n:
        raise ValueError('not enough data for one train/test split')
    folds, pieces = [], []
    variances = []
    for test_start in range(train, n - test + 1, test):
        train_start = 0 if anchored else test_start - train
        train_prices = prices.iloc[train_start:test_start]
        search = grid_search(train_prices, factory, grid, costs, **backtest_kwargs)
        variances.append(search.sharpe_variance)
        strategy = factory(**search.best)
        history = prices.iloc[:test_start + test]
        weights = strategy(history)
        test_prices = history.iloc[test_start:]
        test_weights = weights.iloc[test_start - backtest_kwargs.get('execution_lag', 1):]
        result = run_backtest(test_prices, test_weights, costs, **backtest_kwargs)
        pieces.append(result.returns)
        folds.append(Fold(train_prices.index[0], train_prices.index[-1], test_prices.index[0],
                          test_prices.index[-1], search.best, search.best_sharpe, sharpe_ratio(result.returns)))
    returns = pd.concat(pieces).rename('oos_return')
    return WalkForwardResult(returns, folds, len(expand_grid(grid)), float(np.mean(variances)))


def in_sample_vs_out_of_sample(prices: pd.DataFrame, factory: Factory, grid: dict[str, list],
                               costs: Costs = Costs(), **wf_kwargs) -> dict:
    """Headline comparison: best in-sample Sharpe on the full history versus walk-forward Sharpe."""
    full = grid_search(prices, factory, grid, costs)
    wf = walk_forward(prices, factory, grid, costs=costs, **wf_kwargs)
    return {'in_sample_best': full.best, 'in_sample_sharpe': full.best_sharpe,
            'out_of_sample_sharpe': sharpe_ratio(wf.returns), 'n_trials': full.table.shape[0],
            'trial_sharpe_variance': full.sharpe_variance, 'walk_forward': wf}
