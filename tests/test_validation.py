import numpy as np
import pandas as pd
import pytest

from qbt import signals, synthetic, validation
from qbt.engine import Costs
from qbt.stats import sharpe_ratio


def factory(lookback, skip=0):
    return lambda p: signals.time_series_momentum(p, lookback=lookback, skip=skip, vol_scale=False)


def test_grid_search_covers_every_combination(prices):
    grid = {'lookback': [20, 40], 'skip': [0, 2]}
    result = validation.grid_search(prices, factory, grid, Costs(0, 0))
    assert len(result.table) == 4
    assert result.best in validation.expand_grid(grid)
    assert result.best_sharpe == result.table['sharpe'].max()


def test_walk_forward_windows_tile_the_history_without_overlap():
    prices = synthetic.random_walk(n_days=1000, n_assets=3, seed=5)
    wf = validation.walk_forward(prices, factory, {'lookback': [20, 60]}, train=400, test=200, costs=Costs(0, 0))
    assert len(wf.folds) == 3
    for fold in wf.folds:
        assert fold.train_end < fold.test_start
        assert len(prices.loc[fold.train_start:fold.train_end]) == 400
        assert len(prices.loc[fold.test_start:fold.test_end]) == 200
    for a, b in zip(wf.folds, wf.folds[1:]):
        assert b.test_start > a.test_end
    assert len(wf.returns) == 600
    assert wf.returns.index.is_monotonic_increasing and not wf.returns.index.has_duplicates
    assert wf.n_trials == 2
    # The weights decided on the last training day are executed on the first test day:
    # with a commission, day one of every fold pays the cost of entering the position.
    with_costs = validation.walk_forward(prices, factory, {'lookback': [20, 60]}, train=400, test=200,
                                         costs=Costs(1.0, 0.0))
    for fold in with_costs.folds:
        assert with_costs.returns.loc[fold.test_start] < 0.0


def test_anchored_walk_forward_grows_the_training_window():
    prices = synthetic.random_walk(n_days=1000, n_assets=3, seed=5)
    wf = validation.walk_forward(prices, factory, {'lookback': [20, 60]}, train=400, test=200,
                                 costs=Costs(0, 0), anchored=True)
    assert [f.train_start for f in wf.folds] == [prices.index[0]] * 3
    assert [len(prices.loc[f.train_start:f.train_end]) for f in wf.folds] == [400, 600, 800]


def test_walk_forward_parameters_come_only_from_training_data():
    prices = synthetic.random_walk(n_days=1000, n_assets=3, seed=6)
    base = validation.walk_forward(prices, factory, {'lookback': [20, 60, 120]}, train=400, test=200, costs=Costs(0, 0))
    shocked = prices.copy()
    shocked.iloc[800:] *= np.random.default_rng(9).uniform(0.7, 1.3, size=shocked.iloc[800:].shape)
    again = validation.walk_forward(shocked, factory, {'lookback': [20, 60, 120]}, train=400, test=200, costs=Costs(0, 0))
    # Folds whose train and test windows end before day 800 must be identical.
    for a, b in zip(base.folds[:2], again.folds[:2]):
        assert a.params == b.params and a.in_sample_sharpe == b.in_sample_sharpe
    pd.testing.assert_series_equal(base.returns.iloc[:400], again.returns.iloc[:400])


def test_in_sample_beats_out_of_sample_on_pure_noise():
    """The whole point: selection inflates in-sample Sharpe; walk-forward does not inherit it."""
    in_sample, out_of_sample = [], []
    for seed in range(10):
        prices = synthetic.random_walk(n_days=1500, n_assets=4, seed=100 + seed, regime_switching=False)
        comparison = validation.in_sample_vs_out_of_sample(
            prices, factory, {'lookback': [10, 20, 40, 80, 160], 'skip': [0, 5]}, Costs(0, 0), train=500, test=250)
        in_sample.append(comparison['in_sample_sharpe'])
        out_of_sample.append(comparison['out_of_sample_sharpe'])
    # Picking the best of ten configurations on noise looks like skill; walking forward does not.
    assert np.mean(in_sample) > 0.3
    assert np.mean(in_sample) - np.mean(out_of_sample) > 0.15
    assert abs(np.mean(out_of_sample)) < 0.3


def test_walk_forward_rejects_short_history(prices):
    with pytest.raises(ValueError):
        validation.walk_forward(prices, factory, {'lookback': [20]}, train=500, test=200)


def test_grid_result_deflates_its_own_best(prices):
    result = validation.grid_search(prices, factory, {'lookback': [10, 20, 40, 80]}, Costs(0, 0))
    assert result.sharpe_variance >= 0 and result.n_trials == 4
    assert sharpe_ratio(result.best_returns) == result.best_sharpe
    single = validation.grid_search(prices, factory, {'lookback': [result.best['lookback']]}, Costs(0, 0))
    # One trial is not deflated; four trials are deflated below the one-trial probability.
    assert 0 <= result.deflated_sharpe < single.deflated_sharpe <= 1
