import math

import numpy as np
import pandas as pd
import pytest

from qbt import stats


def test_normal_functions_match_known_values():
    assert stats.norm_cdf(0.0) == pytest.approx(0.5)
    assert stats.norm_cdf(1.959964) == pytest.approx(0.975, abs=1e-6)
    for p in (0.001, 0.025, 0.3, 0.5, 0.8, 0.975, 0.9999):
        assert stats.norm_cdf(stats.norm_ppf(p)) == pytest.approx(p, abs=1e-9)


def test_sharpe_and_standard_error():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 2520))
    sr = stats.sharpe_ratio(r)
    assert sr == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))
    se = stats.sharpe_standard_error(r)
    assert se == pytest.approx(math.sqrt((1 + 0.5 * (sr / math.sqrt(252)) ** 2) / 2520) * math.sqrt(252))
    assert math.isnan(stats.sharpe_ratio(pd.Series([0.01, 0.01, 0.01])))


def test_drawdown_statistics():
    equity = pd.Series([1.0, 1.2, 0.9, 1.0, 1.3, 1.1, 1.2])
    dd = stats.drawdown(equity)
    assert dd.min() == pytest.approx(0.9 / 1.2 - 1)
    assert stats.max_drawdown_duration(equity) == 2
    assert stats.max_drawdown_duration(pd.Series([1.0, 2.0, 3.0])) == 0


def test_psr_and_dsr_behave_sensibly():
    assert stats.probabilistic_sharpe_ratio(0.0, 0.0, 1000) == pytest.approx(0.5)
    assert stats.probabilistic_sharpe_ratio(0.1, 0.0, 1000) > stats.probabilistic_sharpe_ratio(0.1, 0.0, 100)
    assert stats.probabilistic_sharpe_ratio(0.1, 0.0, 1000, skew=-1.0) < stats.probabilistic_sharpe_ratio(0.1, 0.0, 1000)
    # A single trial deflates nothing; more trials deflate more.
    assert stats.deflated_sharpe_ratio(0.1, 1000, 1, 0.01) == pytest.approx(stats.probabilistic_sharpe_ratio(0.1, 0.0, 1000))
    assert stats.deflated_sharpe_ratio(0.1, 1000, 100, 0.01) < stats.deflated_sharpe_ratio(0.1, 1000, 10, 0.01)
    # sqrt(0.01) * ((1 - gamma) * ppf(0.99) + gamma * ppf(1 - 1 / (100 e))) = 0.1 * (0.4228 * 2.3263 + 0.5772 * 2.6789)
    assert stats.expected_max_sharpe(100, 0.01) == pytest.approx(0.2530, abs=0.001)


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    low, high = stats.bootstrap_sharpe_ci(r, n_boot=500, seed=1)
    assert low < stats.sharpe_ratio(r) < high
    assert high - low < 3.0


def test_summarize_on_constant_growth():
    r = pd.Series([0.001] * 504)
    s = stats.summarize(r)
    assert s['ann_return'] == pytest.approx(1.001 ** 252 - 1)
    assert s['max_drawdown'] == 0.0
    assert s['hit_rate'] == 1.0
    assert math.isnan(s['sharpe'])


def test_summarize_with_trials_reports_deflated_sharpe():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0004, 0.01, 1260))
    s = stats.summarize(r, n_trials=50, trial_sr_variance=0.002)
    assert 0 <= s['deflated_sharpe'] <= s['psr_vs_zero'] <= 1
