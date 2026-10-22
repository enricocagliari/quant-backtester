"""End-to-end research session as a script: data, backtest, validation, deflation.

Run from the repository root: python examples/research_notebook.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qbt import signals, synthetic
from qbt.engine import Costs, run_backtest
from qbt.report import format_stats
from qbt.stats import summarize
from qbt.validation import in_sample_vs_out_of_sample

prices = synthetic.trending(n_days=2520, n_assets=8, seed=7)
costs = Costs(commission_bps=1.0, half_spread_bps=2.5)


def factory(lookback=126, skip=5):
    return lambda p: signals.time_series_momentum(p, lookback=lookback, skip=skip)


result = run_backtest(prices, factory()(prices), costs)
print(format_stats(summarize(result.returns, result.turnover), 'Single configuration'))

comparison = in_sample_vs_out_of_sample(prices, factory, {'lookback': [21, 63, 126, 252], 'skip': [0, 5, 10]},
                                        costs, train=756, test=252)
wf = comparison['walk_forward']
print(f"\nBest in-sample: {comparison['in_sample_best']} Sharpe {comparison['in_sample_sharpe']:.2f}")
print(f"Walk-forward out-of-sample Sharpe: {comparison['out_of_sample_sharpe']:.2f}")
print(wf.summary().to_string(index=False))
print()
print(format_stats(summarize(wf.returns, n_trials=comparison['n_trials'],
                             trial_sr_variance=comparison['trial_sharpe_variance']), 'Out of sample, deflated'))
