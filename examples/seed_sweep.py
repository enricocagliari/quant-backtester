"""Repeat the null experiment across seeds so that one lucky or unlucky draw cannot tell the story.

Run from the repository root: python examples/seed_sweep.py [n_seeds]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from qbt import portfolio, signals, synthetic
from qbt.engine import Costs
from qbt.validation import in_sample_vs_out_of_sample

GRID = {'lookback': [21, 63, 126, 252], 'skip': [0, 5, 10]}
COSTS = Costs(commission_bps=1.0, half_spread_bps=2.5)


def factory(lookback, skip):
    def strategy(prices):
        weights = signals.time_series_momentum(prices, lookback=lookback, skip=skip)
        return portfolio.target_volatility(weights, prices, 0.10)
    return strategy


def sweep(generator, n_seeds: int) -> pd.DataFrame:
    rows = []
    for seed in range(n_seeds):
        prices = generator(n_days=2520, n_assets=8, seed=seed)
        c = in_sample_vs_out_of_sample(prices, factory, GRID, COSTS, train=756, test=252, rebalance_threshold=0.01)
        rows.append({'seed': seed, 'is_best_sharpe': c['in_sample_sharpe'],
                     'is_deflated_p': c['in_sample_deflated_sharpe'], 'oos_sharpe': c['out_of_sample_sharpe']})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for name in ('random_walk', 'trending'):
        table = sweep(synthetic.GENERATORS[name], n)
        print(f'## {name}, {n} seeds')
        print(table.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
        mean, sd = table.mean(numeric_only=True), table.std(numeric_only=True)
        print(f"mean in-sample best {mean['is_best_sharpe']:.2f} (sd {sd['is_best_sharpe']:.2f}); "
              f"mean out-of-sample {mean['oos_sharpe']:.2f} (sd {sd['oos_sharpe']:.2f}); "
              f"in-sample > out-of-sample in {(table['is_best_sharpe'] > table['oos_sharpe']).sum()} of {n}; "
              f"mean deflated P(SR > 0) {mean['is_deflated_p']:.1%}", '\n')
