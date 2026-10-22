"""Text and figure output for a backtest."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from qbt.engine import BacktestResult
from qbt.stats import bootstrap_sharpe_ci, drawdown, summarize

LABELS = {
    'periods': ('Trading days', '{:.0f}'), 'ann_return': ('Annual return', '{:.2%}'),
    'ann_vol': ('Annual volatility', '{:.2%}'), 'sharpe': ('Sharpe', '{:.2f}'),
    'sharpe_se': ('Sharpe std. error', '{:.2f}'), 't_stat': ('t-statistic', '{:.2f}'),
    'sortino': ('Sortino', '{:.2f}'), 'max_drawdown': ('Max drawdown', '{:.2%}'),
    'max_drawdown_days': ('Longest drawdown (days)', '{:.0f}'), 'calmar': ('Calmar', '{:.2f}'),
    'skew': ('Skew', '{:.2f}'), 'kurtosis': ('Kurtosis', '{:.2f}'), 'hit_rate': ('Hit rate', '{:.1%}'),
    'ann_turnover': ('Annual turnover', '{:.1f}x'), 'psr_vs_zero': ('P(Sharpe > 0)', '{:.1%}'),
    'n_trials': ('Trials searched', '{:.0f}'), 'deflated_sharpe': ('Deflated Sharpe P(SR > 0)', '{:.1%}'),
}


def format_stats(stats: dict, title: str = 'Performance') -> str:
    lines = [f'## {title}', '', '| Statistic | Value |', '| :--- | ---: |']
    for key, (label, fmt) in LABELS.items():
        if key in stats and stats[key] == stats[key]:
            lines.append(f'| {label} | {fmt.format(stats[key])} |')
    return '\n'.join(lines)


def text_report(result: BacktestResult, title: str = 'Backtest', bootstrap: bool = True, **summary_kwargs) -> str:
    stats = summarize(result.returns, result.turnover, **summary_kwargs)
    parts = [f'# {title}', '', format_stats(stats)]
    if bootstrap:
        low, high = bootstrap_sharpe_ci(result.returns)
        parts.append(f'\nBlock-bootstrap 95% interval for Sharpe: [{low:.2f}, {high:.2f}].')
    years = max(len(result.returns) / 252, 1e-9)
    parts.append(f'Transaction costs cost {result.annual_cost_drag:.2%} of portfolio value per year; '
                 f'financing net {result.financing.sum() / result.config["initial_capital"] / years:+.2%} per year.')
    return '\n'.join(parts) + '\n'


def plot(result: BacktestResult, path: str | Path, title: str = 'Backtest',
         benchmark: pd.Series | None = None) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 1]})
    equity = result.equity / result.config['initial_capital']
    axes[0].plot(equity.index, equity, label='strategy', color='#1f4e79')
    if benchmark is not None:
        bench = (1 + benchmark.reindex(equity.index).fillna(0)).cumprod()
        axes[0].plot(bench.index, bench, label='benchmark', color='#888888', alpha=0.8)
    axes[0].set_ylabel('Growth of 1')
    axes[0].legend(loc='upper left')
    axes[0].set_title(title)
    dd = drawdown(result.equity)
    axes[1].fill_between(dd.index, dd, 0, color='#b23a48', alpha=0.6)
    axes[1].set_ylabel('Drawdown')
    axes[2].plot(result.turnover.index, result.turnover.rolling(21).mean(), color='#4b7f52')
    axes[2].set_ylabel('Turnover (21d avg)')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
