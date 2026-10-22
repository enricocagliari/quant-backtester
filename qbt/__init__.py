"""qbt: a small, auditable backtesting engine for daily securities strategies."""
from qbt.engine import BacktestResult, Costs, run_backtest
from qbt.stats import summarize

__all__ = ['BacktestResult', 'Costs', 'run_backtest', 'summarize']
__version__ = '0.1.0'
