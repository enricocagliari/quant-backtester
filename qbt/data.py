"""Loading and aligning daily price data.

Two layouts are accepted: a single wide CSV (date column plus one column per
ticker) or a directory of per-ticker CSVs in the common ``Date,Open,High,
Low,Close,Volume`` layout, from which the close (or adjusted close when
present) is taken. ``download_stooq`` fetches the per-ticker layout from
Stooq's free daily endpoint; it needs network access and is not exercised
by the test suite.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

STOOQ_URL = 'https://stooq.com/q/d/l/?s={symbol}&i=d'


def load_wide_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index.name = 'date'
    return align(frame.astype(float))


def parse_ohlcv(text: str, prefer_adjusted: bool = True) -> pd.Series:
    frame = pd.read_csv(io.StringIO(text))
    frame.columns = [c.strip().lower().replace(' ', '_') for c in frame.columns]
    if 'date' not in frame.columns:
        raise ValueError('expected a Date column')
    column = 'adj_close' if prefer_adjusted and 'adj_close' in frame.columns else 'close'
    if column not in frame.columns:
        raise ValueError('expected a Close column')
    series = pd.Series(frame[column].astype(float).to_numpy(), index=pd.to_datetime(frame['date']))
    return series[~series.index.duplicated()].sort_index()


def load_ohlcv_dir(directory: str | Path, prefer_adjusted: bool = True) -> pd.DataFrame:
    directory = Path(directory)
    columns = {}
    for path in sorted(directory.glob('*.csv')):
        columns[path.stem.upper()] = parse_ohlcv(path.read_text(encoding='utf-8'), prefer_adjusted)
    if not columns:
        raise FileNotFoundError(f'no CSV files in {directory}')
    return align(pd.DataFrame(columns))


def align(prices: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    """Keep the common date range, forward-fill short gaps, and refuse long ones.

    Forward-filling a missing close is a stale price, which overstates
    tradeability; ``max_gap`` keeps that approximation honest by raising
    when any asset is missing more than that many consecutive sessions.
    """
    prices = prices.sort_index()
    prices = prices.loc[prices.notna().all(axis=1).idxmax():]
    gaps = prices.isna().astype(int)
    run = gaps.cumsum() - gaps.cumsum().where(gaps == 0).ffill().fillna(0)
    worst = int(run.max().max()) if len(run) else 0
    if worst > max_gap:
        raise ValueError(f'an asset is missing {worst} consecutive sessions; inspect the data')
    return prices.ffill()


def download_stooq(symbols: list[str], directory: str | Path) -> list[Path]:
    """Save one CSV per symbol (for example ``spy.us``) from Stooq's daily endpoint."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for symbol in symbols:
        with urllib.request.urlopen(STOOQ_URL.format(symbol=symbol), timeout=30) as response:
            text = response.read().decode('utf-8')
        if 'Date' not in text.splitlines()[0]:
            raise ValueError(f'{symbol}: unexpected response {text[:60]!r}')
        target = directory / f'{symbol.split(".")[0].upper()}.csv'
        target.write_text(text, encoding='utf-8')
        written.append(target)
    return written
