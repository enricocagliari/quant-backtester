import json
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from qbt import data, synthetic


def test_parse_ohlcv_prefers_adjusted_close_and_sorts():
    text = textwrap.dedent('''\
        Date,Open,High,Low,Close,Adj Close,Volume
        2020-01-03,10,11,9,10.5,10.0,100
        2020-01-02,10,11,9,10.2,9.8,100
        2020-01-03,10,11,9,10.5,10.0,100
    ''')
    series = data.parse_ohlcv(text)
    assert list(series.index) == [pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-03')]
    assert series.iloc[-1] == 10.0
    assert data.parse_ohlcv(text, prefer_adjusted=False).iloc[-1] == 10.5
    with pytest.raises(ValueError):
        data.parse_ohlcv('Date,Open\n2020-01-01,1\n')


def test_align_forward_fills_short_gaps_and_rejects_long_ones():
    index = pd.bdate_range('2020-01-01', periods=20)
    frame = pd.DataFrame({'A': np.arange(20, dtype=float) + 1, 'B': np.arange(20, dtype=float) + 1}, index=index)
    frame.iloc[:3, 1] = np.nan  # B lists later: common range starts at day 3
    frame.iloc[8:10, 0] = np.nan  # two missing sessions, filled
    aligned = data.align(frame, max_gap=3)
    assert aligned.index[0] == index[3] and not aligned.isna().any().any()
    assert aligned.loc[index[9], 'A'] == aligned.loc[index[7], 'A']
    frame.iloc[8:14, 0] = np.nan
    with pytest.raises(ValueError):
        data.align(frame, max_gap=3)


def test_load_ohlcv_dir_and_wide_csv_round_trip(tmp_path):
    for name in ('aaa', 'bbb'):
        (tmp_path / f'{name}.csv').write_text('Date,Open,High,Low,Close,Volume\n2020-01-02,1,1,1,2.0,1\n2020-01-03,1,1,1,3.0,1\n')
    frame = data.load_ohlcv_dir(tmp_path)
    assert list(frame.columns) == ['AAA', 'BBB'] and frame.shape == (2, 2)
    wide = tmp_path / 'wide.csv'
    frame.to_csv(wide)
    pd.testing.assert_frame_equal(data.load_wide_csv(wide), frame, check_names=False)
    with pytest.raises(FileNotFoundError):
        data.load_ohlcv_dir(tmp_path / 'empty')


def test_synthetic_generators_are_reproducible_and_positive():
    for name, generator in synthetic.GENERATORS.items():
        a, b = generator(n_days=300, seed=4), generator(n_days=300, seed=4)
        pd.testing.assert_frame_equal(a, b)
        assert (a > 0).all().all() and len(a) == 300
        assert not a.equals(generator(n_days=300, seed=5))


def run_cli(*args):
    return subprocess.run([sys.executable, '-m', 'qbt', *map(str, args)], capture_output=True, text=True)


def test_cli_run_and_walk_forward(tmp_path):
    config = tmp_path / 'c.toml'
    config.write_text(textwrap.dedent('''\
        title = "cli test"
        [data]
        kind = "synthetic"
        generator = "random_walk"
        n_days = 700
        n_assets = 3
        seed = 1
        [strategy]
        name = "time_series_momentum"
        lookback = 40
        skip = 0
        [grid]
        lookback = [20, 40]
        [walk_forward]
        train = 300
        test = 200
    '''))
    out = run_cli('run', config, '--json', tmp_path / 's.json', '--plot', tmp_path / 'p.png')
    assert out.returncode == 0, out.stderr
    assert 'Sharpe' in out.stdout and (tmp_path / 'p.png').stat().st_size > 1000
    assert 'sharpe' in json.loads((tmp_path / 's.json').read_text())
    wf = run_cli('walk-forward', config, '--json', tmp_path / 'wf.json')
    assert wf.returncode == 0, wf.stderr
    assert 'Walk-forward out-of-sample Sharpe' in wf.stdout
    assert len(json.loads((tmp_path / 'wf.json').read_text())['folds']) == 2


def test_cli_synth_and_errors(tmp_path):
    out = run_cli('synth', tmp_path / 'x.csv', '--days', 50, '--assets', 2)
    assert out.returncode == 0 and data.load_wide_csv(tmp_path / 'x.csv').shape == (50, 2)
    bad = run_cli('run', tmp_path / 'missing.toml')
    assert bad.returncode == 2 and 'error:' in bad.stderr
