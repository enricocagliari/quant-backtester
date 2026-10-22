"""Every signal must depend only on the past; every overlay must keep that true."""
import numpy as np
import pandas as pd
import pytest

from qbt import portfolio, signals, synthetic

CASES = {
    'tsmom': lambda p: signals.time_series_momentum(p, lookback=60, skip=5),
    'tsmom_raw': lambda p: signals.time_series_momentum(p, lookback=60, skip=5, vol_scale=False),
    'xsmom': lambda p: signals.cross_sectional_momentum(p, lookback=60, skip=5, top=1),
    'meanrev': lambda p: signals.mean_reversion(p, window=10),
    'pairs': lambda p: signals.pairs_trading(p, 'A00', 'A01', window=30),
    'vol_target': lambda p: portfolio.target_volatility(signals.mean_reversion(p, 10), p, 0.1, window=20),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_truncating_the_future_does_not_change_the_past(prices, name):
    full = CASES[name](prices)
    for cut in (250, 400, 599):
        partial = CASES[name](prices.iloc[:cut])
        pd.testing.assert_frame_equal(full.iloc[:cut], partial, check_exact=False, rtol=1e-10)


@pytest.mark.parametrize('name', sorted(CASES))
def test_gross_exposure_is_bounded(prices, name):
    weights = CASES[name](prices)
    gross = weights.abs().sum(axis=1)
    assert gross.max() <= 3.0 + 1e-9
    assert np.isfinite(weights.fillna(0)).all().all()


def test_cross_sectional_momentum_is_dollar_neutral(prices):
    weights = signals.cross_sectional_momentum(prices, 60, 5, top=1).dropna()
    np.testing.assert_allclose(weights.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(weights.abs().sum(axis=1), 1.0, atol=1e-12)
    with pytest.raises(ValueError):
        signals.cross_sectional_momentum(prices, 60, 5, top=3)


def test_time_series_momentum_follows_the_trend():
    index = pd.bdate_range('2020-01-01', periods=300)
    up = pd.DataFrame({'U': np.linspace(100, 200, 300), 'D': np.linspace(200, 100, 300)}, index=index)
    weights = signals.time_series_momentum(up, lookback=50, skip=0, vol_scale=False).dropna()
    assert (weights['U'] > 0).all() and (weights['D'] < 0).all()


def test_pairs_trading_recovers_the_spread():
    prices = synthetic.cointegrated_pair(n_days=1500, spread_half_life=5, spread_vol=0.03, seed=2)
    beta, z = signals.rolling_ols_zscore(np.log(prices['Y']), np.log(prices['X']), 60)
    assert abs(beta.dropna().median() - 1.0) < 0.15
    weights = signals.pairs_trading(prices, 'Y', 'X', window=60, entry=1.5, exit=0.25)
    open_days = weights['Y'].ne(0)
    assert 0.1 < open_days.mean() < 0.9
    # When the spread is high (z > entry) the position must be short Y and long X.
    high = z > 1.5
    first_entries = high & ~high.shift(1, fill_value=False) & weights['Y'].shift(1).eq(0)
    assert (weights.loc[first_entries, 'Y'] < 0).all() and (weights.loc[first_entries, 'X'] > 0).all()


def test_mean_reversion_fades_deviations():
    index = pd.bdate_range('2020-01-01', periods=50)
    series = np.full(50, 100.0)
    series[-1] = 110.0
    prices = pd.DataFrame({'A': series, 'B': np.full(50, 100.0)}, index=index)
    weights = signals.mean_reversion(prices, window=20)
    assert weights['A'].iloc[-1] < 0 and weights['B'].iloc[-1] == 0


def test_target_volatility_scales_toward_target(prices):
    raw = signals.buy_and_hold(prices)
    scaled = portfolio.target_volatility(raw, prices, target_vol=0.05, window=63, max_leverage=3.0)
    realized = (scaled.shift(1) * prices.pct_change()).sum(axis=1).iloc[70:].std() * np.sqrt(252)
    assert 0.02 < realized < 0.10
    assert scaled.abs().sum(axis=1).max() <= 3.0 + 1e-9


def test_limit_gross_and_cap(prices):
    raw = signals.buy_and_hold(prices, {'A00': 2.0, 'A01': -2.0})
    assert portfolio.limit_gross(raw, 1.0).abs().sum(axis=1).max() == pytest.approx(1.0)
    assert portfolio.cap_weights(raw, 0.5).abs().max().max() == 0.5
