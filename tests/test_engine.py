"""Accounting identities and timing guarantees of the simulator."""
import numpy as np
import pandas as pd
import pytest

from qbt.engine import Costs, run_backtest
from qbt.signals import buy_and_hold, time_series_momentum

ZERO = Costs(commission_bps=0, half_spread_bps=0)


def test_equity_is_cash_plus_holdings(prices):
    result = run_backtest(prices, buy_and_hold(prices), Costs(2.0, 3.0))
    holdings = (result.shares * prices).sum(axis=1)
    np.testing.assert_allclose(result.equity, result.cash + holdings, rtol=1e-12)
    # Weights are measured against post-cost equity, so they exceed one by the cost just paid.
    np.testing.assert_allclose(result.weights.sum(axis=1).iloc[2:], 1.0, atol=1e-3)


def test_buy_and_hold_without_costs_reproduces_asset_return(prices):
    single = prices[['A00']]
    result = run_backtest(single, buy_and_hold(single), ZERO, initial_capital=1000.0)
    # Decided on day 0, bought at the close of day 1; from then on equity tracks the price exactly.
    expected = 1000.0 * single['A00'] / single['A00'].iloc[1]
    np.testing.assert_allclose(result.equity.iloc[1:], expected.iloc[1:], rtol=1e-12)
    assert result.equity.iloc[0] == 1000.0


def test_costs_equal_rate_times_traded_notional(flat_prices):
    weights = buy_and_hold(flat_prices, {'A': 0.6, 'B': 0.4})
    result = run_backtest(flat_prices, weights, Costs(commission_bps=1.0, half_spread_bps=4.0), initial_capital=1e6)
    # One trade on day 1 for the whole million at 5 bps. Paying it lowers equity, so the next day
    # re-targets 0.05% of the book; that geometric tail is below one currency unit in total.
    assert result.costs.iloc[1] == pytest.approx(1e6 * 5e-4)
    assert result.costs.iloc[2:].sum() < 1.0
    assert result.equity.iloc[-1] == pytest.approx(1e6 - 500.0, abs=1.0)
    assert result.turnover.iloc[1] == pytest.approx(1.0)


def test_future_prices_cannot_change_past_trades(prices):
    cut = 300
    shocked = prices.copy()
    shocked.iloc[cut:] *= np.random.default_rng(0).uniform(0.9, 1.1, size=shocked.iloc[cut:].shape)
    # A real signal on each price history, then the engine on each: nothing before the cut may differ.
    full = run_backtest(prices, time_series_momentum(prices, 60, 5), Costs())
    partial = run_backtest(shocked, time_series_momentum(shocked, 60, 5), Costs())
    pd.testing.assert_frame_equal(full.shares.iloc[:cut], partial.shares.iloc[:cut])
    pd.testing.assert_series_equal(full.equity.iloc[:cut], partial.equity.iloc[:cut])


def test_execution_lag_shifts_targets(flat_prices):
    weights = pd.DataFrame(0.0, index=flat_prices.index, columns=flat_prices.columns)
    weights.iloc[3, 0] = 1.0  # decided at close of day 3
    for lag in (1, 2):
        result = run_backtest(flat_prices, weights, ZERO, execution_lag=lag)
        first_trade = result.turnover.ne(0).idxmax()
        assert first_trade == flat_prices.index[3 + lag]
    with pytest.raises(ValueError):
        run_backtest(flat_prices, weights, ZERO, execution_lag=0)


def test_short_positions_pay_borrow_and_cash_earns_interest(flat_prices):
    weights = buy_and_hold(flat_prices, {'A': -1.0})
    costs = Costs(0, 0, borrow_rate_annual=0.0252, cash_rate_annual=0.0)
    result = run_backtest(flat_prices, weights, costs, initial_capital=1e6)
    # Short 1m of A from day 1; each following day pays 1m * 2.52% / 252 = 100.
    assert result.financing.iloc[2] == pytest.approx(-100.0)
    assert result.equity.iloc[-1] == pytest.approx(1e6 - 100.0 * 8)
    interest = run_backtest(flat_prices, buy_and_hold(flat_prices, {'A': 0.0}),
                            Costs(0, 0, cash_rate_annual=0.0252), initial_capital=1e6)
    assert interest.financing.iloc[1] == pytest.approx(100.0)


def test_rebalance_threshold_skips_small_drift(prices):
    weights = buy_and_hold(prices)
    tight = run_backtest(prices, weights, Costs(), rebalance_threshold=0.0)
    loose = run_backtest(prices, weights, Costs(), rebalance_threshold=0.05)
    assert loose.turnover.sum() < tight.turnover.sum()
    assert (loose.turnover.iloc[2:] == 0).mean() > 0.5


def test_market_impact_adds_to_linear_costs(flat_prices):
    weights = buy_and_hold(flat_prices, {'A': 1.0})
    volume = pd.DataFrame(10_000.0, index=flat_prices.index, columns=flat_prices.columns)  # 1m ADV in A
    linear = run_backtest(flat_prices, weights, Costs(1.0, 0.0), volume=volume)
    with_impact = run_backtest(flat_prices, weights, Costs(1.0, 0.0, impact_bps_per_sqrt_pct_adv=1.0), volume=volume)
    # Participation is 100% of ADV: impact = 1 bps * sqrt(100) = 10 bps on 1m notional.
    assert with_impact.costs.iloc[1] - linear.costs.iloc[1] == pytest.approx(1e6 * 10e-4)


def test_rejects_bad_inputs(prices):
    weights = buy_and_hold(prices)
    with pytest.raises(ValueError, match='NaN'):
        holed = prices.copy()
        holed.iloc[10, 0] = np.nan
        run_backtest(holed, weights)
    with pytest.raises(ValueError, match='increasing'):
        run_backtest(prices.iloc[::-1], weights)
    with pytest.raises(RuntimeError, match='bankrupt'):
        crash = prices.copy()
        crash.iloc[5:] = 1e-6
        run_backtest(crash, buy_and_hold(prices, {'A00': 3.0}), ZERO)


def test_nan_and_missing_targets_mean_flat(prices):
    weights = buy_and_hold(prices).iloc[100:200]
    result = run_backtest(prices, weights, ZERO)
    assert (result.shares.iloc[:101] == 0).all().all()
    assert (result.shares.iloc[102:201] != 0).all().all()
    assert (result.shares.iloc[202:] == 0).all().all()
