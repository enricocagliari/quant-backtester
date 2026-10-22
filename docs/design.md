# Design notes

## Timing

The engine works on a sequence of closes. On day t a strategy sees prices up to and including the close of day t and decides a target weight vector w_t. The engine executes w_t at the close of day t + L, where L is `execution_lag` and must be at least 1. There is no setting that trades on the deciding bar.

Worked example with L = 1, one asset, no costs, 1000 of capital:

| Day | Close | Decided weight | Executed weight | Shares after trade | Equity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 1.0 | 0 | 0 | 1000.00 |
| 1 | 110 | 1.0 | 1.0 | 9.0909 | 1000.00 |
| 2 | 99 | 1.0 | 1.0 | 9.0909 | 900.00 |

The strategy decided to be long on day 0 but could not benefit from day 1's rise; it bought at 110 and then took the fall to 99. Equity on day 2 is 1000 × 99 / 110. `tests/test_engine.py::test_buy_and_hold_without_costs_reproduces_asset_return` checks exactly this identity over a long random series.

Signals are built from pandas rolling and shift operations, which only look backwards, plus explicit loops where a state machine or rolling regression is needed. `tests/test_signals.py::test_truncating_the_future_does_not_change_the_past` enforces the contract for every registered signal and overlay by recomputing on a truncated history.

## Accounting

State: cash c and share vector s. Each day with close vector p:

1. Financing on yesterday's positions: c += c × r_cash / 252 + min(s ⊙ p_prev, 0) · 1 × r_borrow / 252. Short market value is negative, so the borrow term reduces cash.
2. Pre-trade value V = c + s · p.
3. Target shares s* = w ⊙ V / p; trade Δ = s* − s, except assets whose weight gap is within `rebalance_threshold`, which do not trade.
4. Transaction cost: Σ|Δ| p × (commission + half-spread) / 10⁴, plus, when volume is supplied, Σ|Δ| p × k / 10⁴ × √(100 × |Δ| p / ADV), the usual square-root impact law with k in basis points at 1% participation.
5. c −= Δ · p + cost; s += Δ; equity E = c + s · p.

Daily return is E_t / E_{t−1} − 1. Because the cost is subtracted before equity is recorded, paying it lowers that day's return. Reported weights are s ⊙ p / E, so after a trade they sum to slightly more than the target by the cost fraction; the test allows for this.

The simulation is a Python loop over days with NumPy vectors across assets. Ten years of daily data for a hundred assets runs in well under a second, which is fast enough that grid searches and walk-forwards stay interactive.

## Statistics

- Sharpe ratio: mean / standard deviation of daily returns times √252, no risk-free adjustment unless `rf` is passed.
- Standard error: Lo (2002), SE(SR_daily) = √((1 + SR_daily² / 2) / T), annualized by √252. This assumes iid returns; the block bootstrap below does not.
- Block bootstrap: circular blocks of 20 days resampled 2000 times; the interval is the 2.5th and 97.5th percentile of the resampled Sharpe ratios.
- Probabilistic Sharpe ratio (Bailey and Lopez de Prado, 2012): P(true SR > SR*) = Φ((SR − SR*) √(T − 1) / √(1 − γ₃ SR + (γ₄ − 1) SR² / 4)) with daily Sharpe, skew γ₃ and raw kurtosis γ₄ of the daily returns.
- Deflated Sharpe ratio (Bailey and Lopez de Prado, 2014): the PSR evaluated against SR* = √V [(1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N e))], the expected maximum Sharpe of N unskilled trials with Sharpe variance V, where γ is the Euler-Mascheroni constant. `grid_search` reports V as the variance of daily Sharpe across the grid; `walk_forward` averages it over folds. This treats grid points as independent trials, which overstates N when neighbouring parameters are highly correlated, so the deflation is conservative.
- The inverse normal CDF is Acklam's rational approximation refined with one Newton step, accurate to about 1e-9, to avoid a SciPy dependency.

## Walk-forward

Given a grid of parameter sets, a training length and a test length:

1. For each test window, run the grid search on the training window immediately before it (rolling) or on all history before it (anchored). The search backtests every parameter set with costs and picks the best Sharpe.
2. Compute the winning strategy's weights on history up to the end of the test window. Because signals only look backwards this is identical to computing them day by day.
3. Backtest the test window only, starting one execution lag before it so the first test day can trade, and keep the returns.
4. Concatenate the out-of-sample returns across folds; the first training window is never scored.

The grid search on the full history is reported alongside. The gap between the two is the selection effect. `tests/test_validation.py::test_in_sample_beats_out_of_sample_on_pure_noise` checks that the gap is positive on random walks across ten seeds.

Starting the test backtest flat and starting each fold with fresh capital are simplifications: a live strategy carries positions across the boundary, and parameter changes between folds would cause a rebalance. Both effects are small at these turnover levels and both bias the result against the strategy, not for it.

## What is deliberately absent

- Intraday data, order books, limit orders, partial fills. The engine assumes a daily close is a tradeable price for a size that only moves it by the modelled cost.
- Corporate actions. The loaders prefer adjusted closes when present; nothing else is done.
- Multiple currencies, futures margining, options.
- Machine-learned signals. The signal module is small on purpose; the engine and validation code are the point.
- Any claim about real-world profitability. The example results are on synthetic data whose structure is known, which is the right place to test whether a research pipeline lies.
