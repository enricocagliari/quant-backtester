"""Command line entry points. Configuration is TOML; see configs/ for examples."""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import pandas as pd

from qbt import data, portfolio, report, signals, synthetic
from qbt.engine import Costs, run_backtest
from qbt.stats import summarize
from qbt.validation import in_sample_vs_out_of_sample


def load_prices(config: dict) -> pd.DataFrame:
    source = config['data']
    kind = source['kind']
    if kind == 'synthetic':
        generator = synthetic.GENERATORS[source['generator']]
        return generator(**{k: v for k, v in source.items() if k not in ('kind', 'generator')})
    if kind == 'wide_csv':
        return data.load_wide_csv(source['path'])
    if kind == 'ohlcv_dir':
        return data.load_ohlcv_dir(source['path'])
    raise ValueError(f'unknown data kind {kind!r}')


def build_strategy(config: dict):
    spec = config['strategy']
    signal = signals.REGISTRY[spec['name']]
    params = {k: v for k, v in spec.items() if k != 'name'}
    overlay = config.get('portfolio', {})

    def factory(**overrides):
        merged = {**params, **overrides}

        def strategy(prices: pd.DataFrame) -> pd.DataFrame:
            weights = signal(prices, **merged)
            if 'max_abs_weight' in overlay:
                weights = portfolio.cap_weights(weights, overlay['max_abs_weight'])
            if 'gross' in overlay:
                weights = portfolio.limit_gross(weights, overlay['gross'])
            if 'target_vol' in overlay:
                weights = portfolio.target_volatility(weights, prices, overlay['target_vol'],
                                                      overlay.get('vol_window', 63),
                                                      overlay.get('max_leverage', 3.0),
                                                      execution_lag=config.get('execution', {}).get('lag', 1))
            return weights
        return strategy
    return factory


def backtest_kwargs(config: dict) -> dict:
    execution = config.get('execution', {})
    return {'costs': Costs(**config.get('costs', {})),
            'initial_capital': execution.get('initial_capital', 1_000_000.0),
            'execution_lag': execution.get('lag', 1),
            'rebalance_threshold': execution.get('rebalance_threshold', 0.0)}


def cmd_run(args) -> int:
    config = tomllib.loads(Path(args.config).read_text(encoding='utf-8'))
    prices = load_prices(config)
    strategy = build_strategy(config)()
    result = run_backtest(prices, strategy(prices), **backtest_kwargs(config))
    title = config.get('title', Path(args.config).stem)
    benchmark = run_backtest(prices, signals.buy_and_hold(prices), **backtest_kwargs(config)).returns
    print(report.text_report(result, title))
    bench = summarize(benchmark)
    print(f"Equal-weight buy-and-hold benchmark: Sharpe {bench['sharpe']:.2f}, "
          f"annual return {bench['ann_return']:.2%}, max drawdown {bench['max_drawdown']:.2%}.")
    if args.plot:
        print(f'Figure written to {report.plot(result, args.plot, title, benchmark)}')
    if args.json:
        Path(args.json).write_text(json.dumps(summarize(result.returns, result.turnover), indent=2))
    return 0


def cmd_walk_forward(args) -> int:
    config = tomllib.loads(Path(args.config).read_text(encoding='utf-8'))
    prices = load_prices(config)
    grid = config['grid']
    wf_config = config.get('walk_forward', {})
    kwargs = backtest_kwargs(config)
    costs = kwargs.pop('costs')
    comparison = in_sample_vs_out_of_sample(prices, build_strategy(config), grid, costs,
                                            train=wf_config.get('train', 756), test=wf_config.get('test', 252),
                                            anchored=wf_config.get('anchored', False), **kwargs)
    wf = comparison['walk_forward']
    title = config.get('title', Path(args.config).stem)
    print(f'# Walk-forward: {title}', '')
    print(f"Grid of {comparison['n_trials']} parameter sets. Best in-sample on full history: "
          f"{comparison['in_sample_best']} with Sharpe {comparison['in_sample_sharpe']:.2f}; "
          f"deflated for {comparison['n_trials']} trials, P(true Sharpe > 0) = "
          f"{comparison['in_sample_deflated_sharpe']:.1%}.")
    print(f"Walk-forward out-of-sample Sharpe: {comparison['out_of_sample_sharpe']:.2f} "
          f"over {len(wf.folds)} folds (not selected, so not deflated).", '')
    print(wf.summary().to_markdown(index=False, floatfmt='.2f') if _has_tabulate() else wf.summary().to_string())
    stats = summarize(wf.returns)
    print('', report.format_stats(stats, 'Out-of-sample statistics'))
    if args.json:
        payload = {**stats, 'in_sample_best': comparison['in_sample_best'],
                   'in_sample_sharpe': comparison['in_sample_sharpe'],
                   'in_sample_deflated_sharpe': comparison['in_sample_deflated_sharpe'],
                   'n_trials': comparison['n_trials'], 'folds': wf.summary().to_dict('records')}
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_synth(args) -> int:
    generator = synthetic.GENERATORS[args.generator]
    frame = generator(n_days=args.days, seed=args.seed) if args.generator == 'cointegrated_pair' else \
        generator(n_days=args.days, n_assets=args.assets, seed=args.seed)
    frame.index.name = 'date'
    frame.to_csv(args.out, float_format='%.4f')
    print(f'wrote {frame.shape[0]} days x {frame.shape[1]} assets to {args.out}')
    return 0


def cmd_download(args) -> int:
    written = data.download_stooq(args.symbols, args.out)
    print('\n'.join(str(p) for p in written))
    return 0


def _has_tabulate() -> bool:
    import importlib.util
    return importlib.util.find_spec('tabulate') is not None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='qbt', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='backtest one configuration')
    run.add_argument('config')
    run.add_argument('--plot', help='write a PNG figure here')
    run.add_argument('--json', help='write summary statistics here')
    run.set_defaults(func=cmd_run)
    wf = sub.add_parser('walk-forward', help='grid search inside rolling training windows')
    wf.add_argument('config')
    wf.add_argument('--json')
    wf.set_defaults(func=cmd_walk_forward)
    synth = sub.add_parser('synth', help='write a synthetic wide CSV')
    synth.add_argument('out')
    synth.add_argument('--generator', choices=sorted(synthetic.GENERATORS), default='random_walk')
    synth.add_argument('--days', type=int, default=2520)
    synth.add_argument('--assets', type=int, default=8)
    synth.add_argument('--seed', type=int, default=0)
    synth.set_defaults(func=cmd_synth)
    download = sub.add_parser('download', help='fetch daily CSVs from Stooq (needs network)')
    download.add_argument('out')
    download.add_argument('symbols', nargs='+', help='for example spy.us tlt.us gld.us')
    download.set_defaults(func=cmd_download)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
