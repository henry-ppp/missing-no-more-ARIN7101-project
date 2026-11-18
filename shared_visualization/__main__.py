"""Make every figure the given reports support, for each mode and seed in them.

    python -m shared_visualization data/output results --out figures
    python -m shared_visualization data/output --out figures --mode quick --rename dae_v2_initial=dae_v2

Figures: a scoreboard per missing rate, accuracy vs distribution, missing-rate curves (needs two or more rates) and
the overall ranking with its table. A figure the reports can't support is skipped with the reason. The distribution
picture and training curves need choices (a column, a model), so they are made from Python; see README.md.
"""
import argparse
import sys

import matplotlib


def main(argv=None):
    matplotlib.use('Agg')
    from . import (HEADLINE, accuracy_vs_distribution, critical_difference, load_results, missing_rate_curves,
                   rank_table, scoreboard, write_tables)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('reports', nargs='+', help='Report files or folders (searched for evaluation*.json)')
    parser.add_argument('--out', required=True, help='Folder for the figures and tables')
    parser.add_argument('--mode', help='Only this mode (default: every mode in the reports)')
    parser.add_argument('--seed', type=int, help='Only this seed (default: every seed in the reports)')
    parser.add_argument('--rename', nargs='*', default=[], metavar='OLD=NEW', help='Rename models in the reports')
    parser.add_argument('--formats', nargs='+', default=['pdf', 'svg', 'png'])
    args = parser.parse_args(argv)
    try:
        rename = dict(item.split('=', 1) for item in args.rename)
    except ValueError:
        parser.error('--rename takes OLD=NEW pairs')

    results = load_results(*args.reports, rename=rename)
    results = results.select(mode=args.mode, seed=args.seed)
    if results.scores.empty:
        parser.error('No scores for that mode and seed')
    saved = list(write_tables(results, args.out))
    headline = [m for m in HEADLINE if (results.scores['metric'] == m).any()]

    def attempt(name, make):
        try:
            plot = make()
        except ValueError as exc:
            print(f'skipped {name}: {exc}')
            return
        saved.extend(plot.save(f'{args.out}/{name}', formats=args.formats))
        plot.close()

    for mode, seed in results.settings('mode', 'seed'):
        part = results.select(mode=mode, seed=seed)
        tag = f'{mode}_seed_{seed}'
        for rate in part.settings('rate'):
            attempt(f'scoreboard_{tag}_rate_{rate:g}', lambda: scoreboard(part, headline, rate=rate))
        attempt(f'accuracy_vs_distribution_{tag}', lambda: accuracy_vs_distribution(part))
        attempt(f'missing_rate_{tag}', lambda: missing_rate_curves(part))
        attempt(f'ranking_{tag}', lambda: critical_difference(part))
        try:
            table = rank_table(part)
            target = f'{args.out}/rank_table_{tag}.csv'
            table.to_csv(target, index=False)
            saved.append(target)
        except ValueError as exc:
            print(f'skipped rank_table_{tag}: {exc}')
    for path in saved:
        print(f'Saved {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
