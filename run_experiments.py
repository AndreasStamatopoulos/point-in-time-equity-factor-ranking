"""Run classic-vs-ML equity ranking experiments on a point-in-time panel."""
import argparse
from pathlib import Path

from finance.experiments import ExperimentConfig, MODELS, run_comparison


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials', type=int, default=12,
                   help='Optuna TPE trials per outer decision date; 0 disables tuning for smoke tests')
    p.add_argument('--rebalance-months', type=int, default=6)
    p.add_argument('--min-train-months', type=int, default=36)
    p.add_argument('--validation-folds', type=int, default=3)
    p.add_argument('--max-iter', type=int, default=200)
    p.add_argument('--models', nargs='+', default=list(MODELS), choices=MODELS)
    p.add_argument('--top-n', type=int, default=5)
    p.add_argument('--cost-bps', type=float, default=5.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, default=None)
    a = p.parse_args()
    cfg = ExperimentConfig(
        min_train_months=a.min_train_months, validation_folds=a.validation_folds,
        rebalance_months=a.rebalance_months, trials=a.trials, max_iter=a.max_iter,
        models=tuple(a.models), top_n=a.top_n, cost_bps=a.cost_bps, seed=a.seed,
    )
    result = run_comparison(Path(__file__).resolve().parent, cfg, a.output)
    print(result.to_string(index=False))
    print('Warning: exploratory small-survivor sample; no evidence of live alpha.')


if __name__ == '__main__':
    main()