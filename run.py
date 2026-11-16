"""CLI entrypoint for no-neural six-signal stock-ranking research.

Run from the repository root after placing verified data/features.csv and
 data/features_metadata.json in the private data/ folder.
"""
import argparse
from pathlib import Path
from finance.model import ModelSettings, run_model


def main():
    parser = argparse.ArgumentParser(description='Walk-forward OLS/Ridge US stock rankings')
    parser.add_argument('--model', choices=['ols', 'ridge'], default='ols')
    parser.add_argument('--top-n', type=int, default=5)
    parser.add_argument('--cost-bps', type=float, default=5.0)
    parser.add_argument('--min-train-months', type=int, default=24)
    parser.add_argument('--ridge-alpha', type=float, default=10.0)
    args = parser.parse_args()
    settings = ModelSettings(model=args.model,top_n=args.top_n,cost_bps=args.cost_bps,
                             min_train_months=args.min_train_months,ridge_alpha=args.ridge_alpha,
                             show_plot=False)
    run_model(Path(__file__).resolve().parent, settings)

if __name__ == '__main__':
    main()
