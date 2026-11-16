from pathlib import Path
from typing import Literal

from finance.model import ModelSettings, run_model

# Orise tis rythmiseis prin deis ta test apotelesmata.
MODEL: Literal["ols", "ridge"] = "ols"
RIDGE_ALPHA = 10.0
MIN_TRAIN_MONTHS = 24
MIN_SECTOR_STOCKS = 1
TOP_FRACTION = 0.25  # compatibility only
TOP_N = 5  # Globally top 5 stocks with complete data
COST_BPS = 5.0
SHOW_PLOT = False

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    run_model(
        ROOT,
        ModelSettings(
            model=MODEL,
            ridge_alpha=RIDGE_ALPHA,
            min_train_months=MIN_TRAIN_MONTHS,
            min_sector_stocks=MIN_SECTOR_STOCKS,
            top_fraction=TOP_FRACTION,
            top_n=TOP_N,
            cost_bps=COST_BPS,
            show_plot=SHOW_PLOT,
        ),
    )
