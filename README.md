# ML Finance — Point-in-Time Stock Ranking (OLS Baseline)

[![Python tests](https://img.shields.io/badge/tests-pytest-blue)](#testing) [![Python](https://img.shields.io/badge/python-3.11%2B-blue)](#quickstart) [![Status](https://img.shields.io/badge/6th_signal-data_pending-orange)](#research-status)

**A research-oriented, no-neural-network stock-ranking application** for forecasting **relative US equity returns over six months** (extendable to 12 months). Built around the **six signals in the original academic assignment**, not the unrelated Fama–French time-series factor returns.

## Original research question
> Can classic financial characteristics rank stocks by their expected *relative* future returns in a strictly chronological, out-of-sample evaluation?

| # | Predictive stock characteristic | Implementation status |
|---|---|---|
| 1 | Momentum 12–1 | Implemented |
| 2 | Book-to-Market | Implemented with SEC/fundamental-data availability controls |
| 3 | Operating Profitability | Implemented as a documented operating income / equity **proxy** |
| 4 | Asset Growth | Implemented from comparable annual balance sheets |
| 5 | Earnings Revisions (analyst EPS forecasts) | **Pipeline implemented, historical vintage data required** |
| 6 | Free Cash Flow Yield | Implemented from cash-flow proxy / market equity |

**Do not confuse earnings revisions with earnings surprise or momentum.** A proper sixth-signal experiment requires dated EPS analyst-consensus snapshots for the *same forecast fiscal period*. Without verified revisions, the software explicitly reports a **five-feature fallback** and does not claim six-factor results.

## Technical highlights
- SEC/XBRL accounting-data and stock-price collectors (requires network access, appropriate request identification, and vendor terms).
- Point-in-time accounting features with source-date controls and historical earnings-revision importer.
- Monthly cross-sectional winsorization/normalization, top-5 ranking and growing-window OLS/Ridge.
- **Training uses only six-month labels fully realized before the prediction date.** Positions enter at the following month-end (monthly approximation).
- Non-overlapping six-month portfolio evaluation, 5-bps-per-side illustrative transaction costs, MAE/RMSE/R², Spearman Rank IC and portfolio reports.
- Python CLI, optional Streamlit dashboard, source-level audit and pytest/CI.

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

The **public repository deliberately excludes market-data archives and historical prices** (vendor redistribution restrictions; also avoids publishing any credentials). Tests run without those files.

If you have the **private replay dataset provided separately**, extract its `data/` directory into this repository's `data/` directory. Then:

```bash
python run.py --model ols --top-n 5 --cost-bps 5
```

Launch optional dashboard:
```bash
python -m pip install -r requirements-dashboard.txt
streamlit run app.py
```

Generate a fresh dataset (online, API access required) using `01_collect_sec.py`, `03_build_features.py`, `05_audit.py`; see [`docs/DATA.md`](docs/DATA.md) and [`README_GR.md`](README_GR.md) for details. To add the sixth input use `07_prepare_estimize.py`, `06_import_revisions.py` and dated vintage approval. **Do not mark an EPS source verified without checking historical timestamps and fiscal-period identity.**

## Research status

A **previously created real-data, 24-stock dataset** can be replayed locally in five-feature mode. Replayed on 2026-11-16 using the included linear package:

| Metric | Historical 5-feature replay |
|---|---:|
| Initial stocks | 24, non-financial, current-survivor sample |
| Fully eligible stocks per date | Often 6–11 (not all 24) |
| OOS stock predictions | 1,045 across 128 completed monthly dates |
| Mean Rank IC | +0.0771 |
| OOS R² | −0.0544 |
| Top-5 illustrative CAGR | 18.62% |
| Equal-weight eligible universe CAGR | 17.63% |
| Non-overlapping 6-month periods | 22 |
| Confirmed six-feature backtest | **Not yet available** |

**Do not interpret the above as validated alpha or trading profits.** Known limitations: current-survivor universe, missing delisted stocks, small cross-sectional coverage, SEC accounting proxies, reconstructed vendor market caps, monthly execution approximations, overlapping forecast-label dependence and no authenticated earnings-revision vintages. The new-source audit for the legacy panel is **blocked** until complete original source records are present. See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Repo layout
```text
finance/                 reusable data, factor, model, audit modules
01_collect_sec.py        SEC/XBRL data ingestion
03_build_features.py     point-in-time feature engineering
06_import_revisions.py   earnings-revision import (vintage-gated)
07_prepare_estimize.py   Estimize vendor import adapter
run.py                   reproducible linear-model CLI
app.py                   optional local Streamlit dashboard
tests/                   unit and time-order safety tests
docs/                    methods, data contract, CV summary
results/                 generated locally; Git-ignored
data/                    templates only; proprietary data Git-ignored
```

## Testing
```bash
python -m pytest -q
```
GitHub Actions runs tests on every push / pull request.

## Attribution and scope
This repository organizes and documents an existing academic codebase; published results are **conditional historical research, not an investable audited track record**. Code is under MIT. Market-data sources and third-party datasets retain their own terms. A neural extension is **out of scope** for this version, following the professor's original assignment.
