# Classic vs ML vs Hybrid: pre-registered comparison protocol

## Research aim

Does a standalone MLP or an MLP trained on **historical out-of-sample OLS residuals** improve future 6-month **cross-sectional relative stock return** forecasts, compared to ordinary linear baselines?

The *original* six signals are Momentum 12–1, Book-to-Market, Operating Profitability (a documented proxy), Asset Growth, Earnings Revisions, and FCF Yield. **Historical EPS consensus vintages have not yet been verified**: the supplied legacy 24-stock panel runs only five signals. No model is called a six-signal model until the real EPS-vintage source passes validation.

## Models

| Name | Form | Tuning |
|---|---|---|
| `ols` | LinearRegression | none |
| `ridge` | L2-penalized regression | alpha grid |
| `lasso` | L1-penalized regression | alpha grid |
| `elastic_net` | joint L1/L2 | alpha and L1 ratio grid |
| `mlp` | MLPRegressor, standard scaler, squared-error training loss | Optuna TPE over layers, decay, LR, batch size |
| `ols_residual_mlp` | OLS forecast + MLP prediction of OOF OLS residuals | same Optuna MLP search, trained only on temporally OOF residuals |

For each decision date, targets are 6-month stock returns minus the mean six-month return of the eligible stock cohort at that decision. MSE is used for validation optimization. Mean Spearman Rank IC, OOS R2, and nonoverlapping gross/net Top-5 portfolio results are **separate evaluation metrics**. All model comparisons use the same test decision dates and stocks.

## Time-based validation and no leakage

1. Only features known on or before decision time can enter `prepare_features`.
2. Before fitting on a decision date `t`, every training row's **full 6-month target must end strictly before `t`** (purge overlapping/immature labels).
3. Each tuning fold trains only on earlier, matured monthly cohorts. Validation is performed on later historical decision dates, spaced approximately 6 months apart. Hyperparameters are selected by **validation MSE only**.
4. All numerical standardizers are fitted **only inside** each training fold; `MLPRegressor(early_stopping=False)` prevents scikit-learn's randomized validation split from inadvertently mixing chronology.
5. For the hybrid model, historical OLS forecasts are generated in **earlier out-of-sample windows** with the same maturity rule. Their residuals are the MLP's training labels. An OLS model fitted on the full (matured) outer training set supplies the baseline prediction; no outer-test residual or label ever enters either fit.
6. No missing EPS revisions are silently imputed or substituted with current analyst values.
7. Portfolio returns include illustrative transaction-cost assumptions; holding periods are nonoverlapping for the portfolio report. Real trading frictions and source-vintage audits remain limitations.

### Important limits

`MSE`, even with correct temporal splits, does **not** optimize net trading profit or rank IC directly. The retrospective 24-stock universe is survivorship biased; some dates have under 10 eligible names, making Top-5 diversification and rank statistics unstable. A credible trading conclusion requires a much larger point-in-time universe, sector controls, delistings, transaction slippage, benchmark returns, cost sensitivity, and repeated testing against a genuinely untouched final holdout. Treat any performance shown as **exploratory**.

## Run locally

From the repo root, **with private data present only locally in `data/`**:

```bash
pip install -r requirements.txt
pytest -q
python run_experiments.py --trials 12 --rebalance-months 6 --min-train-months 36
```

Faster correctness smoke test (does *not* tune; do not use for headline returns):

```bash
python run_experiments.py --trials 0 --max-iter 65 --rebalance-months 12
```

Artifacts created under `results/comparison/` (ignored by Git):
- `comparison.csv`: six directly comparable rows, with MSE/RMSE/MAE/OOS R2/Rank IC and net CAGR.
- `predictions.csv`: per-stock predicted scores and ranks.
- `training_audit.json`: data maturity, dates, tuning hyperparameters and validation scores.
- `portfolio_periods.csv`: non-overlapping six-month test holdings and returns.
- `status.json`: enabled feature set, experimental caveats and configuration.

## Online references / research basis

- Optuna TPE: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
- Optuna tuning/pruning: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
- scikit-learn MLP: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
- TimeSeriesSplit considerations: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Finite-horizon financial targets: avoid training on labels that would not yet have matured at the historical decision time. The implementation applies direct date-based purging instead of relying solely on positional `TimeSeriesSplit(gap=...)`.