"""Leakage-aware comparisons: classic regression, MLP, and OLS + residual MLP.

The unit of validation is a *decision date*, not an individual stock row.
Every fitted training row must have a fully realized 6-month forward label
strictly before the decision date. Only data with verified vintages may enter.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from finance.model import (
    ModelSettings, TARGET, build_periods, evaluated_predictions, load_panel,
    portfolio_metrics, prediction_metrics, prepare_features, read_metadata,
)

MODELS = ("ols", "ridge", "lasso", "elastic_net", "mlp", "ols_residual_mlp")


@dataclass(frozen=True)
class ExperimentConfig:
    min_train_months: int = 36
    validation_folds: int = 3
    rebalance_months: int = 6
    trials: int = 12
    seed: int = 42
    max_iter: int = 200
    min_oof_dates: int = 3
    min_oof_rows: int = 20
    top_n: int = 5
    cost_bps: float = 5.0
    models: tuple[str, ...] = MODELS

    def __post_init__(self):
        if self.min_train_months < 18 or self.validation_folds < 1:
            raise ValueError("Require at least 18 training dates and one validation fold")
        if self.rebalance_months < 1 or self.trials < 0 or self.max_iter < 30:
            raise ValueError("Invalid rebalance, trials or max_iter")
        if self.min_oof_dates < 1 or self.min_oof_rows < 1:
            raise ValueError("Invalid OOF minimum")
        if not set(self.models).issubset(MODELS) or len(set(self.models)) != len(self.models):
            raise ValueError("Unknown or duplicate model")


def mature_rows(frame: pd.DataFrame, decision: pd.Timestamp) -> pd.DataFrame:
    """Drop any decision-date cohort unless ALL its labels matured strictly earlier.

    The group check prevents a changing universe of labels at the boundary.
    """
    mature = (
        frame["date"].lt(decision)
        & frame["label_end_6m"].lt(decision)
        & frame[TARGET].notna()
    )
    complete_cohort = mature.groupby(frame["date"]).transform("all")
    return frame.loc[mature & complete_cohort].copy()


def relative_target(frame: pd.DataFrame) -> pd.Series:
    return frame[TARGET] - frame.groupby("date")[TARGET].transform("mean")


def inner_folds(frame: pd.DataFrame, n_folds: int, min_history: int = 12):
    """Use nonoverlapping validation dates and purge immature six-month labels."""
    dates = sorted(pd.Timestamp(x) for x in frame["date"].unique())
    candidates = []
    for date in dates:
        history = mature_rows(frame, date)
        if history["date"].nunique() >= min_history and len(history) >= 20:
            candidates.append(date)
    chosen = list(reversed(candidates[-1::-6][:n_folds]))
    result = []
    for val_date in chosen:
        prior = mature_rows(frame, val_date)
        validation = frame.loc[frame["date"].eq(val_date)]
        if len(validation) >= 2 and not prior.empty:
            result.append((prior.index.to_numpy(), validation.index.to_numpy(), val_date))
    return result


def estimator(name: str, params: dict[str, Any], seed: int, max_iter: int):
    if name == "ols":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "ridge":
        regressor = Ridge(alpha=float(params.get("alpha", 10.0)))
    elif name == "lasso":
        regressor = Lasso(alpha=float(params.get("alpha", 0.005)), max_iter=5000)
    elif name == "elastic_net":
        regressor = ElasticNet(
            alpha=float(params.get("alpha", 0.005)),
            l1_ratio=float(params.get("l1_ratio", 0.5)), max_iter=5000,
        )
    elif name in {"mlp", "ols_residual_mlp"}:
        regressor = MLPRegressor(
            hidden_layer_sizes=tuple(params.get("layers", (16,))),
            activation="relu", solver="adam",
            alpha=float(params.get("weight_decay", 0.001)),
            learning_rate_init=float(params.get("learning_rate", 0.001)),
            batch_size=int(params.get("batch_size", 32)),
            max_iter=max_iter, random_state=seed,
            early_stopping=False,  # sklearn random holdout leaks future dates
            n_iter_no_change=20, tol=1e-5, shuffle=True,
        )
    else:
        raise ValueError(name)
    return make_pipeline(StandardScaler(), regressor)


def fit_predict(
    name: str, params: dict[str, Any], x_train: pd.DataFrame, y_train: pd.Series,
    x_test: pd.DataFrame, seed: int, max_iter: int,
) -> np.ndarray:
    model = estimator(name, params, seed, max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train.to_numpy(dtype=float))
    return np.asarray(model.predict(x_test), dtype=float)


def linear_params(name: str, frame: pd.DataFrame, x: pd.DataFrame, cfg: ExperimentConfig):
    if name == "ols":
        return {}, None
    folds = inner_folds(frame, cfg.validation_folds)
    if not folds:
        return {"alpha": 0.01, **({"l1_ratio": 0.5} if name == "elastic_net" else {})}, None
    alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0] if name == "ridge" else [0.0001, 0.001, 0.005, 0.01, 0.03, 0.1]
    candidates = ([{"alpha": a, "l1_ratio": r} for a in alphas for r in (0.2, 0.5, 0.8)]
                  if name == "elastic_net" else [{"alpha": a} for a in alphas])
    best, best_score = candidates[0], float("inf")
    y = relative_target(frame)
    for params in candidates:
        scores = []
        for tr, val, _ in folds:
            pred = fit_predict(name, params, x.loc[tr], y.loc[tr], x.loc[val], cfg.seed, cfg.max_iter)
            scores.append(float(np.mean((y.loc[val].to_numpy() - pred) ** 2)))
        score = float(np.mean(scores))
        if score < best_score:
            best, best_score = params, score
    return best, best_score


def mlp_params(frame: pd.DataFrame, x: pd.DataFrame, y: pd.Series, cfg: ExperimentConfig):
    """Optuna TPE; objective is mean purged expanding-window validation MSE."""
    folds = inner_folds(frame, cfg.validation_folds)
    defaults = {"layers": (16,), "weight_decay": 0.001, "learning_rate": 0.001, "batch_size": 32}
    if cfg.trials == 0 or not folds:
        return defaults, None
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        layers_name = trial.suggest_categorical("layers", ["8", "16", "32", "16x8", "32x16"])
        params = {
            "layers": tuple(int(z) for z in layers_name.split("x")),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        }
        errors = []
        for j, (tr, val, _) in enumerate(folds):
            pred = fit_predict("mlp", params, x.loc[tr], y.loc[tr], x.loc[val], cfg.seed, cfg.max_iter)
            errors.append(float(np.mean((y.loc[val].to_numpy() - pred) ** 2)))
            trial.report(float(np.mean(errors)), j)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(errors))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=min(5, cfg.trials), n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=cfg.trials, show_progress_bar=False, n_jobs=1)
    if not study.best_trials:
        return defaults, None
    best = study.best_trial
    params = dict(best.params)
    params["layers"] = tuple(int(z) for z in params["layers"].split("x"))
    return params, float(best.value)


def oof_residuals(frame: pd.DataFrame, x: pd.DataFrame, cfg: ExperimentConfig):
    """Create residual labels ONLY from OLS predictions made with older matured labels.

    At every OOF decision time v, baseline is fitted only with outcomes already
    known before v. The resulting errors themselves become available only after
    their full holding periods mature.
    """
    y = relative_target(frame)
    dates = sorted(pd.Timestamp(v) for v in frame["date"].unique())
    oof = []
    for date in dates[::3]:
        old = mature_rows(frame, date)
        if old["date"].nunique() < 12 or len(old) < 20:
            continue
        val = frame.loc[frame["date"].eq(date)]
        pred = fit_predict("ols", {}, x.loc[old.index], y.loc[old.index],
                           x.loc[val.index], cfg.seed, cfg.max_iter)
        part = val.copy()
        part["oof_residual"] = y.loc[val.index].to_numpy() - pred
        oof.append(part)
    if not oof:
        return pd.DataFrame(columns=[*frame.columns, "oof_residual"])
    return pd.concat(oof).sort_values(["date", "ticker"])


def forecast_for_date(
    panel: pd.DataFrame, normalized: pd.DataFrame, decision: pd.Timestamp,
    cfg: ExperimentConfig,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]] | None:
    history = mature_rows(panel, decision)
    today = panel.loc[panel["date"].eq(decision)]
    if history["date"].nunique() < cfg.min_train_months or len(today) < 5:
        return None
    x_train = normalized.loc[history.index]
    x_test = normalized.loc[today.index]
    y = relative_target(history)
    predictions: dict[str, np.ndarray] = {}
    audit: dict[str, dict[str, Any]] = {}

    for name in cfg.models:
        if name == "ols_residual_mlp":
            continue
        if name in {"lasso", "ridge", "elastic_net"}:
            params, cv_mse = linear_params(name, history, x_train, cfg)
        elif name == "mlp":
            params, cv_mse = mlp_params(history, x_train, y, cfg)
        else:
            params, cv_mse = {}, None
        predictions[name] = fit_predict(name, params, x_train, y, x_test, cfg.seed, cfg.max_iter)
        audit[name] = {"params": params, "inner_cv_mse": cv_mse}

    if "ols_residual_mlp" in cfg.models:
        residuals = oof_residuals(history, x_train, cfg)
        if residuals["date"].nunique() < cfg.min_oof_dates or len(residuals) < cfg.min_oof_rows:
            return None  # enforce same comparison dates, do not disguise fallback as hybrid
        # Validation happens exclusively on historical OOF residuals with full label maturation.
        rx = normalized.loc[residuals.index]
        ry = residuals["oof_residual"]
        params, cv_mse = mlp_params(residuals, rx, ry, cfg)
        residual_pred = fit_predict("mlp", params, rx, ry, x_test, cfg.seed, cfg.max_iter)
        base = predictions.get("ols")
        if base is None:
            base = fit_predict("ols", {}, x_train, y, x_test, cfg.seed, cfg.max_iter)
        predictions["ols_residual_mlp"] = base + residual_pred
        audit["ols_residual_mlp"] = {
            "params": params, "inner_cv_residual_mse": cv_mse,
            "n_oof_rows": len(residuals), "n_oof_dates": int(residuals["date"].nunique()),
            "min_oof_label_end": str(residuals["label_end_6m"].min().date()),
            "max_oof_label_end": str(residuals["label_end_6m"].max().date()),
        }

    common = {
        "decision_date": str(decision.date()),
        "max_train_label_end": str(history["label_end_6m"].max().date()),
        "n_train_dates": int(history["date"].nunique()),
        "n_train_rows": len(history),
        "features": list(normalized.columns),
    }
    for name in audit:
        audit[name] = {**common, **audit[name]}
    return predictions, audit


def run_comparison(root: Path, cfg: ExperimentConfig = ExperimentConfig(), output: Path | None = None):
    """Run all models on identical complete decision-date cross sections."""
    output = output or root / "results" / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata(root / "data" / "features_metadata.json")
    panel = load_panel(root / "data" / "features.csv", metadata.features, metadata.data_asof)
    eligible, normalized, exclusions = prepare_features(panel, metadata.features)
    if eligible.empty:
        raise ValueError("No complete stock cross-sections")
    exclusions.to_csv(output / "excluded_rows.csv", index=False)

    date_values = sorted(pd.Timestamp(x) for x in eligible["date"].unique())
    results = []
    audits = []
    first = None
    last_calendar_month = None
    for date in date_values:
        if first is not None:
            month_n = date.year * 12 + date.month
            if month_n - last_calendar_month < cfg.rebalance_months:
                continue
        computed = forecast_for_date(eligible, normalized, date, cfg)
        if computed is None:
            continue
        model_scores, model_audits = computed
        first = first or date
        last_calendar_month = date.year * 12 + date.month
        today = eligible.loc[eligible["date"].eq(date)].copy()
        for name, scores in model_scores.items():
            current = today.copy()
            current["model"] = name
            current["predicted_excess"] = scores
            current["cross_section_rank"] = current["predicted_excess"].rank(
                ascending=False, method="first"
            ).astype(int)
            results.append(current)
        audits.extend({"model": model_name, **values} for model_name, values in model_audits.items())

    if not results:
        raise ValueError("No dates satisfy minimum history/OOF conditions; reduce min_train_months or use more history")
    predictions = pd.concat(results, ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    (output / "training_audit.json").write_text(
        json.dumps(audits, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )
    comparison = []
    all_periods = []
    for name, group in predictions.groupby("model", sort=False):
        evaluated = evaluated_predictions(group, metadata.data_asof)
        if evaluated.empty:
            continue
        periods = build_periods(
            evaluated, ModelSettings(model="ols", top_n=cfg.top_n, cost_bps=cfg.cost_bps)
        )
        if periods.empty:
            continue
        periods["model"] = name
        all_periods.append(periods)
        metric = prediction_metrics(evaluated).iloc[0]
        pm = portfolio_metrics(periods).set_index("portfolio")
        comparison.append({
            "model": name, "n_features": len(metadata.features),
            "n_predictions": int(metric["n_predictions"]),
            "n_decision_dates": int(metric["n_dates"]),
            "mse": float(metric["rmse"]) ** 2,
            "rmse": float(metric["rmse"]), "mae": float(metric["mae"]),
            "oos_r2": float(metric["r2"]), "mean_rank_ic": float(metric["rank_ic"]),
            "top5_cagr_net": float(pm.loc["strategy", "cagr"]),
            "equal_weight_cagr_net": float(pm.loc["equal_weight", "cagr"]),
            "n_nonoverlap_periods": len(periods),
        })
    if not comparison:
        raise ValueError("No fully matured test windows: no performance results")
    summary = pd.DataFrame(comparison).sort_values("model")
    # Validate the same dates/universe for every model. Changing denominators can fake gains.
    counts = summary[["n_predictions", "n_decision_dates", "n_nonoverlap_periods"]]
    if (counts.nunique() > 1).any():
        raise RuntimeError("Models were evaluated on unequal holdout observations")
    summary.to_csv(output / "comparison.csv", index=False)
    pd.concat(all_periods).to_csv(output / "portfolio_periods.csv", index=False)
    status = {
        "status": "exploratory_replay_not_live_trading",
        "features": list(metadata.features),
        "earnings_revisions_verified": "earnings_revisions" in metadata.features,
        "n_models": len(summary),
        "config": asdict(cfg),
        "limitations": [
            "Legacy survivorship-biased 24-stock universe; feature coverage is much smaller",
            "Sixth EPS-revisions signal is absent unless independent vintage verification passes",
            "Cross-sectionally standardized features, sector effects not fully neutralized",
            "Price execution/costs and corporate-action treatment are approximations",
            "This comparison is not statistically verified alpha or live profitability",
        ],
    }
    (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return summary