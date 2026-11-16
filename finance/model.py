from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
)

from finance.config import FEATURES as BASE_FEATURES

DATE_COLUMNS = (
    "date", "available_at", "entry_date", "label_end_6m"
)
TARGET = "forward_return_6m"
Scalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class ModelSettings:
    model: Literal["ols", "ridge"] = "ols"
    ridge_alpha: float = 10.0
    min_train_months: int = 24
    min_sector_stocks: int = 1
    top_fraction: float = 0.25  # retained for compatibility; global TOP_N overrides it
    top_n: int = 5
    cost_bps: float = 5.0
    show_plot: bool = False

    def __post_init__(self) -> None:
        if self.model not in {"ols", "ridge"}:
            raise ValueError("MODEL must be ols or ridge.")
        if not math.isfinite(self.ridge_alpha) or self.ridge_alpha <= 0:
            raise ValueError("RIDGE_ALPHA must be positive.")
        if self.min_train_months < 2 or self.min_sector_stocks < 1:
            raise ValueError("Insufficient minimum training/group size.")
        if self.top_n < 1:
            raise ValueError("TOP_N must be at least 1.")
        if not 0 < self.top_fraction <= 1:
            raise ValueError("TOP_FRACTION must be in (0, 1].")
        if (
            not math.isfinite(self.cost_bps)
            or not 0 <= self.cost_bps < 10_000
        ):
            raise ValueError("COST_BPS must be in [0, 10000).")


class IncompletePanelError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PanelMetadata:
    data_asof: pd.Timestamp
    features: tuple[str, ...]


@dataclass(slots=True)
class Forecast:
    predictions: pd.DataFrame
    coefficients: pd.Series[float]
    audit: dict[str, Scalar]


def read_metadata(path: Path) -> PanelMetadata:
    payload: object = json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict) or payload.get("data_kind") != "real":
        raise ValueError("Metadata must declare data_kind='real'.")

    status = payload.get("status")
    if status == "incomplete_fundamentals":
        raise IncompletePanelError(
            "Real feature panel is incomplete: "
            f"{payload.get('complete_feature_rows', 0)}/"
            f"{payload.get('total_rows', 'unspecified')} complete rows. "
            f"{payload.get('limitations', 'Accounting data unavailable')}. "
            "Supply SEC_USER_AGENT, collect SEC data and rebuild features. "
            "No model or real backtest was produced."
        )
    if status not in {None, "ready", "complete", "success", "ok"}:
        raise ValueError(f"Dataset is not ready: {status!r}.")

    value = payload.get("data_asof")
    if not isinstance(value, str):
        raise ValueError("Metadata requires data_asof.")
    data_asof = pd.Timestamp(value)
    if pd.isna(data_asof) or data_asof.tzinfo is not None:
        raise ValueError("data_asof must be a timezone-naive date.")

    features: tuple[str, ...] = BASE_FEATURES
    if payload.get("earnings_revisions_verified") is True:
        features = (*BASE_FEATURES, "earnings_revisions")
    return PanelMetadata(data_asof.normalize(), features)


def load_panel(
    path: Path,
    features: tuple[str, ...],
    data_asof: pd.Timestamp,
) -> pd.DataFrame:
    panel = pd.read_csv(
        path, dtype={"ticker": str, "sector": str}
    )
    required = {"ticker", "sector", TARGET, *DATE_COLUMNS, *features}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Missing panel columns: {missing}")

    for column in DATE_COLUMNS:
        panel[column] = pd.to_datetime(
            panel[column], errors="raise", format="ISO8601"
        )
    for column in (*features, TARGET):
        panel[column] = pd.to_numeric(
            panel[column], errors="raise"
        )
        if not np.isfinite(
            panel[column].dropna().to_numpy(dtype=float)
        ).all():
            raise ValueError(f"Nonfinite observations in {column}.")

    if panel[["date", "ticker", "sector"]].isna().any().any():
        raise ValueError("Missing date, ticker or sector.")
    if panel["ticker"].str.startswith("SYN").any():
        raise ValueError("Synthetic demo tickers cannot enter real runs.")
    if panel.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker.")

    if panel.groupby(
        panel["date"].dt.to_period("M")
    )["date"].nunique().gt(1).any():
        raise ValueError("Use one common monthly decision date.")
    if panel["date"].gt(data_asof).any():
        raise ValueError("Decisions exceed data_asof.")
    if panel["available_at"].gt(panel["date"]).any():
        raise ValueError("Future feature availability.")
    if panel["entry_date"].le(panel["date"]).any():
        raise ValueError("Trade entry must follow the decision.")
    if panel["label_end_6m"].le(panel["entry_date"]).any():
        raise ValueError("Label end must follow entry.")
    if panel[TARGET].lt(-1).any():
        raise ValueError("Returns below -100%.")

    has_label = panel[TARGET].notna()
    invalid_label = (
        panel["entry_date"].isna()
        | panel["label_end_6m"].isna()
    )
    if (
        has_label
        & (invalid_label | panel["label_end_6m"].gt(data_asof))
    ).any():
        raise ValueError("Recorded return lacks a mature holding period.")

    for column in ("entry_date", "label_end_6m"):
        if panel.groupby("date")[column].nunique(
            dropna=False
        ).gt(1).any():
            raise ValueError(f"Inconsistent common {column}.")

    dated = (
        panel["entry_date"].notna()
        & panel["label_end_6m"].notna()
    )
    month_distance = (
        (panel["label_end_6m"].dt.year - panel["entry_date"].dt.year)
        * 12
        + panel["label_end_6m"].dt.month
        - panel["entry_date"].dt.month
    )
    if (dated & month_distance.ne(6)).any():
        raise ValueError("Holding periods must span six calendar months.")

    return panel.sort_values(
        ["date", "sector", "ticker"]
    ).reset_index(drop=True)


def prepare_features(
    panel: pd.DataFrame,
    features: tuple[str, ...],
    min_sector_stocks: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Global cross-sectional normalization; no enforced sector allocations."""
    complete = panel[list(features)].notna().all(axis=1) & panel["available_at"].notna()
    sector_complete = complete.groupby([panel["date"], panel["sector"]]).transform("sum")
    mask = complete & sector_complete.ge(min_sector_stocks)
    exclusions = panel.loc[~mask, ["date", "ticker", "sector"]].copy()
    exclusions["reason"] = "insufficient_sector_coverage"
    exclusions.loc[~complete.loc[exclusions.index], "reason"] = "missing_feature"
    eligible = panel.loc[mask].copy()
    if eligible.empty:
        return eligible, pd.DataFrame(index=eligible.index, columns=list(features)), exclusions

    # Mono diathesima features sto idio decision date; oxi y_future.
    grouped = eligible.groupby("date")[list(features)]
    low = grouped.transform("quantile", q=0.05)
    high = grouped.transform("quantile", q=0.95)
    clipped = eligible[list(features)].clip(lower=low, upper=high)
    means = clipped.groupby(eligible["date"]).transform("mean")
    scales = clipped.groupby(eligible["date"]).transform("std", ddof=0)
    normalized = ((clipped - means) / scales.replace(0, 1)).where(scales.ne(0), 0)
    return eligible, normalized, exclusions


def predict_date(
    panel: pd.DataFrame,
    normalized: pd.DataFrame,
    date: pd.Timestamp,
    settings: ModelSettings,
) -> Forecast | None:
    # Oi 6m train labels prepei na exoun oloklirothei PRIN to decision.
    mature = panel["label_end_6m"].lt(date) & panel[TARGET].notna()
    closed = mature.groupby(panel["date"]).transform("all")
    training = panel["date"].lt(date) & closed
    train = panel.loc[training]
    features = list(normalized.columns)
    if train["date"].nunique() < settings.min_train_months or len(train) <= len(features) + 1:
        return None
    current = panel.loc[panel["date"].eq(date)].copy()
    if len(current) < 5:
        return None

    # Stoxos: 6m stock return meion mesos oros olou tou universe stin idia imerominia.
    relative = train[TARGET] - train.groupby("date")[TARGET].transform("mean")
    weights = None

    estimator = (
        Ridge(alpha=settings.ridge_alpha)
        if settings.model == "ridge"
        else LinearRegression()
    )
    estimator.fit(
        normalized.loc[train.index], relative,
        sample_weight=weights,
    )
    current["predicted_excess"] = estimator.predict(
        normalized.loc[current.index]
    )
    current = current.sort_values(
        ["sector", "ticker"], kind="stable"
    )
    current["sector_rank"] = current.groupby(
        "sector"
    )["predicted_excess"].rank(
        ascending=False, method="first"
    ).astype(int)
    current["cross_section_rank"] = current[
        "predicted_excess"
    ].rank(ascending=False, method="first").astype(int)

    design = normalized.loc[train.index].to_numpy(dtype=float)
    condition = float(np.linalg.cond(design))
    prior = panel["date"].lt(date)
    excluded_groups = panel.loc[
        prior & ~closed, ["date", "sector"]
    ].drop_duplicates()

    audit: dict[str, Scalar] = {
        "decision_date": date.date().isoformat(),
        "max_train_date": train["date"].max().date().isoformat(),
        "max_train_label_end": (
            train["label_end_6m"].max().date().isoformat()
        ),
        "n_train_rows": len(train),
        "n_train_dates": int(train["date"].nunique()),
        "model": settings.model,
        "n_features": len(features),
        "design_rank": int(np.linalg.matrix_rank(design)),
        "condition_number": (
            condition if math.isfinite(condition) else None
        ),
        "excluded_training_groups": len(excluded_groups),
    }
    coefficients = pd.Series(
        estimator.coef_, index=features, name="coefficient"
    )
    coefficients.loc["intercept"] = float(estimator.intercept_)
    return Forecast(current, coefficients, audit)


def net_return(gross_return: float, cost_bps: float) -> float:
    cost = cost_bps / 10_000
    return (1 + gross_return) * (1 - cost) / (1 + cost) - 1


def portfolio_weights(current: pd.DataFrame, top_fraction: float, top_n: int = 5) -> pd.Series[float]:
    """Top stocks by six-factor predicted score, independent of sector."""
    weights = pd.Series(0.0, index=current.index)
    count = min(len(current), top_n)
    chosen = current.sort_values(
        ["predicted_excess", "ticker"],
        ascending=[False, True], kind="stable"
    ).head(count)
    weights.loc[chosen.index] = 1.0 / len(chosen)
    return weights


def rank_ic(current: pd.DataFrame) -> float:
    if (
        current["predicted_excess"].nunique() < 2
        or current["true_relative"].nunique() < 2
    ):
        return float("nan")
    return float(
        current["predicted_excess"].corr(
            current["true_relative"], method="spearman"
        )
    )


def evaluated_predictions(
    predictions: pd.DataFrame, data_asof: pd.Timestamp
) -> pd.DataFrame:
    eligible = (
        predictions["label_end_6m"].notna()
        & predictions["label_end_6m"].le(data_asof)
    )
    closed = eligible.groupby(
        predictions["date"]
    ).transform("all")
    if (eligible & ~closed).any():
        raise ValueError("Partial test-date maturity changes the universe.")

    evaluated = predictions.loc[eligible].copy()
    if evaluated[TARGET].isna().any():
        raise ValueError(
            "Missing mature test return. Supply delisting/corporate-action "
            "data before backtesting."
        )
    if evaluated.empty:
        evaluated["true_relative"] = pd.Series(dtype=float)
        return evaluated
    evaluated["true_relative"] = evaluated[TARGET] - evaluated.groupby(
        "date"
    )[TARGET].transform("mean")
    return evaluated


def build_periods(
    evaluated: pd.DataFrame, settings: ModelSettings
) -> pd.DataFrame:
    rows: list[dict[str, Scalar]] = []
    last_exit: pd.Timestamp | None = None

    for decision, current in evaluated.groupby("date", sort=True):
        if not isinstance(decision, pd.Timestamp):
            raise ValueError("Decision dates must be timestamps.")
        entry = pd.Timestamp(current["entry_date"].iloc[0])
        exit_date = pd.Timestamp(current["label_end_6m"].iloc[0])
        if last_exit is not None and entry < last_exit:
            continue

        weights = portfolio_weights(current, settings.top_fraction, settings.top_n)
        gross = float((weights * current[TARGET]).sum())
        equal_stock = float(current[TARGET].mean())
        equal_sector = float(
            current.groupby("sector")[TARGET].mean().mean()
        )
        rows.append(
            {
                "decision_date": decision.date().isoformat(),
                "entry_date": entry.date().isoformat(),
                "exit_date": exit_date.date().isoformat(),
                "cash_gap_days": (
                    0 if last_exit is None else int((entry - last_exit).days)
                ),
                "n_stocks": len(current),
                "n_sectors": int(current["sector"].nunique()),
                "n_holdings": int(weights.gt(0).sum()),
                "model": settings.model,
                "cost_bps_per_side": settings.cost_bps,
                "strategy_gross": gross,
                "strategy_net": net_return(gross, settings.cost_bps),
                "equal_weight_gross": equal_stock,
                "equal_weight_net": net_return(
                    equal_stock, settings.cost_bps
                ),
                "sector_equal_gross": equal_sector,
                "sector_equal_net": net_return(
                    equal_sector, settings.cost_bps
                ),
                "rank_ic": rank_ic(current),
            }
        )
        last_exit = exit_date

    periods = pd.DataFrame(rows)
    if not periods.empty:
        for name in ("strategy", "equal_weight", "sector_equal"):
            periods[f"{name}_wealth"] = (
                1 + periods[f"{name}_net"]
            ).cumprod()
    return periods


def prediction_metrics(evaluated: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope", "n_predictions", "n_dates",
        "mae", "rmse", "r2", "rank_ic",
    ]
    if evaluated.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Scalar]] = []
    for decision, group in evaluated.groupby("date", sort=True):
        if not isinstance(decision, pd.Timestamp):
            raise ValueError("Decision dates must be timestamps.")
        true = group["true_relative"]
        predicted = group["predicted_excess"]
        rows.append(
            {
                "scope": decision.date().isoformat(),
                "n_predictions": len(group),
                "n_dates": 1,
                "mae": float(mean_absolute_error(true, predicted)),
                "rmse": float(
                    np.sqrt(mean_squared_error(true, predicted))
                ),
                "r2": (
                    float(r2_score(true, predicted))
                    if true.nunique() > 1 else None
                ),
                "rank_ic": rank_ic(group),
            }
        )

    per_date = pd.DataFrame(rows)
    true = evaluated["true_relative"]
    predicted = evaluated["predicted_excess"]
    overall: dict[str, Scalar] = {
        "scope": "all_monthly_oos_overlapping_labels",
        "n_predictions": len(evaluated),
        "n_dates": int(evaluated["date"].nunique()),
        "mae": float(mean_absolute_error(true, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(true, predicted))),
        "r2": (
            float(r2_score(true, predicted))
            if true.nunique() > 1 else None
        ),
        "rank_ic": float(per_date["rank_ic"].mean()),
    }
    return pd.concat(
        [pd.DataFrame([overall]), per_date], ignore_index=True
    )


def portfolio_metrics(periods: pd.DataFrame) -> pd.DataFrame:
    if periods.empty:
        return pd.DataFrame(
            columns=[
                "portfolio", "cumulative_return",
                "cagr", "years", "n_periods",
            ]
        )
    first = pd.Timestamp(periods["entry_date"].iloc[0])
    last = pd.Timestamp(periods["exit_date"].iloc[-1])
    years = (last - first).days / 365.25
    if years <= 0:
        raise ValueError("Portfolio evaluation requires positive time.")

    rows: list[dict[str, Scalar]] = []
    for name in ("strategy", "equal_weight", "sector_equal"):
        wealth = float(periods[f"{name}_wealth"].iloc[-1])
        rows.append(
            {
                "portfolio": name,
                "cumulative_return": wealth - 1,
                "cagr": wealth ** (1 / years) - 1,
                "years": years,
                "n_periods": len(periods),
                "cash_gap_days": int(periods["cash_gap_days"].sum()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_uncertainty(
    periods: pd.DataFrame, draws: int = 2_000
) -> pd.DataFrame:
    columns = [
        "statistic", "estimate", "lower_95", "upper_95",
        "n_periods", "method", "status",
    ]
    if len(periods) < 6:
        return pd.DataFrame(
            [
                {
                    "statistic": "all",
                    "n_periods": len(periods),
                    "method": "moving_block_bootstrap",
                    "status": "insufficient_periods_minimum_6",
                }
            ],
            columns=columns,
        )
    if draws < 100:
        raise ValueError("Use at least 100 bootstrap draws.")

    count = len(periods)
    block = 2
    rng = np.random.default_rng(20261106)
    starts = rng.integers(
        0, count - block + 1,
        size=(draws, math.ceil(count / block)),
    )
    indices = (
        starts[:, :, None] + np.arange(block)
    ).reshape(draws, -1)[:, :count]

    duration = (
        pd.to_datetime(periods["exit_date"])
        - pd.to_datetime(periods["entry_date"])
    ).dt.days.to_numpy(dtype=float)
    duration += periods["cash_gap_days"].to_numpy(dtype=float)
    years = duration[indices].sum(axis=1) / 365.25

    strategy = periods["strategy_net"].to_numpy(dtype=float)
    benchmark = periods["equal_weight_net"].to_numpy(dtype=float)
    strategy_draws = (
        np.prod(1 + strategy[indices], axis=1) ** (1 / years) - 1
    )
    benchmark_draws = (
        np.prod(1 + benchmark[indices], axis=1) ** (1 / years) - 1
    )

    metrics = portfolio_metrics(periods).set_index("portfolio")
    strategy_cagr = float(
        metrics.loc[["strategy"], "cagr"].to_numpy(dtype=float)[0]
    )
    benchmark_cagr = float(
        metrics.loc[["equal_weight"], "cagr"].to_numpy(dtype=float)[0]
    )
    samples: list[tuple[str, float, NDArray[np.float64]]] = [
        ("strategy_cagr", strategy_cagr, strategy_draws),
        (
            "strategy_minus_equal_weight_cagr",
            strategy_cagr - benchmark_cagr,
            strategy_draws - benchmark_draws,
        ),
    ]

    ic = periods["rank_ic"].to_numpy(dtype=float)
    if np.isfinite(ic).all():
        samples.append(
            ("mean_rank_ic", float(ic.mean()), ic[indices].mean(axis=1))
        )

    rows: list[dict[str, Scalar]] = []
    for name, estimate, sample in samples:
        lower, upper = np.quantile(sample, [0.025, 0.975])
        rows.append(
            {
                "statistic": name,
                "estimate": estimate,
                "lower_95": float(lower),
                "upper_95": float(upper),
                "n_periods": count,
                "method": (
                    f"moving_block_bootstrap_block2_draws{draws}"
                    "_seed20261106"
                ),
                "status": "exploratory_small_sample_intervals",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _write_status(
    output: Path,
    status: str,
    reason: str,
    **fields: Scalar,
) -> None:
    payload: dict[str, Scalar] = {
        "status": status,
        "reason": reason,
        "real_backtest_results": status == "complete",
        **fields,
    }
    (output / "run_status.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _plot(
    periods: pd.DataFrame, output: Path, show: bool
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5))
    first_entry = pd.Timestamp(periods["entry_date"].iloc[0])
    for name, label in (
        ("strategy", "Model"),
        ("equal_weight", "Equal stock weights"),
        ("sector_equal", "Equal sector weights"),
    ):
        dates = pd.DatetimeIndex(
            [
                first_entry,
                *pd.to_datetime(periods["exit_date"]).tolist(),
            ]
        ).to_numpy()
        axis.plot(
            dates, [1.0, *periods[f"{name}_wealth"].tolist()],
            label=label,
        )
    axis.set(
        title="Real-data six-month walk-forward portfolio | net costs",
        xlabel="Holding-period exit",
        ylabel="Value of initial $1",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "wealth_chart.png", dpi=150)
    if show:
        plt.show()
    plt.close(figure)


def run_model(
    root: Path, settings: ModelSettings = ModelSettings()
) -> None:
    output = root / "results"
    output.mkdir(parents=True, exist_ok=True)

    # Katharizoume mono paragomena arxeia gia na min menoun palia results.
    filenames = (
        "predictions.csv", "latest_ranking.csv",
        "model_coefficients.csv", "train_test_separation.csv",
        "model_exclusions.csv", "backtest_periods.csv",
        "prediction_metrics.csv", "portfolio_metrics.csv",
        "uncertainty.csv", "wealth_chart.png",
    )
    for filename in filenames:
        (output / filename).unlink(missing_ok=True)
    _write_status(
        output, "running", "Validating and evaluating.",
        model=settings.model,
    )

    panel_path = root / "data/features.csv"
    metadata_path = root / "data/features_metadata.json"
    if not panel_path.exists() or not metadata_path.exists():
        reason = "Real feature panel unavailable; run collection/build steps."
        _write_status(
            output, "blocked", reason, model=settings.model
        )
        print(reason)
        return

    try:
        metadata = read_metadata(metadata_path)
        panel = load_panel(
            panel_path, metadata.features, metadata.data_asof
        )
    except IncompletePanelError as error:
        _write_status(
            output, "blocked", str(error), model=settings.model,
        )
        print(str(error))
        return
    except (ValueError, KeyError, json.JSONDecodeError) as error:
        _write_status(
            output, "blocked", str(error), model=settings.model
        )
        raise

    eligible, normalized, exclusions = prepare_features(
        panel, metadata.features, settings.min_sector_stocks
    )
    exclusions.to_csv(
        output / "model_exclusions.csv", index=False
    )
    if eligible.empty:
        reason = "No complete real feature cross-sections."
        _write_status(
            output, "blocked", reason,
            model=settings.model,
            feature_count=len(metadata.features),
        )
        print(reason)
        return

    predictions: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Scalar]] = []
    coefficient_rows: list[pd.DataFrame] = []
    latest_forecast: Forecast | None = None

    for date_value in sorted(eligible["date"].unique()):
        decision = pd.Timestamp(date_value)
        forecast = predict_date(
            eligible, normalized, decision, settings
        )
        if forecast is None:
            continue
        predictions.append(forecast.predictions)
        audit_rows.append(forecast.audit)
        coefficients = forecast.coefficients.rename_axis(
            "feature"
        ).reset_index()
        coefficients["decision_date"] = decision.date().isoformat()
        coefficient_rows.append(coefficients)
        latest_forecast = forecast

    if not predictions or latest_forecast is None:
        reason = (
            "Insufficient mature history: need "
            f"{settings.min_train_months} training dates."
        )
        _write_status(
            output, "blocked", reason,
            model=settings.model,
            feature_count=len(metadata.features),
        )
        print(reason)
        return

    latest_ranking_date = pd.Timestamp(
        latest_forecast.predictions["date"].max()
    )
    latest_panel_date = pd.Timestamp(panel["date"].max())
    ranking_fields: dict[str, Scalar] = {
        "latest_ranking_date": latest_ranking_date.date().isoformat(),
        "latest_panel_date": latest_panel_date.date().isoformat(),
        "latest_ranking_stale": latest_ranking_date < latest_panel_date,
        "latest_ranking_age_days": int(
            (metadata.data_asof - latest_ranking_date).days
        ),
    }
    all_predictions = pd.concat(predictions).sort_values(
        ["date", "sector", "sector_rank"]
    )

    try:
        evaluated = evaluated_predictions(
            all_predictions, metadata.data_asof
        )
        periods = build_periods(evaluated, settings)
    except ValueError as error:
        _write_status(
            output, "blocked", str(error), model=settings.model
        )
        raise

    all_predictions.to_csv(
        output / "predictions.csv", index=False
    )
    latest_forecast.predictions.sort_values(
        ["sector", "sector_rank"]
    ).to_csv(output / "latest_ranking.csv", index=False)
    pd.concat(
        coefficient_rows, ignore_index=True
    ).to_csv(output / "model_coefficients.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(
        output / "train_test_separation.csv", index=False
    )
    prediction_metrics(evaluated).to_csv(
        output / "prediction_metrics.csv", index=False
    )
    portfolio_metrics(periods).to_csv(
        output / "portfolio_metrics.csv", index=False
    )
    bootstrap_uncertainty(periods).to_csv(
        output / "uncertainty.csv", index=False
    )

    if periods.empty:
        reason = "Ranking produced; no completed OOS holding periods."
        _write_status(
            output, "ranking_only", reason,
            model=settings.model,
            feature_count=len(metadata.features),
            **ranking_fields,
        )
        print(reason)
        return

    periods.to_csv(output / "backtest_periods.csv", index=False)
    _plot(periods, output, settings.show_plot)
    _write_status(
        output, "complete",
        "Real-data walk-forward evaluation completed; "
        "interpret with scientific limitations.",
        model=settings.model,
        feature_count=len(metadata.features),
        n_nonoverlapping_periods=len(periods),
        n_prediction_dates=int(all_predictions["date"].nunique()),
        data_asof=metadata.data_asof.date().isoformat(),
        top_fraction=settings.top_fraction,
        top_n=settings.top_n,
        cost_bps_per_side=settings.cost_bps,
        ridge_alpha=(
            settings.ridge_alpha if settings.model == "ridge" else None
        ),
        **ranking_fields,
    )
    print(portfolio_metrics(periods).to_string(index=False))
    print(f"Saved {len(periods)} nonoverlapping periods to {output}.")
