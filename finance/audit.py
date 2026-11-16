from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd

from finance.config import FEATURES

JsonValue: TypeAlias = (
    str | int | float | bool | None
    | list["JsonValue"] | dict[str, "JsonValue"]
)
Severity = Literal["failure", "blocked", "warning"]


@dataclass(frozen=True)
class Finding:
    severity: Severity
    check: str
    message: str
    count: int = 1


class Auditor:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.findings: list[Finding] = []
        self.checks: list[str] = []

    def add(
        self,
        severity: Severity,
        check: str,
        message: str,
        count: int = 1,
    ) -> None:
        self.findings.append(
            Finding(severity, check, message, count)
        )

    def check(
        self, name: str, bad: pd.Series, message: str
    ) -> None:
        self.checks.append(name)
        count = int(bad.fillna(False).sum())
        if count:
            self.add("failure", name, message, count)

    def read(
        self,
        name: str,
        columns: tuple[str, ...],
        required: bool = True,
    ) -> pd.DataFrame:
        path = self.root / name
        if not path.exists():
            if required:
                self.add("blocked", name, "Required input is absent.")
            return pd.DataFrame(columns=list(columns))

        try:
            frame = pd.read_csv(
                path,
                dtype={"ticker": "string", "accession": "string"},
            )
        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
        ) as error:
            self.add("failure", name, f"CSV parse failure: {error}")
            return pd.DataFrame(columns=list(columns))

        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            self.add(
                "failure", name, f"Missing columns: {missing}"
            )
            return pd.DataFrame(columns=list(columns))
        if frame.empty and required:
            self.add("blocked", name, "Input has no observations.")
        return frame

    def dates(
        self,
        frame: pd.DataFrame,
        columns: tuple[str, ...],
        label: str,
    ) -> pd.DataFrame:
        result = frame.copy()
        for column in columns:
            values = pd.to_datetime(
                result[column], errors="coerce", utc=True
            ).dt.tz_localize(None)
            self.check(
                f"{label}.{column}.parse",
                result[column].notna() & values.isna(),
                "Malformed date.",
            )
            result[column] = values
        return result

    def numeric(
        self,
        frame: pd.DataFrame,
        columns: tuple[str, ...],
        label: str,
    ) -> pd.DataFrame:
        result = frame.copy()
        for column in columns:
            values = pd.to_numeric(
                result[column], errors="coerce"
            )
            self.check(
                f"{label}.{column}.numeric",
                result[column].notna() & ~np.isfinite(values),
                "Malformed or nonfinite numeric value.",
            )
            result[column] = values
        return result


def _price_audit(audit: Auditor) -> pd.DataFrame:
    columns = (
        "ticker", "date", "close", "adjusted_close",
        "split_ratio", "future_split_factor", "source", "retrieved_at",
    )
    frame = audit.read("data/prices.csv", columns)
    if frame.empty:
        return frame
    frame = audit.dates(
        frame, ("date", "retrieved_at"), "prices"
    )
    frame = audit.numeric(
        frame,
        ("close", "adjusted_close", "split_ratio", "future_split_factor"),
        "prices",
    )
    audit.check(
        "prices.duplicates",
        frame.duplicated(["ticker", "date"]),
        "Duplicate ticker/date.",
    )
    audit.check(
        "prices.values",
        frame[["close", "adjusted_close"]].isna().any(axis=1)
        | frame["close"].le(0)
        | frame["adjusted_close"].le(0),
        "Prices must be finite and positive.",
    )
    audit.check(
        "prices.splits",
        frame["split_ratio"].isna() | frame["split_ratio"].le(0),
        "Split ratios must be positive.",
    )
    audit.check(
        "prices.split_basis",
        frame["future_split_factor"].isna()
        | frame["future_split_factor"].le(0),
        "Missing or invalid split adjustment basis.",
    )
    audit.check(
        "prices.provenance",
        frame[
            ["ticker", "date", "source", "retrieved_at"]
        ].isna().any(axis=1),
        "Incomplete market provenance.",
    )
    audit.check(
        "prices.retrieval",
        frame["date"].dt.normalize().ge(
            frame["retrieved_at"].dt.normalize()
        ),
        "Daily observation must precede retrieval day.",
    )
    audit.check(
        "prices.synthetic",
        frame["ticker"].str.match(r"^SYN\d+$", na=False),
        "Synthetic stock in real-data inputs.",
    )

    ordered = frame.sort_values(["ticker", "date"])
    prior = ordered.groupby(
        "ticker"
    )["future_split_factor"].shift(1)
    expected = (
        ordered["split_ratio"] * ordered["future_split_factor"]
    )
    audit.check(
        "prices.split_recurrence",
        prior.notna() & ~np.isclose(
            prior, expected, rtol=1e-10, atol=0
        ),
        "Inconsistent adjacent split factors.",
    )
    return frame


def _sec_audit(audit: Auditor) -> pd.DataFrame:
    columns = (
        "ticker", "taxonomy", "concept", "unit", "start", "end",
        "value", "filed", "available_at", "accession", "source",
    )
    frame = audit.read("data/sec_facts.csv", columns)
    if frame.empty:
        return frame
    frame = audit.dates(
        frame, ("start", "end", "filed", "available_at"), "sec"
    )
    frame = audit.numeric(frame, ("value",), "sec")
    audit.check(
        "sec.duplicates",
        frame.duplicated(
            [
                "ticker", "taxonomy", "concept", "unit",
                "start", "end", "value", "filed", "accession",
            ]
        ),
        "Duplicate accounting fact/version.",
    )
    audit.check(
        "sec.provenance",
        frame[
            [
                "ticker", "concept", "unit", "end", "filed",
                "available_at", "accession", "source", "value",
            ]
        ].isna().any(axis=1),
        "Accounting provenance is incomplete.",
    )
    audit.check(
        "sec.period_order",
        frame["start"].gt(frame["end"]),
        "Reversed reporting period.",
    )
    audit.check(
        "sec.filing_order",
        frame["end"].gt(frame["filed"]),
        "Period ends after filing.",
    )
    audit.check(
        "sec.availability",
        frame["available_at"].lt(
            frame["filed"] + pd.Timedelta(days=1)
        ),
        "SEC availability must be at least filed plus one day.",
    )
    audit.add(
        "warning", "sec.vintages",
        "Companyfacts may omit original tags or dimensional contexts; "
        "reconciliation against original filings remains necessary.",
    )
    return frame


def _metadata(
    audit: Auditor,
) -> tuple[pd.Timestamp | None, bool]:
    path = audit.root / "data/features_metadata.json"
    if not path.exists():
        audit.add("blocked", "metadata", "Feature metadata missing.")
        return None, False
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        audit.add("failure", "metadata", f"Invalid JSON: {error}")
        return None, False
    if not isinstance(metadata, dict):
        audit.add("failure", "metadata", "Expected JSON object.")
        return None, False
    if metadata.get("data_kind") != "real":
        audit.add(
            "failure", "metadata.data_kind",
            "Production inputs require data_kind=real.",
        )
    cutoff = metadata.get("data_asof")
    if not isinstance(cutoff, str):
        audit.add("blocked", "metadata.data_asof", "Cutoff missing.")
        return None, metadata.get("earnings_revisions_verified") is True
    try:
        parsed = pd.Timestamp(cutoff)
    except ValueError:
        audit.add("failure", "metadata.data_asof", "Invalid cutoff.")
        return None, False
    if pd.isna(parsed):
        audit.add("failure", "metadata.data_asof", "Missing cutoff.")
        return None, False
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return parsed, metadata.get("earnings_revisions_verified") is True


def _feature_audit(
    audit: Auditor,
    data_asof: pd.Timestamp | None,
    revisions_verified: bool,
) -> pd.DataFrame:
    columns = (
        "date", "ticker", "sector", "available_at",
        "entry_date", "label_end_6m", "forward_return_6m", *FEATURES,
    )
    frame = audit.read("data/features.csv", columns)
    if frame.empty:
        return frame
    frame = audit.dates(
        frame,
        ("date", "available_at", "entry_date", "label_end_6m"),
        "features",
    )
    features = FEATURES + (
        ("earnings_revisions",) if revisions_verified else ()
    )
    if revisions_verified and "earnings_revisions" not in frame:
        audit.add(
            "failure", "features.revisions",
            "Verified revisions are absent from the panel.",
        )
        features = FEATURES
    frame = audit.numeric(
        frame, (*features, "forward_return_6m"), "features"
    )

    audit.check(
        "features.duplicates",
        frame.duplicated(["date", "ticker"]),
        "Duplicate decision/ticker.",
    )
    audit.check(
        "features.identity",
        frame[["date", "ticker", "sector"]].isna().any(axis=1),
        "Missing identity or sector.",
    )
    audit.check(
        "features.future_information",
        frame["available_at"].gt(frame["date"]),
        "Feature availability exceeds decision date.",
    )
    audit.check(
        "features.entry_order",
        frame["entry_date"].notna()
        & frame["entry_date"].le(frame["date"]),
        "Entry must follow decision.",
    )
    audit.check(
        "features.exit_order",
        frame["label_end_6m"].notna()
        & (
            frame["entry_date"].isna()
            | frame["label_end_6m"].le(frame["entry_date"])
        ),
        "Exit must follow entry.",
    )
    audit.check(
        "features.label_provenance",
        frame["forward_return_6m"].notna()
        & (
            frame["entry_date"].isna()
            | frame["label_end_6m"].isna()
        ),
        "Observed returns require entry and exit dates.",
    )
    audit.check(
        "features.return_bounds",
        frame["forward_return_6m"].lt(-1),
        "Return below -100%.",
    )
    audit.check(
        "features.available_at_required",
        frame[list(features)].notna().any(axis=1)
        & frame["available_at"].isna(),
        "Features require availability timestamps.",
    )
    audit.check(
        "features.synthetic",
        frame["ticker"].str.match(r"^SYN\d+$", na=False),
        "Synthetic stock in production panel.",
    )

    if "earnings_revisions" in frame and not revisions_verified:
        audit.check(
            "features.unverified_revisions",
            frame["earnings_revisions"].notna(),
            "Revisions lack verified historical vintage provenance.",
        )
    if data_asof is not None:
        audit.check(
            "features.cutoff",
            frame["date"].gt(data_asof),
            "Decision exceeds observation cutoff.",
        )
        audit.check(
            "features.unmatured_labels",
            frame["forward_return_6m"].notna()
            & frame["label_end_6m"].gt(data_asof),
            "Realized return supplied before its exit date.",
        )
    for _, snapshot in frame.groupby("date"):
        if (
            snapshot["entry_date"].nunique() > 1
            or snapshot["label_end_6m"].nunique() > 1
        ):
            label = snapshot["date"].dt.strftime(
                "%Y-%m-%d"
            ).iloc[0]
            audit.add(
                "failure", "features.common_horizon",
                f"Inconsistent cross-sectional horizon at {label}.",
            )
    return frame


def _source_audit(
    audit: Auditor, panel: pd.DataFrame
) -> pd.DataFrame:
    columns = (
        "date", "ticker", "feature", "component", "value",
        "period_start", "period_end", "filed", "available_at",
        "accession", "concept", "source",
    )
    frame = audit.read("data/feature_sources.csv", columns)
    if frame.empty:
        return frame
    frame = audit.dates(
        frame,
        ("date", "period_start", "period_end", "filed", "available_at"),
        "sources",
    )
    frame = audit.numeric(frame, ("value",), "sources")

    audit.check(
        "sources.provenance",
        frame[
            [
                "date", "ticker", "feature", "component",
                "value", "period_end", "available_at", "source",
            ]
        ].isna().any(axis=1),
        "Incomplete selected-input provenance.",
    )
    audit.check(
        "sources.future_information",
        frame["available_at"].gt(frame["date"]),
        "Future source used in historical feature.",
    )
    audit.check(
        "sources.period_order",
        frame["period_start"].gt(frame["period_end"]),
        "Reversed source period.",
    )
    audit.check(
        "sources.future_period",
        frame["period_end"].gt(frame["date"]),
        "Source period ends after decision.",
    )
    financial = frame["filed"].notna() | frame["accession"].notna()
    audit.check(
        "sources.sec_provenance",
        financial
        & frame[["filed", "accession", "concept"]].isna().any(axis=1),
        "Incomplete selected SEC provenance.",
    )
    audit.check(
        "sources.filing_order",
        financial & frame["period_end"].gt(frame["filed"]),
        "Source period ends after filing.",
    )
    audit.check(
        "sources.sec_availability",
        financial
        & frame["available_at"].lt(
            frame["filed"] + pd.Timedelta(days=1)
        ),
        "SEC input not delayed until filed plus one day.",
    )

    if not panel.empty:
        for feature in FEATURES:
            observed = panel.loc[
                panel[feature].notna(), ["date", "ticker"]
            ]
            supported = frame.loc[
                frame["feature"].eq(feature), ["date", "ticker"]
            ].drop_duplicates()
            joined = observed.merge(
                supported, on=["date", "ticker"],
                how="left", indicator=True,
            )
            audit.check(
                f"sources.{feature}.coverage",
                joined["_merge"].eq("left_only"),
                "Nonmissing feature lacks source evidence.",
            )
    return frame


def _reconstruction_audit(
    audit: Auditor,
    panel: pd.DataFrame,
    sources: pd.DataFrame,
    facts: pd.DataFrame,
) -> None:
    if panel.empty or sources.empty:
        return

    components = {
        "momentum_12_1": ("recent_close", "lookback_close"),
        "book_to_market": ("book_equity", "market_price", "shares"),
        "operating_profitability": ("operating_income", "annual_equity"),
        "asset_growth": ("assets", "prior_assets"),
        "fcf_yield": (
            "cash_from_operations", "capital_expenditures",
            "market_price", "shares",
        ),
    }

    for feature, required in components.items():
        selected = sources.loc[
            sources["feature"].eq(feature)
        ]
        if selected.empty:
            continue
        duplicate = selected.duplicated(
            ["date", "ticker", "component"], keep=False
        )
        audit.check(
            f"mapping.{feature}.duplicates",
            duplicate, "Ambiguous duplicate source component.",
        )
        selected = selected.loc[~duplicate]
        wide = selected.pivot(
            index=["date", "ticker"],
            columns="component", values="value",
        ).reindex(columns=list(required))
        observed = panel.loc[
            panel[feature].notna(), ["date", "ticker", feature]
        ].set_index(["date", "ticker"])
        joined = observed.join(wide, how="left")
        incomplete = joined[list(required)].isna().any(axis=1)
        audit.check(
            f"mapping.{feature}.components",
            incomplete, "Missing feature reconstruction component.",
        )
        valid = joined.loc[~incomplete]
        if valid.empty:
            continue

        if feature == "momentum_12_1":
            denominator = valid["lookback_close"]
            reconstructed = valid["recent_close"] / denominator - 1
        elif feature == "book_to_market":
            denominator = valid["market_price"] * valid["shares"]
            reconstructed = valid["book_equity"] / denominator
        elif feature == "operating_profitability":
            denominator = valid["annual_equity"]
            reconstructed = valid["operating_income"] / denominator
        elif feature == "asset_growth":
            denominator = valid["prior_assets"]
            reconstructed = valid["assets"] / denominator - 1
        else:
            denominator = valid["market_price"] * valid["shares"]
            reconstructed = (
                valid["cash_from_operations"]
                - valid["capital_expenditures"]
            ) / denominator

        audit.check(
            f"mapping.{feature}.denominator",
            denominator.le(0), "Nonpositive feature denominator.",
        )
        audit.check(
            f"mapping.{feature}.value",
            pd.Series(
                ~np.isclose(
                    valid[feature], reconstructed,
                    rtol=1e-9, atol=1e-11,
                ),
                index=valid.index,
            ),
            "Feature differs from reconstructed source value.",
        )

        ends = selected.pivot(
            index=["date", "ticker"],
            columns="component", values="period_end",
        ).reindex(index=valid.index, columns=list(required))
        starts = selected.pivot(
            index=["date", "ticker"],
            columns="component", values="period_start",
        ).reindex(index=valid.index, columns=list(required))

        if feature == "momentum_12_1":
            month = pd.DatetimeIndex(
                valid.index.get_level_values("date")
            ).to_period("M")
            recent_month = pd.Series(
                (month - 1).astype(str), index=valid.index
            )
            old_month = pd.Series(
                (month - 12).astype(str), index=valid.index
            )
            audit.check(
                "mapping.momentum.months",
                ends["recent_close"].dt.strftime(
                    "%Y-%m"
                ).ne(recent_month)
                | ends["lookback_close"].dt.strftime(
                    "%Y-%m"
                ).ne(old_month),
                "Momentum endpoints must be t-1 and t-12.",
            )
        elif feature == "operating_profitability":
            audit.check(
                "mapping.profitability.periods",
                ends["operating_income"].ne(ends["annual_equity"]),
                "Operating income and equity need the same annual end.",
            )
            audit.check(
                "mapping.profitability.duration",
                ~(
                    ends["operating_income"]
                    - starts["operating_income"]
                ).dt.days.between(330, 400),
                "Operating income is not an annual reporting period.",
            )
        elif feature == "asset_growth":
            audit.check(
                "mapping.growth.periods",
                ~(
                    ends["assets"] - ends["prior_assets"]
                ).dt.days.between(350, 380),
                "Asset growth needs comparable annual fiscal ends.",
            )
        elif feature == "fcf_yield":
            audit.check(
                "mapping.fcf.periods",
                ends["cash_from_operations"].ne(
                    ends["capital_expenditures"]
                )
                | starts["cash_from_operations"].ne(
                    starts["capital_expenditures"]
                ),
                "CFO and capex reporting periods differ.",
            )
            audit.check(
                "mapping.fcf.duration",
                ~(
                    ends["cash_from_operations"]
                    - starts["cash_from_operations"]
                ).dt.days.between(330, 400),
                "FCF inputs are not annual.",
            )
            audit.check(
                "mapping.fcf.capex_sign",
                valid["capital_expenditures"].lt(0),
                "PP&E acquisition payments must be nonnegative.",
            )

    financial = sources.loc[sources["filed"].notna()].copy()
    if financial.empty or facts.empty:
        return
    financial["raw_value"] = (
        financial["raw_value"]
        if "raw_value" in financial else financial["value"]
    )
    financial["_source_id"] = np.arange(len(financial))
    keys = [
        "ticker", "concept", "accession",
        "period_start", "period_end",
    ]
    ledger = facts.rename(
        columns={"start": "period_start", "end": "period_end"}
    )
    ledger = ledger[
        keys + ["value", "filed", "available_at"]
    ].drop_duplicates()
    matched = financial.merge(
        ledger, on=keys, how="left", suffixes=("", "_raw")
    )
    matched["_verified"] = (
        np.isclose(
            matched["raw_value"], matched["value_raw"],
            rtol=1e-12, atol=1e-8,
        )
        & matched["filed"].eq(matched["filed_raw"])
        & matched["available_at"].eq(matched["available_at_raw"])
    )
    audit.check(
        "mapping.sec_raw_fact",
        ~matched.groupby("_source_id")["_verified"].any(),
        "Selected accounting source cannot be reconciled to SEC facts.",
    )


def _market_source_audit(
    audit: Auditor,
    sources: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    if sources.empty or prices.empty:
        return
    unique_prices = prices.loc[
        ~prices.duplicated(["ticker", "date"], keep=False)
    ]
    market = sources.loc[
        sources["component"].isin(
            ("recent_close", "lookback_close", "market_price")
        )
    ].copy()

    if not market.empty:
        history = unique_prices.rename(
            columns={"date": "period_end", "source": "price_source"}
        )
        joined = market.merge(
            history[
                [
                    "ticker", "period_end", "close",
                    "adjusted_close", "future_split_factor",
                    "price_source",
                ]
            ],
            on=["ticker", "period_end"], how="left",
            validate="many_to_one",
        )
        nominal = joined["component"].eq("market_price")
        expected = joined["adjusted_close"].where(
            ~nominal,
            joined["close"] * joined["future_split_factor"],
        )
        audit.check(
            "mapping.market_price.available",
            joined["close"].isna(),
            "Market source has no unique price observation.",
        )
        audit.check(
            "mapping.market_price.value",
            pd.Series(
                ~np.isclose(
                    joined["value"], expected,
                    rtol=1e-10, atol=1e-10,
                ),
                index=joined.index,
            ),
            "Market feature differs from independent daily prices.",
        )
        audit.check(
            "mapping.market_price.provenance",
            joined["source"].ne(joined["price_source"]),
            "Market source URL mismatch.",
        )
        audit.check(
            "mapping.market_price.date",
            nominal & joined["period_end"].ne(joined["date"]),
            "Market cap must use the decision-date close.",
        )

    shares = sources.loc[sources["component"].eq("shares")].copy()
    if shares.empty:
        return
    if "raw_value" not in shares:
        audit.add(
            "failure", "mapping.shares.raw_value",
            "Share sources require the originally reported count.",
            len(shares),
        )
        return

    shares = audit.numeric(shares, ("raw_value",), "mapping.shares")
    audit.check(
        "mapping.shares.raw_value",
        shares["raw_value"].isna() | shares["raw_value"].le(0),
        "Reported shares must be finite and positive.",
    )

    alignment_failures: list[bool] = []
    vintage_failures: list[bool] = []
    coverage_failures: list[bool] = []

    for _, row in shares.iterrows():
        history = unique_prices.loc[
            unique_prices["ticker"].eq(row["ticker"])
        ]
        observation = pd.Timestamp(row["period_end"])
        decision = pd.Timestamp(row["date"])
        filed = pd.Timestamp(row["filed"])
        uncovered = (
            history.empty
            or pd.isna(observation)
            or pd.isna(decision)
            or pd.isna(filed)
        )
        if not uncovered:
            uncovered = bool(
                observation < history["date"].min()
                or decision > history["date"].max()
            )
        coverage_failures.append(uncovered)
        if uncovered:
            alignment_failures.append(False)
            vintage_failures.append(False)
            continue

        selected = history.loc[
            history["date"].gt(observation)
            & history["date"].le(decision),
            "split_ratio",
        ]
        expected_count = float(row["raw_value"]) * float(
            selected.to_numpy(dtype=float).prod()
        )
        alignment_failures.append(
            not bool(
                np.isclose(
                    float(row["value"]), expected_count,
                    rtol=1e-10, atol=1e-6,
                )
            )
        )
        ambiguous = (
            history["date"].gt(observation)
            & history["date"].le(filed)
            & history["split_ratio"].ne(1)
        )
        vintage_failures.append(bool(ambiguous.any()))

    audit.check(
        "mapping.shares.coverage",
        pd.Series(coverage_failures),
        "Shares lie outside observed split history.",
    )
    audit.check(
        "mapping.shares.split_adjustment",
        pd.Series(alignment_failures),
        "Shares do not match reported count times intervening splits.",
    )
    audit.check(
        "mapping.shares.vintage",
        pd.Series(vintage_failures),
        "A split between observation and filing makes share units ambiguous.",
    )


def _revisions_audit(audit: Auditor, panel: pd.DataFrame, verified: bool) -> None:
    """Independent checks for dated historical EPS consensus feature provenance."""
    if not verified or panel.empty:
        return
    if "earnings_revisions" not in panel:
        audit.add("failure", "eps.missing_column", "Verified 6th factor column is missing.")
        return
    selected = panel.loc[panel["earnings_revisions"].notna(), ["date", "ticker", "earnings_revisions"]]
    if selected.empty:
        audit.add("blocked", "eps.no_sixth", "Sixth factor approval but zero usable EPS observations.")
        return
    raw_path = audit.root / "data/analyst_consensus_snapshots.csv"
    src_path = audit.root / "data/revisions_sources.csv"
    if not raw_path.exists() or not src_path.exists():
        audit.add("failure", "eps.sources_missing", "Revisions need both PIT consensus and per-row source evidence.")
        return
    src = pd.read_csv(src_path)
    for col in ("date","latest_observed_at","prior_observed_at","target_fiscal_end"):
        if col not in src:
            audit.add("failure", "eps.sources_schema", f"Missing revisions source field: {col}")
            return
        src[col] = pd.to_datetime(src[col], errors="raise")
    if src.duplicated(["ticker","date"]).any():
        audit.add("failure", "eps.sources_duplicate", "More than one revision vintage per stock/date.")
    audit.check("eps.future_snapshot",src["latest_observed_at"].ge(src["date"]),
                "Latest EOD snapshot cannot be traded on the same decision date.")
    audit.check("eps.history_order",src["prior_observed_at"].ge(src["latest_observed_at"]),
                "Prior EPS snapshot must precede latest.")
    audit.check("eps.fiscal_match",src["target_fiscal_end"].le(src["date"]),
                "Target earnings fiscal period must still be in the future.")
    matched = selected.merge(src[["date","ticker","earnings_revisions"]],
                             on=["date","ticker"],how="left",suffixes=("","_source"))
    audit.check("eps.row_source",matched["earnings_revisions_source"].isna(),
                "EPS factor has no source lineage.")
    audit.check("eps.value_reconstruction",~np.isclose(
                matched["earnings_revisions"],matched["earnings_revisions_source"],
                atol=1e-12,rtol=1e-10,equal_nan=False),"EPS factor differs from tracked source.")


def _training_audit(audit: Auditor) -> pd.DataFrame:
    columns = (
        "decision_date", "max_train_date", "max_train_label_end",
        "n_train_rows", "n_train_dates", "model",
    )
    frame = audit.read("results/train_test_separation.csv", columns)
    if frame.empty:
        return frame
    frame = audit.dates(
        frame,
        ("decision_date", "max_train_date", "max_train_label_end"),
        "training",
    )
    frame = audit.numeric(
        frame, ("n_train_rows", "n_train_dates"), "training"
    )
    audit.check(
        "training.missing",
        frame[list(columns)].isna().any(axis=1),
        "Training evidence incomplete.",
    )
    audit.check(
        "training.future_signals",
        frame["max_train_date"].ge(frame["decision_date"]),
        "Training includes a current or future decision.",
    )
    audit.check(
        "training.unmatured_labels",
        frame["max_train_label_end"].ge(frame["decision_date"]),
        "Training includes a label not strictly mature.",
    )
    audit.check(
        "training.empty",
        frame["n_train_rows"].le(0) | frame["n_train_dates"].le(0),
        "Training evidence must have positive rows and dates.",
    )
    return frame


def _coverage(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tickers = sorted(
        set(universe["ticker"].dropna().astype(str))
        | set(prices["ticker"].dropna().astype(str))
        | set(panel["ticker"].dropna().astype(str))
    )
    years = sorted(
        set(pd.to_datetime(prices["date"]).dt.year.dropna().astype(int))
        | set(pd.to_datetime(panel["date"]).dt.year.dropna().astype(int))
    )
    rows: list[dict[str, str | int | float]] = []

    for ticker in tickers:
        ticker_prices = prices.loc[prices["ticker"].eq(ticker)]
        ticker_panel = panel.loc[panel["ticker"].eq(ticker)]
        for year in years or [0]:
            price_year = ticker_prices.loc[
                pd.to_datetime(ticker_prices["date"]).dt.year.eq(year)
            ]
            panel_year = ticker_panel.loc[
                pd.to_datetime(ticker_panel["date"]).dt.year.eq(year)
            ]
            candidates = max(
                len(panel_year),
                int(
                    pd.to_datetime(
                        price_year["date"]
                    ).dt.to_period("M").nunique()
                ),
            )
            for feature in FEATURES:
                present = int(panel_year[feature].notna().sum())
                rows.append(
                    {
                        "ticker": ticker,
                        "year": year,
                        "feature": feature,
                        "candidate_months": candidates,
                        "panel_rows": len(panel_year),
                        "available": present,
                        "missing": candidates - present,
                        "coverage": (
                            present / candidates if candidates else 0.0
                        ),
                    }
                )

    detail = pd.DataFrame(
        rows,
        columns=[
            "ticker", "year", "feature", "candidate_months",
            "panel_rows", "available", "missing", "coverage",
        ],
    )
    sums = ["candidate_months", "panel_rows", "available", "missing"]
    stock = detail.groupby(
        ["ticker", "feature"], as_index=False
    )[sums].sum()
    annual = detail.groupby(
        ["year", "feature"], as_index=False
    )[sums].sum()
    for frame in (stock, annual):
        frame["coverage"] = frame["available"].div(
            frame["candidate_months"].replace(0, np.nan)
        ).fillna(0.0)
    return detail, stock, annual


def _source_coverage(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    facts: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    tickers = sorted(
        set(universe["ticker"].dropna().astype(str))
        | set(prices["ticker"].dropna().astype(str))
        | set(facts["ticker"].dropna().astype(str))
    )
    rows: list[dict[str, str | int]] = []
    for ticker in tickers:
        price_rows = prices.loc[prices["ticker"].eq(ticker)]
        fact_rows = facts.loc[facts["ticker"].eq(ticker)]
        feature_rows = panel.loc[panel["ticker"].eq(ticker)]
        rows.append(
            {
                "ticker": ticker,
                "price_rows": len(price_rows),
                "sec_fact_rows": len(fact_rows),
                "feature_rows": len(feature_rows),
                "complete_feature_rows": int(
                    feature_rows[list(FEATURES)].notna().all(axis=1).sum()
                ),
                "observed_labels": int(
                    feature_rows["forward_return_6m"].notna().sum()
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "ticker", "price_rows", "sec_fact_rows",
            "feature_rows", "complete_feature_rows", "observed_labels",
        ],
    )


def audit_project(root: Path) -> dict[str, JsonValue]:
    audit = Auditor(root)
    universe = audit.read(
        "data/universe.csv", ("ticker", "sector", "selection_basis")
    )
    if not universe.empty:
        audit.check(
            "universe.duplicates",
            universe.duplicated("ticker"),
            "Duplicate universe ticker.",
        )
        audit.check(
            "universe.financials",
            universe["sector"].str.contains(
                "financial", case=False, na=False
            ),
            "Financial stocks are excluded.",
        )
        if not 16 <= universe["ticker"].nunique() <= 24:
            audit.add(
                "warning", "universe.size",
                "Research universe should contain 16–24 stocks.",
            )
    audit.add(
        "warning", "universe.survivorship",
        "Default universe selects surviving stocks and static sectors; "
        "results are conditional on this sample.",
    )

    prices = _price_audit(audit)
    facts = _sec_audit(audit)
    cutoff, verified = _metadata(audit)
    panel = _feature_audit(audit, cutoff, verified)
    sources = _source_audit(audit, panel)
    _reconstruction_audit(audit, panel, sources, facts)
    _market_source_audit(audit, sources, prices)
    _revisions_audit(audit, panel, verified)
    training = _training_audit(audit)

    detail, stock, annual = _coverage(universe, prices, panel)
    source_coverage = _source_coverage(
        universe, prices, facts, panel
    )

    if (
        panel.empty
        or not panel[list(FEATURES)].notna().all(axis=1).any()
    ):
        audit.add(
            "blocked", "model.feature_coverage",
            "No complete real five-feature observations.",
        )

    audit.add(
        "warning", "prices.vendor_vintage",
        "Yahoo data are a retrieved vendor history. Corporate-action "
        "corrections and missing delisting returns remain limitations.",
    )
    audit.add(
        "warning", "inference",
        "Small cross-sections and overlapping six-month labels reduce "
        "effective sample size. Holding-period uncertainty is exploratory.",
    )

    status = (
        "failed"
        if any(item.severity == "failure" for item in audit.findings)
        else "blocked"
        if any(item.severity == "blocked" for item in audit.findings)
        else "passed_with_limitations"
    )

    names = [
        "data/universe.csv", "data/prices.csv",
        "data/sec_facts.csv", "data/features.csv",
        "data/feature_sources.csv", "data/features_metadata.json",
        "results/train_test_separation.csv",
    ]
    hashes: dict[str, JsonValue] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in names if (root / name).exists()
    }
    findings: list[JsonValue] = [
        {
            "severity": finding.severity,
            "check": finding.check,
            "message": finding.message,
            "count": finding.count,
        }
        for finding in audit.findings
    ]
    checks: list[JsonValue] = sorted(set(audit.checks))
    report: dict[str, JsonValue] = {
        "status": status,
        "data_asof": None if cutoff is None else cutoff.date().isoformat(),
        "price_rows": len(prices),
        "sec_fact_rows": len(facts),
        "feature_rows": len(panel),
        "feature_source_rows": len(sources),
        "training_fits_audited": len(training),
        "checks_executed": checks,
        "findings": findings,
        "sha256": hashes,
        "coverage_denominator": (
            "Per ticker/year: max(panel rows, months with prices). "
            "Includes warmup and missing fundamentals."
        ),
    }

    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "audit.json").write_text(
        json.dumps(
            report, indent=2, ensure_ascii=False, allow_nan=False
        ) + "\n",
        encoding="utf-8",
    )
    detail.to_csv(
        results / "coverage_by_stock_year_feature.csv", index=False
    )
    stock.to_csv(
        results / "coverage_by_stock_feature.csv", index=False
    )
    annual.to_csv(
        results / "coverage_by_year_feature.csv", index=False
    )
    source_coverage.to_csv(
        results / "source_coverage.csv", index=False
    )

    lines = [
        "# Source and temporal audit",
        "",
        f"Status: **{status}**.",
        "",
        f"Price rows: {len(prices):,}; SEC facts: {len(facts):,}; "
        f"feature rows: {len(panel):,}; training fits: {len(training):,}.",
        "",
        "Blocked does not mean scientifically validated. "
        "Missing financial inputs and performance are not simulated.",
        "",
    ]
    lines.extend(
        f"- **{item.severity.upper()} — {item.check}** "
        f"({item.count}): {item.message}"
        for item in audit.findings
    )
    lines.extend(
        [
            "",
            "Coverage CSVs include warmup and unavailable fundamentals.",
            "audit.json records executed checks and input SHA-256 hashes.",
            "Synthetic test fixtures are never empirical backtest results.",
            "",
        ]
    )
    (results / "audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report
