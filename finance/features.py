from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from finance.config import FEATURES
from finance.prices import (
    load_prices, nominal_close, split_adjusted_shares,
)
from finance.sec import (
    FACT_COLUMNS, read_facts, select_annual, select_instant,
)

PANEL_COLUMNS = [
    "date", "ticker", "sector", "available_at", "entry_date",
    "label_end_6m", "forward_return_6m", *FEATURES, "earnings_revisions",
]
SOURCE_COLUMNS = [
    "date", "ticker", "feature", "component", "value", "raw_value",
    "period_start", "period_end", "filed", "available_at",
    "accession", "concept", "source",
]

EQUITY = ("StockholdersEquity",)
ASSETS = ("Assets",)
SHARES = (
    "EntityCommonStockSharesOutstanding",
    "CommonStockSharesOutstanding",
)
CFO = ("NetCashProvidedByUsedInOperatingActivities",)
CAPEX = ("PaymentsToAcquirePropertyPlantAndEquipment",)
OPERATING_INCOME = ("OperatingIncomeLoss",)


def optional_date(value: object) -> pd.Timestamp | None:
    if value is None or value is pd.NaT:
        return None
    if not isinstance(value, (date, datetime, str, np.datetime64)):
        raise ValueError("Expected a date or explicit missing date.")
    result = pd.Timestamp(value)
    return None if pd.isna(result) else result


def required_date(value: object) -> pd.Timestamp:
    result = optional_date(value)
    if result is None:
        raise ValueError("Required date is missing.")
    return result


def monthly_schedule(prices: pd.DataFrame) -> pd.DataFrame:
    days = pd.DatetimeIndex(
        prices["date"].unique()
    ).sort_values()
    if days.empty:
        raise ValueError("No trading calendar.")

    calendar = pd.DataFrame({"day": days})
    calendar["month"] = calendar["day"].dt.to_period("M")
    first = calendar.groupby("month")["day"].min()
    last = calendar.groupby("month")["day"].max()

    # O trexon atelis minas den ginetai istoriko signal.
    complete = last.loc[
        last.index < days.max().to_period("M")
    ]
    rows: list[dict[str, pd.Timestamp | None]] = []
    for month, decision in complete.items():
        if not isinstance(month, pd.Period):
            raise ValueError("Expected monthly calendar periods.")
        rows.append(
            {
                "date": required_date(decision),
                "entry_date": optional_date(
                    first.get(month + 1, pd.NaT)
                ),
                "label_end_6m": optional_date(
                    first.get(month + 7, pd.NaT)
                ),
            }
        )
    return pd.DataFrame(
        rows, columns=["date", "entry_date", "label_end_6m"]
    )


def momentum(
    history: pd.DataFrame,
    decision: pd.Timestamp,
) -> tuple[float, pd.Series | None, pd.Series | None]:
    monthly = history.groupby(
        history["date"].dt.to_period("M")
    ).tail(1)
    month = decision.to_period("M")
    recent = monthly.loc[
        monthly["date"].dt.to_period("M").eq(month - 1)
    ]
    old = monthly.loc[
        monthly["date"].dt.to_period("M").eq(month - 12)
    ]
    if len(recent) != 1 or len(old) != 1:
        return float("nan"), None, None
    value = float(
        recent.iloc[0]["adjusted_close"]
        / old.iloc[0]["adjusted_close"] - 1
    )
    return value, recent.iloc[0], old.iloc[0]


def select_shares(
    facts: pd.DataFrame,
    history: pd.DataFrame,
    decision: pd.Timestamp,
) -> pd.Series | None:
    candidates = facts.loc[
        facts["concept"].isin(SHARES)
        & facts["start"].isna()
        & facts["unit"].eq("shares")
        & facts["end"].ge(history["date"].min())
    ].copy()
    split_days = pd.DatetimeIndex(
        history.loc[history["split_ratio"].ne(1), "date"]
    ).sort_values()

    # Filing meta to split mporei na exei anadiatypomenes monades.
    if not candidates.empty and not split_days.empty:
        after_end = split_days.searchsorted(
            candidates["end"], side="right"
        )
        through_filed = split_days.searchsorted(
            candidates["filed"], side="right"
        )
        candidates = candidates.loc[after_end == through_filed]

    return select_instant(
        candidates, SHARES, decision,
        max_age_days=200, unit="shares",
    )


def market_cap(
    history: pd.DataFrame,
    shares: pd.Series,
    decision: pd.Timestamp,
) -> tuple[float, float, float]:
    price = nominal_close(history, decision)
    count = split_adjusted_shares(
        history, float(shares["value"]),
        pd.Timestamp(shares["end"]), decision,
    )
    return price * count, price, count


def _source(
    decision: pd.Timestamp,
    ticker: str,
    feature: str,
    component: str,
    fact: pd.Series,
    value: float | None = None,
) -> dict[str, str | float]:
    return {
        "date": str(decision.date()),
        "ticker": ticker,
        "feature": feature,
        "component": component,
        "value": float(fact["value"]) if value is None else value,
        "raw_value": float(fact["value"]),
        "period_start": (
            "" if pd.isna(fact["start"])
            else str(pd.Timestamp(fact["start"]).date())
        ),
        "period_end": str(pd.Timestamp(fact["end"]).date()),
        "filed": str(pd.Timestamp(fact["filed"]).date()),
        "available_at": str(
            pd.Timestamp(fact["available_at"]).date()
        ),
        "accession": str(fact["accession"]),
        "concept": str(fact["concept"]),
        "source": str(fact["source"]),
    }


def _price_source(
    decision: pd.Timestamp,
    ticker: str,
    feature: str,
    component: str,
    price: pd.Series,
    value: float,
    concept: str,
) -> dict[str, str | float]:
    day = str(pd.Timestamp(price["date"]).date())
    return {
        "date": str(decision.date()),
        "ticker": ticker,
        "feature": feature,
        "component": component,
        "value": value,
        "raw_value": float(
            price["close"]
            if concept == "nominal_close"
            else price["adjusted_close"]
        ),
        "period_start": "",
        "period_end": day,
        "filed": "",
        "available_at": day,
        "accession": "",
        "concept": concept,
        "source": str(price["source"]),
    }


def _historical_assets(
    facts: pd.DataFrame,
    decision: pd.Timestamp,
    current_end: pd.Timestamp,
) -> pd.Series | None:
    ends = facts.loc[
        facts["concept"].eq("Assets") & facts["start"].isna(),
        "end",
    ].drop_duplicates()
    candidates = ends.loc[
        (current_end - ends).dt.days.between(350, 380)
    ]
    if candidates.empty:
        return None
    target = current_end - pd.DateOffset(years=1)
    closest = candidates.iloc[
        (candidates - target).abs().argmin()
    ]
    return select_instant(
        facts, ASSETS, decision, end=closest, max_age_days=950
    )


def annual_assets(
    facts: pd.DataFrame, decision: pd.Timestamp
) -> pd.Series | None:
    annual = facts.loc[
        facts["form"].isin(("10-K", "10-K/A"))
    ]
    anchor = select_instant(annual, ASSETS, decision)
    if anchor is None:
        return None
    return select_instant(
        facts, ASSETS, decision, end=anchor["end"]
    )


def build_features(root: Path) -> pd.DataFrame:
    prices = load_prices(root / "data/prices.csv")
    universe = pd.read_csv(
        root / "data/universe.csv",
        parse_dates=["membership_start", "membership_end"],
    )
    if (
        universe["ticker"].duplicated().any()
        or universe["sector"].isna().any()
    ):
        raise ValueError("Universe requires unique tickers and sectors.")
    if universe["sector"].str.contains(
        "financial", case=False
    ).any():
        raise ValueError("Financial stocks are excluded.")
    if not 16 <= len(universe) <= 24:
        raise ValueError("Configure 16–24 nonfinancial stocks.")

    facts_path = root / "data/sec_facts.csv"
    facts = (
        read_facts(facts_path)
        if facts_path.exists()
        else pd.DataFrame(columns=FACT_COLUMNS)
    )
    schedule = monthly_schedule(prices)
    rows: list[
        dict[str, str | float | pd.Timestamp | None]
    ] = []
    sources: list[dict[str, str | float]] = []
    issues: list[dict[str, str]] = []

    for member in universe.itertuples(index=False):
        ticker = str(member.ticker)
        membership_start = required_date(member.membership_start)
        membership_end = optional_date(member.membership_end)
        history = prices.loc[
            prices["ticker"].eq(ticker)
        ].copy()
        company_facts = facts.loc[
            facts["ticker"].eq(ticker)
        ].copy()

        for event in schedule.itertuples(index=False):
            decision = required_date(event.date)
            entry_date = optional_date(event.entry_date)
            label_end = optional_date(event.label_end_6m)
            if (
                decision < membership_start
                or (
                    membership_end is not None
                    and decision > membership_end
                )
            ):
                continue

            current_price = history.loc[
                history["date"].eq(decision)
            ]
            row: dict[
                str, str | float | pd.Timestamp | None
            ] = {
                "date": decision,
                "ticker": ticker,
                "sector": str(member.sector),
                "available_at": decision,
                "entry_date": entry_date,
                "label_end_6m": label_end,
                "forward_return_6m": float("nan"),
                **dict.fromkeys((*FEATURES, "earnings_revisions"), float("nan")),
            }

            mom, recent, old = momentum(history, decision)
            row["momentum_12_1"] = mom
            if recent is not None and old is not None:
                sources.extend(
                    [
                        _price_source(
                            decision, ticker, "momentum_12_1",
                            "recent_close", recent,
                            float(recent["adjusted_close"]),
                            "adjusted_close",
                        ),
                        _price_source(
                            decision, ticker, "momentum_12_1",
                            "lookback_close", old,
                            float(old["adjusted_close"]),
                            "adjusted_close",
                        ),
                    ]
                )

            if entry_date is not None and label_end is not None:
                entry = history.loc[
                    history["date"].eq(entry_date)
                ]
                exit_price = history.loc[
                    history["date"].eq(label_end)
                ]
                if len(entry) == len(exit_price) == 1:
                    row["forward_return_6m"] = float(
                        exit_price.iloc[0]["adjusted_close"]
                        / entry.iloc[0]["adjusted_close"] - 1
                    )
                else:
                    issues.append(
                        {
                            "ticker": ticker,
                            "date": str(decision.date()),
                            "issue": "missing_exact_label_price",
                        }
                    )

            if not company_facts.empty and len(current_price) == 1:
                equity = select_instant(
                    company_facts, EQUITY, decision
                )
                shares = select_shares(
                    company_facts, history, decision
                )
                cap = price = share_count = float("nan")
                if shares is not None:
                    cap, price, share_count = market_cap(
                        history, shares, decision
                    )

                if (
                    equity is not None
                    and float(equity["value"]) > 0
                    and cap > 0
                    and shares is not None
                ):
                    row["book_to_market"] = float(
                        equity["value"]
                    ) / cap
                    sources.extend(
                        [
                            _source(
                                decision, ticker, "book_to_market",
                                "book_equity", equity,
                            ),
                            _source(
                                decision, ticker, "book_to_market",
                                "shares", shares, share_count,
                            ),
                            _price_source(
                                decision, ticker, "book_to_market",
                                "market_price", current_price.iloc[0],
                                price, "nominal_close",
                            ),
                        ]
                    )

                operating = select_annual(
                    company_facts, OPERATING_INCOME, decision
                )
                if operating is not None:
                    annual_equity = select_instant(
                        company_facts, EQUITY, decision,
                        end=operating["end"],
                    )
                    if (
                        annual_equity is not None
                        and float(annual_equity["value"]) > 0
                    ):
                        row["operating_profitability"] = float(
                            operating["value"] / annual_equity["value"]
                        )
                        sources.extend(
                            [
                                _source(
                                    decision, ticker,
                                    "operating_profitability",
                                    "operating_income", operating,
                                ),
                                _source(
                                    decision, ticker,
                                    "operating_profitability",
                                    "annual_equity", annual_equity,
                                ),
                            ]
                        )

                cfo = select_annual(
                    company_facts, CFO, decision
                )
                if cfo is not None:
                    capex = select_annual(
                        company_facts, CAPEX, decision,
                        end=cfo["end"],
                    )
                    if (
                        capex is not None
                        and capex["start"] == cfo["start"]
                        and float(capex["value"]) >= 0
                        and cap > 0
                        and shares is not None
                    ):
                        row["fcf_yield"] = float(
                            cfo["value"] - capex["value"]
                        ) / cap
                        sources.extend(
                            [
                                _source(
                                    decision, ticker, "fcf_yield",
                                    "cash_from_operations", cfo,
                                ),
                                _source(
                                    decision, ticker, "fcf_yield",
                                    "capital_expenditures", capex,
                                ),
                                _source(
                                    decision, ticker, "fcf_yield",
                                    "shares", shares, share_count,
                                ),
                                _price_source(
                                    decision, ticker, "fcf_yield",
                                    "market_price", current_price.iloc[0],
                                    price, "nominal_close",
                                ),
                            ]
                        )

                assets = annual_assets(company_facts, decision)
                if assets is not None:
                    previous = _historical_assets(
                        company_facts, decision,
                        pd.Timestamp(assets["end"]),
                    )
                    if (
                        previous is not None
                        and float(previous["value"]) > 0
                        and float(assets["value"]) > 0
                    ):
                        row["asset_growth"] = float(
                            assets["value"] / previous["value"] - 1
                        )
                        sources.extend(
                            [
                                _source(
                                    decision, ticker, "asset_growth",
                                    "assets", assets,
                                ),
                                _source(
                                    decision, ticker, "asset_growth",
                                    "prior_assets", previous,
                                ),
                            ]
                        )

            rows.append(row)

    panel = pd.DataFrame(
        rows, columns=PANEL_COLUMNS
    ).sort_values(["date", "ticker"])

    if np.isinf(
        panel[
            list(FEATURES) + ["forward_return_6m"]
        ].to_numpy(dtype=float)
    ).any():
        raise ValueError("Infinite feature or return.")

    panel.to_csv(
        root / "data/features.csv",
        index=False, date_format="%Y-%m-%d",
    )
    pd.DataFrame(
        sources, columns=SOURCE_COLUMNS
    ).to_csv(root / "data/feature_sources.csv", index=False)
    pd.DataFrame(
        issues, columns=["ticker", "date", "issue"]
    ).to_csv(root / "data/feature_issues.csv", index=False)

    complete_rows = int(
        panel[list(FEATURES)].notna().all(axis=1).sum()
    )
    metadata = {
        "data_kind": "real",
        "status": (
            "ready" if complete_rows else "incomplete_fundamentals"
        ),
        "data_asof": str(prices["date"].max().date()),
        "features": list(FEATURES),
        "earnings_revisions_verified": False,
        "complete_feature_rows": complete_rows,
        "total_rows": len(panel),
        "universe_stocks": int(universe["ticker"].nunique()),
        "survivorship_bias": (
            "Current survivor sample; static sectors. "
            "Historical constituents are not reconstructed."
        ),
        "accounting_basis": (
            "Annual standard US-GAAP concepts. Operating income/book "
            "equity is a proxy, not exact Fama-French profitability."
        ),
        "equity_basis": (
            "StockholdersEquity excludes noncontrolling interests "
            "but may include preferred equity."
        ),
        "availability": "SEC filed date plus one calendar day.",
        "price_basis": (
            "Adjusted close for returns. Yahoo close split adjustment "
            "undone for nominal historical market capitalization."
        ),
        "shares_basis": (
            "Latest known instantaneous shares, maximum age 200 days; "
            "splits adjusted through the decision only."
        ),
        "fundamental_max_age_days": 550,
        "limitations": (
            "No SEC fundamentals available"
            if facts.empty
            else "Current SEC archives may omit original XBRL vintages."
        ),
    }
    (root / "data/features_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Real monthly panel: {len(panel)} rows; "
        f"{complete_rows} complete rows. {metadata['status']}"
    )
    return panel
