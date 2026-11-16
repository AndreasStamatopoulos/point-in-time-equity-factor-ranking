"""Fail-closed data-contract audits for point-in-time equity research.

This module validates source tables *before* feature computation. A present-day
universe snapshot is never an acceptable substitute for historical memberships.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


class DatasetIntegrityError(ValueError):
    """Raised when source data cannot support a defensible historical backtest."""


@dataclass(frozen=True)
class AuditResult:
    securities: int
    sectors: int
    price_observations: int
    filing_observations: int
    membership_observations: int
    earliest: str
    latest: str


def _required(df: pd.DataFrame, cols: set[str], kind: str) -> None:
    missing = sorted(cols - set(df.columns))
    if missing:
        raise DatasetIntegrityError(f"{kind}: missing columns {missing}")
    if df.empty:
        raise DatasetIntegrityError(f"{kind}: empty source")


def _date(df: pd.DataFrame, cols: tuple[str, ...], kind: str) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
        if df[col].isna().any() and col not in {"valid_to"}:
            raise DatasetIntegrityError(f"{kind}: invalid or missing {col}")
    return df


def audit_history(memberships: pd.DataFrame, prices: pd.DataFrame,
                  filings: pd.DataFrame, *, asof: str,
                  minimum_securities: int = 100, minimum_sectors: int = 10) -> AuditResult:
    """Verify PIT provenance, dates, stable IDs and return/corporate-action consistency.

    Data vendor must attest that total_return_index includes splits, dividends,
    rights/distributions and terminal delisting proceeds. Historical backtests
    may not silently substitute last valid prices for delisted shares.
    """
    _required(memberships, {"security_id", "ticker", "cik", "sector", "valid_from",
                            "valid_to", "membership_known_at", "source", "source_type"}, "memberships")
    _required(prices, {"security_id", "date", "close_unadjusted", "close_split_adjusted",
                       "total_return_index", "split_ratio", "cash_dividend", "delisted",
                       "terminal_return_included", "source"}, "prices")
    _required(filings, {"security_id", "period_end", "accepted_at", "accession",
                        "book_equity", "assets", "operating_income", "operating_cash_flow",
                        "capex", "shares_outstanding", "shares_asof", "source"}, "filings")
    m = _date(memberships, ("valid_from", "valid_to", "membership_known_at"), "memberships")
    p = _date(prices, ("date",), "prices")
    f = _date(filings, ("period_end", "accepted_at", "shares_asof"), "filings")
    cutoff = pd.Timestamp(asof).normalize()
    for label, frame, col in (("prices", p, "date"), ("filings", f, "accepted_at"),
                              ("memberships", m, "membership_known_at")):
        if frame[col].gt(cutoff).any():
            raise DatasetIntegrityError(f"{label}: data dated after asof={cutoff.date()}")
    if not m["source_type"].eq("historical_membership_events").all():
        raise DatasetIntegrityError("Historical backtest requires historical_membership_events, NOT current constituents")
    if m["valid_to"].notna().any() and (m.loc[m["valid_to"].notna(), "valid_to"] <= m.loc[m["valid_to"].notna(), "valid_from"]).any():
        raise DatasetIntegrityError("Membership valid_to must be after valid_from")
    if m["membership_known_at"].gt(m["valid_from"]).any():
        raise DatasetIntegrityError("Membership became known after its declared start date")
    if m.duplicated(["security_id", "valid_from"]).any():
        raise DatasetIntegrityError("Duplicate membership events")
    if p.duplicated(["security_id", "date"]).any():
        raise DatasetIntegrityError("Duplicate price observations")
    if f.duplicated(["security_id", "accession", "period_end"]).any():
        raise DatasetIntegrityError("Duplicate filing rows")
    for col in ["close_unadjusted", "close_split_adjusted", "total_return_index", "split_ratio"]:
        values = pd.to_numeric(p[col], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or (values < 0 if col == "total_return_index" else values <= 0).any():
            raise DatasetIntegrityError(f"Invalid prices.{col}")
        p[col] = values
    for col in ("book_equity", "assets", "operating_income", "operating_cash_flow", "capex", "shares_outstanding"):
        f[col] = pd.to_numeric(f[col], errors="coerce")
        if f[col].notna().any() and not np.isfinite(f[col].dropna().to_numpy()).all():
            raise DatasetIntegrityError(f"Invalid filings.{col}")
    if f["period_end"].gt(f["accepted_at"]).any() or f["shares_asof"].gt(f["accepted_at"]).any():
        raise DatasetIntegrityError("Filing facts or share count timestamp occurs after accepted_at")
    if f["accepted_at"].lt(f["period_end"]).any():
        raise DatasetIntegrityError("Filing accepted before fiscal period end")
    if f["shares_outstanding"].dropna().le(0).any() or f["assets"].dropna().le(0).any():
        raise DatasetIntegrityError("Nonpositive shares outstanding or assets")
    if not p["delisted"].isin([True, False, 0, 1]).all():
        raise DatasetIntegrityError("delisted flag must be Boolean")
    if not p["terminal_return_included"].isin([True, False, 0, 1]).all():
        raise DatasetIntegrityError("terminal_return_included flag must be Boolean")
    if (p["delisted"].astype(bool) & ~p["terminal_return_included"].astype(bool)).any():
        raise DatasetIntegrityError("Delisting without verified final distribution/return")
    # Split-only back-adjusted close must be consistent with unadjusted price and
    # a split-event multiplier on the effective ex-date. Equity data vendors may
    # encode splits differently; here a 4:1 split is represented by 4.0.
    p = p.sort_values(["security_id", "date"]).reset_index(drop=True)
    grouped = p.groupby("security_id", sort=False)
    prev_adj_raw = grouped["close_split_adjusted"].shift().div(grouped["close_unadjusted"].shift())
    this_adj_raw = p["close_split_adjusted"] / p["close_unadjusted"]
    jump = this_adj_raw / prev_adj_raw
    # For a 4:1 split, old adj/raw factor is 1/4 of new; jump=4.
    bad = prev_adj_raw.notna() & (~np.isclose(jump.to_numpy(), p["split_ratio"].to_numpy(), rtol=0.035, atol=0.01))
    if bad.any():
        first = p.loc[bad, ["security_id", "date", "split_ratio"]].iloc[0].to_dict()
        raise DatasetIntegrityError(f"Split-only adjusted/unadjusted close inconsistency: {first}")
    # Total return index is vendor-provided, never synthesized from a price-only
    # series. An ex-dividend day is naturally represented by this index.
    if len(m["sector"].unique()) < minimum_sectors:
        raise DatasetIntegrityError(f"Only {m['sector'].nunique()} sectors; require {minimum_sectors}")
    if m["security_id"].nunique() < minimum_securities:
        raise DatasetIntegrityError(f"Only {m['security_id'].nunique()} securities; require {minimum_securities}")
    ids = set(m["security_id"])
    if not ids.issubset(set(p["security_id"])):
        raise DatasetIntegrityError("Missing price history for membership securities")
    if not ids.issubset(set(f["security_id"])):
        raise DatasetIntegrityError("Missing filings for membership securities")
    return AuditResult(m["security_id"].nunique(), m["sector"].nunique(), len(p),
                       len(f), len(m), str(p["date"].min().date()), str(p["date"].max().date()))


def eligible_asof(memberships: pd.DataFrame, decision: str | pd.Timestamp) -> pd.DataFrame:
    """Returns companies actually in historical membership as of decision date."""
    _required(memberships, {"security_id", "sector", "valid_from", "valid_to",
                            "membership_known_at", "source_type"}, "memberships")
    m = _date(memberships, ("valid_from", "valid_to", "membership_known_at"), "memberships")
    if not m["source_type"].eq("historical_membership_events").all():
        raise DatasetIntegrityError("Current snapshot is FORWARD-ONLY; no historical membership reconstruction")
    t = pd.Timestamp(decision).normalize()
    active = (m["valid_from"] <= t) & (m["valid_to"].isna() | (t < m["valid_to"])) & (m["membership_known_at"] <= t)
    result = m.loc[active].copy()
    if result.duplicated("security_id").any():
        raise DatasetIntegrityError("Overlapping membership events for same security")
    return result


def total_return(prices: pd.DataFrame, security_id: str, entry: str, exit: str) -> float:
    """6m holding return from vendor-authenticated total-return index.

    An unavailable exit value is NOT forward-filled, particularly for delistings.
    """
    subset = prices[prices["security_id"].eq(security_id)].copy()
    subset["date"] = pd.to_datetime(subset["date"])
    e, x = pd.Timestamp(entry), pd.Timestamp(exit)
    rows = subset.set_index("date")
    if e not in rows.index or x not in rows.index:
        raise DatasetIntegrityError(f"{security_id}: missing entry/exit TRI; never drop delisted stock")
    if subset.duplicated("date").any():
        raise DatasetIntegrityError(f"{security_id}: duplicate trading date")
    inside = rows.loc[(rows.index >= e) & (rows.index <= x)]
    if inside["total_return_index"].isna().any():
        raise DatasetIntegrityError(f"{security_id}: missing TRI inside holding period")
    if (inside["delisted"].astype(bool) & ~inside["terminal_return_included"].astype(bool)).any():
        raise DatasetIntegrityError(f"{security_id}: delisting return not certified")
    if (inside["total_return_index"] < 0).any() or rows.loc[e, "total_return_index"] <= 0:
        raise DatasetIntegrityError(f"{security_id}: invalid TRI")
    return float(rows.loc[x, "total_return_index"] / rows.loc[e, "total_return_index"] - 1)


def asof_filings(filings: pd.DataFrame, decision: str | pd.Timestamp,
                 reporting_lag_days: int = 1) -> pd.DataFrame:
    """SEC filing available strictly after acceptance, with next-day lag by default."""
    _required(filings, {"security_id", "period_end", "accepted_at", "accession"}, "filings")
    f = _date(filings, ("period_end", "accepted_at"), "filings")
    t = pd.Timestamp(decision).normalize()
    if (f["accepted_at"] < f["period_end"]).any():
        raise DatasetIntegrityError("Impossible filing chronology")
    eligible = f.loc[f["accepted_at"] + pd.Timedelta(days=reporting_lag_days) <= t].copy()
    return eligible.sort_values(["security_id", "period_end", "accepted_at"]).drop_duplicates(
        ["security_id", "period_end"], keep="last"
    )