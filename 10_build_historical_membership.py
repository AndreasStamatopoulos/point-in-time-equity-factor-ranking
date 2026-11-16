"""Build dated S&P 500 membership snapshots from pinned free historical intervals.

This is *membership only*: no prices, splits, dividends, SEC fundamentals, or EPS estimates.
The source is retrospectively reconstructed and is NOT a formally audited PIT security master.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import date
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "data/universe/historical/sp500_membership_intervals_fja.csv"
OUTPUT = Path(__file__).resolve().parent / "data/universe/generated"
SOURCE_COMMIT = "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7"
SOURCE_NAME = "fja05680/sp500"


def read_intervals(path: Path = SOURCE) -> list[tuple[str, date, date | None]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            ticker = raw["ticker"].strip().upper()
            start = date.fromisoformat(raw["start_date"])
            end = date.fromisoformat(raw["end_date"]) if raw["end_date"].strip() else None
            if not ticker or end is not None and end < start:
                raise ValueError(f"Invalid membership interval: {raw}")
            rows.append((ticker, start, end))
    if not rows:
        raise ValueError("Empty membership source")
    return rows


def members_on(intervals: list[tuple[str, date, date | None]], asof: date) -> list[str]:
    """Treat source end_date as exclusive to avoid trading after removal.

    Boundary-day semantics should be audited against primary index events.
    """
    tickers = [ticker for ticker, start, end in intervals
               if start <= asof and (end is None or asof < end)]
    duplicates = [k for k, n in Counter(tickers).items() if n > 1]
    if duplicates:
        raise ValueError(f"Overlapping membership intervals on {asof}: {duplicates[:10]}")
    return sorted(tickers)


def month_ends(start: date, end: date):
    if end < start:
        raise ValueError("End date must not precede start date")
    year, month = start.year, start.month
    while year * 12 + month <= end.year * 12 + end.month:
        nxt_y, nxt_m = (year + 1, 1) if month == 12 else (year, month + 1)
        last = date.fromordinal(date(nxt_y, nxt_m, 1).toordinal() - 1)
        if start <= last <= end:
            yield last
        year, month = nxt_y, nxt_m


def build(intervals, start: date, end: date, out_dir: Path, check_counts=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    membership = out_dir / "sp500_membership_monthly.csv"
    audit = out_dir / "sp500_membership_audit.csv"
    n_dates = 0
    with membership.open("w", newline="", encoding="utf-8") as mf, audit.open("w", newline="", encoding="utf-8") as af:
        mw, aw = csv.writer(mf), csv.writer(af)
        mw.writerow(["decision_date", "ticker", "source", "source_commit"])
        aw.writerow(["decision_date", "n_tickers", "source"])
        for d in month_ends(start, end):
            members = members_on(intervals, d)
            if check_counts and not 480 <= len(members) <= 515:
                raise ValueError(f"Unexpected S&P 500 constituent count at {d}: {len(members)}")
            for symbol in members:
                mw.writerow([d.isoformat(), symbol, SOURCE_NAME, SOURCE_COMMIT])
            aw.writerow([d.isoformat(), len(members), SOURCE_NAME])
            n_dates += 1
    return membership, audit, n_dates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    path, audit, n = build(read_intervals(args.source), date.fromisoformat(args.start),
                           date.fromisoformat(args.end), args.output)
    print(f"{n} dated S&P 500 snapshots -> {path}")
    print(f"Membership quality audit -> {audit}")
    print("WARNING: reconstructed membership, not audited PIT prices/fundamentals")


if __name__ == "__main__":
    main()
