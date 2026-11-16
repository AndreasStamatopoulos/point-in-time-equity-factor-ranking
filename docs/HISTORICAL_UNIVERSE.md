# Historical S&P 500 membership: open source, isolated from legacy baseline

## Source

Pinned GitHub source: [fja05680/sp500](https://github.com/fja05680/sp500), commit
`a2430f2af0c79ddf0748e91de11bdeb1616ab5a7`,
file `sp500_ticker_start_end.csv`.

Our repository includes the **unmodified membership interval CSV**, stored as
`data/universe/historical/sp500_membership_intervals_fja.csv`. The source
contains **1,262 membership intervals**, including former components and
re-entering securities. This is **not** an adjusted stock-price dataset.

This source is reconstructed from a historical base and manually compiled S&P
500 constituent changes. Its own author states that change records are not
always complete. Treat its intervals as **historical membership candidates**,
not a verified institutional PIT index master. In particular, ticker changes,
mergers, and effective-date boundaries need to be independently audited.

## Keep the two studies separate

- **Legacy 24-stock experiment:** unchanged; stored in private replay data.
- **Historical S&P membership universe:** starts at 2015, supports all dated
  members (about 500 securities per date), including historical exits.
- **Current 165-stock snapshot:** a separate contemporary reference only.
  Do not backfill it into historical periods, and do not mix with this dataset.

## Build monthly eligibility snapshots

From the repository root:

```bash
python 10_build_historical_membership.py --start 2015-01-01 --end 2025-12-31
python -m pytest -q
```

This writes **locally** (intentionally excluded from Git):

- `data/universe/generated/sp500_membership_monthly.csv`:
  `decision_date,ticker,source,source_commit`.
- `data/universe/generated/sp500_membership_audit.csv`:
  date and membership count.

The script treats `start_date` as inclusive and `end_date` as **exclusive**
as a conservative implementation convention. Exact boundary semantics need
primary-index-event verification; trading eligibility around event dates should
exclude uncertain cases. The snapshots are calendar month-end labels;
the future trading/portfolio builder must map to actual market sessions.

## Validations already performed

- Pinned file format; parsable ISO dates and non-inverted intervals.
- No simultaneous duplicate ticker at each queried decision date.
- Re-entry example: AMD absent in January 2016 and present in January 2018.
- Five historical January snapshots between 2015 and 2025 have 499–506 names.
- Fail if a monthly membership count is outside 480–515 (heuristic).

A passing test does **not** prove complete survivorship-bias elimination.

## Missing before any 500-stock predictive backtest

1. **Historical prices and terminal returns** for every eligible security,
   including defunct companies, stock splits, dividends, delistings, mergers
   and other corporate actions.
2. **Stable security/issuer identifiers** and historically correct ticker
   mappings (a symbol alone is not a permanent security identifier).
3. **Point-in-time filing financials**, with acceptance timestamps, no
   backfilled restatements, and causal publication/processing delays.
4. Historical analyst EPS consensus revisions, if running all six features.
5. Comparable factor definitions across sectors; banks and REITs require
   separate treatment or explicitly documented exclusions.
6. Causal universe and reconstitution checks against primary effective dates.

**Important:** Do not train on present-day survivors, fill missing delisting
returns with zero, or use current analyst estimates in past rows. The free
GitHub membership source does **not** supply these additional datasets.
