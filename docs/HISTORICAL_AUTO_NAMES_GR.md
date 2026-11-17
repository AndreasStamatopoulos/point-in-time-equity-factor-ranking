# Automatic historical ticker → company-name hints (v0.5)

**Purpose:** populate `company_name_hint` for the unresolved historical S&P 500 tickers, then propose candidate SEC CIKs. **This does not approve any CIK.** A CIK identifies a SEC filer, not a share class; a ticker may be reused, renamed, retired, or backfilled in third-party historical constituents.

## Sources

1. [Wikipedia S&P 500 selected changes](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies): the **added/removed Ticker + Security** rows. Wikipedia is mutable; a downloaded snapshot is cached locally at `data/cache/historical_names/wikipedia_sp500_changes.html`. Parser checks the table schema and refuses suspiciously few entries. A Wikipedia entry is *search evidence*, not security/CiK identity proof.
2. [pierrebrunelle historical monthly SP500 constituents](https://github.com/pierrebrunelle/sp500-historical-constituents), pinned git SHA `58269b88822abd549b8f8642f0d71ca9be1353da`. Monthly TSVs from January 2016 through April 2026, fetched and cached under `data/cache/historical_names/monthly/`. Default is one of every three months (`--snapshot-step-months 3`); `1` fetches all 124. **WARNING: this series retrospectively rewrites certain historical ticker labels** (e.g. 2016 ELV/META for ANTM/FB). It is useful only for searching names, not as a true ticker timeline or an issuer security master.
3. `data/universe/historical/ticker_name_hints.csv` contains 11 previous hand-entered and 9 extra *user-reported, unverified* company-name hints. These have no pre-approved CIK. A company-name lookup can return multiple CIK candidates (e.g. an acquisition), which must be distinguished manually.

## Instructions (PowerShell)

```powershell
git pull origin main
python -m pip install -r requirements-data.txt
python -m pytest -q
$env:SEC_USER_AGENT="Actual Name actual-email@example.com"

# First run: populate the full queue and show historical company-name coverage,
# using Wikipedia and a quarterly sample of pinned historical monthly names.
python 12_build_historical_crosswalk.py --skip-indexes --snapshot-step-months 3

# Second run: search a limited selection of SEC indexes using the cached names.
python 12_build_historical_crosswalk.py --index-years 2015,2018,2021,2025 --index-quarters 1 --snapshot-step-months 3

# Optional: retrieve all 124 historical name snapshots for maximum recall,
# without resending SEC index requests.
python 12_build_historical_crosswalk.py --skip-indexes --snapshot-step-months 1
```

The SEC indexes are cached as `.idx.gz` and later runs reuse them. Once the first pass is complete, targeted additional years/quarters can be added. A `--offline --skip-indexes` run reads whatever historical-name cache is already present; uncached sources show in the error field rather than silently generating invented names. Use `--no-historical-names` to disable automatic historical name collection.

## Audit output

- `results/historical_crosswalk/unresolved_review_queue.csv`: **all** historical unresolved tickers, even if a batch is selected, with primary hint, alternatives, provenance and hint count. If a ticker has none, keep it blank and flag for review.
- `results/historical_crosswalk/historical_name_hints.csv`: all distinct sourced company-name hints; **all marked UNVERIFIED_SEARCH_HINT**.
- `results/historical_crosswalk/candidate_crosswalk.csv`: candidate CIKs from SEC legal-name matches, each marked **`candidate_unverified`**, plus truly unresolved rows when no candidate was found. Do not mark `verified` without independent, dated ticker/security evidence.
- `results/historical_crosswalk/crosswalk_audit.json`: `name_hints_loaded`, `auto_name_hints_from_historical_sources`, `unique_unresolved_with_name_hints`, `selected_symbols_with_company_name_hint`, `historical_name_source_errors`, SEC index row counts, matched ticker/CIK pairs, cache hits, requests and SEC errors.

**Do not confuse the number of `candidate_unverified` rows with the number of distinct identified tickers.** There can be multiple candidates per ticker, and the source membership intervals can create repeated rows. Look at `symbols_with_candidate_cik` as well.

## Scientific limitations and safeguards

- SEC `master.idx` contains CIK and company name, not historical ticker. `submissions` can corroborate former **legal names** but does not certify historical tickers.
- The pinned monthly list has **look-ahead ticker corrections**; use these strings for *identity research only*. They must never be ingested as time-stamped market observations or feature input.
- Old/new corporation names can be very different. Do not merge predecessor and acquirer under one security_id or CIK without transaction-level evidence.
- No automatic approval, security_id assignment, or inclusion in ML training. The canonical `verified_crosswalk.csv` stays local and requires human evidence.
- If Wikipedia or the pinned GitHub source becomes unavailable, audit its error and use cached files; do not report missing hints as confirmed security delistings.

## Curated SEC legal-name hints (v0.6, November 2026)

Added 40 curated legal-name search hints (AABA through WYND) to
`data/universe/historical/ticker_name_hints.csv`, replacing the older
AABA and BCR hints without duplicating their ticker rows. There are now
58 ticker hints in the file, 40 with source `CURATED_SEC_LEGAL_NAMES`.

These names are **search strings only**: neither the name match nor an
SEC CIK candidate establishes ticker/share-class identity. If FRC or
SBNY fail in the four selected Q1 indexes, inspect additional filing
quarters, alternate SEC entity names, and bank-regulatory sources;
do not assume those companies never filed with the SEC.

### Re-run after pulling

```powershell
git pull origin main
$env:SEC_USER_AGENT="Actual Name actual-email@example.com"
python -m pytest -q
python 12_build_historical_crosswalk.py --index-years 2015,2018,2021,2025 --index-quarters 1
```

Review the unique tickers in
`results/historical_crosswalk/candidate_crosswalk.csv` and the
`crosswalk_audit.json` errors before interpreting coverage. The
historical unresolved inventory is recomputed from the **latest**
`results/free_source_coverage` outputs. If a reviewed local crosswalk
was previously loaded into script 11, the unresolved inventory will
change; the count of 40 is not a permanent expected assertion.

### User-supplied 107-row crosswalk schema

The reviewed CSV shape matches `finance.historical_identity.COLUMNS`:
`security_id,cik,historical_ticker,valid_from,valid_to,company_name,corporate_action,mapping_source,verification_status`.

The parser checks column presence, status, dates, CIK, identifier
non-emptiness, source-URL syntax, and ticker-interval overlaps. It
**does not verify primary-source sufficiency or corporate-action
accuracy**. In particular, `master.idx | submissions` URLs may
establish name→CIK but **not historical ticker→share class→CIK**.
The source CSV's labels `verified` are assertions by its author, not
independent historical-identity evidence. Keep the file local and
gitignored until a dated share-class/ticker review has been completed.
Do not bulk-promote `candidate_unverified` rows.

To validate locally **without publishing any crosswalk**, copy the
manually reviewed file to
`data/universe/historical/verified_crosswalk.csv` and run:

```powershell
python -c "from pathlib import Path; from finance.historical_identity import verified_crosswalk; p=Path('data/universe/historical/verified_crosswalk.csv'); d=verified_crosswalk(p); print('Parsed rows:', sum(map(len,d.values())))"
```

Only after independent verification should step 11 consume that path.
The `--prices` mode is an exploratory Yahoo price-presence probe, not
a PIT-adjusted total-return collection or delisting validation.
