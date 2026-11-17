# Free data coverage and missing-data audit — methodology

## Status of the dataset

The 24-stock historical regression baseline, the forward-only current 165-stock sector-balanced snapshot, and the historical S&P 500 membership list are separate products. The membership list is not a market-price, issuer-CIK, GICS, or SEC fundamentals panel. **No full 2015–2026 SEC CompanyFacts or stock-price download was completed in the authoring environment**: direct HTTPS calls to SEC and free price endpoints failed; do not interpret `not fetched` as `unavailable`, `delisted`, or `missing`.

The committed report counts exact ticker matches against a **separately pinned third-party GitHub mirror of a current SEC ticker/CIK directory**. Those CIKs are *candidates*, not historically verified permanent identity joins. All historical symbol changes and mergers must be reconciled using SEC CIK, corporate actions and a distinct security/share-class identifier. The SEC itself says its current ticker lookup has limited scope and is periodically updated, not a full historic security master.

## Reproduce the annual audit and probe raw SEC data

From repository root, generate membership dates using `10_build_historical_membership.py` or read `data/universe/historical/sp500_membership_intervals_fja.csv`. Install the optional data collector requirements, and identify yourself with a **real** contact in `SEC_USER_AGENT`:

```bash
python -m pip install -r requirements-data.txt
# Linux/macOS:
export SEC_USER_AGENT='Your Real Name your-real-email@example.com'
# Windows PowerShell:
# $env:SEC_USER_AGENT='Your Real Name your-real-email@example.com'
python 11_collect_sec_price_audit.py --max-symbols 20
# Optional exploratory prices (not delisting-certified):
python -m pip install yfinance
python 11_collect_sec_price_audit.py --max-symbols 20 --prices
```

The program writes `results/free_source_coverage/annual_coverage.csv`, `per_security_coverage.csv` and `audit_metadata.json`. Its default **20-symbol limit is a probe**, not a measurement of all S&P stocks. Increase only after inspecting errors and rate limits. SEC requests use an identifying User-Agent and a modest inter-request delay; avoid launching large parallel SEC jobs. The raw tag-presence flags are NOT a five-feature validity determination.

### Point-in-time requirements before training

1. **Issuer ID:** CIK (zero-padded 10 digits) for SEC filings. Retain a **separate security/share-class ID** for prices; mergers can create successor CIKs; a ticker can be reassigned. Never use ticker as a unique long-lived key.
2. **Filings:** preserve `accn`, fiscal `start/end`, `filed`, form, unit and value *per individual observation*. Select as-of vintage by **filing date**, not a restated `frame` or latest annual value. For daily decisions use acceptance timestamps from submissions, plus processing latency. `filed` alone is date-level and insufficient for same-day trading.
3. **PIT price return:** require historical raw closes, split events, dividends/distributions, spin-offs, and verified terminal delisting proceeds. Yahoo/Stooq-style raw OHLCV is not a certified all-in total return stream. Avoid zero return, forward fill and dropping disappeared tickers when no terminal data are provided.
4. **Historical sectors:** need a vintage-dated GICS/industry classification. The **current 165 snapshot**, while balanced (15 per each of 11 sectors), cannot supply historical GICS membership and cannot establish 5-feature availability for 2015–2026.
5. **Sector comparability:** prefer sector-aware normalizations, financial-specific definitions (e.g. bank CFO/FCF is structurally different), REIT funds-from-operations, and formal missingness reporting. Do not silently replace the original six requested signals without declaring a new experimental design.
6. **Survivorship:** the fja membership history includes firms that left the S&P, but index exit is **not identical** to stock delisting. Current-ticker CIK misses are **not proof of delisting**. Verify status using exchanges/vendor securities master and terminal returns.
7. **Readiness gate:** do not train or report an 'unbiased 150+ stock comparison' until valid comparable price, filing and identity histories clear the original `finance/data_contract.py` checks, amended as necessary for real-world complex corporate actions.

### How to read the actual report

- `memberships`: historical year-end ticker count from pinned membership intervals.
- `current_sec_ticker_match_candidate`: exact-normalized ticker match to a **2025-12-08** GitHub mirror of SEC ticker/CIK directory; NOT a historically authenticated mapping.
- `unresolved_alias_or_no_current_match`: no direct match; includes renames, mergers, ceased trading, private/off-market companies, ticker reuse, and source omissions.
- `exited_sp500_by_2025_12`: those constituents whose membership ends by the date; **NOT** a count of delisted stocks or missing prices.
- `free_price_coverage`: **NOT MEASURED** because provider endpoints were not accessible to the execution environment. No delisting-based coverage gap count can honestly be supplied yet.
- `five_features_pass_by_historical_sector`: **NOT MEASURED** because historic GICS and point-in-time filings/total returns remain unverified.

## Provenance

- S&P membership intervals: `fja05680/sp500`, commit `a2430f2af0c79ddf0748e91de11bdeb1616ab5a7` (1,262 intervals); retrospective reconstruction subject to index-event errors.
- SEC ticker/CIK candidate directory mirror: `Ancalagan/sec-data`, commit `33917b0d6869175cc88337c785b08604726e5437` (2025-12-08), original provenance attributed to SEC but not independently validated against live SEC on this run.
- Current 165-stock GICS snapshot: existing project's forward-only `data/universe_165_current.csv`; 11 sectors × 15 securities each. This should not be backfilled.
- SEC source docs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces and https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data.