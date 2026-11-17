# Historical S&P 500 source-level coverage audit — 17 November 2026

**Status: PARTIAL / BLOCKED AT REMOTE SEC AND PRICE FETCH. No verified 2015–2026 fundamentals or price panel has been assembled yet.**

Pinned source data: 1,262 historical S&P membership intervals from `fja05680/sp500` at commit `a2430f2af0c79ddf0748e91de11bdeb1616ab5a7`; 10,196 directory rows from a **2025-12-08 GitHub mirror of SEC ticker→CIK**, `Ancalagan/sec-data` at `33917b0d6869175cc88337c785b08604726e5437`. This latter directory is a **current (as of that mirror) lookup**, not a historical security identity master, so its matches are candidate joins only. SEC CompanyFacts and free price-provider endpoints were not accessible from the authoring runtime. No per-stock split/dividend/delisting observations were downloaded.

| Year-end | Historical members | Exact current-ticker CIK candidates | Unresolved/historical aliases | Members subsequently exited S&P by 2025-12 |
|---|---:|---:|---:|---:|
| 2015 | 502 | 402 | 100 | 180 |
| 2016 | 506 | 418 | 88 | 165 |
| 2017 | 505 | 426 | 79 | 145 |
| 2018 | 505 | 436 | 69 | 131 |
| 2019 | 505 | 449 | 56 | 108 |
| 2020 | 505 | 459 | 46 | 93 |
| 2021 | 505 | 466 | 39 | 80 |
| 2022 | 503 | 482 | 21 | 57 |
| 2023 | 503 | 489 | 14 | 39 |
| 2024 | 503 | 496 | 7 | 20 |
| 2025 | 503 | 503 | 0 | 0 |

**Do not equate an S&P exit, missing current SEC ticker, or unavailable Yahoo price bar with a verified stock delisting.** The number of price histories missing *because of delisting* is presently **undetermined**, not zero. All per-year observed-price counts and per-sector five-feature pass rates are also **not measured**, not zero.

The separate current 165-security snapshot contains exactly 15 securities per each of 11 GICS sectors. It is not a valid historical GICS dataset. Therefore the historical counts of issuers passing all five features — especially Financials and Real Estate — are not yet measurable. Sector-specific accounting definitions must be reviewed before pooling FCF Yield and Asset Growth with other industries.

For detailed evidence, see [yearly audit](../data/audit/free_source_annual_audit.csv), [per-security CIK candidate audit 2015–2021](../data/audit/sec_cik_candidates_2015_2021.csv), [sector status](../data/audit/current_sector_feature_status.csv), and [local collection instructions](FREE_DATA_AUDIT_METHOD.md).

**Next gate:** run the SEC EDGAR + exploratory price probe using a real SEC User-Agent, record failures by provider, obtain a verified historical ticker/CIK/security crosswalk and corporate actions, then quantify missing terminal delisting returns and validated 5-feature coverage. Do not run the six models on this partial data.
