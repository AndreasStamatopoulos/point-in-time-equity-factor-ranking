# Data quality / point-in-time universe protocol (Phase 2)

## Status

- **Delivered:** a **current, 165-security, 11-sector** equal-count S&P 500-derived research shortlist, with CIK identifiers; data-contract audit code, 9 dedicated tests, and a historical vendor-CSV pipeline skeleton. The snapshot is **forward-only**.
- **NOT delivered:** certified historical memberships, complete adjusted prices/total returns, delisting proceeds, annual filings for all 165 over 10+ years, or historical EPS revisions. No completed 165-stock backtest or no-leakage claim is justified until the vendor inputs pass the audit.
- Existing 24-stock baseline and its numerical results remain separate.

## Universe choice

1. Current S&P 500-derived shortlist: 15 securities from each of the eleven GICS sectors, selected deterministically across sorted ticker lists, deduplicated by CIK (one share class per issuer). **It is not an equal-sector-weighted portfolio**; it is an equal-sector-count universe. Each decision can separately impose equal sector weights. The contemporary list can be used prospectively; *do not backfill historical index membership*.
2. For historical experiments, **preferred**: CRSP/WRDS security master + share/delisting returns, historical SEC filings, and properly timestamped index membership. If unavailable, investigate Massive historical daily ticker references and corporate actions, but do not assume a ticker reference snapshot equals past S&P membership. A vendor's historical constituent membership events are needed for a literal S&P 500 study.
3. To reduce sector and survivorship bias, a broader investable-US-equity universe with point-in-time listings and an ex-ante liquidity threshold can be better than fixed 2026 S&P members; it must include stocks that were later delisted. A balanced top-15 by sector computed **at each historical decision date** is valid only when it uses that date's listings and strictly historical liquidity data. The current shortlist cannot satisfy this.

## Immutable identity and provenance

- `security_id`: stable vendor security identifier (not a ticker). `cik`: stable issuer SEC identifier where available. Multiple ticker symbols/share classes can map to one CIK. Keep raw ticker history separately.
- All raw source files are private/ignored by git, versioned by provider/delivery timestamp and checksum. Store dates and source ID in each derived observation.
- `memberships.csv`: `security_id,ticker,cik,sector,valid_from,valid_to,membership_known_at,source,source_type`, `source_type=historical_membership_events`. `valid_to` exclusive; historical sector assignments as known at decision time.
- `prices.csv`: `security_id,date,close_unadjusted,close_split_adjusted,total_return_index,split_ratio,cash_dividend,delisted,terminal_return_included,source`. `split_ratio` is the **forward share multiplier on the ex-date** (4-for-1 = 4), with 1 on other dates; when compressing raw daily bars to monthly, aggregate splits correctly. `close_split_adjusted` is split-only, not split-and-dividend adjusted; it must be on a consistent scale. The *vendor* TRI must capture splits, regular/special dividends, spinoffs and terminal delisting proceeds. The builder does not infer total returns from raw `Close` or `Adj Close`.
- `filings.csv`: `security_id,period_end,accepted_at,accession,book_equity,assets,operating_income,operating_cash_flow,capex,shares_outstanding,shares_asof,source`. Values are same currency (usually USD) and coherent fiscal period; **use annual or TTM consistently**, not annual and quarterly totals mixed. A separate `period_type` column and period-length validation should be mandatory for production. Retain amendment vintages, not restated modern historical figures. The current adapter is a **research scaffold**, not a certified full point-in-time accounting harmonizer.

## Hard exclusions (fail closed)

- A contemporary security list used as historical universe, ticker-only security matching, future-known memberships, duplicate security/date records, restatements visible before acceptance, or newer filing values backfilled onto older dates.
- Price histories that miss mature six-month exits, or omit delisted stocks. If shares disappear, use vendor terminal delisting return and explicit cash-reinvestment convention, never forward-fill stale traded prices. The initial adapter rejects missing terminal TRI instead of inventing a value.
- Split ratio not reconciling the series of split-only adjusted and unadjusted closes; return index missing or nonpositive; unadjusted prices accidentally substituted for total returns.
- Numerator-vs-denominator mismatch: unadjusted shares multiplied by split-adjusted stock price. Market cap must use contemporaneous unadjusted close times split-adjusted share count; returns use vendor TRI. Corporate-action rules must be reconciled against vendor documentation.
- Maturity gap: a six-month future label is used for training only after its *end* has passed. Neural tuning and scaling fit within inner past-only folds. No global `fit_transform` using later dates.

## Metrics and interpretation

| Feature | Correct measurement | Main timestamp trap |
|---|---|---|
| Momentum 12-1 | Total-return index at month-end t−2 divided by index at t−13, minus one (return months t−12 ... t−2, skipping t−1) | Using future dividend adjustments inconsisently; latest-month reversal contamination |
| Book-to-Market | Latest **reported** positive book equity / contemporaneous market cap | Modern restatements; shares and price split mismatch |
| Operating Profitability | Annual operating income / book equity; **proxy**, not exact FF RMW | Quarterly vs annual units; future filings |
| Asset Growth | Latest annual assets / prior year annual assets −1 | Fiscal calendar and revised prior-year balances |
| Free Cash Flow Yield | (Annual CFO − capex) / market cap | Capex sign/units; shifted shares |
| Earnings Revisions | Future-price-free change in consensus EPS vintages from past dated analyst estimates | Current-only earnings estimates and historical restatements |

**Prediction target:** total six-month price+distribution return, relative to the eligible cohort at the corresponding decision date. Inner validation and outer testing operate only on matured labels. Rank IC, sector-neutral rank IC, OOS R², long-short spread, top-decile net performance, turnover, max drawdown and uncertainty intervals should be reported alongside MSE. Comparisons against SPY must use **total-return** benchmark and the same trading interval.

## Execute only after vendor data is ready

```bash
pip install -r requirements.txt
python -m pytest -q
python 08_build_pit_dataset.py --input-dir data/vendor --asof 2026-10-31 --start 2017-01-01
python run_experiments.py --trials 12 --rebalance-months 6
```

No local vendor files implies the builder **exits with an informative error**; this is deliberate.

## Sources

- Current universe snapshot: https://github.com/datasets/s-and-p-500-companies (snapshot pinned to a source commit, date recorded in CSV)
- CRSP/WRDS delisting return / shares / distribution fields: https://wrds-www.wharton.upenn.edu/demo/crsp/form/
- Massive historical stock reference master: https://massive.com/knowledge-base/article/does-massive-have-a-downloadable-security-master
- SEC Company Facts: https://www.sec.gov/search-filings/edgar-application-programming-interfaces