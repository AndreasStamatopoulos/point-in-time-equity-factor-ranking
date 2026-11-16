# Methodology, assumptions and audit limits

## 1. Cross-sectional objective
Decision time is month-end *t*. Signal vector X(i,t) uses available market/accounting data. Target is next-period **six-month realized stock return minus same-date eligible-universe mean**, chosen for ranking rather than exact-price estimation. Entry is approximated at month-end *t+1*, exit at *t+7*. In live execution, daily tradable open/close prices and spreads are needed.

For an OOS decision *t*, only training signals from earlier dates with realized label ending strictly before *t* are allowed. Features are clipped and standardized per cross section at *t* without future returns. The estimation step uses sklearn LinearRegression (OLS); a Ridge option is available as a non-neural benchmark.

## 2. Six input characteristics, not FF factor returns
- Momentum 12–1 = P(t−1)/P(t−12) − 1 (split-adjusted price proxy).
- B/M = most recent known book equity / historical market equity (shares and unadjusted price proxies).
- Operating profitability = reported operating income / book equity proxy (**not identical to FF operating profitability**).
- Asset growth = (reported annual assets / prior comparable reported annual assets) − 1.
- Earnings revisions = analyst EPS consensus revision over a configured historical lookback, **same estimated fiscal period**, proven known at *t*. Not earnings surprise.
- Free cash flow yield = (operating cash flow − capital expenditure) / historical market equity (proxy).

## 3. Backtest
Top 5 ranked eligible stocks, equal weight, approx 5 basis points per transaction side. Non-overlapping six-month portfolio periods. Benchmark: equal-weight eligible universe and equal-sector eligible universe. Mean Spearman Rank IC is measured over completed overlapping prediction months; it is not an independent t-statistic over 1,045 IID observations.

## 4. Data provenance limitations
- SEC filings have availability dates; split-adjustment and historical market-cap reconstruction are proxies.
- As-filed XBRL concepts may require audited fallbacks; some firms have missing fields.
- No authenticated EPS revision vintages are bundled: six-feature model **not complete**.
- Historic universe is current surviving firms; delistings and original universe memberships absent. This is a major source of upward bias.
- SEC source-level validation on the legacy imported panel is not fully reproducible without its original raw files; full audit reports **blocked**. Unit tests are not a substitute.
- Only a small subset has complete cross sections (often 6–11 stock candidates).
- This demo has 6-month return labels; implementation of additional 12-month labels remains a planned extension, not claimed completed.
- Costs exclude market impact, realistic bid/ask, taxes and turnover-specific execution complexities.

## 5. Future research, separate from original baseline
Only after validating six historical feature streams, compare OLS/Ridge vs linear-residual correction and neural factor models on the **same splits**. No out-of-sample improvement or profit is currently established.
