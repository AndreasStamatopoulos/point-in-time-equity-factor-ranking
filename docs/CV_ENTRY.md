# CV / LinkedIn / GitHub presentation

## Suggested GitHub repo name
`point-in-time-equity-factor-ranking`

## One-line résumé entry (English)
**Quantitative Equity Research Project (Python)** — Built an SEC/XBRL-informed, point-in-time US equity-ranking pipeline using five computed fundamental/price signals and an extensible sixth analyst-earnings-revision input; implemented walk-forward OLS/Ridge, chronological label controls, portfolio backtesting, and automated tests.

## Portfolio description (English)
Designed a reproducible Python research system for ranking stocks by future six-month relative returns rather than point-estimating prices. Implemented price momentum, book-to-market, profitability, asset growth and free-cash-flow-yield features, with a gated interface for historical earnings revisions. Added date-aware walk-forward validation, rank-based evaluation, cost-adjusted illustrative portfolio returns and source-provenance checks. Current six-signal implementation is pending point-in-time analyst estimates; historical five-signal results have limited coverage and survivorship bias.

## Interview talking points
- Contrast cross-sectional signal features and Fama–French factor-return series.
- Explain why an analyst EPS *historical revision* is not current EPS or earnings surprise.
- Describe temporal leakage through label maturation and as-of filings.
- Justify Rank IC vs R² for stock ranking, and why CAGR alone does not establish alpha.
- Explain source-level audit limitations and next steps (delistings, point-in-time universe, full six-signal dataset).

## Avoid claiming
Do not claim profitable investment strategy, real live returns, confirmed six-feature results, positive factor-adjusted alpha, statistically significant outperformance, or completed neural-network work.
