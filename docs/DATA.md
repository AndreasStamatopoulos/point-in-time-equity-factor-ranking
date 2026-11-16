# Local datasets and sixth-feature requirements

## Public repo policy
No proprietary price history, earnings vendor data, API keys, or SEC bulk caches are tracked. The private replay archive fills `data/features.csv` and `data/features_metadata.json`; those paths are ignored by Git.

## `data/features.csv` contract
One row per ticker per monthly decision date. Columns: `ticker`, `sector`, `date`, `available_at`, `entry_date`, `label_end_6m`, `forward_return_6m`, `momentum_12_1`, `book_to_market`, `operating_profitability`, `asset_growth`, `fcf_yield`. Add `earnings_revisions` *only with verified analyst-vintage data*.

`available_at` cannot be later than `date`. `entry_date` must be after decision. `label_end_6m` must follow entry. Forward returns of not-yet-completed holding periods must be missing. Numerical features are per stock and decision date, and returns are decimals (0.10 = +10%).

## `data/features_metadata.json`
Required fields include `data_kind: "real"`, `status: "ready"`, `data_asof` (YYYY-MM-DD) and `earnings_revisions_verified` boolean. The sixth feature is enabled **only** with genuine approved historical vintages; cannot be created from a single current consensus number.

## EPS consensus source
Prefer academic I/B/E/S Summary History through institutional WRDS/LSEG access, or a licensed timestamped consensus provider such as Estimize. Align ticker identity, forecast fiscal-period end, observation date and share-split conventions. Obtain permission before publishing any proprietary data. Original template: `data/analyst_consensus_snapshots_TEMPLATE.csv`; scripts `07_prepare_estimize.py` and `06_import_revisions.py` implement guarded ingestion.

## Fresh acquisition
`01_collect_sec.py` and `03_build_features.py` require internet, SEC identity header and source-provider access. Set credentials only through local environment variables; do not commit them. The offline real replay does **not** need internet.
