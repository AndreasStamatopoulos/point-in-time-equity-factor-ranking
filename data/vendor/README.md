# Private point-in-time vendor inputs — DO NOT COMMIT

Put three verified historical vendor CSVs here: `memberships.csv`, `prices.csv`, `filings.csv`.

Schemas, stable security IDs, corporate-action conventions and PIT timestamp
requirements are in [`docs/UNIVERSE_DATA_QUALITY.md`](../../docs/UNIVERSE_DATA_QUALITY.md).

The **165-name current snapshot is NOT a substitute** for these historical files.
These CSVs are deliberately ignored by git. Never commit paid vendor records.