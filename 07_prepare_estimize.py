"""Map a licensed Estimize combined_consensus.csv into the project EPS format."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/vendor/combined_consensus.csv"
OUTPUT = ROOT / "data/analyst_consensus_snapshots.csv"

if __name__ == "__main__":
    if not SOURCE.exists():
        raise FileNotFoundError(f"Place licensed Estimize file at {SOURCE}; never use today's current estimates as past snapshots.")
    if OUTPUT.exists():
        raise FileExistsError(f"{OUTPUT} already exists. Rename or review it before replacing.")
    df = pd.read_csv(SOURCE)
    required = {"ticker", "date", "period_ends_on", "estimize.eps.weighted", "reports_at"}
    if not required.issubset(df):
        raise ValueError(f"Estimize columns missing: {sorted(required-set(df.columns))}")
    dates = pd.to_datetime(df["date"], errors="raise", utc=True)
    reports = pd.to_datetime(df["reports_at"], errors="coerce", utc=True)
    target = pd.to_datetime(df["period_ends_on"], errors="raise")
    eps = pd.to_numeric(df["estimize.eps.weighted"], errors="coerce")
    # EOD snapshot tradable only NEXT day.  Exclude announcements already occurred.
    valid = eps.notna() & np.isfinite(eps) & reports.gt(dates) & target.gt(dates.dt.tz_localize(None))
    universe = pd.read_csv(ROOT / "data/universe.csv")
    valid &= df["ticker"].isin(universe["ticker"])
    clean = pd.DataFrame({
        "ticker": df.loc[valid,"ticker"].values,
        "observed_at": dates.loc[valid].dt.strftime("%Y-%m-%d").values,
        "target_fiscal_end": target.loc[valid].dt.strftime("%Y-%m-%d").values,
        "eps_consensus": eps.loc[valid].values,
        "source": "Estimize_licensed_combined_consensus_PIT",
    })
    if clean.empty: raise ValueError("No eligible historical EPS estimates after verification filters.")
    if clean.duplicated(["ticker","observed_at","target_fiscal_end"]).any():
        raise ValueError("Repeated ticker/date/period; requires manual source reconciliation.")
    clean.to_csv(OUTPUT, index=False)
    print(f"Prepared {len(clean)} historical EPS snapshots. Review provider provenance and manually approve via eps_vintage_approval.json, then run 06_import_revisions.py.")
