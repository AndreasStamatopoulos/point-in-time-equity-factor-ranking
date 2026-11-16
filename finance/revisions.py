"""Import licensed point-in-time EPS consensus snapshots; NEVER fabricate revisions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

COLUMNS = ("ticker", "observed_at", "target_fiscal_end", "eps_consensus", "source")


def revision_at(snapshots: pd.DataFrame, ticker: str, decision: pd.Timestamp):
    """Revision of the SAME future fiscal period, from ~90 days ago to today."""
    cutoff = decision - pd.Timedelta(days=1)  # EOD observations only tradable next session
    subset = snapshots.loc[
        snapshots["ticker"].eq(ticker)
        & snapshots["observed_at"].le(cutoff)
        & snapshots["target_fiscal_end"].gt(decision)
    ]
    if subset.empty:
        return None
    # Epilegoume tin epomeni akoma anoikti oikonomiki periodo.
    for fiscal_end in sorted(subset["target_fiscal_end"].unique()):
        s = subset.loc[subset["target_fiscal_end"].eq(fiscal_end)].sort_values("observed_at")
        recent = s.loc[s["observed_at"].le(cutoff)].tail(1)
        older_asof = cutoff - pd.Timedelta(days=90)
        prior = s.loc[s["observed_at"].le(older_asof)].tail(1)
        if recent.empty or prior.empty:
            continue
        a, b = recent.iloc[0], prior.iloc[0]
        if (cutoff-a["observed_at"]).days > 45 or (older_asof-b["observed_at"]).days > 45:
            continue
        if a["observed_at"] <= b["observed_at"] or a["source"] != b["source"] or abs(float(b["eps_consensus"])) < 0.10:
            continue
        # Ekfrasi posostiaias anatheorisis (oxi earnings surprise).
        return {
            "earnings_revisions": (float(a["eps_consensus"])-float(b["eps_consensus"]))/abs(float(b["eps_consensus"])),
            "latest_observed_at": str(a["observed_at"].date()),
            "prior_observed_at": str(b["observed_at"].date()),
            "target_fiscal_end": str(pd.Timestamp(fiscal_end).date()),
            "source": str(a["source"]),
        }
    return None


def import_revisions(root: Path):
    file = root / "data/analyst_consensus_snapshots.csv"
    approval = root / "data/eps_vintage_approval.json"
    if not file.exists():
        print("EPS not imported: no licensed PIT historical consensus CSV.")
        return
    if not approval.exists():
        raise ValueError("No data/eps_vintage_approval.json (provider, evidence, approved).");
    meta = json.loads(approval.read_text(encoding="utf-8"))
    if meta.get("approved") is not True or not meta.get("provider") or not meta.get("historical_snapshot_evidence"):
        raise ValueError("PIT source provenance not approved. Do not backfill current estimates.")
    df = pd.read_csv(file)
    missing = set(COLUMNS) - set(df.columns)
    if missing: raise ValueError(f"Missing EPS consensus columns: {sorted(missing)}")
    if df.empty: raise ValueError("EPS dataset is empty.")
    if df[list(COLUMNS)].isna().any().any(): raise ValueError("Missing required consensus data.")
    for col in ("observed_at", "target_fiscal_end"):
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="raise")
    df["eps_consensus"] = pd.to_numeric(df["eps_consensus"], errors="raise")
    if not np.isfinite(df["eps_consensus"]).all(): raise ValueError("Nonfinite consensus.")
    if df.duplicated(["ticker","observed_at","target_fiscal_end"]).any():
        raise ValueError("Duplicate vintage/period; do not silently deduplicate.")
    if (df["observed_at"] > pd.Timestamp.today().normalize()).any(): raise ValueError("Future consensus observation.")
    panel_file = root / "data/features.csv"
    if not panel_file.exists(): raise FileNotFoundError("Run 03_build_features.py first.")
    panel = pd.read_csv(panel_file, parse_dates=["date"])
    if "earnings_revisions" not in panel: panel["earnings_revisions"] = np.nan
    rows=[]
    for index, item in panel[["ticker", "date"]].iterrows():
        found = revision_at(df, item["ticker"], item["date"])
        if found is None: continue
        panel.loc[index, "earnings_revisions"] = found["earnings_revisions"]
        rows.append({"ticker":item["ticker"],"date":str(item["date"].date()),**found})
    sources = pd.DataFrame(rows,columns=["ticker","date","earnings_revisions","latest_observed_at","prior_observed_at","target_fiscal_end","source"])
    sources.to_csv(root / "data/revisions_sources.csv",index=False)
    panel.to_csv(panel_file,index=False,date_format="%Y-%m-%d")
    metadata_file=root / "data/features_metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    count=int(panel[list(metadata["features"])+["earnings_revisions"]].notna().all(axis=1).sum())
    metadata.update({"earnings_revisions_verified":bool(count), "revisions_source":meta["provider"], "six_complete_rows":count,
      "eps_source_evidence":meta["historical_snapshot_evidence"]})
    metadata_file.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"PIT revisions imported: {len(rows)} observations, {count} complete six-feature rows.")
    if count == 0: print("No complete 6-feature rows: model remains FIVE-feature only.")
