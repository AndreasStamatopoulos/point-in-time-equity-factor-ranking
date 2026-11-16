"""Construct a six-month return ranking panel from fully audited vendor inputs.

No provider data is bundled. This adapter deliberately fails if historical
membership, corporate-action-adjusted total returns or PIT filings are absent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd

from finance.data_contract import DatasetIntegrityError, audit_history, eligible_asof, asof_filings, total_return

FEATURES = ("momentum_12_1", "book_to_market", "operating_profitability",
            "asset_growth", "free_cash_flow_yield")


def _last_before(prices: pd.DataFrame, security_id: str, target: pd.Timestamp,
                 tolerance_days: int = 8) -> pd.Series | None:
    rows = prices[(prices.security_id == security_id) & (prices.date <= target)].sort_values('date')
    if rows.empty or (target - rows.iloc[-1]['date']).days > tolerance_days:
        return None
    return rows.iloc[-1]


def _first_on_or_after(prices: pd.DataFrame, security_id: str, target: pd.Timestamp,
                       tolerance_days: int = 8) -> pd.Series | None:
    rows = prices[(prices.security_id == security_id) & (prices.date >= target)].sort_values('date')
    if rows.empty or (rows.iloc[0]['date'] - target).days > tolerance_days:
        return None
    return rows.iloc[0]


def _fundamental_features(filings: pd.DataFrame, prices: pd.DataFrame,
                          sid: str, date: pd.Timestamp) -> tuple[dict, pd.Timestamp] | None:
    available = asof_filings(filings.loc[filings.security_id == sid], date)
    if available.empty:
        return None
    latest = available.sort_values(['period_end','accepted_at']).iloc[-1]
    if (date - latest.period_end).days > 540:
        return None
    older = available[(available.period_end <= latest.period_end - pd.Timedelta(days=330)) &
                      (available.period_end >= latest.period_end - pd.Timedelta(days=400))]
    if older.empty:
        return None
    prior = older.sort_values('period_end').iloc[-1]
    values = [latest.get(z) for z in ('book_equity','assets','operating_income','operating_cash_flow',
                                      'capex','shares_outstanding')]
    if any(pd.isna(v) or not np.isfinite(float(v)) for v in values):
        return None
    be,assets,op,cfo,capex,shares = map(float, values)
    if be <= 0 or assets <= 0 or shares <= 0 or pd.isna(prior.assets) or float(prior.assets)<=0:
        return None
    # A filing can describe shares as of a prior date. Convert into date-t shares
    # using precisely observed intervening forward split factors.
    since = prices[(prices.security_id == sid) &
                   (prices.date > latest.shares_asof) & (prices.date <= date)]
    shares_now = shares * since.split_ratio.prod()
    price = _last_before(prices, sid, date)
    if price is None:
        return None
    market_cap = float(price.close_unadjusted) * shares_now
    if market_cap <= 0:
        return None
    # Profitability is an OP/BE proxy (not an exact FF RMW construction).
    return ({'book_to_market':be/market_cap,
             'operating_profitability':op/be,
             'asset_growth':assets/float(prior.assets)-1,
             'free_cash_flow_yield':(cfo-capex)/market_cap},
            latest.accepted_at + pd.Timedelta(days=1))


def _momentum(prices: pd.DataFrame, sid: str, date: pd.Timestamp) -> float | None:
    # Conventional 12-1: exclude the last completed month (skip most recent month).
    # For decision at November month-end, use September end of the
    # current year vs October end of the previous year: 11 return months
    # (t-12 through t-2), excluding the latest completed month t-1.
    last_month = date.to_period('M') - 2
    old_month = date.to_period('M') - 13
    recent = _last_before(prices, sid, last_month.end_time.normalize())
    old = _last_before(prices, sid, old_month.end_time.normalize())
    if recent is None or old is None:
        return None
    return float(recent.total_return_index / old.total_return_index - 1)


def build_monthly_panel(memberships: pd.DataFrame, prices: pd.DataFrame,
                        filings: pd.DataFrame, *, data_asof: str,
                        start: str, min_securities_per_date: int = 100,
                        min_sectors_per_date: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Emit finance.model-compatible panel plus per-date coverage audit.

    Important: all source tables must be complete PIT histories. Current S&P
    constituents cannot be used retrospectively. The 6m target is computed from
    the first tradable session of next month to the first session 6 months later.
    Strict failure on missing matured exit, including delistings (never drop losers).
    """
    check = audit_history(memberships, prices, filings, asof=data_asof,
                          minimum_securities=min_securities_per_date,
                          minimum_sectors=min_sectors_per_date)
    _ = check
    p=prices.copy(); p['date']=pd.to_datetime(p['date']); p=p.sort_values(['security_id','date'])
    f=filings.copy()
    for x in ('period_end','accepted_at','shares_asof'):
        f[x]=pd.to_datetime(f[x])
    cutoff=pd.Timestamp(data_asof)
    dates = pd.date_range(start=start,end=cutoff,freq='ME')
    records=[];coverage=[]
    for decision in dates:
        # True decision is last shared US equity trading day for that month.
        traded=p.loc[p.date.le(decision)&p.date.ge(decision-pd.Timedelta(days=8)),'date']
        if traded.empty:
            continue
        t=traded.max()
        if t.to_period('M')!=decision.to_period('M'):
            continue
        universe=eligible_asof(memberships,t)
        candidates=[]; missing={'no_price_or_momentum':0,'no_asof_fundamentals':0}
        for sec in universe.itertuples(index=False):
            sid=str(sec.security_id)
            mom=_momentum(p,sid,t)
            if mom is None or not np.isfinite(mom):
                missing['no_price_or_momentum']+=1;continue
            facts=_fundamental_features(f,p,sid,t)
            if facts is None:
                missing['no_asof_fundamentals']+=1;continue
            metrics, fact_available_at=facts
            candidates.append({'date':t,'ticker':sec.ticker,'security_id':sid,'sector':sec.sector,
                               'available_at':max(t, fact_available_at),
                               'momentum_12_1':mom,**metrics})
        coverage.append({'date':str(t.date()),'universe':len(universe),'feature_complete':len(candidates),
                         'sectors':len(set(x['sector'] for x in candidates)),**missing})
        if len(candidates)<min_securities_per_date or len(set(x['sector'] for x in candidates))<min_sectors_per_date:
            raise DatasetIntegrityError(f"{t.date()}: only {len(candidates)} feature-complete stocks; minimum {min_securities_per_date}; see coverage")
        # Use consistent market trading entry/end dates (first shared day of target month).
        next_month = t.to_period('M')+1
        end_month = next_month + 6
        cal=p['date'].drop_duplicates().sort_values()
        start_dates=cal[cal.dt.to_period('M').eq(next_month)]
        end_dates=cal[cal.dt.to_period('M').eq(end_month)]
        entry=start_dates.iloc[0] if not start_dates.empty else pd.NaT
        exit=end_dates.iloc[0] if not end_dates.empty else pd.NaT
        matured=pd.notna(exit) and exit <= cutoff
        for row in candidates:
            record={**row,'entry_date':entry,'label_end_6m':exit,'forward_return_6m':np.nan,
                    'earnings_revisions':np.nan}
            if matured:
                # Failure here is intentional; unavailable delisting returns cannot
                # disappear from the OOS universe.
                record['forward_return_6m']=total_return(p,row['security_id'],str(entry.date()),str(exit.date()))
            records.append(record)
    return pd.DataFrame(records),pd.DataFrame(coverage)


def run_from_directory(root: Path, *, asof: str, start: str, out_dir: Path) -> None:
    """CSV contract: private inputs in `root`; outputs remain local (gitignored)."""
    expected=['memberships.csv','prices.csv','filings.csv']
    for item in expected:
        if not (root/item).is_file():
            raise DatasetIntegrityError(f"Missing historical vendor file {root/item}; no synthetic backtest generated")
    tables=[pd.read_csv(root/item) for item in expected]
    panel,coverage=build_monthly_panel(*tables,data_asof=asof,start=start)
    out_dir.mkdir(parents=True,exist_ok=True)
    # Drop issuer ID only for existing model compatibility; retain separately in audit
    panel.drop(columns=['security_id']).to_csv(out_dir/'features.csv',index=False)
    coverage.to_csv(out_dir/'coverage.csv',index=False)
    metadata={'data_kind':'real','data_asof':asof,'status':'ready','earnings_revisions_verified':False,
              'provider_source':'AUDITED HISTORICAL VENDOR CSV','limitations':['earnings revisions unavailable']}
    (out_dir/'features_metadata.json').write_text(json.dumps(metadata,indent=2))