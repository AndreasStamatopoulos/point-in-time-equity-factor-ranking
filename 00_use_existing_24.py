"""Optional: re-use earlier 24-stock real panel for separate comparison; NO SEC calls."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from finance.config import FEATURES

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'data/reference_existing_24_factor_panel.csv'

if __name__=='__main__':
    orig=pd.read_csv(SOURCE,parse_dates=['date','label_end_6m'])
    mapping={'mom_12_1':'momentum_12_1','btm':'book_to_market','op_prof':'operating_profitability',
             'asset_growth':'asset_growth','fcf_yield':'fcf_yield','stock_return_6m':'forward_return_6m'}
    panel=orig.rename(columns=mapping)
    panel['available_at']=panel['date']
    # Palio dataset: entry at following MONTH-END; labels from original 24-stock backtest.
    panel['entry_date']=panel['date']+pd.offsets.MonthEnd(1)
    panel['earnings_revisions']=np.nan
    cols=['date','ticker','sector','available_at','entry_date','label_end_6m','forward_return_6m',*FEATURES,'earnings_revisions']
    panel=panel[cols].copy()
    cutoff=pd.Timestamp('2026-10-31')
    panel.loc[panel['label_end_6m'].gt(cutoff),'forward_return_6m']=np.nan
    panel.to_csv(ROOT/'data/features.csv',index=False,date_format='%Y-%m-%d')
    # An to paio universe exei 24 diaforetikes metoxes, enimeronoume to list.
    # Xwris epivevaiwmeno CIK den trexoume ton neo SEC collector autoamata.
    universe=pd.read_csv(ROOT/'data/reference_existing_24_universe.csv')
    cik=pd.read_csv(ROOT/'data/reference_existing_24_ticker_cik.csv',dtype={'cik':str})
    universe=universe.merge(cik[['ticker','cik']],on='ticker',how='left')
    if universe['cik'].isna().any(): raise ValueError('Missing CIK for legacy stock')
    universe['membership_start']='2012-01-01'
    universe['membership_end']=''
    universe['selection_basis']='legacy_current_survivor_sample'
    universe.to_csv(ROOT/'data/universe.csv',index=False)

    n=int(panel[list(FEATURES)].notna().all(axis=1).sum())
    metadata={'data_kind':'real','status':'ready','data_asof':str(cutoff.date()),'features':list(FEATURES),
      'earnings_revisions_verified':False,'complete_feature_rows':n,'total_rows':len(panel),
      'limitations':'LEGACY 24 STOCK dataset reconstructed from prior SEC snapshot pipeline, not fresh Codex SEC fact mapping. Sources in original user archive. Shares/raw close are vendor proxies; survivorship bias. Entry is NEXT MONTH END.'}
    (ROOT/'data/features_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'Imported {len(panel)} real legacy 24-stock monthly observations, {n} complete 5-feature rows.')
    print('Next run 04_run_ols.py. Historical results will NOT exactly match older code due changed global ranking.')
    print('WARNING: 05_audit.py needs fresh raw SEC and daily price sources for full source-level audit.')
