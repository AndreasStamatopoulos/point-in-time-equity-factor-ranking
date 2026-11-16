import importlib.util
from pathlib import Path
import pandas as pd


def loader():
    file=Path(__file__).resolve().parents[1]/'09_generate_current_universe.py'
    spec=importlib.util.spec_from_file_location('current_universe',file)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_balanced_distinct_sectors_and_ciks():
    m=loader()
    rows=[]
    for sector in ['Energy','Health Care']:
        for i in range(4):
            rows.append({'Symbol':sector[:2]+str(i),'Security':'Company '+str(i),
                        'GICS Sector':sector,'GICS Sub-Industry':'Synthetic for unit test',
                        'CIK':sector[:2]+str(i)})
    result=m.select_current_universe(pd.DataFrame(rows),per_sector=3,snapshot_asof='2026-11-17')
    assert result.shape[0]==6
    assert result.cik.nunique()==6
    assert result.sector.value_counts().to_dict()=={'Energy':3,'Health Care':3}
    assert result.research_usage.eq('FORWARD_ONLY_CURRENT_SNAPSHOT_NOT_HISTORICAL_BACKTEST').all()