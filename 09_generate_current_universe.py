"""Rebuild prospective-only balanced S&P 500 sector snapshot from pinned upstream CSV.

This script MUST NOT generate retrospective historical index membership.
Requires network access on the user's machine; no API key needed.
"""
from __future__ import annotations
import argparse
import io
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen
import pandas as pd

SOURCE_SHA='05c998c3e714b8dc9b621239bbdef92bccefc376'
SOURCE_URL=f'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/{SOURCE_SHA}/data/constituents.csv'


def select_current_universe(source: pd.DataFrame, *, per_sector: int=15,
                            snapshot_asof: str | None=None) -> pd.DataFrame:
    required={'Symbol','Security','GICS Sector','GICS Sub-Industry','CIK'}
    if not required.issubset(source.columns):
        raise ValueError(f'Missing source columns: {sorted(required-set(source.columns))}')
    used=set();rows=[]
    for sector, group in sorted(source.groupby('GICS Sector'),key=lambda x:x[0]):
        group=group.sort_values('Symbol')
        candidates=[x for _,x in group.iterrows() if pd.notna(x.CIK) and str(x.CIK) not in used]
        if len(candidates)<per_sector:
            raise ValueError(f'{sector}: fewer than {per_sector} distinct CIKs')
        taken=set()
        for j in range(per_sector):
            index=int(j*(len(candidates)-1)/(per_sector-1)+0.5) if per_sector>1 else 0
            x=candidates[index]
            if str(x.CIK) in used or str(x.CIK) in taken:
                x=next(v for v in candidates if str(v.CIK) not in used and str(v.CIK) not in taken)
            used.add(str(x.CIK));taken.add(str(x.CIK))
            rows.append(dict(ticker=x.Symbol,company=x.Security,sector=sector,
                             subindustry=x['GICS Sub-Industry'],cik=str(x.CIK),
                             source_commit=SOURCE_SHA,snapshot_asof=snapshot_asof or date.today().isoformat(),
                             research_usage='FORWARD_ONLY_CURRENT_SNAPSHOT_NOT_HISTORICAL_BACKTEST'))
    return pd.DataFrame(rows).sort_values(['sector','ticker']).reset_index(drop=True)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('data/universe_165_current.csv'))
    p.add_argument('--snapshot-asof',default=date.today().isoformat())
    a=p.parse_args()
    req=Request(SOURCE_URL,headers={'User-Agent':'EquityResearchDataset/0.2 (research only)'})
    with urlopen(req,timeout=30) as r:
        raw=r.read().decode('utf-8')
    source=pd.read_csv(io.StringIO(raw),dtype={'CIK':str})
    result=select_current_universe(source,snapshot_asof=a.snapshot_asof)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(a.output,index=False)
    print(f'Created {len(result)} forward-only securities / {result.sector.nunique()} sectors: {a.output}')


if __name__ == '__main__':
    main()