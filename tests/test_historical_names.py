"""Ensure historical ticker names stay evidence-only and never become verified joins."""
import csv
from pathlib import Path
from types import SimpleNamespace
import importlib.util

import pytest

from finance.historical_names import (
    NameHint, HistoricalNameSources, MONTHLY_SHA, WIKIPEDIA_URL,
    changes_from_wikipedia, names_from_month, names_for_unresolved, months,
)
from finance.historical_identity import index_matches

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('crosswalk_12', ROOT / '12_build_historical_crosswalk.py')
build12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build12)


def fixture_html():
    rows = ''.join(f'<tr><td>2020-01-01</td><td>XYZ</td><td>New Company {j}</td>'
                   f'<td>AGN</td><td>Allergan plc</td><td>merger</td></tr>' for j in range(22))
    return ('<table class="wikitable"><tr><th>Date</th><th colspan="2">Added</th>'
            '<th colspan="2">Removed</th><th>Reason</th></tr>'
            '<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>'
            +rows+'</table>')


def test_wikipedia_removed_company_name_and_ticker():
    r = changes_from_wikipedia(fixture_html())
    assert len(r) == 44
    assert r[1].ticker == 'AGN' and r[1].name == 'Allergan plc'
    assert r[1].source_kind == 'WIKIPEDIA_REMOVAL'
    assert r[1].source == WIKIPEDIA_URL


def test_monthly_snapshots_are_explicitly_retrospective_and_not_cik_identity():
    raw = '\n'.join(f'T{i}\tCompany {i}\t0.2' for i in range(500))
    r = names_from_month(raw, '201601')
    assert len(r) == 500
    assert r[0].source_kind == 'RETROSPECTIVE_MONTHLY_NAME_ONLY'
    assert MONTHLY_SHA in r[0].source


def test_month_names_only_match_membership_window():
    names = [NameHint('AGN','Allergan','source','RETROSPECTIVE_MONTHLY_NAME_ONLY','201601'),
             NameHint('AGN','Future name','source','RETROSPECTIVE_MONTHLY_NAME_ONLY','202501')]
    spans = [{'ticker':'AGN','start_date':'2015-01-01','end_date':'2020-01-01'}]
    selected=names_for_unresolved(names, {'AGN'}, spans)
    assert [v.name for v in selected['AGN']] == ['Allergan']


def test_multiple_names_per_ticker_return_unverified_cik_candidates():
    filings=[{'cik':111,'name':'ALLERGAN PLC /IE/'},
             {'cik':222,'name':'ALLERGAN INC /DE/'}]
    hits=index_matches(iter(filings), {'AGN':['Allergan plc','Allergan Inc.']})
    assert set(hits['AGN']) == {111,222}


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def test_auto_hint_pipeline_but_no_auto_verification(tmp_path, monkeypatch):
    annual=[{'year':year,'memberships':1} for year in range(2015,2027)]
    per=[{'year':year,'ticker':'AGN','cik_status':'unresolved_current_ticker_or_rename', 'cik_candidate':''}
         for year in range(2015,2027)]
    history=[{'ticker':'AGN','start_date':'2015-01-01','end_date':'2020-01-01'}]
    a,p,h=[tmp_path/n for n in ('annual.csv','per.csv','history.csv')]
    write_csv(a,annual);write_csv(p,per);write_csv(h,history)
    monkeypatch.setattr(HistoricalNameSources, 'collect',
        lambda self,**kwargs: [NameHint('AGN','Allergan plc',WIKIPEDIA_URL,'WIKIPEDIA_REMOVAL','2020-01-01')])
    opts=SimpleNamespace(annual=a,per_security=p,memberships=h,hints=tmp_path/'missing.csv',
         verified_crosswalk=tmp_path/'verified.csv',output=tmp_path/'output',cache_dir=tmp_path/'cache',
         historical_names=True,offline=True,skip_wikipedia=False,skip_monthly=True,snapshot_step_months=3,
         max_symbols=0,offset=0,skip_indexes=False,index_years='2018',index_quarters='1',
         name_match_threshold=0.92,max_candidates=5)
    class Client:
        cache_hits=0;network_requests=0
        def master_index(self,*args):
            return iter([{'cik':111,'name':'ALLERGAN PLC/IE/', 'source':'https://sec.gov/index','form':'10-K'}])
        def submissions(self,cik):
            return {'name':'Allergan','formerNames':[],'tickers':[]}
    meta=build12.build(opts,Client())
    assert meta['total_unique_unresolved_from_all_annual_snapshots']==1
    assert meta['unique_unresolved_with_name_hints']==1
    assert meta['sec_name_candidates_not_verified']==1
    with (tmp_path/'output'/'candidate_crosswalk.csv').open(newline='') as fh:
        data=list(csv.DictReader(fh))
    assert data[0]['verification_status']=='candidate_unverified'
    assert data[0]['cik']=='0000000111'
    with (tmp_path/'output'/'unresolved_review_queue.csv').open(newline='') as fh:
        queue=list(csv.DictReader(fh))
    assert queue[0]['company_name_hint']=='Allergan plc'


def test_wikipedia_schema_failure_is_reported_and_no_silent_fabrication(tmp_path):
    class Session:
        def __init__(self): self.headers={}
        def get(self,url,timeout):
            class Response:
                content=b'<html><body>changed markup</body></html>'
                def raise_for_status(self): pass
            return Response()
    source=HistoricalNameSources(tmp_path,session=Session())
    assert source.collect(wikipedia=True, monthly=False)==[]
    assert 'Wikipedia' in source.errors[0]


def test_complete_month_schedule():
    assert len(months(1))==124
    assert months(1)[0]=='201601' and months(1)[-1]=='202604'
    assert 30 < len(months(3)) < 50