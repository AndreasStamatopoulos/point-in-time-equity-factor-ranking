"""Offline no-leakage and identity safety tests for historical SEC crosswalk."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec11 = importlib.util.spec_from_file_location('audit11', ROOT / '11_collect_sec_price_audit.py')
audit11 = importlib.util.module_from_spec(spec11)
spec11.loader.exec_module(audit11)
spec12 = importlib.util.spec_from_file_location('build12', ROOT / '12_build_historical_crosswalk.py')
build12 = importlib.util.module_from_spec(spec12)
spec12.loader.exec_module(build12)
from finance.historical_identity import (SECClient, verified_crosswalk, find_verified,
                                        index_matches, COLUMNS)


def csv_file(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=columns)
        wr.writeheader(); wr.writerows(rows)


def test_unresolved_extracted_all_years_not_only_probed():
    yearly = [{'year':y,'memberships':2} for y in range(2015, 2027)]
    per = []
    for y in range(2015, 2027):
        per.extend([{'year':y, 'ticker':'OLD', 'cik_status':'unresolved_current_ticker_or_rename', 'cik_candidate':''},
                    {'year':y, 'ticker':'LIVE', 'cik_status':'current_ticker_unique_candidate', 'cik_candidate':'123'}])
    out = build12.unresolved_from_audit(per, yearly)
    assert out == {'OLD': set(range(2015, 2027))}
    assert len(out) == 1


def test_incomplete_per_security_audit_fails_closed():
    with pytest.raises(ValueError, match='Incomplete per_security'):
        build12.unresolved_from_audit([{'year':2015,'ticker':'A', 'cik_status':'unresolved', 'cik_candidate':''}],
                                      [{'year':2015,'memberships':503}])


def test_verified_crosswalk_rejects_candidate_and_ambiguous_security_id(tmp_path):
    path=tmp_path/'verified.csv'
    row=dict(zip(COLUMNS, ['issuer/share-class-77','100','OLD','2015-01-01','2021-01-01',
                           'Oldco','ticker rename','https://www.sec.gov/x','candidate_unverified']))
    csv_file(path,[row],COLUMNS)
    with pytest.raises(ValueError,match='non-verified'):
        verified_crosswalk(path)
    row['verification_status']='verified'; row['security_id']='OLD'
    csv_file(path,[row],COLUMNS)
    with pytest.raises(ValueError,match='security_id'):
        verified_crosswalk(path)


def test_verified_crosswalk_valid_dates_and_overlap(tmp_path):
    path=tmp_path/'verified.csv'
    row=dict(zip(COLUMNS, ['security-abc','100','OLD','2015-01-01','2021-01-01',
                           'Oldco','rename','https://www.sec.gov/x','verified']))
    csv_file(path,[row],COLUMNS)
    approved = verified_crosswalk(path)
    assert find_verified(approved,'OLD','2020-12-31')['cik']==100
    assert find_verified(approved,'OLD','2021-01-01') is None  # exclusive end
    row2={**row,'valid_from':'2020-12-31','valid_to':'2022-01-01','cik':'200'}
    csv_file(path,[row,row2],COLUMNS)
    with pytest.raises(ValueError,match='Overlapping'):
        verified_crosswalk(path)


def test_verified_crosswalk_precedes_current_sec_ticker(tmp_path):
    path=tmp_path/'verified.csv'
    row=dict(zip(COLUMNS,['share-class-abc','100','OLD','2015-01-01','2021-01-01',
                          'Oldco','renamed','https://www.sec.gov/x','verified']))
    csv_file(path,[row],COLUMNS)
    verified=verified_crosswalk(path)
    cmap={'OLD':{999}}
    assert audit11.resolve_identity('OLD','2018-12-31',cmap,verified)==(100,'verified_historical_cik',True,'share-class-abc')
    assert audit11.resolve_identity('OLD','2022-12-31',cmap,verified)==(999,'current_ticker_unique_candidate',False,'')


def test_name_match_is_candidate_only_and_proves_no_ticker():
    filings=[{'cik':123,'name':'ALLERGAN PLC','filed':'2016-03-01',
              'source':'https://www.sec.gov/Archives/x','form':'10-K','filename':'x'}]
    r=index_matches(iter(filings),{'AGN':'Allergan plc'})
    assert r['AGN'][123]['match_type']=='exact_company_name'
    assert r['AGN'][123]['cik']==123
    assert index_matches(iter(filings),{'XYZ':'Completely Different Firm'}) == {}


class Response:
    def __init__(self,content,status=200):
        self.content=content;self.status_code=status;self.headers={}
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError('HTTP '+str(self.status_code))

class Session:
    def __init__(self, body):
        self.headers={};self.calls=0;self.body=body
    def get(self,url,timeout):
        self.calls+=1
        return Response(self.body)


def test_sec_index_master_parse_cached_resumable(tmp_path):
    body=("Description: Master Index of EDGAR Dissemination Feed\n"
          "CIK|Company Name|Form Type|Date Filed|Filename\n"
          "-----------------------------------------------\n"
          "123|ALLERGAN PLC|10-K|2015-01-31|edgar/data/123/x.txt\n"
          "123|ALLERGAN PLC|S-1|2015-01-31|edgar/data/123/y.txt\n").encode()
    session=Session(body)
    client=SECClient(tmp_path,'Name contact@example.org',session=session,delay=0.12)
    rows=list(client.master_index(2015,1))
    assert len(rows)==1 and rows[0]['cik']==123
    assert session.calls==1
    offline=SECClient(tmp_path,'',offline=True,delay=0.12)
    assert list(offline.master_index(2015,1))==rows
    assert offline.cache_hits==1


def test_sec_submissions_persistent_by_cik(tmp_path):
    s=Session(b'{"cik": 123, "name": "Some Co", "formerNames": []}')
    c=SECClient(tmp_path,'Name contact@example.org',session=s,delay=0.12)
    assert c.submissions(123)['cik']==123
    assert c.submissions(123)['cik']==123
    assert s.calls==1
    assert SECClient(tmp_path,'',offline=True,delay=0.12).submissions(123)['name']=='Some Co'


def test_price_probe_cache_survives_retry(tmp_path, monkeypatch):
    calls=[]
    def price(t,s,e):
        calls.append(t)
        return {'price_status':'no_bars_unverified','price_bars':0}
    monkeypatch.setattr(audit11,'price_probe',price)
    p1,hit1=audit11._cached_price(tmp_path,'AABA',start='2015-01-01',end='2026-11-01',do_probe=True)
    p2,hit2=audit11._cached_price(tmp_path,'AABA',start='2015-01-01',end='2026-11-01',do_probe=True)
    assert p1==p2 and calls==['AABA']
    assert hit1 is False and hit2 is True
    assert p2['price_status']=='no_bars_unverified'  # not delisting proof


def test_build_review_queue_and_cik_candidate_never_auto_verified(tmp_path):
    annual=[{'year':y,'memberships':1} for y in range(2015,2027)]
    per=[{'year':y, 'ticker':'AGN','cik_status':'unresolved_current_ticker_or_rename',
          'cik_candidate':''} for y in range(2015,2027)]
    aa=tmp_path/'annual.csv'; pp=tmp_path/'per.csv'; hh=tmp_path/'history.csv'; hints=tmp_path/'hints.csv'
    csv_file(aa,annual,['year','memberships'])
    csv_file(pp,per,['year','ticker','cik_status','cik_candidate'])
    csv_file(hh,[{'ticker':'AGN','start_date':'2010-01-01','end_date':'2020-01-01'}],
             ['ticker','start_date','end_date'])
    csv_file(hints,[{'historical_ticker':'AGN','company_name_hint':'Allergan plc','cik_hint':'',
                     'hint_source':'https://www.sec.gov/edgar/search/'}],
                     ['historical_ticker','company_name_hint','cik_hint','hint_source'])
    from types import SimpleNamespace
    args=SimpleNamespace(annual=aa,per_security=pp,memberships=hh,hints=hints,
                         verified_crosswalk=tmp_path/'notyet.csv',output=tmp_path/'output',
                         max_symbols=0,offset=0,skip_indexes=False,index_years='2018',index_quarters='1',
                         name_match_threshold=.92,max_candidates=4)
    class FakeClient:
        cache_hits=0;network_requests=0
        def master_index(self,y,q):
            yield {'cik':1234,'name':'ALLERGAN PLC','source':'https://www.sec.gov/Archives/foo',
                   'score':0,'filed':'2018-01-02','filename':'foo','form':'10-K'}
        def submissions(self,cik):
            return {'name':'Allergan PLC','tickers':['AGN'],
                    'formerNames':[{'name':'Allergan plc','from':'2012-01-01','to':'2020-01-01'}]}
    meta=build12.build(args,FakeClient())
    assert meta['total_unique_unresolved_from_all_annual_snapshots']==1
    rows=list(csv.DictReader((args.output/'candidate_crosswalk.csv').open()))
    assert len(rows)==1
    assert rows[0]['cik']=='0000001234'
    assert rows[0]['verification_status']=='candidate_unverified'
    assert rows[0]['security_id']==''
    queue=list(csv.DictReader((args.output/'unresolved_review_queue.csv').open()))
    assert len(queue)==1 and queue[0]['historical_ticker']=='AGN'