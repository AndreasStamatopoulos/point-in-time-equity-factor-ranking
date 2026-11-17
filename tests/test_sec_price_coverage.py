import importlib.util
from pathlib import Path
SPEC=importlib.util.spec_from_file_location('mod',Path(__file__).resolve().parents[1]/'11_collect_sec_price_audit.py')
mod=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(mod)

def test_historical_membership_and_reentry():
    intervals=[{'ticker':'AAA','start_date':'2010-01-01','end_date':'2017-01-01'},
               {'ticker':'AAA','start_date':'2020-01-01','end_date':''},
               {'ticker':'BBB','start_date':'2017-01-01','end_date':''}]
    assert mod.members_at(intervals,'2015-12-31')==['AAA']
    assert mod.members_at(intervals,'2018-12-31')==['BBB']
    assert mod.members_at(intervals,'2020-12-31')==['AAA','BBB']

def test_sec_cik_normalization_not_historical_assertion():
    raw={'0':{'cik_str':320193,'ticker':'AAPL','title':'Apple'},'1':{'cik_str':1067983,'ticker':'BRK-B'}}
    idx=mod.make_ticker_index(raw)
    assert idx['AAPL']=={320193}
    assert idx['BRK.B']=={1067983}

def test_companyfacts_vintage_uses_filed_not_period_only():
    obj={'facts':{'us-gaap':{'Assets':{'units':{'USD':[
       {'form':'10-K','filed':'2021-03-01','end':'2020-12-31','val':100},
       {'form':'10-K/A','filed':'2022-03-01','end':'2020-12-31','val':120}]}}}}}
    before=mod.concept_flags(obj,'2020-12-31')
    after=mod.concept_flags(obj,'2021-12-31')
    assert before['assets'] is False
    assert after['assets'] is True
    assert after['operating_income'] is False

def test_duplicate_rejected():
    d=[{'ticker':'AAA','start_date':'2015-01-01','end_date':''}]*2
    try:mod.members_at(d,'2016-01-01')
    except ValueError:pass
    else:raise AssertionError('duplicate not blocked')