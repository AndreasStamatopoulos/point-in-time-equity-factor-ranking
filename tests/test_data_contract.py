import pandas as pd
import pytest
from finance.data_contract import (DatasetIntegrityError, audit_history,
                                   eligible_asof, total_return, asof_filings)


def tables(n=100):
    ids = [f'S{i:03d}' for i in range(n)]
    m = pd.DataFrame({'security_id':ids, 'ticker':ids, 'cik':range(100,100+n),
                      'sector':[f'Sector{i%11:02}' for i in range(n)],
                      'valid_from':['2020-01-01']*n, 'valid_to':[None]*n,
                      'membership_known_at':['2019-12-30']*n,
                      'source':['vendor']*n, 'source_type':['historical_membership_events']*n})
    p=[]
    for sid in ids:
        for date in ['2020-01-02','2020-02-03']:
            p.append(dict(security_id=sid,date=date,close_unadjusted=100.,
                          close_split_adjusted=100.,total_return_index=100.,
                          split_ratio=1.,cash_dividend=0.,delisted=False,
                          terminal_return_included=False,source='vendor'))
    f=pd.DataFrame(dict(security_id=ids,period_end=['2019-09-30']*n,
        accepted_at=['2019-11-05']*n,accession=[f'ACC{i}' for i in range(n)],
        book_equity=[1000.]*n,assets=[2000.]*n,operating_income=[100.]*n,
        operating_cash_flow=[200.]*n,capex=[40.]*n,shares_outstanding=[100.]*n,
        shares_asof=['2019-09-30']*n,source=['vendor']*n))
    return m,pd.DataFrame(p),f


def test_valid_pit_audit():
    m,p,f=tables()
    x=audit_history(m,p,f,asof='2021-01-01')
    assert (x.securities,x.sectors)==(100,11)


def test_current_snapshot_prohibited():
    m,p,f=tables();m.loc[0,'source_type']='current_snapshot'
    with pytest.raises(DatasetIntegrityError,match='NOT current'):
        audit_history(m,p,f,asof='2021-01-01')
    with pytest.raises(DatasetIntegrityError,match='FORWARD-ONLY'):
        eligible_asof(m,'2020-01-31')


def test_late_filing_does_not_leak():
    m,p,f=tables(n=1)
    f.loc[0,'accepted_at']='2020-01-31'
    assert asof_filings(f,'2020-01-31').empty
    assert len(asof_filings(f,'2020-02-02'))==1


def test_delisting_without_proceeds_rejected():
    m,p,f=tables();p.loc[(p.security_id=='S000') & (p.date=='2020-02-03'),'delisted']=True
    with pytest.raises(DatasetIntegrityError,match='Delisting'):
        audit_history(m,p,f,asof='2021-01-01')


def test_stock_split_good_and_bad():
    m,p,f=tables()
    idx=(p.security_id=='S000') & (p.date=='2020-01-02')
    p.loc[idx,'close_unadjusted']=400
    p.loc[idx,'close_split_adjusted']=100
    idx2=(p.security_id=='S000') & (p.date=='2020-02-03')
    p.loc[idx2,'split_ratio']=4
    audit_history(m,p,f,asof='2021-01-01')
    p.loc[idx2,'split_ratio']=1
    with pytest.raises(DatasetIntegrityError,match='Split-only'):
        audit_history(m,p,f,asof='2021-01-01')


def test_missing_terminal_price_must_not_forward_fill():
    _,p,_=tables(n=1)
    with pytest.raises(DatasetIntegrityError,match='missing entry/exit'):
        total_return(p,'S000','2020-01-02','2020-03-02')


def test_vendor_total_return_includes_dividend():
    _,p,_=tables(n=1)
    p.loc[p.date=='2020-02-03','total_return_index']=101.25
    assert total_return(p,'S000','2020-01-02','2020-02-03')==pytest.approx(.0125)


def test_membership_future_not_backdated():
    m,_,_=tables(n=1)
    assert eligible_asof(m,'2019-12-31').empty
    assert len(eligible_asof(m,'2020-01-31'))==1


def test_membership_known_late_rejected():
    m,p,f=tables();m.loc[0,'membership_known_at']='2020-02-01'
    with pytest.raises(DatasetIntegrityError,match='became known after'):
        audit_history(m,p,f,asof='2021-01-01')


def test_bankruptcy_total_loss_supported_when_final_return_certified():
    m,p,f=tables(n=1)
    i=p.date=='2020-02-03'
    p.loc[i,'total_return_index']=0
    p.loc[i,'delisted']=True
    p.loc[i,'terminal_return_included']=True
    # The dataset auditor allows -100% terminal returns. Raw close still requires
    # positive observation; vendors can carry non-tradable post-exit cash separately.
    assert total_return(p,'S000','2020-01-02','2020-02-03')==-1


def test_momentum_skips_most_recent_month():
    from finance.pit_panel import _momentum
    import numpy as np
    months=pd.date_range('2019-10-31','2020-11-30',freq='ME')
    p=pd.DataFrame({'security_id':['S000']*len(months),'date':months,
                    'total_return_index':np.linspace(100,130,len(months))})
    # decision Nov 2020 => old Oct 2019, recent Sept 2020; Oct 2020 ignored.
    result=_momentum(p,'S000',pd.Timestamp('2020-11-30'))
    assert result==pytest.approx(p.loc[p.date.eq('2020-09-30'),'total_return_index'].iloc[0]/100-1)
    p.loc[p.date.eq('2020-10-31'),'total_return_index']=9999
    assert _momentum(p,'S000',pd.Timestamp('2020-11-30'))==pytest.approx(result)