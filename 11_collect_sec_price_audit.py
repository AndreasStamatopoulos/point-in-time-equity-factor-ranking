"""Exploratory free-source coverage probe (NOT a verified PIT equity panel).

Run from repository root after installing requirements-data.txt and exporting
SEC_USER_AGENT='Full Name real.email@domain'. Writes only audit/probe files.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


SEC_MAP = 'https://www.sec.gov/files/company_tickers.json'
SEC_FACTS = 'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json'
CONCEPTS = {
 'assets': ('us-gaap', ('Assets',)),
 'book_equity': ('us-gaap', ('StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest')),
 'operating_income': ('us-gaap', ('OperatingIncomeLoss',)),
 'operating_cash_flow': ('us-gaap', ('NetCashProvidedByUsedInOperatingActivities',)),
 'capex': ('us-gaap', ('PaymentsToAcquirePropertyPlantAndEquipment',)),
 'shares': ('dei', ('EntityCommonStockSharesOutstanding',)),
}

def read_memberships(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty historical membership source')
    return rows


def members_at(rows, at):
    tickers=[r['ticker'].strip().upper() for r in rows if r['start_date']<=at and (not r['end_date'] or r['end_date']>at)]
    if len(tickers)!=len(set(tickers)):
        raise ValueError('Overlapping duplicate ticker membership at '+at)
    return sorted(tickers)


def normalize(ticker):
    return ticker.upper().replace('-', '.')


def make_ticker_index(raw):
    # Accept both official SEC company_tickers.json and exchange mapping formats.
    arr = list(raw.values()) if isinstance(raw, dict) else raw
    out=defaultdict(set)
    for x in arr:
        if not isinstance(x, dict): continue
        ticker=x.get('ticker'); cik=x.get('cik_str', x.get('cik'))
        if ticker and cik is not None:
            out[normalize(str(ticker))].add(int(cik))
    return dict(out)


def concept_flags(facts, asof):
    """Raw tag availability only. NOT 5-feature validity or coherent fiscal periods."""
    groups=facts.get('facts', {})
    out={}
    for name,(namespace,tags) in CONCEPTS.items():
        present=False
        for tag in tags:
            for unit, vals in groups.get(namespace,{}).get(tag,{}).get('units',{}).items():
                if name=='shares' and unit!='shares': continue
                if name!='shares' and unit!='USD': continue
                for v in vals:
                    if v.get('filed','9999')<=asof and v.get('end','9999')<=asof and v.get('form') in ('10-K','10-Q','10-K/A','10-Q/A'):
                        present=True;break
                if present:break
            if present:break
        out[name]=present
    return out


def fetch_json(session,url,timeout=25):
    r=session.get(url,timeout=timeout)
    r.raise_for_status()
    return r.json()


def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def price_probe(ticker,start,end):
    """Free Yahoo/yfinance presence probe, NOT institutional adjusted total returns.

    Missing bars do NOT prove delisting. Current symbol can fail after rename.
    """
    try:
        import yfinance as yf
        frame=yf.Ticker(ticker.replace('.','-')).history(start=start,end=end,auto_adjust=False,actions=True,timeout=20)
        if frame.empty:
            return {'price_status':'no_bars_unverified','price_bars':0,'price_split_events':None,'price_dividend_events':None}
        return {'price_status':'bars_observed_unverified_for_pit', 'price_bars':len(frame),
                'price_split_events':int(frame['Stock Splits'].fillna(0).ne(0).sum()) if 'Stock Splits' in frame else None,
                'price_dividend_events':int(frame['Dividends'].fillna(0).ne(0).sum()) if 'Dividends' in frame else None}
    except Exception as e:
        return {'price_status':'provider_error','price_bars':None,'price_split_events':None,
                'price_dividend_events':None,'price_error':type(e).__name__+': '+str(e)[:180]}


def run(args):
    root=Path(__file__).resolve().parent
    history=read_memberships(root/'data/universe/historical/sp500_membership_intervals_fja.csv')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    years=list(range(2015,2027)); dates={y: (f'{y}-12-31' if y<2026 else '2026-10-31') for y in years}
    annual={y:members_at(history,dates[y]) for y in years}
    ua=os.environ.get('SEC_USER_AGENT','').strip()
    if not ua or '@' not in ua:
        raise SystemExit('Set SEC_USER_AGENT to a genuine identifying name and contact email (SEC fair-access policy).')
    import requests  # optional dependency: install requirements-data.txt
    session=requests.Session();session.headers.update({'User-Agent':ua,'Accept-Encoding':'gzip, deflate','Host':'www.sec.gov'})
    sec_error=None; sec_map={}
    try:
        js=fetch_json(session,SEC_MAP);sec_map=make_ticker_index(js)
        (out/'source_sec_tickers.json').write_text(json.dumps({'source':SEC_MAP,'retrieved_at_utc':date.today().isoformat(),
             'warning':'current SEC ticker aliases, NOT a historical CIK crosswalk'}),encoding='utf-8')
    except Exception as e:
        sec_error=type(e).__name__+': '+str(e)[:180]
    all_symbols=sorted(set().union(*annual.values()))
    resolved={t:next(iter(sec_map[normalize(t)])) for t in all_symbols if len(sec_map.get(normalize(t),set()))==1}
    if args.probe_selection=='historical_exits_first':
        prioritized=sorted(all_symbols,key=lambda t:(all(not(t in annual[y]) for y in range(2015,2022)),
             any(t in annual[y] for y in range(2022,2027)),t))
    else: prioritized=all_symbols
    selected=prioritized[:args.max_symbols] if args.max_symbols>0 else prioritized
    facts_cache={};prices_cache={}
    SEC_host='data.sec.gov'
    # Switch host by request for SEC XBRL endpoint.
    for ticker in selected:
        cik=resolved.get(ticker)
        if cik is not None and cik not in facts_cache:
            try:
                session.headers['Host']=SEC_host
                facts_cache[cik]=fetch_json(session,SEC_FACTS.format(cik=cik))
            except Exception as e:
                facts_cache[cik]={'_error':type(e).__name__+': '+str(e)[:180]}
            finally:session.headers.pop('Host',None)
            time.sleep(max(args.sec_delay,0.11))
        if args.prices:
            prices_cache[ticker]=price_probe(ticker,'2015-01-01','2026-11-01')
    details=[];summary=[]
    for y in years:
        symbols=annual[y]; flags_counter=Counter(); prices_counter=Counter(); unresolved=[]
        for ticker in symbols:
            candidate=sec_map.get(normalize(ticker),set());cik=next(iter(candidate)) if len(candidate)==1 else None
            status=('current_ticker_unique_candidate' if cik is not None else 'ambiguous' if len(candidate)>1 else
                'unresolved_current_ticker_or_rename' if sec_map else 'sec_ticker_download_blocked')
            if cik is None:unresolved.append(ticker)
            sec_data=facts_cache.get(cik)
            if ticker not in selected: fs='not_probed';flags={k:None for k in CONCEPTS}
            elif not cik:fs='unresolved_cik';flags={k:None for k in CONCEPTS}
            elif not sec_data or '_error' in sec_data:fs='sec_companyfacts_error_or_not_probed';flags={k:None for k in CONCEPTS}
            else:fs='fetched_raw_tag_presence_only';flags=concept_flags(sec_data,dates[y]);flags_counter.update(k for k,v in flags.items() if v)
            px=prices_cache.get(ticker,{'price_status':'not_probed'})
            prices_counter[px['price_status']]+=1
            details.append({'year':y,'decision_date':dates[y],'ticker':ticker,'cik_candidate':cik or '',
                'cik_status':status,'cik_historically_verified':False,'sec_facts_status':fs,
                **{f'raw_{k}_present':flags[k] for k in CONCEPTS},
                'price_status':px['price_status'],'price_bars':px.get('price_bars'),
                'possible_delisting':'UNDETERMINED','historical_sector':'UNAVAILABLE',
                'five_features_validity':'NOT_AUDITED'})
        count_status=Counter(r['cik_status'] for r in details if r['year']==y)
        summary.append({'year':y,'decision_date':dates[y],'memberships':len(symbols),
            'sec_current_ticker_candidates':count_status['current_ticker_unique_candidate'],
            'unresolved_cik_or_alias':len(unresolved),
            'sec_raw_tags_probed_ciks':len({r['cik_candidate'] for r in details if r['year']==y and r['sec_facts_status']=='fetched_raw_tag_presence_only'}),
            'price_bars_observed_unverified':prices_counter['bars_observed_unverified_for_pit'],
            'price_no_bars_unverified':prices_counter['no_bars_unverified'],
            'price_provider_errors':prices_counter['provider_error'],
            'price_not_probed':prices_counter['not_probed'],
            'delisted_stock_missing_prices_verified':'UNDETERMINED',
            'five_features_verified':'NOT_AUDITED',
            'historical_gics_available':False})
    save_csv(out/'annual_coverage.csv',summary,list(summary[0]))
    save_csv(out/'per_security_coverage.csv',details,list(details[0]))
    meta={'price_source':'yfinance Yahoo experimental; no guaranteed delisting or corporate-action coverage',
          'sec_source':SEC_MAP,'sec_source_error':sec_error,
          'source_of_membership':'pinned fja05680/sp500 intervals','probe_symbols':len(selected),
          'ticker_cik_mapping_status':'current aliases, historical crosswalk unverified',
          'sector_feature_pass':'not auditable without historical sector records + validated fundamentals and price adjustment',
          'delist_reason_audit':'not established by missing free price bars'}
    (out/'audit_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Audit files:',out,'Probe symbols:',len(selected))
    if sec_error:print('SEC ERROR:',sec_error)
    print('Historical dataset remains NOT ready for training. Missing is NOT the same as delisted.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='results/free_source_coverage')
    p.add_argument('--max-symbols',type=int,default=20)
    p.add_argument('--sec-delay',type=float,default=0.3)
    p.add_argument('--prices',action='store_true',help='Probe Yahoo daily history, not auditable total returns')
    p.add_argument('--probe-selection',choices=['historical_exits_first','alphabetical'],default='historical_exits_first')
    run(p.parse_args())
if __name__=='__main__': main()