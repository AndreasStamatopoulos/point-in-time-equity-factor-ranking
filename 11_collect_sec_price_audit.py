"""Exploratory free-source SEC/price coverage audit (NOT a verified PIT panel).

Historical identity precedence: MANUALLY verified crosswalk for a date, then
current SEC ticker directory as unverified CIK candidate, otherwise unresolved.
Caches SEC CompanyFacts by CIK and yfinance probes by symbol. Supports batches
with --offset/--max-symbols; cached probes survive interruption and reruns.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from finance.historical_identity import SECClient, find_verified, verified_crosswalk

ROOT = Path(__file__).resolve().parent
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
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Empty historical membership source')
    return rows


def members_at(rows, at):
    tickers = [r['ticker'].strip().upper() for r in rows
               if r['start_date'] <= at and (not r['end_date'] or r['end_date'] > at)]
    if len(tickers) != len(set(tickers)):
        raise ValueError('Overlapping duplicate ticker membership at ' + at)
    return sorted(tickers)


def normalize(ticker):
    return ticker.upper().replace('-', '.')


def make_ticker_index(raw):
    if isinstance(raw, dict) and 'fields' in raw and 'data' in raw:
        arr = [dict(zip(raw['fields'], row)) for row in raw['data']]
    else:
        arr = list(raw.values()) if isinstance(raw, dict) else raw
    out = defaultdict(set)
    for x in arr:
        if not isinstance(x, dict):
            continue
        ticker, cik = x.get('ticker'), x.get('cik_str', x.get('cik'))
        if ticker and cik is not None:
            out[normalize(str(ticker))].add(int(cik))
    return dict(out)


def concept_flags(facts, asof):
    """Raw tag availability only. NOT 5-feature validity or coherent fiscal periods."""
    groups = facts.get('facts', {})
    out = {}
    for name, (namespace, tags) in CONCEPTS.items():
        present = False
        for tag in tags:
            for unit, vals in groups.get(namespace, {}).get(tag, {}).get('units', {}).items():
                if name == 'shares' and unit != 'shares':
                    continue
                if name != 'shares' and unit != 'USD':
                    continue
                for v in vals:
                    if (v.get('filed', '9999') <= asof and v.get('end', '9999') <= asof and
                            v.get('form') in ('10-K', '10-Q', '10-K/A', '10-Q/A')):
                        present = True
                        break
                if present:
                    break
            if present:
                break
        out[name] = present
    return out


def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    with temp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def price_probe(ticker, start, end):
    """Yahoo presence probe only; no certified total returns or delisting proceeds."""
    try:
        import yfinance as yf
        frame = yf.Ticker(ticker.replace('.', '-')).history(
            start=start, end=end, auto_adjust=False, actions=True, timeout=20)
        if frame.empty:
            return {'price_status': 'no_bars_unverified', 'price_bars': 0,
                    'price_split_events': None, 'price_dividend_events': None}
        return {'price_status': 'bars_observed_unverified_for_pit', 'price_bars': len(frame),
                'price_split_events': int(frame['Stock Splits'].fillna(0).ne(0).sum()) if 'Stock Splits' in frame else None,
                'price_dividend_events': int(frame['Dividends'].fillna(0).ne(0).sum()) if 'Dividends' in frame else None}
    except Exception as exc:
        return {'price_status': 'provider_error', 'price_bars': None,
                'price_split_events': None, 'price_dividend_events': None,
                'price_error': f'{type(exc).__name__}: {str(exc)[:180]}'}


def _cached_price(cache_dir: Path, ticker: str, *, start: str, end: str,
                  do_probe: bool, refresh: bool = False):
    safe = re.sub(r'[^A-Z0-9_.-]', '_', ticker.upper())
    path = cache_dir / 'yfinance' / f'{safe}.json'
    if path.exists() and not refresh:
        try:
            wrapper = json.loads(path.read_text(encoding='utf-8'))
            if wrapper['ticker'] == ticker and wrapper['start'] == start and wrapper['end'] == end:
                return wrapper['result'], True
        except (ValueError, KeyError, OSError):
            pass
    if not do_probe:
        return {'price_status': 'not_probed'}, False
    result = price_probe(ticker, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps({'ticker': ticker, 'start': start, 'end': end, 'result': result},
                               indent=2), encoding='utf-8')
    temp.replace(path)
    return result, False


def resolve_identity(ticker, at, sec_map, crosswalk, *, sec_error=None):
    """Verified join beats present-day ticker match. Date-specific and fail-closed."""
    row = find_verified(crosswalk, ticker, at)
    if row is not None:
        return int(row['cik']), 'verified_historical_cik', True, row['security_id']
    candidates = sec_map.get(normalize(ticker), set())
    if len(candidates) == 1:
        return next(iter(candidates)), 'current_ticker_unique_candidate', False, ''
    if len(candidates) > 1:
        return None, 'ambiguous', False, ''
    if sec_error and not sec_map:
        return None, 'sec_ticker_download_blocked', False, ''
    return None, 'unresolved_current_ticker_or_rename', False, ''


def run(args):
    history = read_memberships(ROOT / 'data/universe/historical/sp500_membership_intervals_fja.csv')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)
    years = list(range(2015, 2027))
    dates = {y: (f'{y}-12-31' if y < 2026 else '2026-10-31') for y in years}
    annual = {y: members_at(history, dates[y]) for y in years}
    # Optional crosswalk file has a template in Git; the real approval CSV stays local.
    crosswalk = verified_crosswalk(Path(args.verified_crosswalk))
    ua = os.environ.get('SEC_USER_AGENT', '').strip()
    client = SECClient(cache, ua, delay=args.sec_delay, offline=getattr(args, 'offline', False))
    sec_error = None
    sec_map = {}
    try:
        if args.refresh:
            (cache / 'sec' / 'current_company_tickers.json').unlink(missing_ok=True)
        sec_map = make_ticker_index(client.json(SEC_MAP, 'current_company_tickers'))
    except Exception as exc:
        sec_error = f'{type(exc).__name__}: {str(exc)[:180]}'
    all_symbols = sorted(set().union(*annual.values()))
    if args.probe_selection == 'historical_exits_first':
        prioritized = sorted(all_symbols, key=lambda t: (
            all(t not in annual[y] for y in range(2015, 2022)),
            any(t in annual[y] for y in range(2022, 2027)), t))
    else:
        prioritized = all_symbols
    selected = (prioritized[args.offset: args.offset + args.max_symbols]
                if args.max_symbols > 0 else prioritized[args.offset:])
    selected_set = set(selected)
    facts_cache = {}
    facts_errors = {}
    price_status = {}
    price_hits = 0
    # Probe unique issuer CIKs only; verified mapping takes precedence for each date.
    for ticker in selected:
        historical_ids = {resolve_identity(ticker, dates[y], sec_map, crosswalk, sec_error=sec_error)[0]
                          for y in years if ticker in annual[y]}
        for cik in sorted(x for x in historical_ids if x is not None):
            if cik in facts_cache:
                continue
            try:
                if args.refresh:
                    (cache / 'sec' / 'companyfacts' / f'CIK{cik:010d}.json').unlink(missing_ok=True)
                facts_cache[cik] = client.companyfacts(cik)
            except Exception as exc:
                facts_errors[cik] = f'{type(exc).__name__}: {str(exc)[:200]}'
        price_status[ticker], hit = _cached_price(cache, ticker,
            start='2015-01-01', end='2026-11-01', do_probe=bool(args.prices), refresh=args.refresh)
        price_hits += int(hit)

    # Past completed batches continue to contribute to the new output.
    for ticker in all_symbols:
        if ticker not in price_status:
            price_status[ticker], hit = _cached_price(cache, ticker,
                start='2015-01-01', end='2026-11-01', do_probe=False)
            price_hits += int(hit)
    details, summary = [], []
    for y in years:
        symbols = annual[y]
        prices_counter = Counter()
        unresolved_count = 0
        for ticker in symbols:
            cik, status, historic_verified, security_id = resolve_identity(
                ticker, dates[y], sec_map, crosswalk, sec_error=sec_error)
            if cik is None:
                unresolved_count += 1
            facts = facts_cache.get(cik)
            facts_path = cache / 'sec' / 'companyfacts' / f'CIK{cik:010d}.json' if cik else None
            if facts is None and facts_path and facts_path.exists():
                try:
                    facts = json.loads(facts_path.read_text(encoding='utf-8'))
                    facts_cache[cik] = facts
                except (ValueError, OSError):
                    pass
            if cik is None:
                fs = 'unresolved_cik'
            elif facts is not None:
                fs = 'fetched_raw_tag_presence_only'
            elif cik in facts_errors:
                fs = 'sec_companyfacts_error'
            else:
                fs = 'not_probed'
            flags = concept_flags(facts, dates[y]) if facts is not None else {k: None for k in CONCEPTS}
            px = price_status[ticker]
            prices_counter[px['price_status']] += 1
            details.append({'year': y, 'decision_date': dates[y], 'ticker': ticker,
                'security_id': security_id, 'cik_candidate': cik or '',
                'cik_status': status, 'cik_historically_verified': historic_verified,
                'sec_facts_status': fs, **{f'raw_{k}_present': flags[k] for k in CONCEPTS},
                'price_status': px['price_status'], 'price_bars': px.get('price_bars'),
                'possible_delisting': 'UNDETERMINED', 'historical_sector': 'UNAVAILABLE',
                'five_features_validity': 'NOT_AUDITED'})
        current_candidates = sum(r['cik_status'] == 'current_ticker_unique_candidate'
                                 for r in details if r['year'] == y)
        verified_count = sum(r['cik_historically_verified'] for r in details if r['year'] == y)
        n_raw = len({r['cik_candidate'] for r in details if r['year'] == y
                     and r['sec_facts_status'] == 'fetched_raw_tag_presence_only'})
        summary.append({'year': y, 'decision_date': dates[y], 'memberships': len(symbols),
            'sec_current_ticker_candidates': current_candidates,
            'verified_historical_cik': verified_count,
            'unresolved_cik_or_alias': unresolved_count,
            'sec_raw_tags_probed_ciks': n_raw,
            'price_bars_observed_unverified': prices_counter['bars_observed_unverified_for_pit'],
            'price_no_bars_unverified': prices_counter['no_bars_unverified'],
            'price_provider_errors': prices_counter['provider_error'],
            'price_not_probed': prices_counter['not_probed'],
            'delisted_stock_missing_prices_verified': 'UNDETERMINED',
            'five_features_verified': 'NOT_AUDITED',
            'historical_gics_available': False})
    save_csv(out / 'annual_coverage.csv', summary, list(summary[0]))
    save_csv(out / 'per_security_coverage.csv', details, list(details[0]))
    meta = {'price_source': 'yfinance Yahoo experimental; no guaranteed delisting or corporate-action coverage',
        'sec_source': SEC_MAP, 'sec_source_error': sec_error,
        'source_of_membership': 'pinned fja05680/sp500 intervals',
        'probe_symbols_this_run': len(selected), 'probe_offset': args.offset,
        'cached_price_probes_including_earlier_batches': price_hits,
        'cached_sec_json_reads_or_cache_hits': client.cache_hits,
        'sec_network_requests': client.network_requests, 'companyfacts_errors': facts_errors,
        'verified_crosswalk_path': str(args.verified_crosswalk),
        'ticker_cik_mapping_status': 'manual verified_crosswalk first; current aliases unverified',
        'sector_feature_pass': 'not auditable without historical sector records + validated fundamentals and price adjustment',
        'delist_reason_audit': 'not established by missing free price bars'}
    (out / 'audit_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('Audit files:', out, 'symbols this batch:', len(selected), 'offset:', args.offset)
    if sec_error:
        print('SEC map error:', sec_error)
    print('WARNING: exploratory only; not a verified PIT dataset; no delisting conclusion')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=str(ROOT / 'results/free_source_coverage'))
    p.add_argument('--cache-dir', default=str(ROOT / 'data/cache'))
    p.add_argument('--verified-crosswalk', default=str(ROOT / 'data/universe/historical/verified_crosswalk.csv'))
    p.add_argument('--max-symbols', type=int, default=20, help='0=all unique historical symbols')
    p.add_argument('--offset', type=int, default=0, help='Batch offset; caches survive subsequent batches')
    p.add_argument('--sec-delay', type=float, default=0.4)
    p.add_argument('--prices', action='store_true', help='Yahoo presence only, not auditable total return')
    p.add_argument('--offline', action='store_true', help='Use previously cached SEC files only')
    p.add_argument('--refresh', action='store_true', help='Refetch selected sources, replacing cache (do not use during resume)')
    p.add_argument('--probe-selection', choices=['historical_exits_first', 'alphabetical'], default='historical_exits_first')
    args = p.parse_args()
    if args.max_symbols < 0 or args.offset < 0:
        p.error('max-symbols/offset must be >=0')
    run(args)


if __name__ == '__main__':
    main()