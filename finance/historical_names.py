"""Unverified historical ticker -> issuer-name hints; NEVER a ticker/CIK proof.

Sources:
* Wikipedia 'Selected changes' table (removed/added ticker + name pairs)
* Reconstructed monthly SP500 constituents (2016-01 to 2026-04), pinned commit.

NOTE: the monthly dataset retroactively replaces some historical tickers with
current ticker symbols (e.g. ELV in 2016 rather than ANTM); its name is *not*
proof that a symbol traded in a particular month. This is a SEARCH-HINT ONLY.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
import hashlib
import re
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

WIKIPEDIA_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
MONTHLY_REPO = 'pierrebrunelle/sp500-historical-constituents'
MONTHLY_SHA = '58269b88822abd549b8f8642f0d71ca9be1353da'
MONTHLY_URL = ('https://raw.githubusercontent.com/' + MONTHLY_REPO + '/' + MONTHLY_SHA
               + '/data/spy{yyyymm}')


@dataclass(frozen=True)
class NameHint:
    ticker: str
    name: str
    source: str
    source_kind: str
    observed_date: str = ''


def norm_symbol(text: str) -> str:
    return text.strip().upper().replace('-', '.').replace('/', '.')


def _clean(text: str) -> str:
    # Drop Wikipedia footnote superscripts without stripping name content.
    return re.sub(r'\s+', ' ', re.sub(r'\[[^]]*\]', '', text)).strip()


def changes_from_wikipedia(html: str) -> list[NameHint]:
    """Parse only structured Add/Remove changes, never unrelated tables.

    Handles standard 6-column rows and a rowspan date cell (5-column rows).
    Every row is search evidence, not an authenticated old-ticker/security join.
    """
    soup = BeautifulSoup(html, 'html.parser')
    matched_table = None
    for table in soup.find_all('table'):
        header_text = ' '.join(x.get_text(' ', strip=True).lower()
                               for x in table.find_all('th', limit=18))
        if all(t in header_text for t in ('added', 'removed', 'ticker', 'security')):
            matched_table = table
            break
    if matched_table is None:
        raise ValueError('S&P 500 Added/Removed ticker/security table not found; Wikipedia layout changed')
    observed = []
    last_date = ''
    for tr in matched_table.find_all('tr'):
        vals = [_clean(td.get_text(' ', strip=True)) for td in tr.find_all('td', recursive=False)]
        if not vals:
            continue
        # Standard table: Date, added ticker, added security, removed ticker,
        # removed security, reason. Date omitted where HTML rowspan applies.
        if len(vals) >= 6:
            when, add_t, add_name, rem_t, rem_name = vals[:5]
            last_date = when or last_date
        elif len(vals) == 5 and last_date:
            add_t, add_name, rem_t, rem_name = vals[:4]
            when = last_date
        else:
            continue
        for ticker, name, kind in ((add_t, add_name, 'WIKIPEDIA_ADDITION'),
                                   (rem_t, rem_name, 'WIKIPEDIA_REMOVAL')):
            ticker = norm_symbol(ticker)
            # The published table often uses blank cells when no addition/removal.
            if not re.fullmatch(r'[A-Z][A-Z0-9.]{0,7}', ticker) or not name:
                continue
            observed.append(NameHint(ticker=ticker, name=name, source=WIKIPEDIA_URL,
                                     source_kind=kind, observed_date=when))
    if len(observed) < 20:
        raise ValueError('Suspiciously few Wikipedia constituent-change rows parsed')
    return observed


def names_from_month(text: str, yyyymm: str) -> list[NameHint]:
    """Read constituent-month names; do NOT treat retroactive tickers as actual PIT symbols."""
    result = []
    for line in text.splitlines():
        cols = line.split('\t')
        if len(cols) < 2:
            continue
        ticker, name = norm_symbol(cols[0]), _clean(cols[1])
        if re.fullmatch(r'[A-Z][A-Z0-9.]{0,7}', ticker) and name:
            result.append(NameHint(ticker, name, MONTHLY_URL.format(yyyymm=yyyymm),
                                   'RETROSPECTIVE_MONTHLY_NAME_ONLY', yyyymm))
    if len(result) < 470:
        raise ValueError(f'Invalid monthly SP500 snapshot for {yyyymm}: {len(result)} rows')
    return result


def months(step: int = 3) -> list[str]:
    if step not in (1, 3, 6, 12):
        raise ValueError('snapshot step must be 1, 3, 6 or 12')
    result = []
    for year in range(2016, 2027):
        for month in range(1, 13):
            if (year, month) > (2026, 4):
                break
            if (month - 1) % step == 0:
                result.append(f'{year}{month:02d}')
    return result


class HistoricalNameSources:
    def __init__(self, cache_dir: Path, *, offline=False, session=None):
        self.cache_dir = Path(cache_dir)
        self.offline = offline
        self.session = session or requests.Session()
        self.session.headers.update({'User-Agent': 'MLFinance-Research/0.5 (historical-name audit)',
                                     'Accept': 'text/html,text/plain,*/*'})
        self.errors: list[str] = []
        self.network_requests = 0
        self.cache_hits = 0

    def get(self, url: str, filename: str) -> str:
        path = self.cache_dir / 'historical_names' / filename
        if path.exists():
            self.cache_hits += 1
            return path.read_text(encoding='utf-8')
        if self.offline:
            raise FileNotFoundError(f'Offline and no historical-name cache: {path}')
        response = self.session.get(url, timeout=(10, 45))
        self.network_requests += 1
        response.raise_for_status()
        body = response.content.decode('utf-8')
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + '.part')
        partial.write_text(body, encoding='utf-8')
        partial.replace(path)
        return body

    def collect(self, *, wikipedia=True, monthly=True, monthly_step=3) -> list[NameHint]:
        found = []
        if wikipedia:
            try:
                body = self.get(WIKIPEDIA_URL, 'wikipedia_sp500_changes.html')
                found.extend(changes_from_wikipedia(body))
            except Exception as exc:
                self.errors.append(f'Wikipedia: {type(exc).__name__}: {str(exc)[:180]}')
        if monthly:
            for mo in months(monthly_step):
                try:
                    body = self.get(MONTHLY_URL.format(yyyymm=mo), f'monthly/{mo}.tsv')
                    found.extend(names_from_month(body, mo))
                except Exception as exc:
                    self.errors.append(f'month {mo}: {type(exc).__name__}: {str(exc)[:170]}')
        return found


def names_for_unresolved(source_rows: Iterable[NameHint], unresolved: set[str],
                         intervals: list[dict]) -> dict[str, list[NameHint]]:
    """Limit hints to historical membership windows; never rewrite ticker aliases.

    For Wikipedia addition/removal ticker, require explicit ticker exact match.
    For the monthly source, require overlap between membership and observation
    month, but keep source labelled retrospective to avoid false PIT claims.
    """
    intervals_by_symbol = defaultdict(list)
    for row in intervals:
        t = norm_symbol(row.get('ticker',''))
        if t in unresolved:
            intervals_by_symbol[t].append((row['start_date'], row.get('end_date', '')))
    selected = defaultdict(dict)
    for h in source_rows:
        if h.ticker not in unresolved:
            continue
        if h.source_kind == 'RETROSPECTIVE_MONTHLY_NAME_ONLY':
            first = h.observed_date[:4] + '-' + h.observed_date[4:] + '-01'
            # Month-end inclusive vs membership stop uncertain; window overlap.
            if not any(start[:7] <= first[:7] and (not end or first[:7] <= end[:7])
                       for start, end in intervals_by_symbol.get(h.ticker, [])):
                continue
        selected[h.ticker][(h.name.casefold(), h.source_kind)] = h
    return {sym: list(items.values()) for sym, items in selected.items()}