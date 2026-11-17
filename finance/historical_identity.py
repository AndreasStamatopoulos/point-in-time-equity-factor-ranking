"""SEC identity tools. Never promote candidate ticker/CIK joins to verified automatically.

A CIK identifies the SEC registrant, not a permanent tradable share-class security.
SEC quarterly master.idx contains *company names and CIKs*, never ticker history.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests

COLUMNS = ("security_id", "cik", "historical_ticker", "valid_from", "valid_to",
           "company_name", "corporate_action", "mapping_source", "verification_status")
VERIFIED = "verified"
SEC_INDEX = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FILING_FORMS = frozenset(("10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A",
                          "20-F", "20-F/A", "40-F"))


def symbol(s: str) -> str:
    return s.strip().upper().replace("-", ".")


def normalize_company(name: str) -> str:
    name = re.sub(r"[^A-Z0-9 ]", " ", (name or "").upper())
    tokens = [x for x in name.split() if x not in {
        "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED",
        "PLC", "LLC", "THE", "HOLDING", "HOLDINGS", "GROUP", "LP", "SA", "NV"}]
    return " ".join(tokens)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")
        return list(reader)


def atomic_csv(path: Path, rows: list[dict], columns: list[str] | tuple[str, ...]):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    with part.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(part, path)


def _atomic(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_bytes(data)
    os.replace(part, path)


class SECClient:
    """Cache-by-source, rate-limited SEC HTTP client with resumable JSON/index access."""
    def __init__(self, cache_dir: Path, user_agent: str, *, delay: float = 0.4,
                 offline: bool = False, session=None, max_attempts: int = 3):
        if not offline and ("@" not in user_agent or not user_agent.strip()):
            raise ValueError("SEC_USER_AGENT must be a real name and contact email")
        if delay < 0.12:
            raise ValueError("SEC request delay must be >= 0.12s")
        self.cache_dir = Path(cache_dir)
        self.offline = offline
        self.delay = delay
        self.max_attempts = max_attempts
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent,
                                     "Accept-Encoding": "gzip, deflate", "Accept": "application/json, */*"})
        self._last_request = 0.0
        self.cache_hits = 0
        self.network_requests = 0

    def _request(self, url: str) -> bytes:
        if self.offline:
            raise FileNotFoundError(f"Offline and no cached SEC source for {url}")
        for attempt in range(self.max_attempts):
            delay = self.delay - (time.monotonic() - self._last_request)
            if delay > 0:
                time.sleep(delay)
            self._last_request = time.monotonic()
            self.network_requests += 1
            try:
                response = self.session.get(url, timeout=(10, 65))
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt + 1 == self.max_attempts:
                        response.raise_for_status()
                    retry_header = response.headers.get("Retry-After", "")
                    try:
                        wait = max(float(retry_header), self.delay * 2 ** (attempt + 1))
                    except ValueError:
                        wait = self.delay * 2 ** (attempt + 1)
                    time.sleep(min(max(wait, 1.0), 30.0))
                    continue
                response.raise_for_status()
                return response.content
            except (requests.Timeout, requests.ConnectionError):
                if attempt + 1 == self.max_attempts:
                    raise
                time.sleep(min(self.delay * 2 ** (attempt + 1), 10))
        raise RuntimeError(f"Could not retrieve {url}")

    def json(self, url: str, key: str) -> dict:
        path = self.cache_dir / "sec" / (key + ".json")
        if path.exists():
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    self.cache_hits += 1
                    return obj
            except (ValueError, OSError):
                path.unlink(missing_ok=True)
        raw = self._request(url)
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected SEC JSON object at {url}")
        _atomic(path, raw)
        return obj

    def submissions(self, cik: int) -> dict:
        return self.json(SEC_SUBMISSIONS.format(cik=int(cik)), f"submissions/CIK{int(cik):010d}")

    def companyfacts(self, cik: int) -> dict:
        return self.json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json",
                         f"companyfacts/CIK{int(cik):010d}")

    def master_index(self, year: int, quarter: int):
        if not 1994 <= year <= date.today().year or quarter not in (1, 2, 3, 4):
            raise ValueError("Invalid SEC index year/quarter")
        key = f"{year}Q{quarter}"
        path = self.cache_dir / "sec" / "master_indexes" / f"{key}.idx.gz"
        if path.exists():
            try:
                raw = gzip.decompress(path.read_bytes()).decode("latin-1")
                self.cache_hits += 1
            except (OSError, UnicodeError):
                path.unlink(missing_ok=True)
                raw = self._request(SEC_INDEX.format(year=year, q=quarter)).decode("latin-1")
                _atomic(path, gzip.compress(raw.encode("latin-1")))
        else:
            raw = self._request(SEC_INDEX.format(year=year, q=quarter)).decode("latin-1")
            if "CIK|Company Name|Form Type|Date Filed|Filename" not in raw:
                raise ValueError(f"Malformed SEC master.idx {key}: header not found")
            _atomic(path, gzip.compress(raw.encode("latin-1")))
        if "CIK|Company Name|Form Type|Date Filed|Filename" not in raw:
            raise ValueError(f"Malformed SEC master.idx cache: {key}")
        for line in raw.splitlines():
            parts = line.split("|", 4)
            if len(parts) != 5 or not parts[0].strip().isdigit():
                continue
            cik, name, form, filed, filename = (p.strip() for p in parts)
            if form not in FILING_FORMS:
                continue
            yield {"cik": int(cik), "name": name, "form": form,
                   "filed": filed, "filename": filename,
                   "source": SEC_INDEX.format(year=year, q=quarter)}


def _normalized_interval(row: dict) -> tuple[str, str, str]:
    ticker = symbol(row["historical_ticker"])
    start = (row.get("valid_from") or "").strip()
    end = (row.get("valid_to") or "").strip()
    if not ticker or not start:
        raise ValueError("verified crosswalk entry requires historical_ticker and valid_from")
    date.fromisoformat(start)
    if end:
        date.fromisoformat(end)
        if end <= start:
            raise ValueError("valid_to must be exclusive and strictly after valid_from")
    return ticker, start, end


def verified_crosswalk(path: Path | None) -> dict[str, list[dict]]:
    """Validate manually approved, evidenced joins. Never accept candidate rows as verified."""
    if path is None or not Path(path).exists():
        return {}
    entries = read_csv(path)
    verified: dict[str, list[dict]] = {}
    for row in entries:
        if any(col not in row for col in COLUMNS):
            raise ValueError(f"Verified crosswalk must have all columns {COLUMNS}")
        if row.get("verification_status", "").strip().lower() != VERIFIED:
            raise ValueError("Verified crosswalk contains a non-verified entry; candidates MUST NOT be used")
        ticker, start, end = _normalized_interval(row)
        cik = (row.get("cik") or "").strip()
        if not cik.isdigit() or int(cik) <= 0:
            raise ValueError(f"Missing/invalid CIK for {ticker}")
        # A symbol is not an ID; even SEC CIK alone is not a share-class ID.
        sid = (row.get("security_id") or "").strip()
        if not sid or sid.upper() in {ticker, cik, cik.zfill(10)}:
            raise ValueError(f"security_id must distinguish tradable security from ticker/CIK: {ticker}")
        source = (row.get("mapping_source") or "").strip()
        if not source.startswith("https://"):
            raise ValueError(f"Verified {ticker} requires an explicit https primary-source evidence URL")
        row = {**row, "historical_ticker": ticker, "cik": int(cik),
               "valid_from": start, "valid_to": end, "security_id": sid}
        verified.setdefault(ticker, []).append(row)
    # Any same-symbol overlapping intervals with different CIK/security need manual review.
    for ticker, group in verified.items():
        group.sort(key=lambda x: x["valid_from"])
        for earlier, later in zip(group, group[1:]):
            if not earlier["valid_to"] or earlier["valid_to"] > later["valid_from"]:
                raise ValueError(f"Overlapping verified identity intervals for {ticker}")
    return verified


def find_verified(verified: dict, ticker: str, at: str) -> dict | None:
    candidates = [x for x in verified.get(symbol(ticker), [])
                  if x["valid_from"] <= at and (not x["valid_to"] or at < x["valid_to"])]
    if len(candidates) > 1:
        raise ValueError(f"Multiple verified CIK joins for {ticker} as of {at}")
    return candidates[0] if candidates else None


def index_matches(rows: list[dict], hints: dict[str, str], *, threshold: float = 0.91):
    """Filing-name matches are CIK *candidates*, not proof of historical ticker."""
    results: dict[str, dict[int, dict]] = {}
    names = {sym: normalize_company(n) for sym, n in hints.items() if n}
    for filing in rows:
        reg_name = normalize_company(filing["name"])
        if not reg_name:
            continue
        for sym, hint in names.items():
            if not hint:
                continue
            ratio = SequenceMatcher(None, hint, reg_name).ratio()
            if hint == reg_name or ratio >= threshold:
                match = {**filing, "match_type": "exact_company_name" if hint == reg_name else "similar_company_name",
                         "score": round(ratio, 5)}
                prior = results.setdefault(sym, {}).get(filing["cik"])
                if prior is None or match["score"] > prior["score"]:
                    results[sym][filing["cik"]] = match
    return results