"""Generate a reviewable historical ticker-to-CIK candidate crosswalk.

SEC EDGAR master.idx gives filer name+CIK (NOT ticker); SEC submissions gives
former legal names and *current* ticker(s). Both can corroborate candidates,
never automatically verify an historical ticker mapping or share-class ID.

Requires full per_security_coverage.csv from step 11 and annual_coverage.csv.
Run with --offline for an inventory-only dry run, --index-years for SEC lookup.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from finance.historical_names import (HistoricalNameSources, names_for_unresolved)
from finance.historical_identity import (COLUMNS, SECClient, atomic_csv,
    find_verified, index_matches, normalize_company, read_csv, symbol,
    verified_crosswalk)

ROOT = Path(__file__).resolve().parent
DEFAULT_COVERAGE = ROOT / "results/free_source_coverage"
DEFAULT_HISTORY = ROOT / "data/universe/historical/sp500_membership_intervals_fja.csv"
DEFAULT_HINTS = ROOT / "data/universe/historical/ticker_name_hints.csv"
DEFAULT_VERIFIED = ROOT / "data/universe/historical/verified_crosswalk.csv"


def unresolved_from_audit(per_security: list[dict], annual: list[dict],
                          *, first_year: int = 2015, last_year: int = 2026) -> dict[str, set[int]]:
    """Inventory every distinct unresolved ticker, not just --max-symbols 20."""
    if not annual or not per_security:
        raise ValueError("Both complete annual_coverage.csv and per_security_coverage.csv are required")
    valid_years = {int(x["year"]): int(x["memberships"]) for x in annual
                   if first_year <= int(x["year"]) <= last_year}
    observed = Counter(int(x["year"]) for x in per_security
                       if first_year <= int(x["year"]) <= last_year)
    for yr in range(first_year, last_year + 1):
        if yr not in valid_years or observed[yr] != valid_years[yr]:
            raise ValueError(f"Incomplete per_security audit {yr}: got {observed[yr]} of {valid_years.get(yr)}"
                             "; rerun 11 before building crosswalk")
    unresolved = defaultdict(set)
    for row in per_security:
        yr = int(row["year"])
        if not first_year <= yr <= last_year:
            continue
        state = row.get("cik_status", "").lower()
        cik = str(row.get("cik_candidate", "")).strip()
        if state.startswith("unresolved") or state == "ambiguous" or (not cik and state != "current_ticker_unique_candidate"):
            unresolved[symbol(row["ticker"])].add(yr)
    return dict(sorted(unresolved.items()))


def intervals_for(ticker: str, membership: list[dict]) -> list[dict]:
    items = []
    for x in membership:
        if symbol(x["ticker"]) == ticker:
            # Source intervals: start inclusive / end exclusive (see step 10).
            if x["start_date"] > "2026-12-31" or x["end_date"] and x["end_date"] < "2015-01-01":
                continue
            items.append(x)
    return items


def load_hints(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    result = {}
    for row in read_csv(path):
        ticker = symbol(row.get("historical_ticker", ""))
        if ticker:
            result[ticker] = row
    return result


def proposal(ticker, entry, cik="", company="", source="", *, score="", note=""):
    return {"security_id": "", "cik": str(cik).zfill(10) if cik else "",
            "historical_ticker": ticker,
            "valid_from": entry["start_date"], "valid_to": entry["end_date"],
            "company_name": company, "corporate_action": "UNKNOWN_REQUIRES_REVIEW",
            "mapping_source": source, "verification_status": "candidate_unverified" if cik else "unresolved",
            "match_score": score, "review_note": note}


def build(args, client: SECClient):
    # A failed current SEC directory fetch would falsely classify all names as
    # unresolved; reject those audits instead of generating a misleading queue.
    prior_metadata = Path(args.per_security).with_name("audit_metadata.json")
    if prior_metadata.exists():
        meta = json.loads(prior_metadata.read_text(encoding="utf-8"))
        if meta.get("sec_source_error"):
            raise ValueError("Audit SEC directory lookup failed; rerun step 11 before building crosswalk")
    annual = read_csv(Path(args.annual))
    per = read_csv(Path(args.per_security))
    outstanding = unresolved_from_audit(per, annual)
    history = read_csv(Path(args.memberships))
    hints = load_hints(Path(args.hints))
    verified = verified_crosswalk(Path(args.verified_crosswalk))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Historical ticker-to-company names are SEARCH HINTS, not SEC CIK proof.
    # Wiki selected changes give the original removed/added ticker. The separate
    # monthly repository may backfill current tickers into earlier snapshots,
    # so source provenance is retained in all review tables.
    historical = HistoricalNameSources(Path(getattr(args, 'cache_dir', ROOT / 'data/cache')), offline=getattr(args, 'offline', False))
    discovered = {}
    if getattr(args, 'historical_names', False):
        observed = historical.collect(
            wikipedia=not getattr(args, 'skip_wikipedia', False),
            monthly=not getattr(args, 'skip_monthly', False),
            monthly_step=getattr(args, 'snapshot_step_months', 3))
        discovered = names_for_unresolved(observed, set(outstanding), history)

    name_variants = {}
    for ticker in outstanding:
        names = {}
        manual = (hints.get(ticker, {}).get('company_name_hint') or '').strip()
        if manual:
            names[manual.casefold()] = (manual, hints[ticker].get('hint_source',''))
        for hint in discovered.get(ticker, []):
            names.setdefault(hint.name.casefold(), (hint.name, hint.source))
        name_variants[ticker] = list(names.values())

    name_rows = [{'historical_ticker': t, 'company_name_hint': n, 'hint_source': src,
                  'verification_status': 'UNVERIFIED_SEARCH_HINT'}
                 for t, values in name_variants.items() for n, src in values]
    atomic_csv(out / 'historical_name_hints.csv', name_rows,
               ['historical_ticker', 'company_name_hint', 'hint_source', 'verification_status'])

    queue = []
    for ticker, years in outstanding.items():
        hint = hints.get(ticker, {})
        variants = name_variants.get(ticker, [])
        queue.append({"historical_ticker": ticker,
                      "years_unresolved": ",".join(map(str, sorted(years))),
                      "company_name_hint": variants[0][0] if variants else "",
                      "company_name_variants": " | ".join(n for n, src in variants),
                      "hint_count": len(variants),
                      "cik_hint": hint.get("cik_hint", ""),
                      "hint_source": variants[0][1] if variants else hint.get("hint_source", ""),
                      "review_status": "NEEDS_EXTERNAL_TICKER_IDENTITY_EVIDENCE"})
    atomic_csv(out / "unresolved_review_queue.csv", queue,
               ["historical_ticker", "years_unresolved", "company_name_hint", "company_name_variants",
                "hint_count", "cik_hint", "hint_source", "review_status"])

    selected = list(outstanding)
    if args.max_symbols > 0:
        selected = selected[args.offset: args.offset + args.max_symbols]
    elif args.offset:
        selected = selected[args.offset:]

    candidates_by_sym = defaultdict(dict)
    errors = []
    hint_names = {ticker: [name for name, src in name_variants.get(ticker, [])]
                  for ticker in selected if name_variants.get(ticker)}
    index_diagnostics = []
    if not args.skip_indexes:
        if not hint_names and not any(hints.get(t, {}).get("cik_hint", "").strip() for t in selected):
            raise ValueError("No historical company-name/CIK hints among selected symbols. "
                             "Check --hints or fill ticker_name_hints.csv before downloading SEC indexes.")
        years = [int(x) for x in args.index_years.split(",") if x.strip()]
        quarters = [int(x) for x in args.index_quarters.split(",") if x.strip()]
        for year in years:
            for quarter in quarters:
                stat = {"index": f"{year}Q{quarter}", "filing_rows_scanned": 0,
                        "matched_tickers": 0, "matched_cik_pairs": 0}
                try:
                    def counted_filings():
                        for filing in client.master_index(year, quarter):
                            stat["filing_rows_scanned"] += 1
                            yield filing
                    matches = index_matches(counted_filings(), hint_names,
                                            threshold=args.name_match_threshold)
                    stat["matched_tickers"] = len(matches)
                    stat["matched_cik_pairs"] = sum(len(v) for v in matches.values())
                    index_diagnostics.append(stat)
                except Exception as exc:
                    errors.append({"source": f"EDGAR master.idx {year}Q{quarter}",
                                   "error": f"{type(exc).__name__}: {str(exc)[:220]}"})
                    continue
                for ticker, by_cik in matches.items():
                    for cik, hit in by_cik.items():
                        prev = candidates_by_sym[ticker].get(cik)
                        if prev is None or hit["score"] > prev["score"]:
                            candidates_by_sym[ticker][cik] = hit

    # Explicit CIK hints are *unverified* hints, never automatic approvals.
    for ticker in selected:
        raw_cik = hints.get(ticker, {}).get("cik_hint", "").strip()
        if raw_cik.isdigit():
            cik = int(raw_cik)
            candidates_by_sym[ticker].setdefault(cik, {
                "cik": cik, "name": hints[ticker].get("company_name_hint", ""),
                "score": 0, "source": hints[ticker].get("hint_source", ""),
                "match_type": "manual_unverified_cik_hint"})

    # Retrieve submissions *only* for shortlisted CIKs from EDGAR index/name hints.
    submissions = {}
    ciks = sorted({cik for tick in selected for cik in candidates_by_sym[tick]})
    for cik in ciks:
        try:
            submissions[cik] = client.submissions(cik)
        except Exception as exc:
            errors.append({"source": f"SEC submissions CIK{cik:010d}",
                           "error": f"{type(exc).__name__}: {str(exc)[:220]}"})

    proposals = []
    for ticker in selected:
        intervals = intervals_for(ticker, history)
        if not intervals:
            errors.append({"source": f"membership intervals {ticker}", "error": "ticker absent from pinned intervals"})
            continue
        by_cik = candidates_by_sym[ticker]
        # Include verified entries ONLY if already present in explicitly reviewed file.
        for interval in intervals:
            verified_at = find_verified(verified, ticker, interval["start_date"])
            if verified_at:
                proposals.append({**verified_at, "match_score": "", "review_note": "manually verified; do not reinterpret ticker/cik"})
            if not by_cik:
                proposals.append(proposal(ticker, interval, note="No reliable SEC name-to-CIK candidate: needs historical company name/evidence"))
            for cik, hit in sorted(by_cik.items(), key=lambda item: -item[1]["score"])[:args.max_candidates]:
                sub = submissions.get(cik, {})
                former = sub.get("formerNames", [])
                historical_names = hint_names.get(ticker, [])
                normalized_names = {normalize_company(n) for n in historical_names}
                formerly = [x for x in former if normalize_company(x.get("name", "")) in normalized_names]
                notes = [hit.get("match_type", ""), "SEC filings give NAME->CIK, NOT historical ticker->CIK",
                         "hint_sources=" + " | ".join(src for n, src in name_variants.get(ticker, [])[:4])]
                if formerly:
                    notes.append("former legal name in SEC submissions")
                if sub.get("tickers"):
                    notes.append("current_submissions_tickers=" + ",".join(sub["tickers"][:8]))
                if cik not in submissions:
                    notes.append("SEC submissions NOT fetched/validated")
                source = " | ".join(filter(None, [hit.get("source"),
                          f"https://data.sec.gov/submissions/CIK{cik:010d}.json" if cik in submissions else ""]))
                proposals.append(proposal(ticker, interval, cik, sub.get("name") or hit.get("name", ""),
                                          source, score=hit.get("score", ""), note="; ".join(notes)))

    atomic_csv(out / "candidate_crosswalk.csv", proposals,
               list(COLUMNS) + ["match_score", "review_note"])
    status = Counter(p["verification_status"] for p in proposals)
    meta = {"total_unique_unresolved_from_all_annual_snapshots": len(outstanding),
            "symbols_processed_this_run": len(selected), "unresolved_without_candidate_rows": status["unresolved"],
            "sec_name_candidates_not_verified": status["candidate_unverified"],
            "manually_verified_rows_from_input": status["verified"],
            "name_hints_loaded": len(hints),
            "auto_name_hints_from_historical_sources": sum(len(v) for v in discovered.values()),
            "unique_unresolved_with_name_hints": sum(bool(name_variants[t]) for t in outstanding),
            "selected_symbols_with_company_name_hint": len(hint_names),
            "selected_symbols_without_company_name_hint": len(selected) - len(hint_names),
            "historical_names_cache_hits": historical.cache_hits,
            "historical_names_network_requests": historical.network_requests,
            "historical_name_source_errors": historical.errors,
            "symbols_with_candidate_cik": sum(bool(candidates_by_sym.get(t)) for t in selected),
            "edgar_index_diagnostics": index_diagnostics,
            "cache_hits": client.cache_hits, "network_requests": client.network_requests,
            "sec_errors": errors, "warning": "SEC master.idx and formerNames never certify ticker identity."
            " No model training permitted until verified_crosswalk explicitly approved."}
    (out / "crosswalk_audit.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "sec_errors"}, indent=2))
    if errors:
        print(f"WARNING: {len(errors)} SEC source errors, see crosswalk_audit.json")
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annual", type=Path, default=DEFAULT_COVERAGE / "annual_coverage.csv")
    p.add_argument("--per-security", type=Path, default=DEFAULT_COVERAGE / "per_security_coverage.csv")
    p.add_argument("--memberships", type=Path, default=DEFAULT_HISTORY)
    p.add_argument("--hints", type=Path, default=DEFAULT_HINTS)
    p.add_argument("--verified-crosswalk", type=Path, default=DEFAULT_VERIFIED)
    p.add_argument("--output", type=Path, default=ROOT / "results/historical_crosswalk")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache")
    p.add_argument("--max-symbols", type=int, default=0, help="0=all; affects SEC candidate lookup only, never review inventory")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--index-years", default="2015,2018,2021,2025")
    p.add_argument("--index-quarters", default="1")
    p.add_argument("--max-candidates", type=int, default=5)
    p.add_argument("--name-match-threshold", type=float, default=0.92)
    p.add_argument("--sec-delay", type=float, default=0.4)
    p.add_argument("--offline", action="store_true", help="Use cached SEC files only; report unavailable lookups honestly")
    p.add_argument("--skip-indexes", action="store_true", help="Prepare review queue without SEC index downloads")
    p.add_argument("--historical-names", action=argparse.BooleanOptionalAction, default=True,
                   help="Auto-download/cache historical S&P ticker+company-name hints (default on)")
    p.add_argument("--skip-wikipedia", action="store_true", help="Skip Wikipedia historical change table")
    p.add_argument("--skip-monthly", action="store_true", help="Skip pinned 2016-2026 reconstructed monthly name snapshots")
    p.add_argument("--snapshot-step-months", type=int, choices=[1, 3, 6, 12], default=3,
                   help="Read every 3rd historical monthly names file by default; 1=all 124 months")
    args = p.parse_args()
    if args.max_symbols < 0 or args.offset < 0 or args.max_candidates < 1:
        p.error("invalid limits")
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    client = SECClient(args.cache_dir, ua, delay=args.sec_delay, offline=args.offline)
    build(args, client)


if __name__ == "__main__":
    main()