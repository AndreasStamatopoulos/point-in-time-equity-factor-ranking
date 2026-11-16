from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, TypeAlias
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

JSONValue: TypeAlias = (
    str | int | float | bool | None
    | list["JSONValue"] | dict[str, "JSONValue"]
)
Endpoint: TypeAlias = Literal["EARNINGS", "EARNINGS_ESTIMATES"]
Fetcher: TypeAlias = Callable[[str, str, str], bytes]

ENDPOINTS: tuple[Endpoint, ...] = (
    "EARNINGS", "EARNINGS_ESTIMATES"
)
DAILY_LIMIT = 25
DOC_URL = "https://www.alphavantage.co/documentation/#earnings-estimates"
SUPPORT_URL = "https://www.alphavantage.co/support/#support"

LIMITATION = (
    "Historical point-in-time EPS consensus revisions are not verified. "
    "EARNINGS estimates/surprises are not consensus vintages. "
    "EARNINGS_ESTIMATES advertises revision history but does not establish "
    "historical as-of queries and independently verified availability. "
    "Current snapshots must not be backfilled. Use five features."
)


class AlphaVantageError(RuntimeError):
    def __init__(self, status: str, message: str) -> None:
        self.status = status
        super().__init__(message)


def _json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON keys must be strings.")
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    raise ValueError("Invalid JSON value.")


def _read_json(path: Path) -> dict[str, JSONValue]:
    value = _json_value(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}.")
    return value


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{uuid4().hex}.tmp"
    )
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_json(
    path: Path, value: dict[str, JSONValue]
) -> None:
    _write_bytes(
        path,
        (
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        ).encode(),
    )


@contextmanager
def _lock(
    path: Path, timeout: float = 35.0
) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            descriptor = os.open(
                path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise AlphaVantageError(
                    "cache_locked",
                    "Alpha Vantage cache is locked. Check running "
                    "collectors before removing a stale lock.",
                ) from None
            time.sleep(0.1)
    os.close(descriptor)
    try:
        yield
    finally:
        path.unlink()


def _reserve_request(
    ledger_path: Path,
    now: datetime,
    external_requests: int = 0,
) -> None:
    if not 0 <= external_requests <= DAILY_LIMIT:
        raise ValueError("External requests must be 0–25.")
    ledger = (
        _read_json(ledger_path)
        if ledger_path.exists() else {"requests": []}
    )
    stored = ledger.get("requests")
    if not isinstance(stored, list) or not all(
        isinstance(item, str) for item in stored
    ):
        raise ValueError("Invalid quota ledger; do not reset it.")

    cutoff = now - timedelta(hours=24)
    retained: list[JSONValue] = []
    for item in stored:
        if not isinstance(item, str):
            raise ValueError("Invalid quota timestamp.")
        stamp = datetime.fromisoformat(item)
        if (
            stamp.tzinfo is None
            or stamp > now + timedelta(minutes=5)
        ):
            raise ValueError("Invalid quota timestamp chronology.")
        if stamp > cutoff:
            retained.append(item)

    if len(retained) + external_requests >= DAILY_LIMIT:
        raise AlphaVantageError(
            "quota_exhausted",
            "25-request rolling 24-hour quota exhausted; no request sent.",
        )

    retained.append(now.isoformat())
    _write_json(
        ledger_path,
        {
            "requests": retained,
            "limit": DAILY_LIMIT,
            "window_hours": 24,
        },
    )


def _fetch(endpoint: str, ticker: str, api_key: str) -> bytes:
    query = urlencode(
        {"function": endpoint, "symbol": ticker, "apikey": api_key}
    )
    request = Request(
        f"https://www.alphavantage.co/query?{query}",
        headers={"User-Agent": "MLFinanceResearch/1.0"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            body: object = response.read()
            if not isinstance(body, bytes):
                raise AlphaVantageError(
                    "invalid_response", "Expected response bytes."
                )
            return body
    except HTTPError as error:
        status = (
            "rate_limited" if error.code == 429 else "http_error"
        )
        raise AlphaVantageError(
            status,
            f"Alpha Vantage HTTP {error.code}; request counted.",
        ) from None
    except (URLError, TimeoutError, OSError):
        raise AlphaVantageError(
            "network_error",
            "Alpha Vantage network failure; request counted. "
            "Credentials and request URL suppressed.",
        ) from None


def _response_status(payload: dict[str, JSONValue]) -> str:
    for name in ("Error Message", "Note", "Information"):
        if name not in payload:
            continue
        message = payload[name]
        text = message.lower() if isinstance(message, str) else ""
        if "premium" in text or "subscription" in text:
            return "premium_required"
        if (
            name == "Note"
            or "rate limit" in text
            or "frequency" in text
            or "25 requests" in text
        ):
            return "rate_limited"
        return "api_error"
    return "available"


@dataclass(frozen=True)
class CachedResponse:
    endpoint: Endpoint
    ticker: str
    payload: dict[str, JSONValue]
    fetched_at: str
    source: str
    status: str
    cached: bool


def collect_endpoint(
    root: Path,
    endpoint: Endpoint,
    ticker: str,
    *,
    refresh: bool = False,
    external_requests: int = 0,
    fetcher: Fetcher = _fetch,
) -> CachedResponse:
    if (
        endpoint not in ENDPOINTS
        or re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", ticker) is None
    ):
        raise ValueError("Unsupported endpoint or ticker.")

    cache = root / "data/raw/alphavantage"
    directory = cache / endpoint / ticker
    index = directory / "latest.json"
    source = (
        "https://www.alphavantage.co/query"
        f"?function={endpoint}&symbol={ticker}"
    )

    with _lock(cache / ".lock"):
        if index.exists() and not refresh:
            metadata = _read_json(index)
            filename = metadata.get("file")
            fetched_at = metadata.get("fetched_at")
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(fetched_at, str)
            ):
                raise ValueError("Invalid Alpha Vantage cache index.")
            response_path = directory / filename
            if (
                hashlib.sha256(
                    response_path.read_bytes()
                ).hexdigest()
                != metadata.get("sha256")
            ):
                raise ValueError("Alpha Vantage cache checksum mismatch.")
            payload = _read_json(response_path)
            return CachedResponse(
                endpoint, ticker, payload, fetched_at,
                source, _response_status(payload), True,
            )

        api_key = os.environ.get(
            "ALPHAVANTAGE_API_KEY", ""
        ).strip()
        if not api_key or api_key.lower() in {
            "demo", "your_key", "your_api_key"
        }:
            raise AlphaVantageError(
                "missing_api_key",
                "Set ALPHAVANTAGE_API_KEY; no request sent.",
            )

        now = datetime.now(timezone.utc)
        _reserve_request(
            cache / "request_ledger.json", now, external_requests
        )
        raw = fetcher(endpoint, ticker, api_key)
        redacted = api_key.encode() in raw
        if redacted:
            raw = raw.replace(
                api_key.encode(), b"[REDACTED_API_KEY]"
            )

        try:
            value = _json_value(json.loads(raw))
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            raise AlphaVantageError(
                "invalid_response",
                "Invalid Alpha Vantage JSON; request counted.",
            ) from None

        if not isinstance(value, dict):
            raise AlphaVantageError(
                "invalid_response", "Expected a JSON object."
            )
        if (
            _response_status(value) == "available"
            and value.get("symbol") != ticker
        ):
            raise AlphaVantageError(
                "symbol_mismatch",
                "Alpha Vantage response identifies a different ticker.",
            )

        digest = hashlib.sha256(raw).hexdigest()
        filename = (
            f"{now.strftime('%Y%m%dT%H%M%S%fZ')}"
            f"_{digest[:12]}.json"
        )
        _write_bytes(directory / filename, raw)
        _write_json(
            index,
            {
                "file": filename,
                "fetched_at": now.isoformat(),
                "sha256": digest,
                "endpoint": endpoint,
                "ticker": ticker,
                "source": source,
                "status": _response_status(value),
                "credential_redacted": redacted,
            },
        )
        return CachedResponse(
            endpoint, ticker, value, now.isoformat(),
            source, _response_status(value), False,
        )


def _inspect(response: CachedResponse) -> dict[str, JSONValue]:
    key = (
        "quarterlyEarnings"
        if response.endpoint == "EARNINGS" else "estimates"
    )
    rows = response.payload.get(key, [])
    records = rows if isinstance(rows, list) else []
    fields: set[str] = set()
    periods: set[str] = set()
    dated_rows = 0
    invalid_dates = 0
    reported_before_end = 0

    for row in records:
        if not isinstance(row, dict):
            continue
        fields.update(row)
        fiscal = row.get("fiscalDateEnding", row.get("date"))
        if isinstance(fiscal, str):
            try:
                datetime.strptime(fiscal, "%Y-%m-%d")
            except ValueError:
                invalid_dates += 1
            else:
                periods.add(fiscal)

        reported = row.get("reportedDate")
        if isinstance(reported, str):
            try:
                datetime.strptime(reported, "%Y-%m-%d")
            except ValueError:
                invalid_dates += 1
            else:
                dated_rows += 1
                if (
                    isinstance(fiscal, str)
                    and fiscal in periods
                    and reported < fiscal
                ):
                    reported_before_end += 1

    field_names: list[JSONValue] = sorted(fields)
    fiscal_dates: list[JSONValue] = sorted(periods)
    return {
        "endpoint": response.endpoint,
        "ticker": response.ticker,
        "status": response.status,
        "cached": response.cached,
        "source": response.source,
        "fetched_at": response.fetched_at,
        "record_count": len(records),
        "fields": field_names,
        "fiscal_period_count": len(periods),
        "fiscal_periods": fiscal_dates,
        "reported_date_rows": dated_rows,
        "invalid_date_fields": invalid_dates,
        "reported_before_fiscal_end": reported_before_end,
        "provider_message": next(
            (
                response.payload[key]
                for key in ("Note", "Information", "Error Message")
                if key in response.payload
            ),
            None,
        ),
        "historical_consensus_vintages_verified": False,
    }


def collect_eps(
    root: Path,
    ticker: str = "IBM",
    *,
    refresh: bool = False,
    external_requests: int = 0,
) -> dict[str, JSONValue]:
    reports: list[JSONValue] = []

    for endpoint in ENDPOINTS:
        try:
            response = collect_endpoint(
                root, endpoint, ticker,
                refresh=refresh,
                external_requests=external_requests,
            )
        except AlphaVantageError as error:
            reports.append(
                {
                    "endpoint": endpoint,
                    "ticker": ticker,
                    "status": error.status,
                    "reason": str(error),
                }
            )
            if error.status in {
                "missing_api_key", "quota_exhausted",
                "rate_limited", "network_error", "cache_locked",
            }:
                break
        else:
            reports.append(_inspect(response))
            if response.status == "rate_limited":
                break

    report: dict[str, JSONValue] = {
        "status": "five_feature_fallback",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "earnings_revisions_available": False,
        "model_feature_count": 5,
        "limitation": LIMITATION,
        "endpoint_reports": reports,
        "documentation": [DOC_URL, SUPPORT_URL],
        "quota": (
            "25 requests per rolling 24 hours across this project's "
            "collectors; reserve account use from other software."
        ),
        "earnings_surprise": (
            "Not used. Surprise is distinct from revisions."
        ),
        "vintage_import": (
            "Not enabled without independent review of immutable "
            "historical consensus vintages and fiscal-period matching."
        ),
    }
    _write_json(root / "results/eps_availability.json", report)
    details = "\n".join(
        f"- {item['endpoint']}: {item['status']}"
        for item in reports if isinstance(item, dict)
    )
    _write_bytes(
        root / "results/eps_availability.md",
        (
            "# EPS availability\n\n"
            + LIMITATION + "\n\n" + details
            + "\n\nToday's estimates must not replace historical "
            "estimates. Fiscal-end and reported dates do not establish "
            "consensus vintage availability.\n\n"
            + f"Sources: {DOC_URL}\n{SUPPORT_URL}\n"
        ).encode(),
    )
    return report
