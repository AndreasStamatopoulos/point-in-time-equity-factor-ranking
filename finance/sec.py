from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TypeAlias
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from finance.config import SEC_CONCEPT_UNITS

JsonValue: TypeAlias = (
    str | int | float | bool | None
    | list["JsonValue"] | dict[str, "JsonValue"]
)

FACT_COLUMNS = [
    "ticker", "taxonomy", "concept", "unit", "start", "end",
    "value", "filed", "available_at", "accession", "form",
    "fy", "fp", "frame", "source", "retrieved_at",
]

ALLOWED_FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A")


class SecConfigurationError(ValueError):
    pass


class SecAccessError(RuntimeError):
    pass


@dataclass(frozen=True)
class CachedResponse:
    document: dict[str, JsonValue]
    source: str
    retrieved_at: str
    path: Path


def _validated_json(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite SEC JSON number.")
        return value
    if isinstance(value, list):
        return [_validated_json(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("SEC JSON keys must be strings.")
            result[key] = _validated_json(item)
        return result
    raise ValueError("Unsupported SEC JSON value.")


def _object(value: JsonValue, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {label}.")
    return value


def _string(value: JsonValue, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Expected text: {label}.")
    return value


def _date(value: JsonValue, label: str) -> str:
    text = _string(value, label)
    datetime.strptime(text, "%Y-%m-%d")
    return text


def _read_document(path: Path) -> dict[str, JsonValue]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    return _object(_validated_json(raw), str(path))


def _write_json(path: Path, payload: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_user_agent(value: str | None) -> str:
    agent = (value or "").strip()
    email = re.search(
        r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
        r"@([A-Za-z0-9.-]+)\.[A-Za-z]{2,}\b",
        agent,
    )
    placeholders = (
        "example.com", "example.org", "example.net",
        "your-email", "your_email", "replace", "<", ">",
    )
    reserved = email is not None and email.group(0).lower().endswith(
        (".invalid", ".test", ".local", ".localhost", ".example")
    )
    if (
        email is None
        or email.start() < 3
        or reserved
        or "\n" in agent
        or "\r" in agent
        or any(item in agent.lower() for item in placeholders)
    ):
        raise SecConfigurationError(
            "Set SEC_USER_AGENT to a project name followed by your "
            "real contact email. No SEC request was sent."
        )
    return agent


class SecClient:
    def __init__(
        self,
        cache_dir: Path,
        user_agent: str | None = None,
        attempts: int = 5,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be positive.")
        self.cache_dir = cache_dir
        self.user_agent = (
            user_agent
            if user_agent is not None
            else os.environ.get("SEC_USER_AGENT")
        )
        self.attempts = attempts
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = cache_dir / "access_state.sqlite3"
        with closing(
            sqlite3.connect(self.state_path, timeout=60)
        ) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS access "
                "(id INTEGER PRIMARY KEY CHECK(id=1), "
                "last_request REAL NOT NULL, blocked INTEGER NOT NULL)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO access VALUES (1, 0, 0)"
            )

    def _reserve(self) -> None:
        with closing(
            sqlite3.connect(self.state_path, timeout=60)
        ) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT last_request, blocked FROM access WHERE id=1"
            ).fetchone()
            if row is None:
                raise SecAccessError("SEC rate-limit state is missing.")
            if int(row[1]):
                raise SecAccessError(
                    "SEC previously returned HTTP 403. Collection remains "
                    "stopped. Resolve access with SEC before explicitly "
                    "resetting the local access state."
                )
            wait = float(row[0]) + 0.51 - time.time()
            if wait > 60:
                raise SecAccessError(
                    "System clock moved backwards; collection stopped."
                )
            if wait > 0:
                time.sleep(wait)
            connection.execute(
                "UPDATE access SET last_request=? WHERE id=1",
                (time.time(),),
            )

    def _block(self) -> None:
        with closing(
            sqlite3.connect(self.state_path, timeout=60)
        ) as connection, connection:
            connection.execute("UPDATE access SET blocked=1 WHERE id=1")

    @staticmethod
    def _retry_delay(error: HTTPError, attempt: int) -> float:
        delay = float(2 ** attempt)
        value = error.headers.get("Retry-After")
        if value:
            if value.strip().isdigit():
                delay = max(delay, float(value))
            else:
                parsed = parsedate_to_datetime(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                delay = max(
                    delay,
                    (parsed - datetime.now(timezone.utc)).total_seconds(),
                )
        return delay

    def get(
        self, endpoint: str, cik: str, refresh: bool = False
    ) -> CachedResponse:
        if (
            endpoint not in ("companyfacts", "submissions")
            or not re.fullmatch(r"\d{1,10}", cik)
        ):
            raise ValueError("Invalid SEC endpoint or CIK.")

        padded = cik.zfill(10)
        directory = self.cache_dir / endpoint
        directory.mkdir(parents=True, exist_ok=True)
        metadata_files = sorted(
            directory.glob(f"CIK{padded}_*.meta.json")
        )

        if metadata_files and not refresh:
            snapshots = [
                (_read_document(path), path) for path in metadata_files
            ]
            metadata, metadata_path = max(
                snapshots,
                key=lambda pair: _string(
                    pair[0].get("retrieved_at"), "retrieved_at"
                ),
            )
            filename = _string(metadata.get("file"), "file")
            if Path(filename).name != filename:
                raise ValueError("Unsafe SEC cache filename.")
            path = metadata_path.parent / filename
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            if checksum != _string(metadata.get("sha256"), "sha256"):
                raise ValueError("SEC cache integrity failure.")
            return CachedResponse(
                _read_document(path),
                _string(metadata.get("source"), "source"),
                _string(metadata.get("retrieved_at"), "retrieved_at"),
                path,
            )

        agent = validate_user_agent(self.user_agent)
        prefix = (
            "api/xbrl/companyfacts"
            if endpoint == "companyfacts"
            else "submissions"
        )
        source = f"https://data.sec.gov/{prefix}/CIK{padded}.json"
        request = Request(
            source,
            headers={
                "User-Agent": agent,
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
        )

        for attempt in range(self.attempts):
            self._reserve()
            try:
                with urlopen(request, timeout=45) as response:
                    content = response.read()
            except HTTPError as error:
                if error.code == 403:
                    self._block()
                    raise SecAccessError(
                        "SEC returned HTTP 403. Further SEC collection "
                        "is stopped; resolve identification/access with SEC."
                    ) from error
                if error.code == 429 or 500 <= error.code <= 599:
                    if attempt + 1 == self.attempts:
                        raise SecAccessError(
                            f"SEC HTTP {error.code} after retries: {source}"
                        ) from error
                    time.sleep(self._retry_delay(error, attempt))
                    continue
                raise SecAccessError(
                    f"SEC HTTP {error.code}: {source}"
                ) from error
            except (URLError, TimeoutError) as error:
                raise SecAccessError(
                    f"SEC connection failed: {source}; "
                    f"{type(error).__name__}."
                ) from error

            raw: object = json.loads(content)
            document = _object(_validated_json(raw), source)
            retrieved = datetime.now(timezone.utc).isoformat()
            digest = hashlib.sha256(content).hexdigest()
            path = directory / f"CIK{padded}_{digest}.json"

            if not path.exists():
                with path.open("xb") as stream:
                    stream.write(content)
                _write_json(
                    path.with_suffix(".meta.json"),
                    {
                        "file": path.name,
                        "sha256": digest,
                        "source": source,
                        "retrieved_at": retrieved,
                        "endpoint": endpoint,
                        "cik": padded,
                    },
                )
            else:
                existing = _read_document(path.with_suffix(".meta.json"))
                retrieved = _string(
                    existing.get("retrieved_at"), "retrieved_at"
                )
            return CachedResponse(document, source, retrieved, path)

        raise SecAccessError("SEC request produced no response.")


def normalize_companyfacts(
    ticker: str, response: CachedResponse
) -> pd.DataFrame:
    facts = _object(response.document.get("facts"), "facts")
    records: list[dict[str, str | float | int | None]] = []

    for taxonomy, concepts_value in facts.items():
        if taxonomy not in ("us-gaap", "dei"):
            continue
        concepts = _object(concepts_value, taxonomy)
        for concept, details_value in concepts.items():
            if concept not in SEC_CONCEPT_UNITS:
                continue
            details = _object(details_value, concept)
            units = _object(details.get("units"), f"{concept}.units")
            for unit, entries in units.items():
                if unit != SEC_CONCEPT_UNITS[concept]:
                    continue
                if not isinstance(entries, list):
                    raise ValueError("Expected a SEC fact list.")
                for value in entries:
                    entry = _object(value, concept)
                    if entry.get("form") not in ALLOWED_FORMS:
                        continue
                    amount = entry.get("val")
                    if (
                        isinstance(amount, bool)
                        or not isinstance(amount, (float, int))
                        or not math.isfinite(amount)
                    ):
                        raise ValueError(
                            f"Invalid SEC value: {ticker}/{concept}."
                        )
                    end = _date(entry.get("end"), "end")
                    filed = _date(entry.get("filed"), "filed")
                    start = (
                        _date(entry["start"], "start")
                        if "start" in entry else None
                    )
                    if start is not None and start > end:
                        raise ValueError("Reversed SEC reporting period.")
                    if filed < end:
                        raise ValueError("SEC filing precedes period end.")
                    accession = _string(entry.get("accn"), "accn")
                    if not re.fullmatch(
                        r"\d{10}-\d{2}-\d{6}", accession
                    ):
                        raise ValueError("Invalid SEC accession.")
                    fy = entry.get("fy")
                    if fy is not None and (
                        isinstance(fy, bool) or not isinstance(fy, int)
                    ):
                        raise ValueError("Invalid SEC fiscal year.")

                    records.append(
                        {
                            "ticker": ticker,
                            "taxonomy": taxonomy,
                            "concept": concept,
                            "unit": unit,
                            "start": start,
                            "end": end,
                            "value": float(amount),
                            "filed": filed,
                            "available_at": (
                                datetime.strptime(filed, "%Y-%m-%d")
                                + timedelta(days=1)
                            ).date().isoformat(),
                            "accession": accession,
                            "form": _string(entry.get("form"), "form"),
                            "fy": fy,
                            "fp": (
                                _string(entry["fp"], "fp")
                                if entry.get("fp") is not None else ""
                            ),
                            "frame": (
                                _string(entry["frame"], "frame")
                                if entry.get("frame") is not None else ""
                            ),
                            "source": response.source,
                            "retrieved_at": response.retrieved_at,
                        }
                    )

    frame = pd.DataFrame.from_records(
        records, columns=FACT_COLUMNS
    ).drop_duplicates()
    for column in ("start", "end", "filed", "available_at"):
        frame[column] = pd.to_datetime(
            frame[column], format="%Y-%m-%d"
        )
    return frame


def read_facts(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={
            "ticker": str, "accession": str, "taxonomy": str,
            "concept": str, "unit": str,
        },
    )
    missing = set(FACT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing SEC columns: {sorted(missing)}.")
    for column in ("start", "end", "filed", "available_at"):
        frame[column] = pd.to_datetime(
            frame[column], format="%Y-%m-%d", errors="raise"
        )
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    required = [
        "ticker", "taxonomy", "concept", "unit", "end", "filed",
        "available_at", "accession", "source", "retrieved_at", "value",
    ]
    if frame[required].isna().any().any():
        raise ValueError("SEC facts have missing required fields.")
    pd.to_datetime(frame["retrieved_at"], utc=True, errors="raise")
    if not frame["accession"].str.fullmatch(
        r"\d{10}-\d{2}-\d{6}"
    ).all():
        raise ValueError("Invalid SEC accession.")
    if not frame["source"].str.startswith(
        "https://data.sec.gov/"
    ).all():
        raise ValueError("Invalid SEC source provenance.")
    if (
        frame["start"].notna()
        & frame["start"].gt(frame["end"])
    ).any():
        raise ValueError("Reversed SEC reporting period.")
    if not frame["value"].map(math.isfinite).all():
        raise ValueError("SEC facts contain nonfinite values.")
    if (
        frame["available_at"].lt(
            frame["filed"] + pd.Timedelta(days=1)
        ).any()
        or frame["end"].gt(frame["filed"]).any()
    ):
        raise ValueError("Invalid SEC availability chronology.")
    return frame


def asof_facts(
    facts: pd.DataFrame, asof: pd.Timestamp
) -> pd.DataFrame:
    eligible = facts.loc[
        facts["available_at"].le(asof)
        & facts["end"].le(asof)
        & facts["form"].isin(ALLOWED_FORMS)
    ].copy()
    keys = ["ticker", "taxonomy", "concept", "unit", "start", "end"]
    conflicts = eligible.groupby(
        keys + ["filed", "accession"], dropna=False
    )["value"].nunique()
    if conflicts.gt(1).any():
        raise ValueError(
            "Conflicting SEC values within one accession and period; "
            "manual mapping required."
        )
    return eligible.sort_values(
        ["filed", "accession"]
    ).drop_duplicates(keys, keep="last")


def _eligible(
    facts: pd.DataFrame,
    concepts: tuple[str, ...],
    asof: pd.Timestamp,
    unit: str,
) -> pd.DataFrame:
    if facts["ticker"].nunique() > 1:
        raise ValueError("Selectors require one ticker.")
    available = asof_facts(facts, asof)
    return available.loc[
        available["concept"].isin(concepts)
        & available["unit"].eq(unit)
    ].copy()


def _best(
    eligible: pd.DataFrame,
    concepts: tuple[str, ...],
    asof: pd.Timestamp,
    end: pd.Timestamp | None,
    max_age_days: int,
) -> pd.Series | None:
    if end is not None:
        eligible = eligible.loc[eligible["end"].eq(end)].copy()
    eligible = eligible.loc[
        (asof - eligible["end"]).dt.days.le(max_age_days)
    ].copy()
    if eligible.empty:
        return None
    eligible["_priority"] = eligible["concept"].map(
        {concept: index for index, concept in enumerate(concepts)}
    )
    return eligible.sort_values(
        ["end", "_priority", "filed", "accession"],
        ascending=[False, True, False, False],
    ).iloc[0].drop(labels="_priority")


def select_instant(
    facts: pd.DataFrame,
    concepts: tuple[str, ...],
    asof: pd.Timestamp,
    end: pd.Timestamp | None = None,
    max_age_days: int = 550,
    unit: str = "USD",
) -> pd.Series | None:
    eligible = _eligible(facts, concepts, asof, unit)
    return _best(
        eligible.loc[eligible["start"].isna()],
        concepts, asof, end, max_age_days,
    )


def select_annual(
    facts: pd.DataFrame,
    concepts: tuple[str, ...],
    asof: pd.Timestamp,
    end: pd.Timestamp | None = None,
    max_age_days: int = 550,
    unit: str = "USD",
) -> pd.Series | None:
    eligible = _eligible(facts, concepts, asof, unit)
    duration = (eligible["end"] - eligible["start"]).dt.days + 1
    return _best(
        eligible.loc[duration.between(330, 400)],
        concepts, asof, end, max_age_days,
    )


def collect_sec(
    root: Path, refresh: bool = False
) -> pd.DataFrame:
    universe = pd.read_csv(
        root / "data/universe.csv",
        dtype={"ticker": str, "cik": str},
    )
    if (
        not {"ticker", "cik"}.issubset(universe.columns)
        or universe[["ticker", "cik"]].isna().any().any()
    ):
        raise ValueError("Universe requires ticker and CIK.")
    if universe["ticker"].duplicated().any():
        raise ValueError("Duplicate universe ticker.")

    client = SecClient(root / "data/raw/sec")
    frames: list[pd.DataFrame] = []
    sources: list[JsonValue] = []

    for row in universe.itertuples(index=False):
        ticker, cik = str(row.ticker), str(row.cik)
        companyfacts = client.get("companyfacts", cik, refresh)
        submissions = client.get("submissions", cik, refresh)
        for response in (companyfacts, submissions):
            response_cik = response.document.get("cik")
            if (
                not isinstance(response_cik, (str, int))
                or isinstance(response_cik, bool)
                or str(response_cik).zfill(10) != cik.zfill(10)
            ):
                raise ValueError(
                    f"SEC issuer identity mismatch for {ticker}."
                )
        frame = normalize_companyfacts(ticker, companyfacts)
        if frame.empty:
            raise ValueError(f"No mapped SEC facts for {ticker}.")
        frames.append(frame)
        sources.append(
            {
                "ticker": ticker,
                "facts_source": companyfacts.source,
                "facts_retrieved_at": companyfacts.retrieved_at,
                "submissions_source": submissions.source,
                "submissions_retrieved_at": submissions.retrieved_at,
                "fact_rows": len(frame),
            }
        )

    if not frames:
        raise ValueError("Universe is empty.")

    result = pd.concat(frames, ignore_index=True)
    target = root / "data/sec_facts.csv"
    temporary = target.with_suffix(".csv.tmp")
    result.to_csv(
        temporary, index=False, date_format="%Y-%m-%d"
    )
    temporary.replace(target)
    _write_json(
        root / "results/sec_collection_status.json",
        {
            "status": "complete",
            "synthetic": False,
            "tickers": len(frames),
            "rows": len(result),
            "availability_rule": "SEC filed date plus one calendar day.",
            "normalized_scope": (
                "Seven mapped US-GAAP/DEI concepts; "
                "full responses retained in raw JSON."
            ),
            "sources": sources,
        },
    )
    return result


def main(root: Path, refresh: bool = False) -> None:
    try:
        result = collect_sec(root, refresh)
    except (SecConfigurationError, SecAccessError) as error:
        _write_json(
            root / "results/sec_collection_status.json",
            {
                "status": "blocked",
                "reason": str(error),
                "synthetic": False,
            },
        )
        raise
    print(
        f"SEC collection complete: {len(result):,} facts; "
        f"{result.ticker.nunique()} tickers."
    )
