from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TypeAlias
from uuid import uuid4

import pandas as pd

Json: TypeAlias = (
    None | bool | int | float | str
    | list["Json"] | dict[str, "Json"]
)

PRICE_COLUMNS = [
    "ticker", "date", "close", "adjusted_close", "volume",
    "split_ratio", "future_split_factor", "source", "retrieved_at",
]


def mapping(value: Json) -> dict[str, Json]:
    if not isinstance(value, dict):
        raise ValueError("Expected a market-data JSON object.")
    return value


def sequence(value: Json) -> list[Json]:
    if not isinstance(value, list):
        raise ValueError("Expected a market-data JSON array.")
    return value


def number(value: Json) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("Expected a market-data number.")
    if not math.isfinite(value):
        raise ValueError("Nonfinite market-data number.")
    return float(value)


def _retrieval_timestamp(value: str) -> datetime:
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("Retrieval timestamp requires a timezone.")
    if stamp > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise ValueError("Retrieval timestamp is in the future.")
    return stamp


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(
        f".{path.name}.{uuid4().hex}.tmp"
    )
    temporary.write_bytes(payload)
    temporary.replace(path)


def parse_chart(
    payload: Json,
    ticker: str,
    source: str,
    retrieved_at: str,
) -> pd.DataFrame:
    chart = mapping(mapping(payload)["chart"])
    if chart.get("error") is not None:
        raise ValueError(
            f"Market provider error for {ticker}: {chart['error']}"
        )
    results = sequence(chart["result"])
    if len(results) != 1:
        raise ValueError(f"Expected one price history for {ticker}.")

    result = mapping(results[0])
    metadata = mapping(result["meta"])
    if metadata.get("symbol") != ticker:
        raise ValueError("Market provider symbol mismatch.")
    if (
        metadata.get("currency") != "USD"
        or metadata.get("instrumentType") != "EQUITY"
    ):
        raise ValueError("Expected USD equity prices.")

    exchange_timezone = metadata.get("exchangeTimezoneName")
    if not isinstance(exchange_timezone, str):
        raise ValueError("Missing exchange timezone.")
    retrieved = pd.Timestamp(
        _retrieval_timestamp(retrieved_at)
    ).tz_convert(exchange_timezone)

    timestamps = sequence(result["timestamp"])
    indicators = mapping(result["indicators"])
    quote = mapping(sequence(indicators["quote"])[0])
    closes = sequence(quote["close"])
    volumes = sequence(quote["volume"])
    adjusted = sequence(
        mapping(sequence(indicators["adjclose"])[0])["adjclose"]
    )
    if not timestamps or not (
        len(timestamps) == len(closes)
        == len(adjusted) == len(volumes)
    ):
        raise ValueError("Empty or mismatched price arrays.")

    instants = pd.to_datetime(
        [number(value) for value in timestamps],
        unit="s", utc=True,
    ).tz_convert(exchange_timezone)
    if (instants > retrieved + pd.Timedelta(minutes=5)).any():
        raise ValueError("Future market observation.")
    days = instants.date
    if len(set(days)) != len(days):
        raise ValueError("Duplicate daily market observations.")

    splits: dict[date, float] = {}
    events = mapping(
        mapping(result.get("events", {})).get("splits", {})
    )
    for event in events.values():
        item = mapping(event)
        split_day = pd.Timestamp(
            number(item["date"]), unit="s", tz="UTC"
        ).tz_convert(exchange_timezone).date()
        numerator = number(item["numerator"])
        denominator = number(item["denominator"])
        if (
            numerator <= 0 or denominator <= 0
            or split_day > retrieved.date()
        ):
            raise ValueError("Invalid or future stock split.")
        ratio = numerator / denominator
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("Invalid stock split ratio.")
        if (
            min(days) <= split_day <= max(days)
            and split_day not in days
        ):
            raise ValueError("Split event has no daily price row.")
        splits[split_day] = splits.get(split_day, 1.0) * ratio

    rows: list[dict[str, str | float]] = []
    for day, close, adj, volume in zip(
        days, closes, adjusted, volumes, strict=True
    ):
        if close is None or adj is None:
            raise ValueError(
                f"Missing price for {ticker} on {day}; inspect raw JSON."
            )
        close_value = number(close)
        adjusted_value = number(adj)
        volume_value = (
            number(volume) if volume is not None else float("nan")
        )
        if (
            close_value <= 0
            or adjusted_value <= 0
            or volume_value < 0
        ):
            raise ValueError("Invalid price or volume.")

        # Kratame ta splits eos to plires snapshot tou paroxou.
        future_factor = math.prod(
            ratio for split_day, ratio in splits.items()
            if split_day > day
        )
        if not math.isfinite(future_factor) or future_factor <= 0:
            raise ValueError("Invalid cumulative split factor.")
        rows.append(
            {
                "ticker": ticker,
                "date": str(day),
                "close": close_value,
                "adjusted_close": adjusted_value,
                "volume": volume_value,
                "split_ratio": splits.get(day, 1.0),
                "future_split_factor": future_factor,
                "source": source,
                "retrieved_at": retrieved_at,
            }
        )

    return pd.DataFrame(
        rows, columns=PRICE_COLUMNS
    ).sort_values("date").reset_index(drop=True)


def collect_prices(
    root: Path,
    start: str = "2010-01-01",
    end: date | None = None,
) -> pd.DataFrame:
    basis_day = datetime.now(timezone.utc).date()
    cutoff = end or basis_day - timedelta(days=1)
    begin = date.fromisoformat(start)
    if cutoff < begin or cutoff >= basis_day:
        raise ValueError("Use a completed historical price interval.")

    universe = pd.read_csv(
        root / "data/universe.csv", dtype={"ticker": str}
    )
    if universe.empty or universe["ticker"].duplicated().any():
        raise ValueError("Empty universe or duplicate ticker.")

    cache = root / "data/raw/prices"
    cache.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, str | int]] = []

    for ticker in universe["ticker"]:
        if (
            not isinstance(ticker, str)
            or re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", ticker) is None
        ):
            raise ValueError("Invalid universe ticker.")

        query = urllib.parse.urlencode(
            {
                "period1": int(
                    datetime.combine(
                        begin, datetime.min.time(), timezone.utc
                    ).timestamp()
                ),
                "period2": int(
                    datetime.combine(
                        basis_day + timedelta(days=1),
                        datetime.min.time(), timezone.utc,
                    ).timestamp()
                ),
                "interval": "1d",
                "events": "div,splits",
                "includeAdjustedClose": "true",
            }
        )
        source = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{urllib.parse.quote(ticker)}?{query}"
        )
        path = cache / f"{ticker}_{begin}_basis{basis_day}.json"
        meta_path = path.with_suffix(".meta.json")

        if path.exists():
            if not meta_path.exists():
                raise ValueError("Price cache provenance is missing.")
            raw = path.read_bytes()
            metadata = mapping(
                json.loads(meta_path.read_text(encoding="utf-8"))
            )
            if metadata.get("sha256") != hashlib.sha256(raw).hexdigest():
                raise ValueError("Price cache checksum mismatch.")
            if (
                metadata.get("source") != source
                or metadata.get("split_basis_request_date") != str(basis_day)
            ):
                raise ValueError("Price cache adjustment-basis mismatch.")
            retrieval = metadata.get("retrieved_at")
            if not isinstance(retrieval, str):
                raise ValueError("Missing cache retrieval timestamp.")
            retrieved_at = retrieval
        else:
            request = urllib.request.Request(
                source,
                headers={"User-Agent": "MLFinanceResearch/1.0"},
            )
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(
                        request, timeout=45
                    ) as response:
                        raw = response.read()
                    break
                except urllib.error.HTTPError as error:
                    if 500 <= error.code <= 599 and attempt < 3:
                        time.sleep(2 ** attempt)
                        continue
                    raise RuntimeError(
                        f"Yahoo HTTP {error.code} for {ticker}; stopped."
                    ) from None

            retrieved_at = datetime.now(timezone.utc).isoformat()
            parse_chart(json.loads(raw), ticker, source, retrieved_at)
            _atomic_write(path, raw)
            _atomic_write(
                meta_path,
                json.dumps(
                    {
                        "source": source,
                        "retrieved_at": retrieved_at,
                        "split_basis_request_date": str(basis_day),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    },
                    indent=2,
                ).encode(),
            )
            time.sleep(0.6)

        frame = parse_chart(
            json.loads(raw), ticker, source, retrieved_at
        )
        frame = frame.loc[
            frame["date"].between(str(begin), str(cutoff))
        ]
        if frame.empty:
            raise ValueError(f"No prices for {ticker}.")
        frames.append(frame)
        manifest.append(
            {
                "ticker": ticker,
                "rows": len(frame),
                "start": str(frame.date.min()),
                "end": str(frame.date.max()),
                "source": source,
                "retrieved_at": retrieved_at,
                "split_basis_request_date": str(basis_day),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        print(f"Market data {ticker}: {len(frame)} observations")

    prices = pd.concat(
        frames, ignore_index=True
    ).sort_values(["ticker", "date"])
    _atomic_write(
        root / "data/prices.csv",
        prices.to_csv(index=False).encode(),
    )
    _atomic_write(
        root / "data/price_sources.csv",
        pd.DataFrame(manifest).to_csv(index=False).encode(),
    )
    return prices


def load_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path)
    if (
        prices.empty
        or not set(PRICE_COLUMNS).issubset(prices.columns)
    ):
        raise ValueError("Invalid prices schema or empty dataset.")
    prices["date"] = pd.to_datetime(
        prices["date"], format="%Y-%m-%d", errors="raise"
    )
    if (
        prices["date"].isna().any()
        or prices.duplicated(["ticker", "date"]).any()
    ):
        raise ValueError("Missing date or duplicate prices.")

    for column in ("ticker", "source", "retrieved_at"):
        if (
            prices[column].isna().any()
            or prices[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"Missing price provenance: {column}.")
    if not prices["ticker"].astype(str).str.fullmatch(
        r"[A-Z][A-Z0-9.-]{0,14}"
    ).all():
        raise ValueError("Invalid ticker in prices.")

    for source in prices["source"].unique():
        address = urllib.parse.urlparse(str(source))
        if (
            address.scheme != "https"
            or not address.netloc
            or address.username is not None
        ):
            raise ValueError("Invalid price source URL.")

    retrievals = prices["retrieved_at"].map(
        lambda value: _retrieval_timestamp(str(value))
    )
    retrieval_days = pd.to_datetime(
        retrievals, utc=True
    ).dt.tz_localize(None).dt.normalize()
    if prices["date"].ge(retrieval_days).any():
        raise ValueError("Incomplete retrieval-day observation.")

    for column in (
        "close", "adjusted_close", "split_ratio", "future_split_factor"
    ):
        prices[column] = pd.to_numeric(
            prices[column], errors="raise"
        )
        if (
            not prices[column].map(math.isfinite).all()
            or prices[column].le(0).any()
        ):
            raise ValueError(f"Invalid {column}.")

    prices["volume"] = pd.to_numeric(
        prices["volume"], errors="raise"
    )
    volume = prices["volume"].dropna()
    if volume.lt(0).any() or not volume.map(math.isfinite).all():
        raise ValueError("Invalid volume.")

    prices = prices.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)
    for _, history in prices.groupby("ticker", sort=False):
        factors = history["future_split_factor"].to_numpy(dtype=float)
        ratios = history["split_ratio"].to_numpy(dtype=float)
        if any(
            not math.isclose(left, ratio * right, rel_tol=1e-10)
            for left, right, ratio in zip(
                factors[:-1], factors[1:], ratios[1:], strict=True
            )
        ):
            raise ValueError("Inconsistent historical split factors.")
    return prices


def _single_ticker(history: pd.DataFrame) -> None:
    if (
        history.empty
        or "ticker" not in history
        or history["ticker"].nunique() != 1
    ):
        raise ValueError("Price history requires exactly one ticker.")


def nominal_close(
    history: pd.DataFrame, day: pd.Timestamp
) -> float:
    _single_ticker(history)
    current = history.loc[history["date"].eq(day)]
    if len(current) != 1:
        raise ValueError(f"Expected one price on {day.date()}.")
    if "future_split_factor" not in current:
        raise ValueError("Missing split basis; recollect prices.")
    value = float(
        current.iloc[0]["close"]
        * current.iloc[0]["future_split_factor"]
    )
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid nominal close.")
    return value


def split_adjusted_shares(
    history: pd.DataFrame,
    shares: float,
    observed: pd.Timestamp,
    decision: pd.Timestamp,
) -> float:
    _single_ticker(history)
    if observed > decision or not math.isfinite(shares) or shares <= 0:
        raise ValueError(
            "Shares must be positive and measured before the decision."
        )
    if (
        observed < history["date"].min()
        or decision > history["date"].max()
    ):
        raise ValueError("Shares lie outside the split-event history.")
    splits = history.loc[
        history["date"].gt(observed)
        & history["date"].le(decision),
        "split_ratio",
    ]
    count = shares * float(splits.to_numpy(dtype=float).prod())
    if not math.isfinite(count) or count <= 0:
        raise ValueError("Invalid adjusted share count.")
    return count
