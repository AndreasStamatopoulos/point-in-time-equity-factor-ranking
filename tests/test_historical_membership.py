"""Tests for pinned historical S&P 500 membership loader (stdlib only)."""
import importlib.util
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("historical_membership", ROOT / "10_build_historical_membership.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_membership_counts_on_historical_dates():
    entries = mod.read_intervals()
    for day in ("2015-01-30", "2018-01-31", "2020-01-31", "2023-01-31", "2025-01-31"):
        members = mod.members_on(entries, date.fromisoformat(day))
        assert 480 <= len(members) <= 515
        assert len(members) == len(set(members))


def test_reentry_and_delisting_intervals():
    entries = mod.read_intervals()
    amd = [(begin, end) for ticker, begin, end in entries if ticker == "AMD"]
    assert len(amd) >= 2
    assert "AMD" not in mod.members_on(entries, date(2016, 1, 31))
    assert "AMD" in mod.members_on(entries, date(2018, 1, 31))


def test_start_end_boundary_and_no_future_member():
    entries = [("AAA", date(2020, 2, 1), date(2021, 1, 1))]
    assert mod.members_on(entries, date(2020, 1, 31)) == []
    assert mod.members_on(entries, date(2020, 2, 1)) == ["AAA"]
    assert mod.members_on(entries, date(2021, 1, 1)) == []


def test_overlapping_same_symbol_is_rejected():
    entries = [("AAA", date(2019, 1, 1), None), ("AAA", date(2020, 1, 1), None)]
    try:
        mod.members_on(entries, date(2020, 2, 1))
    except ValueError as exc:
        assert "Overlapping" in str(exc)
    else:
        raise AssertionError("Duplicate ticker not rejected")


def test_month_ends_do_not_exceed_end():
    dates = list(mod.month_ends(date(2020, 1, 1), date(2020, 3, 31)))
    assert dates == [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)]
