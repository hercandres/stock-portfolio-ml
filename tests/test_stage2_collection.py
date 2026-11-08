"""Stage 2 tests.

Payloads in this module are mocked Alpaca responses or synthetic fixtures.
Prices are placeholders such as 1.0. They are not market data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from data_collection.alpaca_bars import (
    AlpacaBarsClient,
    AlpacaRequestError,
    AlpacaResponseError,
    BarFetch,
    MinuteBar,
    bar_is_storable,
    history_window,
    parse_bar,
)
from data_collection.credentials import resolve_alpaca_keys
from data_collection.seed import seed_monitored_stocks
from data_collection.sync_history import run_collection
from data_collection.universe import MONITORED_STOCKS
from database.connection import get_connection
from database.initialize import initialize_database

NY = ZoneInfo("America/New_York")
NOW = datetime(2026, 11, 8, 16, 0, tzinfo=timezone.utc)
SECRET = "test-secret-value"
KEYS = {"APCA_API_KEY_ID": "test-key-id", "APCA_API_SECRET_KEY": SECRET}


def _at(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=NY).astimezone(timezone.utc)


def _fixture_bar(when: datetime, **overrides: object) -> dict[str, object]:
    """One mocked bar. The numbers are placeholders, not a market print."""
    payload: dict[str, object] = {
        "t": when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "o": 1.0,
        "h": 1.0,
        "l": 1.0,
        "c": 1.0,
        "v": 1,
        "n": 99,
        "vw": 123.45,
    }
    payload.update(overrides)
    return payload


class RecordingTransport:
    def __init__(self, responses: list[tuple[int, dict[str, object]]]) -> None:
        self.responses = responses
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []

    def __call__(self, url: str, headers: dict[str, str]) -> tuple[int, dict[str, str], str]:
        self.urls.append(url)
        self.headers.append(dict(headers))
        index = min(len(self.urls) - 1, len(self.responses) - 1)
        status, body = self.responses[index]
        return status, {}, json.dumps(body)


def _client(transport: RecordingTransport) -> AlpacaBarsClient:
    return AlpacaBarsClient(
        KEYS["APCA_API_KEY_ID"],
        KEYS["APCA_API_SECRET_KEY"],
        transport=transport,
        pause=lambda _seconds: None,
    )


def test_universe_is_one_editable_list_of_25() -> None:
    symbols = [symbol for symbol, _name in MONITORED_STOCKS]
    assert len(symbols) == 25
    assert len(set(symbols)) == 25
    assert "GOOGL" in symbols
    assert "GOOG" not in symbols
    assert "BRK.B" in symbols
    assert "BRK-B" not in symbols
    assert symbols[0] == "AAPL"
    assert symbols[9:11] == ["LLY", "BRK.B"]


def test_seed_sets_monitored_regular_session_and_is_idempotent(db_path: Path) -> None:
    first_now = datetime(2026, 11, 1, 15, 0, tzinfo=timezone.utc)
    later = datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)
    with get_connection(db_path) as connection:
        assert seed_monitored_stocks(connection, first_now) == 25
        connection.execute(
            "UPDATE stocks SET company_name = ?, is_monitored = 0 WHERE symbol = ?",
            ("Changed Fixture", "AAPL"),
        )
        seed_monitored_stocks(connection, later)
        rows = connection.execute("SELECT * FROM stocks").fetchall()
        by_symbol = {row["symbol"]: row for row in rows}
    assert set(by_symbol) == {symbol for symbol, _name in MONITORED_STOCKS}
    for symbol, company_name in MONITORED_STOCKS:
        row = by_symbol[symbol]
        assert row["company_name"] == company_name
        assert row["is_monitored"] == 1
        assert row["timezone"] == "America/New_York"
        assert row["regular_session_open"] == "09:30"
        assert row["regular_session_close"] == "16:00"
        assert row["extended_hours_enabled"] == 0
        assert row["created_at"] == "2026-11-01T15:00:00Z"


def test_history_window_stops_before_the_recent_15_minutes() -> None:
    start, end = history_window(NOW)
    assert end == datetime(2026, 11, 8, 15, 44, tzinfo=timezone.utc)
    assert start == datetime(2025, 11, 8, 15, 44, tzinfo=timezone.utc)
    assert (end - start).days == 365


def test_regular_session_filter_drops_extended_close_and_weekend() -> None:
    regular = _at(2026, 11, 4, 10, 0)
    opening = _at(2026, 11, 4, 9, 30)
    premarket = _at(2026, 11, 4, 9, 29)
    closing_print = _at(2026, 11, 4, 16, 0)
    summer = _at(2026, 7, 15, 9, 30)
    weekend = _at(2026, 11, 8, 10, 0)
    assert bar_is_storable(regular, NOW)
    assert bar_is_storable(opening, NOW)
    assert bar_is_storable(summer, NOW)
    assert not bar_is_storable(premarket, NOW)
    assert not bar_is_storable(closing_print, NOW)
    assert not bar_is_storable(weekend, NOW)
    incomplete = datetime(2026, 11, 8, 15, 44, tzinfo=timezone.utc)
    assert not bar_is_storable(incomplete, datetime(2026, 11, 8, 15, 44, 30, tzinfo=timezone.utc))


def test_client_requests_iex_split_adjusted_minute_bars_and_drops_extra_fields() -> None:
    first = _at(2026, 11, 4, 10, 0)
    second = _at(2026, 11, 4, 10, 1)
    transport = RecordingTransport(
        [
            (
                200,
                {
                    "bars": [_fixture_bar(first)],
                    "symbol": "TEST1",
                    "next_page_token": "next-fixture",
                },
            ),
            (
                200,
                {"bars": [_fixture_bar(second)], "symbol": "TEST1", "next_page_token": None},
            ),
        ]
    )
    fetched = _client(transport).historical_minute_bars("TEST1", NOW)
    assert len(fetched.bars) == 2
    assert fetched.raw_bar_count == 2
    assert [bar.timestamp for bar in fetched.bars] == [
        "2026-11-04T15:00:00Z",
        "2026-11-04T15:01:00Z",
    ]
    assert fetched.bars[0].as_row()[-1] == "1m"
    assert not hasattr(fetched.bars[0], "trade_count")
    joined = " ".join(transport.urls)
    assert "/v2/stocks/TEST1/bars?" in joined
    assert "timeframe=1Min" in joined
    assert "feed=iex" in joined
    assert "adjustment=split" in joined
    assert "feed=sip" not in joined
    assert "/trades" not in joined
    assert "/quotes" not in joined
    assert "page_token=next-fixture" in transport.urls[1]
    assert transport.headers[0]["APCA-API-KEY-ID"] == "test-key-id"
    assert transport.headers[0]["APCA-API-SECRET-KEY"] == SECRET
    assert SECRET not in joined


def test_client_retries_429_and_hides_the_secret_on_403() -> None:
    regular = _at(2026, 11, 4, 10, 0)
    retry = RecordingTransport(
        [
            (429, {"message": "slow down"}),
            (200, {"bars": [_fixture_bar(regular)], "symbol": "TEST1", "next_page_token": None}),
        ]
    )
    fetched = _client(retry).historical_minute_bars("TEST1", NOW)
    assert len(retry.urls) == 2
    assert len(fetched.bars) == 1
    forbidden = RecordingTransport([(403, {"message": "forbidden"})])
    with pytest.raises(AlpacaRequestError) as caught:
        _client(forbidden).historical_minute_bars("TEST1", NOW)
    assert caught.value.status == 403
    assert "forbidden" in str(caught.value)
    assert SECRET not in str(caught.value)
    assert len(forbidden.urls) == 1


def test_client_rejects_a_non_iex_feed_and_invalid_ohlc() -> None:
    with pytest.raises(ValueError):
        AlpacaBarsClient("test-key-id", SECRET, feed="sip", pause=lambda _seconds: None)
    with pytest.raises(AlpacaResponseError):
        parse_bar("TEST1", _fixture_bar(_at(2026, 11, 4, 10, 0), h=1.0, l=2.0))


def test_client_keeps_raw_count_when_no_bar_is_in_the_regular_session() -> None:
    weekend = _at(2026, 11, 8, 10, 0)
    transport = RecordingTransport(
        [(200, {"bars": [_fixture_bar(weekend)], "symbol": "TEST1", "next_page_token": None})]
    )
    fetched = _client(transport).historical_minute_bars("TEST1", NOW)
    assert fetched.raw_bar_count == 1
    assert fetched.bars == ()


def test_missing_keys_seed_the_universe_and_do_not_download(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_stage2.db"

    class Boom:
        def historical_minute_bars(self, symbol: str, now: datetime) -> BarFetch:
            raise AssertionError(f"download started for {symbol}")

    outcome = run_collection(
        db_path,
        tmp_path / "missing.env",
        environ={},
        client=Boom(),
        now=NOW,
    )
    assert outcome.downloaded is False
    assert outcome.missing_keys == ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")
    assert outcome.results == ()
    with get_connection(db_path) as connection:
        count = connection.execute("SELECT COUNT(*) AS n FROM stocks").fetchone()["n"]
        prices = connection.execute("SELECT COUNT(*) AS n FROM historical_prices").fetchone()["n"]
        status_count = connection.execute("SELECT COUNT(*) AS n FROM sync_status").fetchone()["n"]
    assert count == 25
    assert prices == 0
    assert status_count == 0


def test_credentials_ignore_blank_values_and_prefer_the_process(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "APCA_API_KEY_ID=\nAPCA_API_SECRET_KEY=file-secret\n",
        encoding="utf-8",
    )
    resolved, missing = resolve_alpaca_keys(env_path, {})
    assert resolved is None
    assert missing == ["APCA_API_KEY_ID"]
    resolved, missing = resolve_alpaca_keys(
        env_path,
        {"APCA_API_KEY_ID": "process-key", "APCA_API_SECRET_KEY": "process-secret"},
    )
    assert missing == []
    assert resolved == {
        "APCA_API_KEY_ID": "process-key",
        "APCA_API_SECRET_KEY": "process-secret",
    }


def test_one_ticker_failure_does_not_block_another_and_duplicates_do_not_double(
    tmp_path: Path,
) -> None:
    regular = _at(2026, 11, 4, 10, 0)
    saved = parse_bar("TEST2", _fixture_bar(regular))
    db_path = tmp_path / "synthetic_stage2.db"

    class Scripted:
        def historical_minute_bars(self, symbol: str, now: datetime) -> BarFetch:
            if symbol == "TEST1":
                raise AlpacaRequestError("TEST1", 500, "fixture failure")
            return BarFetch(bars=(saved,), raw_bar_count=1)

    symbols = (
        ("TEST1", "Synthetic Fixture TEST1"),
        ("TEST2", "Synthetic Fixture TEST2"),
    )
    first = run_collection(
        db_path,
        tmp_path / "missing.env",
        environ=KEYS,
        client=Scripted(),
        now=NOW,
        symbols=symbols,
    )
    assert [(item.symbol, item.status, item.inserted) for item in first.results] == [
        ("TEST1", "error", 0),
        ("TEST2", "success", 1),
    ]
    with get_connection(db_path) as connection:
        stored = connection.execute(
            "SELECT ticker, close, volume, interval FROM historical_prices"
        ).fetchall()
        statuses = connection.execute(
            "SELECT symbol, status, error_text, latest_success_timestamp FROM sync_status"
        ).fetchall()
    assert [(row["ticker"], row["close"], row["volume"], row["interval"]) for row in stored] == [
        ("TEST2", 1.0, 1, "1m")
    ]
    by_status = {row["symbol"]: row for row in statuses}
    assert by_status["TEST1"]["status"] == "error"
    assert "fixture failure" in by_status["TEST1"]["error_text"]
    assert by_status["TEST1"]["latest_success_timestamp"] is None
    assert by_status["TEST2"]["status"] == "success"
    assert by_status["TEST2"]["latest_success_timestamp"] == saved.timestamp
    assert by_status["TEST2"]["error_text"] is None

    second = run_collection(
        db_path,
        tmp_path / "missing.env",
        environ=KEYS,
        client=Scripted(),
        now=NOW,
        symbols=(("TEST2", "Synthetic Fixture TEST2"),),
    )
    assert second.results[0].status == "success"
    assert second.results[0].inserted == 0
    with get_connection(db_path) as connection:
        count = connection.execute("SELECT COUNT(*) AS n FROM historical_prices").fetchone()["n"]
    assert count == 1


def test_empty_response_is_not_stored_as_bars(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_stage2.db"

    class Empty:
        def historical_minute_bars(self, symbol: str, now: datetime) -> BarFetch:
            return BarFetch(bars=(), raw_bar_count=0)

    class Filtered:
        def historical_minute_bars(self, symbol: str, now: datetime) -> BarFetch:
            return BarFetch(bars=(), raw_bar_count=2)

    symbols = (("TEST1", "Synthetic Fixture TEST1"),)
    empty = run_collection(
        db_path,
        tmp_path / "missing.env",
        environ=KEYS,
        client=Empty(),
        now=NOW,
        symbols=symbols,
    )
    assert empty.results[0].status == "empty"
    assert "no 1-minute bars" in empty.results[0].message
    filtered = run_collection(
        db_path,
        tmp_path / "missing.env",
        environ=KEYS,
        client=Filtered(),
        now=NOW,
        symbols=symbols,
    )
    assert filtered.results[0].status == "empty"
    assert "regular-session" in filtered.results[0].message
    with get_connection(db_path) as connection:
        prices = connection.execute("SELECT COUNT(*) AS n FROM historical_prices").fetchone()["n"]
        latest = connection.execute(
            "SELECT latest_success_timestamp FROM sync_status WHERE symbol = ?",
            ("TEST1",),
        ).fetchone()["latest_success_timestamp"]
    assert prices == 0
    assert latest is None


def test_db_path_fixture_still_initializes(db_path: Path) -> None:
    """The shared fixture still builds an empty schema before a seed test uses it."""
    assert initialize_database(db_path) == db_path
    with get_connection(db_path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    assert "historical_prices" in {row["name"] for row in tables}
    assert set(MinuteBar.__dataclass_fields__) == {
        "ticker",
        "started_at",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
