"""Stage 3 tests.

Payloads here are mocked Alpaca responses or synthetic fixtures. Prices are
placeholders such as 1.0. They are not market data. No test opens a real
socket or leaves a session thread running.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from data_collection.alpaca_bars import (
    MIN_REQUEST_INTERVAL_SECONDS,
    AlpacaBarsClient,
    AlpacaRequestError,
    BarFetch,
    bar_is_storable,
    history_window,
    is_contiguous_regular_bar,
    live_bar_is_storable,
    missing_bar_window,
    parse_bar,
)
from data_collection.incremental import run_incremental
from data_collection.live_session import (
    IEX_STREAM_URL,
    MarketDataSession,
    open_iex_socket,
    subscription_message,
)
from data_collection.retention import SIZE_CAP_BYTES, SIZE_WARNING_BYTES, apply_retention
from data_collection.sync_history import record_sync_status
from database.connection import get_connection, insert_historical_prices
from database.initialize import initialize_database

NY = ZoneInfo("America/New_York")
NOW = datetime(2026, 11, 8, 16, 0, tzinfo=timezone.utc)
LIVE_NOW = datetime(2026, 11, 4, 15, 0, tzinfo=timezone.utc)
SECRET = "test-secret-value"
KEYS = {"APCA_API_KEY_ID": "test-key-id", "APCA_API_SECRET_KEY": SECRET}
SYMBOLS = (
    ("TEST1", "Synthetic Fixture TEST1"),
    ("TEST2", "Synthetic Fixture TEST2"),
)


def _at(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=NY).astimezone(timezone.utc)


def _fixture_bar(when: datetime, **overrides: object) -> dict[str, object]:
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


def _seed_stock(db_path: Path, symbol: str = "TEST1") -> None:
    initialize_database(db_path)
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO stocks (symbol, company_name, is_monitored, created_at)
            VALUES (?, ?, 1, ?)
            """,
            (symbol, f"Synthetic Fixture {symbol}", "2026-11-04T15:00:00Z"),
        )


class RecordingTransport:
    def __init__(self, responses: list[tuple[int, dict[str, object]]]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def __call__(self, url: str, headers: dict[str, str]) -> tuple[int, dict[str, str], str]:
        self.urls.append(url)
        index = min(len(self.urls) - 1, len(self.responses) - 1)
        status, body = self.responses[index]
        return status, {}, json.dumps(body)


class ScriptedBars:
    def __init__(self, fetches: dict[str, BarFetch | Exception] | None = None) -> None:
        self.fetches = fetches or {}
        self.calls: list[tuple[str, datetime, datetime]] = []

    def minute_bars_between(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        now: datetime,
    ) -> BarFetch:
        self.calls.append((symbol, start, end))
        outcome = self.fetches.get(symbol, BarFetch((), 0))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class ScriptedSocket:
    def __init__(self, messages: list[str], session: MarketDataSession | None = None) -> None:
        self.messages = list(messages)
        self.sent: list[str] = []
        self.closed = False
        self._session = session

    def send(self, data: str) -> None:
        self.sent.append(data)

    def recv(self) -> str:
        if self.messages:
            return self.messages.pop(0)
        if self._session is not None:
            self._session._stop.set()
        raise ConnectionError("drop")

    def close(self) -> None:
        self.closed = True


def _message(payload: object) -> str:
    return json.dumps(payload)


def _handshake() -> list[str]:
    return [
        _message([{"T": "success", "msg": "connected"}]),
        _message([{"T": "success", "msg": "authenticated"}]),
        _message([{"T": "subscription", "bars": ["TEST1"], "quotes": ["TEST1"]}]),
    ]


def test_historical_pace_stays_under_200_calls_per_minute() -> None:
    assert MIN_REQUEST_INTERVAL_SECONDS > 0
    assert 60 / MIN_REQUEST_INTERVAL_SECONDS <= 200
    assert IEX_STREAM_URL == "wss://stream.data.alpaca.markets/v2/iex"


def test_missing_window_uses_the_cursor_and_not_the_full_year() -> None:
    cursor = _at(2026, 11, 6, 15, 59)
    window = missing_bar_window(NOW, cursor, cursor)
    assert window is not None
    start, end = window
    full_start, full_end = history_window(NOW)
    assert start == cursor + timedelta(minutes=1)
    assert end == full_end
    assert start > full_start


def test_stored_history_without_a_cursor_does_not_request_the_full_year() -> None:
    newest = _at(2026, 11, 4, 10, 0)
    window = missing_bar_window(NOW, None, newest)
    assert window is not None
    start, _end = window
    full_start, _full_end = history_window(NOW)
    assert start == newest + timedelta(minutes=1)
    assert start > full_start
    assert missing_bar_window(NOW, None, None) == history_window(NOW)


def test_friday_close_is_contiguous_with_monday_open() -> None:
    friday = _at(2026, 11, 6, 15, 59)
    monday_open = _at(2026, 11, 9, 9, 30)
    monday_later = _at(2026, 11, 9, 9, 31)
    assert is_contiguous_regular_bar(friday, monday_open)
    assert not is_contiguous_regular_bar(friday, monday_later)


def test_incremental_requests_only_the_gap_and_keeps_iex_limits(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_stage3.db"
    _prepare_cursor(db_path, "TEST1", "2026-11-06T20:59:00Z")
    weekend = _at(2026, 11, 8, 10, 0)
    transport = RecordingTransport(
        [(200, {"bars": [_fixture_bar(weekend)], "symbol": "TEST1", "next_page_token": None})]
    )
    client = AlpacaBarsClient("test-key-id", SECRET, transport=transport, pause=lambda _s: None)
    outcome = run_incremental(
        db_path,
        tmp_path / "missing.env",
        environ=KEYS,
        client=client,
        now=NOW,
        symbols=(("TEST1", "Synthetic Fixture TEST1"),),
    )
    assert outcome.results[0].status == "success"
    assert outcome.results[0].inserted == 0
    assert "Nothing new" in outcome.results[0].message
    joined = " ".join(transport.urls)
    assert "feed=iex" in joined
    assert "adjustment=split" in joined
    assert "timeframe=1Min" in joined
    assert "start=2026-11-06T21%3A00%3A00Z" in joined
    assert "end=2026-11-08T15%3A44%3A00Z" in joined
    assert "feed=sip" not in joined
    assert "feed=otc" not in joined
    assert "/trades" not in joined
    assert "/quotes" not in joined
    with get_connection(db_path) as connection:
        prices = connection.execute("SELECT COUNT(*) AS n FROM historical_prices").fetchone()["n"]
        cursor = connection.execute(
            "SELECT latest_success_timestamp, status FROM sync_status WHERE symbol = 'TEST1'"
        ).fetchone()
    assert prices == 0
    assert cursor["status"] == "success"
    assert cursor["latest_success_timestamp"] == "2026-11-06T20:59:00Z"


def test_closed_poll_skips_http_when_no_session_minute_is_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_stage3.db"
    _prepare_cursor(db_path, "TEST1", "2026-11-06T20:59:00Z")
    client = ScriptedBars()
    outcome = run_incremental(
        db_path,
        tmp_path / "missing.env",
        environ=KEYS,
        client=client,
        now=NOW,
        symbols=(("TEST1", "Synthetic Fixture TEST1"),),
        skip_when_closed=True,
    )
    assert client.calls == []
    assert outcome.results[0].status == "success"
    assert outcome.results[0].inserted == 0


def test_one_ticker_failure_does_not_stop_the_gap_fill(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_stage3.db"
    saved = parse_bar("TEST2", _fixture_bar(_at(2026, 11, 4, 10, 0)))
    client = ScriptedBars(
        {
            "TEST1": AlpacaRequestError("TEST1", 404, "fixture missing"),
            "TEST2": BarFetch((saved,), 1),
        }
    )
    first = run_incremental(
        db_path,
        tmp_path / "missing.env",
        environ=KEYS,
        client=client,
        now=NOW,
        symbols=SYMBOLS,
    )
    assert [item.symbol for item in first.results] == ["TEST1", "TEST2"]
    assert first.results[0].status == "error"
    assert first.results[1].status == "success"
    assert first.results[1].inserted == 1
    assert client.calls[0][0] == "TEST1"
    assert client.calls[1][0] == "TEST2"
    # The second pass starts after the saved bar, so the year is not requested again.
    second = run_incremental(
        db_path,
        tmp_path / "missing.env",
        environ=KEYS,
        client=client,
        now=NOW,
        symbols=(("TEST2", "Synthetic Fixture TEST2"),),
    )
    assert second.results[0].inserted == 0
    assert client.calls[-1][1] == saved.started_at + timedelta(minutes=1)
    with get_connection(db_path) as connection:
        count = connection.execute("SELECT COUNT(*) AS n FROM historical_prices").fetchone()["n"]
    assert count == 1


def test_history_without_cursor_starts_after_the_newest_bar(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_stage3.db"
    _seed_stock(db_path, "TEST1")
    newest = parse_bar("TEST1", _fixture_bar(_at(2026, 11, 4, 10, 0)))
    with get_connection(db_path) as connection:
        insert_historical_prices(connection, [newest.as_row()])
    client = ScriptedBars()
    outcome = run_incremental(
        db_path,
        tmp_path / "missing.env",
        environ=KEYS,
        client=client,
        now=NOW,
        symbols=(("TEST1", "Synthetic Fixture TEST1"),),
    )
    assert outcome.results[0].status == "success"
    assert client.calls[0][1] == newest.started_at + timedelta(minutes=1)


def test_already_current_cursor_does_not_call_alpaca(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_stage3.db"
    _end = history_window(NOW)[1]
    _prepare_cursor(db_path, "TEST1", _end.strftime("%Y-%m-%dT%H:%M:%SZ"))
    client = ScriptedBars()
    outcome = run_incremental(
        db_path,
        tmp_path / "missing.env",
        environ=KEYS,
        client=client,
        now=NOW,
        symbols=(("TEST1", "Synthetic Fixture TEST1"),),
    )
    assert client.calls == []
    assert outcome.results[0].status == "success"
    assert outcome.inserted == 0


def test_missing_keys_do_not_download(tmp_path: Path) -> None:
    client = ScriptedBars()
    outcome = run_incremental(
        tmp_path / "synthetic_stage3.db",
        tmp_path / "missing.env",
        environ={},
        client=client,
        now=NOW,
        symbols=(("TEST1", "Synthetic Fixture TEST1"),),
    )
    assert outcome.missing_keys == ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")
    assert outcome.results == ()
    assert client.calls == []


def test_recent_live_minute_is_stored_but_historical_backfill_skips_it() -> None:
    recent = _at(2026, 11, 4, 9, 50)
    assert live_bar_is_storable(recent, LIVE_NOW)
    assert not bar_is_storable(recent, LIVE_NOW)


def test_retention_deletes_only_rows_older_than_365_days(db_path: Path) -> None:
    _seed_stock(db_path, "TEST1")
    cutoff = (NOW - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
    older = "2025-11-08T15:59:59Z"
    with get_connection(db_path) as connection:
        insert_historical_prices(
            connection,
            [
                ("TEST1", older, 1, 1, 1, 1, 1, "1m"),
                ("TEST1", cutoff, 1, 1, 1, 1, 1, "1m"),
                ("TEST1", "2026-11-04T15:00:00Z", 1, 1, 1, 1, 1, "1m"),
            ],
        )
    result = apply_retention(db_path, NOW)
    assert result.deleted_rows == 1
    assert result.cutoff == cutoff
    with get_connection(db_path) as connection:
        kept = connection.execute(
            "SELECT timestamp FROM historical_prices ORDER BY timestamp"
        ).fetchall()
    assert [row["timestamp"] for row in kept] == [cutoff, "2026-11-04T15:00:00Z"]


def test_retention_logs_size_and_warns_above_4gb(
    db_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "data_collection.retention.database_file_size",
        lambda _path: SIZE_WARNING_BYTES + 1,
    )
    with caplog.at_level(logging.INFO, logger="data_collection.retention"):
        apply_retention(db_path, NOW)
    warnings = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "exceeds 4 GB" in warnings[0]
    assert "5 GB" in warnings[0]
    assert str(SIZE_CAP_BYTES) in warnings[0]
    assert any("database file size is" in record.message for record in caplog.records)

    caplog.clear()
    monkeypatch.setattr(
        "data_collection.retention.database_file_size",
        lambda _path: SIZE_WARNING_BYTES,
    )
    with caplog.at_level(logging.INFO, logger="data_collection.retention"):
        apply_retention(db_path, NOW)
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)


def test_stream_writes_completed_bars_keeps_quotes_in_memory_and_reconnects(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "synthetic_stage3.db"
    _seed_stock(db_path, "TEST1")
    recent = _at(2026, 11, 4, 9, 50)
    bar = _fixture_bar(
        recent,
        **{
            "T": "b",
            "S": "TEST1",
            "t": recent.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        },
    )
    quote = {"T": "q", "S": "TEST1", "bp": 1.0, "ap": 1.25, "t": "2026-11-04T14:50:00Z"}
    trade = {"T": "t", "S": "TEST1", "p": 9.5, "s": 1, "t": "2026-11-04T14:50:01Z"}
    updated = {"T": "u", "S": "TEST1", **_fixture_bar(_at(2026, 11, 4, 9, 51))}
    sockets: list[ScriptedSocket] = []
    catch_ups: list[bool] = []

    def factory() -> ScriptedSocket:
        if not sockets:
            socket = ScriptedSocket(_handshake())
        else:
            socket = ScriptedSocket(
                _handshake() + [_message([bar, bar, quote, trade, updated])],
                session=session,
            )
        sockets.append(socket)
        return socket

    session = MarketDataSession(
        db_path,
        tmp_path / "missing.env",
        symbols=("TEST1",),
        socket_factory=factory,
        catch_up=catch_ups.append,
        clock=lambda: LIVE_NOW,
        backoff=lambda _attempt: 0,
        api_key_id="test-key-id",
        api_secret_key=SECRET,
        join_timeout=2,
    )
    session.run()
    assert len(sockets) == 2
    assert catch_ups == [False, False]
    subscribe = json.loads(sockets[1].sent[1])
    assert subscribe == subscription_message(("TEST1",))
    assert "trades" not in subscribe
    assert sockets[0].closed
    assert sockets[1].closed
    quote_snapshot = session.quote_for("TEST1")
    assert quote_snapshot is not None
    assert quote_snapshot.bid == 1.0
    assert quote_snapshot.ask == 1.25
    with get_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT timestamp, close, interval FROM historical_prices"
        ).fetchall()
        cursor = connection.execute(
            "SELECT latest_success_timestamp FROM sync_status WHERE symbol = 'TEST1'"
        ).fetchone()
    assert [(row["timestamp"], row["close"], row["interval"]) for row in rows] == [
        ("2026-11-04T14:50:00Z", 1.0, "1m")
    ]
    assert cursor["latest_success_timestamp"] == "2026-11-04T14:50:00Z"
    assert not any(thread.name == "market-data-session" and thread.is_alive() for thread in threading.enumerate())


def test_live_bar_does_not_advance_the_cursor_across_a_gap(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_stage3.db"
    _prepare_cursor(db_path, "TEST1", "2026-11-04T14:40:00Z")
    later = _fixture_bar(_at(2026, 11, 4, 9, 50), **{"T": "b", "S": "TEST1"})
    socket = ScriptedSocket(_handshake() + [_message([later])])

    def factory() -> ScriptedSocket:
        socket._session = session
        return socket

    session = MarketDataSession(
        db_path,
        tmp_path / "missing.env",
        symbols=("TEST1",),
        socket_factory=factory,
        catch_up=lambda _closed: None,
        clock=lambda: LIVE_NOW,
        backoff=lambda _attempt: 0,
        api_key_id="test-key-id",
        api_secret_key=SECRET,
    )
    session.run()
    with get_connection(db_path) as connection:
        count = connection.execute("SELECT COUNT(*) AS n FROM historical_prices").fetchone()["n"]
        cursor = connection.execute(
            "SELECT latest_success_timestamp FROM sync_status WHERE symbol = 'TEST1'"
        ).fetchone()["latest_success_timestamp"]
    assert count == 1
    assert cursor == "2026-11-04T14:40:00Z"


def test_auth_failure_does_not_reconnect(tmp_path: Path) -> None:
    calls: list[int] = []

    def factory() -> ScriptedSocket:
        calls.append(1)
        if len(calls) > 1:
            raise AssertionError("reconnect after auth failure")
        return ScriptedSocket(
            [
                _message([{"T": "success", "msg": "connected"}]),
                _message([{"T": "error", "code": 402, "msg": "auth failed"}]),
            ]
        )

    session = MarketDataSession(
        tmp_path / "synthetic_stage3.db",
        tmp_path / "missing.env",
        symbols=("TEST1",),
        socket_factory=factory,
        catch_up=lambda _closed: None,
        clock=lambda: LIVE_NOW,
        api_key_id="test-key-id",
        api_secret_key=SECRET,
    )
    session.run()
    assert calls == [1]


def test_start_and_stop_join_the_session_thread(tmp_path: Path) -> None:
    hold = _HoldingSocket()
    session = MarketDataSession(
        tmp_path / "synthetic_stage3.db",
        tmp_path / "missing.env",
        symbols=("TEST1",),
        socket_factory=lambda: hold,
        catch_up=lambda _closed: None,
        clock=lambda: LIVE_NOW,
        api_key_id="test-key-id",
        api_secret_key=SECRET,
        join_timeout=2,
    )
    try:
        session.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not any("subscribe" in item for item in hold.sent):
            time.sleep(0.02)
        assert any("subscribe" in item for item in hold.sent)
        assert "trades" not in hold.sent[1]
    finally:
        session.stop()
    assert session._thread is not None
    assert not session._thread.is_alive()
    assert hold.closed


def test_missing_keys_and_non_iex_url_do_not_open_a_socket(tmp_path: Path) -> None:
    def factory() -> ScriptedSocket:
        raise AssertionError("socket opened")

    session = MarketDataSession(
        tmp_path / "synthetic_stage3.db",
        tmp_path / "missing.env",
        symbols=("TEST1",),
        socket_factory=factory,
        catch_up=lambda _closed: None,
        environ={},
    )
    session.run()
    with pytest.raises(ValueError, match="free IEX stream only"):
        open_iex_socket("wss://stream.data.alpaca.markets/v2/sip")
    with pytest.raises(ValueError, match="30 symbols"):
        MarketDataSession(
            tmp_path / "synthetic_stage3.db",
            tmp_path / "missing.env",
            symbols=tuple(f"TEST{index}" for index in range(31)),
        )


class _HoldingSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._closed = threading.Event()
        self._messages = _handshake()

    def send(self, data: str) -> None:
        self.sent.append(data)

    def recv(self) -> str:
        if self._messages:
            return self._messages.pop(0)
        if self._closed.wait(2):
            raise ConnectionError("closed")
        raise TimeoutError("stop did not close the socket")

    def close(self) -> None:
        self.closed = True
        self._closed.set()


def _prepare_cursor(db_path: Path, symbol: str, timestamp: str) -> None:
    _seed_stock(db_path, symbol)
    record_sync_status(
        db_path,
        symbol,
        "success",
        timestamp,
        None,
        NOW,
    )
