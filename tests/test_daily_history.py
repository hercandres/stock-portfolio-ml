"""Daily bars and the rolling minute year.

Every row here is synthetic. Nothing reads or writes database/stock_market.db.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from data_collection.daily_bars import (
    AlpacaDailyClient,
    AlpacaResponseError,
    DailyFetch,
    keep_trading_date,
    latest_session_date,
    parse_daily_bar,
    trading_date,
)
from data_collection.retention import apply_retention
from data_collection.sync_daily import run_daily_sync
from database.connection import (
    get_connection,
    insert_completed_minute_bars,
    insert_daily_prices,
    insert_historical_prices,
)
from database.initialize import initialize_database

NOW = datetime(2026, 11, 9, 15, 0, tzinfo=timezone.utc)
SECRET = "test-secret-value"
KEYS = {"APCA_API_KEY_ID": "test-key-id", "APCA_API_SECRET_KEY": SECRET}


def _seed_stock(db_path: Path, symbol: str) -> None:
    initialize_database(db_path)
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO stocks (symbol, company_name, is_monitored, created_at)
            VALUES (?, 'Synthetic Fixture', 1, '2026-01-01T00:00:00Z')
            """,
            (symbol,),
        )


def _day(symbol: str, stamp: str, close: float = 1.0) -> dict[str, object]:
    return {"t": stamp, "o": close, "h": close, "l": close, "c": close, "v": 3, "n": 9, "vw": 4}


class RecordingTransport:
    def __init__(self, responses: list[tuple[int, dict[str, object]]]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def __call__(self, url: str, headers: dict[str, str]) -> tuple[int, dict[str, str], str]:
        self.urls.append(url)
        index = min(len(self.urls) - 1, len(self.responses) - 1)
        status, body = self.responses[index]
        return status, {}, json.dumps(body)


def test_daily_request_is_split_adjusted_iex_from_2016() -> None:
    transport = RecordingTransport(
        [
            (
                200,
                {
                    "bars": [_day("TEST1", "2016-01-04T00:00:00Z", 2)],
                    "next_page_token": "next-fixture",
                },
            ),
            (200, {"bars": [_day("TEST1", "2026-11-06T05:00:00Z", 3)], "next_page_token": None}),
        ]
    )
    client = AlpacaDailyClient(
        "test-key-id",
        SECRET,
        transport=transport,
        pause=lambda _seconds: None,
    )
    fetched = client.historical_daily_bars("TEST1", NOW)
    assert [bar.trading_date for bar in fetched.bars] == ["2016-01-04", "2026-11-06"]
    assert fetched.bars[0].as_row()[1] == "2016-01-04"
    joined = " ".join(transport.urls)
    assert "timeframe=1Day" in joined
    assert "start=2016-01-01T00%3A00%3A00Z" in joined
    assert "feed=iex" in joined
    assert "adjustment=split" in joined
    assert "feed=sip" not in joined
    assert "/trades" not in joined
    assert "page_token=next-fixture" in transport.urls[1]
    assert SECRET not in joined


def test_open_session_day_is_dropped_and_weekend_is_dropped() -> None:
    # Monday 10:00 ET. Friday is the latest finished session.
    assert latest_session_date(NOW).isoformat() == "2026-11-06"
    today = parse_daily_bar("TEST1", _day("TEST1", "2026-11-09T05:00:00Z"))
    friday = parse_daily_bar("TEST1", _day("TEST1", "2026-11-06T05:00:00Z"))
    saturday = parse_daily_bar("TEST1", _day("TEST1", "2026-11-07T05:00:00Z"))
    assert trading_date(datetime(2016, 1, 4, tzinfo=timezone.utc)) == "2016-01-04"
    assert keep_trading_date(today.trading_date, NOW) is False
    assert keep_trading_date(friday.trading_date, NOW) is True
    assert keep_trading_date(saturday.trading_date, NOW) is False
    assert keep_trading_date("2015-12-31", NOW) is False


def test_invalid_daily_ohlc_is_rejected() -> None:
    with pytest.raises(AlpacaResponseError):
        parse_daily_bar("TEST1", {**_day("TEST1", "2026-11-06T05:00:00Z"), "h": 1, "l": 2})
    with pytest.raises(ValueError):
        AlpacaDailyClient("test-key-id", SECRET, feed="sip", pause=lambda _seconds: None)


def test_new_minute_drops_only_that_tickers_bar_from_one_year_earlier(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_roll.db"
    _seed_stock(db_path, "TEST1")
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO stocks (symbol, company_name, is_monitored, created_at)
            VALUES ('TEST2', 'Synthetic Fixture', 1, '2026-01-01T00:00:00Z')
            """
        )
        insert_historical_prices(
            connection,
            [
                ("TEST1", "2025-11-09T14:31:00Z", 1, 1, 1, 1, 1, "1m"),
                ("TEST1", "2025-11-09T14:30:00Z", 1, 1, 1, 1, 1, "1m"),
                ("TEST2", "2025-11-09T14:31:00Z", 1, 1, 1, 1, 1, "1m"),
            ],
        )
        insert_daily_prices(connection, [("TEST1", "2016-01-04", 2, 2, 2, 2, 5)])
        inserted = insert_completed_minute_bars(
            connection,
            [("TEST1", "2026-11-09T14:31:00Z", 3, 3, 3, 3, 4, "1m")],
        )
    assert inserted == 1
    with get_connection(db_path) as connection:
        minutes = connection.execute(
            """
            SELECT ticker, timestamp FROM historical_prices
            ORDER BY ticker, timestamp
            """
        ).fetchall()
        daily = connection.execute("SELECT ticker, date FROM daily_prices").fetchall()
    assert [(row["ticker"], row["timestamp"]) for row in minutes] == [
        ("TEST1", "2025-11-09T14:30:00Z"),
        ("TEST1", "2026-11-09T14:31:00Z"),
        ("TEST2", "2025-11-09T14:31:00Z"),
    ]
    assert [(row["ticker"], row["date"]) for row in daily] == [("TEST1", "2016-01-04")]


def test_retention_keeps_daily_rows_and_old_minutes_go(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_retention.db"
    _seed_stock(db_path, "TEST1")
    with get_connection(db_path) as connection:
        insert_historical_prices(
            connection,
            [("TEST1", "2020-01-02T15:00:00Z", 1, 1, 1, 1, 1, "1m")],
        )
        insert_daily_prices(connection, [("TEST1", "2016-01-04", 2, 3, 1, 2.5, 8)])
    result = apply_retention(db_path, NOW)
    assert result.deleted_rows == 1
    with get_connection(db_path) as connection:
        minutes = connection.execute("SELECT COUNT(*) AS n FROM historical_prices").fetchone()["n"]
        daily = connection.execute(
            "SELECT date, close FROM daily_prices"
        ).fetchone()
    assert minutes == 0
    assert daily["date"] == "2016-01-04"
    assert daily["close"] == 2.5


def test_daily_sync_writes_daily_prices_and_leaves_minutes(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_sync.db"
    _seed_stock(db_path, "TEST1")
    with get_connection(db_path) as connection:
        insert_historical_prices(
            connection,
            [("TEST1", "2026-11-06T15:00:00Z", 9, 9, 9, 9, 9, "1m")],
        )

    class Scripted:
        def historical_daily_bars(self, symbol: str, now: datetime) -> DailyFetch:
            bar = parse_daily_bar(symbol, _day(symbol, "2026-06-12T04:00:00Z", 4))
            return DailyFetch(bars=(bar,), raw_bar_count=1)

    outcome = run_daily_sync(
        db_path,
        tmp_path / "missing.env",
        environ=KEYS,
        client=Scripted(),
        now=NOW,
        symbols=(("TEST1", "Synthetic Fixture TEST1"),),
    )
    assert outcome.results[0].status == "success"
    assert outcome.results[0].earliest == "2026-06-12"
    with get_connection(db_path) as connection:
        daily = connection.execute("SELECT date, close FROM daily_prices").fetchone()
        minute = connection.execute(
            "SELECT timestamp, close FROM historical_prices"
        ).fetchone()
    assert daily["date"] == "2026-06-12"
    assert daily["close"] == 4
    assert minute["timestamp"] == "2026-11-06T15:00:00Z"
    assert minute["close"] == 9


def test_missing_keys_do_not_download_daily_bars(tmp_path: Path) -> None:
    calls: list[str] = []

    class Scripted:
        def historical_daily_bars(self, symbol: str, now: datetime) -> DailyFetch:
            calls.append(symbol)
            return DailyFetch(bars=(), raw_bar_count=0)

    outcome = run_daily_sync(
        tmp_path / "synthetic_missing.db",
        tmp_path / "missing.env",
        environ={},
        client=Scripted(),
        now=NOW,
        symbols=(("TEST1", "Synthetic Fixture TEST1"),),
    )
    assert outcome.missing_keys == ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")
    assert outcome.results == ()
    assert calls == []


def test_duplicate_daily_date_refreshes_the_same_row(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_upsert.db"
    _seed_stock(db_path, "TEST1")
    with get_connection(db_path) as connection:
        insert_daily_prices(connection, [("TEST1", "2016-01-04", 1, 1, 1, 1, 1)])
        insert_daily_prices(connection, [("TEST1", "2016-01-04", 2, 3, 1, 2.5, 6)])
        count = connection.execute("SELECT COUNT(*) AS n FROM daily_prices").fetchone()["n"]
        close = connection.execute("SELECT close FROM daily_prices").fetchone()["close"]
    assert count == 1
    assert close == 2.5
    # 365 days before the new minute, used so the arithmetic stays obvious.
    assert (
        datetime(2026, 11, 9, 14, 31, tzinfo=timezone.utc) - timedelta(days=365)
    ).strftime("%Y-%m-%dT%H:%M:%SZ") == "2025-11-09T14:31:00Z"
