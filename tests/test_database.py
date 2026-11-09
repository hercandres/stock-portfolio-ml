"""Stage 1 database tests.

Every row in this module is a synthetic fixture. Tickers look like TEST1.
These rows are not market data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from database.connection import (
    backup_database,
    database_file_size,
    delete_historical_prices_before,
    get_connection,
    insert_historical_prices,
)
from database.initialize import initialize_database

CREATED_AT = "2026-01-01T00:00:00Z"

EXPECTED_TABLES = {
    "stocks",
    "historical_prices",
    "daily_prices",
    "portfolio_holdings",
    "sync_status",
    "model_runs",
    "ml_predictions",
}

EXPECTED_COLUMNS = {
    "stocks": {
        "symbol",
        "company_name",
        "is_monitored",
        "timezone",
        "regular_session_open",
        "regular_session_close",
        "extended_hours_enabled",
        "created_at",
    },
    "historical_prices": {
        "id",
        "ticker",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "interval",
    },
    "daily_prices": {
        "id",
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    },
    "portfolio_holdings": {
        "id",
        "symbol",
        "quantity",
        "purchase_price",
        "purchase_time",
        "notes",
        "created_at",
    },
    "sync_status": {
        "symbol",
        "latest_success_timestamp",
        "last_attempt_timestamp",
        "status",
        "error_text",
    },
    "model_runs": {
        "id",
        "model_name",
        "started_at",
        "finished_at",
        "cutoff_timestamp",
        "metrics",
        "status",
    },
    "ml_predictions": {
        "id",
        "model_run_id",
        "ticker",
        "prediction_value",
        "horizon",
        "predicted_at",
        "created_at",
    },
}


def _add_stock(connection: sqlite3.Connection, symbol: str) -> None:
    connection.execute(
        """
        INSERT INTO stocks (symbol, company_name, created_at)
        VALUES (?, ?, ?)
        """,
        (symbol, f"Synthetic Fixture {symbol}", CREATED_AT),
    )


def _bar(
    ticker: str,
    timestamp: str,
    interval: str = "1m",
) -> tuple[str, str, float, float, float, float, int, str]:
    """One obviously fake price row. The numbers are placeholders."""
    return (ticker, timestamp, 1.0, 1.0, 1.0, 1.0, 1, interval)


def _price_rows(connection: sqlite3.Connection) -> list[tuple[str, str, str]]:
    rows = connection.execute(
        """
        SELECT ticker, timestamp, interval
        FROM historical_prices
        ORDER BY ticker, timestamp, interval
        """
    ).fetchall()
    return [(row["ticker"], row["timestamp"], row["interval"]) for row in rows]


def test_schema_creates_required_tables(db_path: Path) -> None:
    with get_connection(db_path) as connection:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        tables = {row["name"] for row in table_rows}
        assert EXPECTED_TABLES <= tables
        for table, columns in EXPECTED_COLUMNS.items():
            info = connection.execute(f"PRAGMA table_info({table})").fetchall()
            found = {row["name"] for row in info}
            assert columns <= found
        index_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
        indexes = {row["name"] for row in index_rows}
        assert "idx_historical_prices_ticker_timestamp" in indexes
        assert "idx_historical_prices_interval" in indexes
        assert "idx_daily_prices_ticker_date" in indexes
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        assert foreign_keys == 1


def test_new_stock_defaults_to_regular_us_session(db_path: Path) -> None:
    with get_connection(db_path) as connection:
        _add_stock(connection, "TEST1")
        row = connection.execute(
            "SELECT * FROM stocks WHERE symbol = ?",
            ("TEST1",),
        ).fetchone()
    assert row["company_name"] == "Synthetic Fixture TEST1"
    assert row["is_monitored"] == 0
    assert row["timezone"] == "America/New_York"
    assert row["regular_session_open"] == "09:30"
    assert row["regular_session_close"] == "16:00"
    assert row["extended_hours_enabled"] == 0


def test_reconnect_persists_rows(db_path: Path) -> None:
    with get_connection(db_path) as connection:
        _add_stock(connection, "TEST1")
        insert_historical_prices(
            connection,
            [_bar("TEST1", "2026-04-01T15:30:00Z")],
        )
    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT company_name, close, volume
            FROM stocks
            JOIN historical_prices ON historical_prices.ticker = stocks.symbol
            WHERE stocks.symbol = ?
            """,
            ("TEST1",),
        ).fetchone()
    assert row["company_name"] == "Synthetic Fixture TEST1"
    assert row["close"] == 1.0
    assert row["volume"] == 1


def test_initialize_again_keeps_existing_rows(db_path: Path) -> None:
    with get_connection(db_path) as connection:
        _add_stock(connection, "TEST1")
    initialize_database(db_path)
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT company_name FROM stocks WHERE symbol = ?",
            ("TEST1",),
        ).fetchone()
    assert row["company_name"] == "Synthetic Fixture TEST1"


def test_batch_insert_commits_both_rows(db_path: Path) -> None:
    with get_connection(db_path) as connection:
        _add_stock(connection, "TEST1")
        inserted = insert_historical_prices(
            connection,
            [
                _bar("TEST1", "2026-03-01T00:00:00Z"),
                _bar("TEST1", "2026-03-01T00:01:00Z"),
            ],
        )
        stored = _price_rows(connection)
    assert inserted == 2
    assert stored == [
        ("TEST1", "2026-03-01T00:00:00Z", "1m"),
        ("TEST1", "2026-03-01T00:01:00Z", "1m"),
    ]


def test_batch_insert_rolls_back_when_one_row_fails(db_path: Path) -> None:
    with get_connection(db_path) as connection:
        _add_stock(connection, "TEST1")
        insert_historical_prices(
            connection,
            [_bar("TEST1", "2026-03-01T00:00:00Z")],
        )
        with pytest.raises(sqlite3.IntegrityError):
            insert_historical_prices(
                connection,
                [
                    _bar("TEST1", "2026-03-02T00:00:00Z"),
                    _bar("TEST1", "2026-03-02T00:00:00Z"),
                ],
            )
        stored = _price_rows(connection)
    assert stored == [("TEST1", "2026-03-01T00:00:00Z", "1m")]


def test_unique_constraint_rejects_duplicates_and_allows_other_intervals(
    db_path: Path,
) -> None:
    with get_connection(db_path) as connection:
        _add_stock(connection, "TEST1")
        insert_historical_prices(
            connection,
            [_bar("TEST1", "2026-05-01T00:00:00Z", "1m")],
        )
        insert_historical_prices(
            connection,
            [_bar("TEST1", "2026-05-01T00:00:00Z", "5m")],
        )
        with pytest.raises(sqlite3.IntegrityError):
            insert_historical_prices(
                connection,
                [_bar("TEST1", "2026-05-01T00:00:00Z", "1m")],
            )
        stored = _price_rows(connection)
    assert stored == [
        ("TEST1", "2026-05-01T00:00:00Z", "1m"),
        ("TEST1", "2026-05-01T00:00:00Z", "5m"),
    ]


def test_failed_insert_does_not_block_another_ticker(db_path: Path) -> None:
    with get_connection(db_path) as connection:
        _add_stock(connection, "TEST1")
        _add_stock(connection, "TEST2")
        insert_historical_prices(
            connection,
            [_bar("TEST1", "2026-02-01T00:00:00Z")],
        )
        with pytest.raises(sqlite3.IntegrityError):
            insert_historical_prices(
                connection,
                [_bar("TEST1", "2026-02-01T00:00:00Z")],
            )
        insert_historical_prices(
            connection,
            [_bar("TEST2", "2026-02-01T00:00:01Z")],
        )
        stored = _price_rows(connection)
    assert stored == [
        ("TEST1", "2026-02-01T00:00:00Z", "1m"),
        ("TEST2", "2026-02-01T00:00:01Z", "1m"),
    ]


def test_wal_mode_is_active(db_path: Path) -> None:
    with get_connection(db_path) as connection:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"
    raw = sqlite3.connect(db_path)
    try:
        persisted = raw.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        raw.close()
    assert str(persisted).lower() == "wal"


def test_backup_file_contains_the_rows(db_path: Path, tmp_path: Path) -> None:
    with get_connection(db_path) as connection:
        _add_stock(connection, "TEST1")
        _add_stock(connection, "TEST2")
        insert_historical_prices(
            connection,
            [
                _bar("TEST1", "2026-07-01T00:00:00Z"),
                _bar("TEST2", "2026-07-01T00:01:00Z"),
            ],
        )
    destination = tmp_path / "synthetic_backup.db"
    backup_database(db_path, destination)
    backup = sqlite3.connect(destination)
    backup.row_factory = sqlite3.Row
    try:
        rows = backup.execute(
            """
            SELECT ticker, timestamp
            FROM historical_prices
            ORDER BY ticker, timestamp
            """
        ).fetchall()
    finally:
        backup.close()
    assert [(row["ticker"], row["timestamp"]) for row in rows] == [
        ("TEST1", "2026-07-01T00:00:00Z"),
        ("TEST2", "2026-07-01T00:01:00Z"),
    ]


def test_database_file_size_is_byte_length(db_path: Path) -> None:
    with get_connection(db_path) as connection:
        _add_stock(connection, "TEST1")
    size = database_file_size(db_path)
    assert isinstance(size, int)
    assert size == db_path.stat().st_size
    assert size > 0


def test_retention_keeps_recent_rows_and_drops_older_ones(db_path: Path) -> None:
    cutoff = "2024-06-01T00:00:00Z"
    with get_connection(db_path) as connection:
        _add_stock(connection, "TEST1")
        _add_stock(connection, "TEST2")
        insert_historical_prices(
            connection,
            [
                _bar("TEST1", "2024-05-31T23:59:59Z"),
                _bar("TEST1", "2024-06-01T00:00:00Z"),
                _bar("TEST1", "2024-06-01T00:00:01Z"),
                _bar("TEST2", "2020-01-01T00:00:00Z"),
                _bar("TEST2", "2025-01-01T00:00:00Z"),
            ],
        )
        connection.execute(
            """
            INSERT INTO portfolio_holdings (
                symbol, quantity, purchase_price, purchase_time, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "TEST1",
                3,
                1.25,
                "2020-01-01T00:00:00Z",
                "synthetic fixture note",
                CREATED_AT,
            ),
        )
        deleted = delete_historical_prices_before(connection, cutoff)
        remaining = _price_rows(connection)
        holdings = connection.execute(
            "SELECT COUNT(*) AS n FROM portfolio_holdings"
        ).fetchone()["n"]
    assert deleted == 2
    assert remaining == [
        ("TEST1", "2024-06-01T00:00:00Z", "1m"),
        ("TEST1", "2024-06-01T00:00:01Z", "1m"),
        ("TEST2", "2025-01-01T00:00:00Z", "1m"),
    ]
    assert holdings == 1


def test_retention_rejects_a_non_utc_cutoff_and_deletes_nothing(db_path: Path) -> None:
    with get_connection(db_path) as connection:
        _add_stock(connection, "TEST1")
        insert_historical_prices(
            connection,
            [_bar("TEST1", "2026-08-01T00:00:00Z")],
        )
        with pytest.raises(ValueError):
            delete_historical_prices_before(connection, "not-a-timestamp")
        remaining = _price_rows(connection)
    assert remaining == [("TEST1", "2026-08-01T00:00:00Z", "1m")]


def test_holdings_and_predictions_are_separate_synthetic_tables(db_path: Path) -> None:
    with get_connection(db_path) as connection:
        _add_stock(connection, "TEST1")
        connection.execute(
            """
            INSERT INTO portfolio_holdings (
                symbol, quantity, purchase_price, purchase_time, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("TEST1", 2, 1.0, "2026-01-02T00:00:00Z", "synthetic holding", CREATED_AT),
        )
        connection.execute(
            """
            INSERT INTO sync_status (
                symbol, latest_success_timestamp, last_attempt_timestamp, status, error_text
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("TEST1", None, "2026-01-02T00:00:00Z", "error", "synthetic failure text"),
        )
        connection.execute(
            """
            INSERT INTO model_runs (
                model_name, started_at, finished_at, cutoff_timestamp, metrics, status
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "synthetic-fixture-model",
                "2026-01-03T00:00:00Z",
                "2026-01-03T00:05:00Z",
                "2026-01-03T00:00:00Z",
                '{"synthetic": 1}',
                "succeeded",
            ),
        )
        connection.execute(
            """
            INSERT INTO ml_predictions (
                model_run_id, ticker, prediction_value, horizon, predicted_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (1, "TEST1", 1.0, "1d", "2026-01-03T00:05:00Z", "2026-01-03T00:05:00Z"),
        )
        monitored = connection.execute(
            "SELECT is_monitored FROM stocks WHERE symbol = ?",
            ("TEST1",),
        ).fetchone()["is_monitored"]
        holding_count = connection.execute(
            "SELECT COUNT(*) AS n FROM portfolio_holdings"
        ).fetchone()["n"]
        prediction = connection.execute(
            "SELECT prediction_value, horizon FROM ml_predictions"
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO ml_predictions (
                    model_run_id, ticker, prediction_value, horizon, predicted_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (99, "TEST1", 1.0, "1d", "2026-01-04T00:00:00Z", "2026-01-04T00:00:00Z"),
            )
    assert monitored == 0
    assert holding_count == 1
    assert prediction["prediction_value"] == 1.0
    assert prediction["horizon"] == "1d"
