"""SQLite connection helper for one local writer.

Stage 1 stores rows only. It does not download prices or train models.
Timestamps are UTC ISO-8601 text: YYYY-MM-DDTHH:MM:SSZ. That fixed width
sorts in chronological order, which is what retention relies on.
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
UTC_DATE_FORMAT = "%Y-%m-%d"
_UTC_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_UTC_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_INSERT_PRICE_SQL = """
INSERT INTO historical_prices (
    ticker, timestamp, open, high, low, close, volume, interval
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_PRICE_IGNORE_SQL = """
INSERT OR IGNORE INTO historical_prices (
    ticker, timestamp, open, high, low, close, volume, interval
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

# A second download of the same day refreshes the open, high, low, close, and volume.
_UPSERT_DAILY_SQL = """
INSERT INTO daily_prices (
    ticker, date, open, high, low, close, volume
) VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(ticker, date) DO UPDATE SET
    open = excluded.open,
    high = excluded.high,
    low = excluded.low,
    close = excluded.close,
    volume = excluded.volume
"""

# Serializes writers inside this process. SQLite still enforces one writer
# across processes; busy_timeout is set on each connection for that case.
_write_lock = threading.Lock()


def utc_now_iso() -> str:
    """Return the current UTC time as YYYY-MM-DDTHH:MM:SSZ."""
    return datetime.now(timezone.utc).strftime(UTC_TIMESTAMP_FORMAT)


def ensure_utc_timestamp(value: str) -> str:
    """Return value when it is a real UTC timestamp in the storage format."""
    if not isinstance(value, str) or _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise ValueError(
            "timestamp must be UTC ISO-8601 text in the form YYYY-MM-DDTHH:MM:SSZ"
        )
    datetime.strptime(value, UTC_TIMESTAMP_FORMAT)
    return value


def ensure_utc_date(value: str) -> str:
    """Return value when it is a real calendar date, YYYY-MM-DD."""
    if not isinstance(value, str) or _UTC_DATE_RE.fullmatch(value) is None:
        raise ValueError("date must be text in the form YYYY-MM-DD")
    datetime.strptime(value, UTC_DATE_FORMAT)
    return value


def timestamp_days_earlier(timestamp: str, days: int) -> str:
    """Same clock time, ``days`` earlier. Used to drop a minute from one year ago."""
    if days < 0:
        raise ValueError("days must be zero or positive")
    started = datetime.strptime(ensure_utc_timestamp(timestamp), UTC_TIMESTAMP_FORMAT)
    started = started.replace(tzinfo=timezone.utc)
    return (started - timedelta(days=days)).strftime(UTC_TIMESTAMP_FORMAT)


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open one autocommit connection with foreign keys and WAL enabled.

    Prefer ``get_connection`` for writes so this process keeps a single writer.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
    if foreign_keys != 1:
        connection.close()
        raise sqlite3.OperationalError("foreign keys could not be enabled")
    journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
    if str(journal_mode).lower() != "wal":
        connection.close()
        raise sqlite3.OperationalError(f"WAL mode was not enabled (got {journal_mode})")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


@contextmanager
def get_connection(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Yield this process's single-writer connection.

    The lock is not re-entrant. Do not call ``get_connection``,
    ``backup_database``, or ``initialize_database`` from inside this block.
    Closing the connection rolls back a transaction that was left open.
    """
    with _write_lock:
        connection = connect(db_path)
        try:
            yield connection
        finally:
            connection.close()


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run one immediate transaction, rolling back if the body raises."""
    if connection.in_transaction:
        raise RuntimeError("a transaction is already open on this connection")
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise


def insert_historical_prices(
    connection: sqlite3.Connection,
    rows: Sequence[Sequence[object]],
) -> int:
    """Insert price rows with ``executemany`` in one transaction.

    Each row is ``(ticker, timestamp, open, high, low, close, volume, interval)``.
    A failure rolls the whole batch back and leaves the connection usable.
    """
    payload = tuple(_normalize_price_row(row) for row in rows)
    if not payload:
        return 0
    with transaction(connection):
        connection.executemany(_INSERT_PRICE_SQL, payload)
    return len(payload)


def insert_historical_prices_ignoring_duplicates(
    connection: sqlite3.Connection,
    rows: Sequence[Sequence[object]],
) -> int:
    """Insert price rows and skip rows that repeat the unique key.

    The unique key is still ``(ticker, timestamp, interval)``. A duplicate
    is ignored. Any other failure rolls the batch back. Returns how many
    rows were newly stored.
    """
    payload = tuple(_normalize_price_row(row) for row in rows)
    if not payload:
        return 0
    with transaction(connection):
        before = connection.total_changes
        connection.executemany(_INSERT_PRICE_IGNORE_SQL, payload)
        return int(connection.total_changes - before)


def insert_completed_minute_bars(
    connection: sqlite3.Connection,
    rows: Sequence[Sequence[object]],
) -> int:
    """Insert 1-minute bars, then drop each ticker's bar from 365 days earlier.

    A new minute replaces the minute from one year ago, so
    ``historical_prices`` stays about one year long. Rows that are already
    stored are left alone, and their older partner is not deleted again.
    ``daily_prices`` is a different table and is not touched.
    """
    normalized = [_normalize_price_row(row) for row in rows]
    payload = tuple(sorted(normalized, key=lambda item: (item[0], item[1])))
    if not payload:
        return 0
    inserted = 0
    with transaction(connection):
        for row in payload:
            ticker, timestamp, _open, _high, _low, _close, _volume, interval = row
            before = connection.total_changes
            connection.execute(_INSERT_PRICE_IGNORE_SQL, row)
            if connection.total_changes == before:
                continue
            inserted += 1
            # Only the rolling minute table loses the bar from one year ago.
            if interval != "1m":
                continue
            year_ago = timestamp_days_earlier(timestamp, 365)
            connection.execute(
                """
                DELETE FROM historical_prices
                WHERE ticker = ? AND interval = '1m' AND timestamp = ?
                """,
                (ticker, year_ago),
            )
    return inserted


def insert_daily_prices(
    connection: sqlite3.Connection,
    rows: Sequence[Sequence[object]],
) -> int:
    """Insert or refresh daily bars. Each row is ticker, date, open, high, low, close, volume.

    The date is one trading day, YYYY-MM-DD. The same ticker and date updates
    the prices instead of adding a second row. Minute bars are not written here.
    """
    payload = tuple(_normalize_daily_row(row) for row in rows)
    if not payload:
        return 0
    with transaction(connection):
        connection.executemany(_UPSERT_DAILY_SQL, payload)
    return len(payload)


def delete_historical_prices_before(
    connection: sqlite3.Connection,
    cutoff: str,
) -> int:
    """Delete 1-minute rows strictly older than ``cutoff``.

    Rows at the cutoff and rows newer than the cutoff stay. ``cutoff`` uses
    the same UTC text format as stored minute timestamps. Daily bars live in
    ``daily_prices`` and this delete does not read that table.
    """
    cutoff_text = ensure_utc_timestamp(cutoff)
    with transaction(connection):
        connection.execute(
            """
            DELETE FROM historical_prices
            WHERE interval = '1m' AND timestamp < ?
            """,
            (cutoff_text,),
        )
        deleted = int(connection.execute("SELECT changes()").fetchone()[0])
    return deleted


def backup_database(db_path: str | Path, destination: str | Path) -> Path:
    """Copy the database to ``destination`` with the SQLite backup API."""
    source_path = Path(db_path)
    dest_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(f"database file not found: {source_path}")
    if source_path.resolve() == dest_path.resolve():
        raise ValueError("backup destination must be a different file")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(source_path) as source:
        target = sqlite3.connect(dest_path)
        try:
            with target:
                source.backup(target)
        finally:
            target.close()
    return dest_path


def database_file_size(db_path: str | Path) -> int:
    """Return the main database file size in bytes.

    WAL and shared-memory sidecars are separate files and are not included.
    """
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"database file not found: {path}")
    return path.stat().st_size


def _normalize_daily_row(
    row: Sequence[object],
) -> tuple[str, str, float, float, float, float, int]:
    if len(row) != 7:
        raise ValueError(
            "a daily price row must have ticker, date, open, high, low, close, and volume"
        )
    ticker, trading_date, open_, high, low, close, volume = row
    if not isinstance(trading_date, str):
        raise ValueError("date must be text in the form YYYY-MM-DD")
    open_price = _price(open_, "open")
    high_price = _price(high, "high")
    low_price = _price(low, "low")
    close_price = _price(close, "close")
    if high_price < low_price or open_price < low_price or open_price > high_price:
        raise ValueError("daily open and high must sit inside the low-high range")
    if close_price < low_price or close_price > high_price:
        raise ValueError("daily close must sit inside the low-high range")
    return (
        _required_text(ticker, "ticker"),
        ensure_utc_date(trading_date),
        open_price,
        high_price,
        low_price,
        close_price,
        _volume(volume),
    )


def _normalize_price_row(
    row: Sequence[object],
) -> tuple[str, str, float, float, float, float, int, str]:
    if len(row) != 8:
        raise ValueError(
            "a historical price row must have ticker, timestamp, open, high, "
            "low, close, volume, and interval"
        )
    ticker, timestamp, open_, high, low, close, volume, interval = row
    if not isinstance(timestamp, str):
        raise ValueError(
            "timestamp must be UTC ISO-8601 text in the form YYYY-MM-DDTHH:MM:SSZ"
        )
    return (
        _required_text(ticker, "ticker"),
        ensure_utc_timestamp(timestamp),
        _price(open_, "open"),
        _price(high, "high"),
        _price(low, "low"),
        _price(close, "close"),
        _volume(volume),
        _required_text(interval, "interval"),
    )


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or value.strip() == "" or value != value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _price(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _volume(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("volume must be a non-negative integer")
    return value
