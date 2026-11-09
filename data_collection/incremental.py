"""Catch up missing IEX minute bars, then apply rolling retention.

The request for each ticker starts the minute after
``sync_status.latest_success_timestamp``. Stored history with an empty
cursor starts after the newest stored bar. The full 365-day window is
requested only when that ticker has no history. A closed market, weekend,
or holiday is success with nothing new. One ticker's failure does not stop
the others.

This module does not stay running and does not register a scheduler.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from database.connection import (
    UTC_TIMESTAMP_FORMAT,
    get_connection,
    insert_completed_minute_bars,
)
from database.initialize import DEFAULT_DB_PATH, initialize_database

from .alpaca_bars import (
    MIN_REQUEST_INTERVAL_SECONDS,
    AlpacaBarsClient,
    BarFetch,
    missing_bar_window,
    range_has_regular_session,
)
from .credentials import REQUIRED_ALPACA_KEYS, resolve_alpaca_keys
from .retention import apply_retention
from .seed import seed_monitored_stocks
from .sync_history import (
    DEFAULT_ENV_PATH,
    SymbolSync,
    _public_error,
    _seed_explicit,
    record_sync_status,
)
from .universe import MONITORED_STOCKS

class RangeBarsSource(Protocol):
    """Historical bars for one inclusive window. Tests supply fakes."""

    def minute_bars_between(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        now: datetime,
    ) -> BarFetch:
        """Return completed bars for ``symbol`` inside the window."""
        ...


@dataclass(frozen=True)
class IncrementalRun:
    database_path: Path
    seeded_symbols: int
    missing_keys: tuple[str, ...]
    results: tuple[SymbolSync, ...]
    deleted_rows: int
    database_bytes: int

    @property
    def inserted(self) -> int:
        return sum(item.inserted for item in self.results)


def stored_timestamp(value: str) -> datetime:
    """Parse one stored UTC timestamp. The text is already second-aligned."""
    return datetime.strptime(value, UTC_TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)


def run_incremental(
    db_path: str | Path,
    env_path: str | Path,
    *,
    environ: dict[str, str] | None = None,
    client: RangeBarsSource | None = None,
    now: datetime | None = None,
    symbols: tuple[tuple[str, str], ...] | None = None,
    skip_when_closed: bool = False,
) -> IncrementalRun:
    """Seed, request only missing bars, and delete prices older than 365 days.

    ``skip_when_closed`` avoids a historical request when the missing window
    contains no regular-session minute. The one-shot command leaves it false
    so a weekend run still asks Alpaca. The open session uses it on the quiet
    poll so a weekend does not call Alpaca every few minutes.
    """
    if MIN_REQUEST_INTERVAL_SECONDS <= 0 or (60 / MIN_REQUEST_INTERVAL_SECONDS) > 200:
        raise RuntimeError("historical requests must stay under 200 calls per minute")
    moment = now if now is not None else datetime.now(timezone.utc)
    path = initialize_database(db_path)
    universe = symbols if symbols is not None else MONITORED_STOCKS
    with get_connection(path) as connection:
        if symbols is None:
            seeded = seed_monitored_stocks(connection, moment)
        else:
            seeded = _seed_explicit(connection, universe, moment)
    env = os.environ if environ is None else environ
    resolved, missing = resolve_alpaca_keys(Path(env_path), env)
    results: list[SymbolSync] = []
    if not missing:
        if client is None:
            assert resolved is not None
            client = AlpacaBarsClient(
                resolved[REQUIRED_ALPACA_KEYS[0]],
                resolved[REQUIRED_ALPACA_KEYS[1]],
            )
        for symbol, _company_name in universe:
            results.append(
                _sync_symbol(path, client, symbol, moment, skip_when_closed=skip_when_closed)
            )
    retention = apply_retention(path, moment)
    return IncrementalRun(
        path,
        seeded,
        tuple(missing),
        tuple(results),
        retention.deleted_rows,
        retention.database_bytes,
    )


def _sync_symbol(
    db_path: Path,
    client: RangeBarsSource,
    symbol: str,
    now: datetime,
    *,
    skip_when_closed: bool,
) -> SymbolSync:
    cursor_text, newest_text = _markers(db_path, symbol)
    window = missing_bar_window(
        now,
        stored_timestamp(cursor_text) if cursor_text else None,
        stored_timestamp(newest_text) if newest_text else None,
    )
    if window is None:
        record_sync_status(db_path, symbol, "success", None, None, now)
        return SymbolSync(
            symbol,
            "success",
            0,
            cursor_text,
            "Already current. Nothing new to store.",
        )
    start, end = window
    if skip_when_closed and not range_has_regular_session(start, end):
        record_sync_status(db_path, symbol, "success", None, None, now)
        return SymbolSync(
            symbol,
            "success",
            0,
            cursor_text,
            "Market is closed. Nothing new to store.",
        )
    try:
        fetched = client.minute_bars_between(symbol, start, end, now)
    except Exception as exc:
        message = _public_error(exc)
        record_sync_status(db_path, symbol, "error", None, message, now)
        return SymbolSync(symbol, "error", 0, cursor_text, message)
    if not fetched.bars:
        record_sync_status(db_path, symbol, "success", None, None, now)
        return SymbolSync(
            symbol,
            "success",
            0,
            cursor_text,
            "No new regular-session bars. Nothing new to store.",
        )
    try:
        with get_connection(db_path) as connection:
            inserted = insert_completed_minute_bars(
                connection,
                [bar.as_row() for bar in fetched.bars],
            )
    except Exception as exc:
        message = _public_error(exc)
        record_sync_status(db_path, symbol, "error", None, message, now)
        return SymbolSync(symbol, "error", 0, cursor_text, message)
    latest = max(bar.timestamp for bar in fetched.bars)
    message = f"Saved {inserted} new 1-minute bars."
    record_sync_status(db_path, symbol, "success", latest, None, now)
    return SymbolSync(symbol, "success", inserted, latest, message)


def _markers(db_path: Path, symbol: str) -> tuple[str | None, str | None]:
    with get_connection(db_path) as connection:
        status = connection.execute(
            "SELECT latest_success_timestamp FROM sync_status WHERE symbol = ?",
            (symbol,),
        ).fetchone()
        newest = connection.execute(
            """
            SELECT MAX(timestamp) AS latest
            FROM historical_prices
            WHERE ticker = ? AND interval = '1m'
            """,
            (symbol,),
        ).fetchone()
    cursor = None if status is None else status["latest_success_timestamp"]
    latest = None if newest is None else newest["latest"]
    return cursor, latest


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    outcome = run_incremental(DEFAULT_DB_PATH, DEFAULT_ENV_PATH)
    if outcome.missing_keys:
        print("No download was started. Alpaca API keys are missing.")
        print("Put APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env. Do not commit .env.")
        for name in outcome.missing_keys:
            print(f"  {name}")
        return 2
    print(f"Inserted {outcome.inserted} new rows into historical_prices.")
    print(
        "Retention deleted "
        f"{outcome.deleted_rows} rows. Database file size is {outcome.database_bytes} bytes."
    )
    for item in outcome.results:
        print(f"{item.symbol}: {item.status} inserted={item.inserted} ({item.message})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
