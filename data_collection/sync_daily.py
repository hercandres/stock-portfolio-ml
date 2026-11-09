"""Download daily bars into ``daily_prices``.

Minute bars are not requested and not deleted. One ticker's failure does
not stop the others. A missing Alpaca key stops the process before any
HTTP call. The secret is never printed.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

from database.connection import get_connection, insert_daily_prices
from database.initialize import DEFAULT_DB_PATH, initialize_database

from .credentials import REQUIRED_ALPACA_KEYS, resolve_alpaca_keys
from .daily_bars import DAILY_START, AlpacaDailyClient, DailyFetch, latest_session_date
from .seed import seed_monitored_stocks
from .sync_history import DEFAULT_ENV_PATH, _public_error, _seed_explicit
from .universe import MONITORED_STOCKS


class DailySource(Protocol):
    """Anything that returns daily bars for one symbol. Tests supply fakes."""

    def historical_daily_bars(self, symbol: str, now: datetime) -> DailyFetch:
        """Return the daily bars to store for ``symbol`` as of ``now``."""
        ...


class DailyGapSource(Protocol):
    """Daily bars from one start time. The live session uses this for catch-up."""

    def daily_bars_from(self, symbol: str, start: datetime, now: datetime) -> DailyFetch:
        """Return finished daily bars at or after ``start``."""
        ...


@dataclass(frozen=True)
class SymbolDaily:
    symbol: str
    status: str
    stored: int
    earliest: str | None
    latest: str | None
    message: str


@dataclass(frozen=True)
class DailyRun:
    database_path: Path
    seeded_symbols: int
    missing_keys: tuple[str, ...]
    results: tuple[SymbolDaily, ...]

    @property
    def downloaded(self) -> bool:
        return not self.missing_keys


def run_daily_sync(
    db_path: str | Path,
    env_path: str | Path,
    *,
    environ: dict[str, str] | None = None,
    client: DailySource | None = None,
    now: datetime | None = None,
    symbols: tuple[tuple[str, str], ...] | None = None,
) -> DailyRun:
    """Seed the watchlist, then store daily bars for each symbol.

    ``client`` is set by tests. The command-line entry point builds the
    Alpaca client only after both keys are present.
    """
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
    if missing:
        return DailyRun(path, seeded, tuple(missing), ())
    if client is None:
        assert resolved is not None
        client = AlpacaDailyClient(
            resolved[REQUIRED_ALPACA_KEYS[0]],
            resolved[REQUIRED_ALPACA_KEYS[1]],
        )
    results = [_sync_one(path, client, symbol, moment) for symbol, _name in universe]
    return DailyRun(path, seeded, (), tuple(results))


def catch_up_daily(
    db_path: str | Path,
    env_path: str | Path,
    *,
    environ: dict[str, str] | None = None,
    client: DailyGapSource | None = None,
    now: datetime | None = None,
    symbols: tuple[tuple[str, str], ...] | None = None,
) -> DailyRun:
    """Store finished daily bars that are not in ``daily_prices`` yet.

    Each ticker starts the day after its newest stored date. Minute bars are
    not requested and not deleted. Nothing is asked when the table is already
    through the latest finished session.
    """
    moment = now if now is not None else datetime.now(timezone.utc)
    path = initialize_database(db_path)
    universe = symbols if symbols is not None else MONITORED_STOCKS
    env = os.environ if environ is None else environ
    resolved, missing = resolve_alpaca_keys(Path(env_path), env)
    if missing:
        return DailyRun(path, 0, tuple(missing), ())
    if client is None:
        assert resolved is not None
        client = AlpacaDailyClient(
            resolved[REQUIRED_ALPACA_KEYS[0]],
            resolved[REQUIRED_ALPACA_KEYS[1]],
        )
    results = [_catch_one(path, client, symbol, moment) for symbol, _name in universe]
    return DailyRun(path, len(universe), (), tuple(results))


def _catch_one(db_path: Path, client: DailyGapSource, symbol: str, now: datetime) -> SymbolDaily:
    latest = _latest_daily_date(db_path, symbol)
    start_day = DAILY_START if latest is None else date.fromisoformat(latest) + timedelta(days=1)
    if start_day > latest_session_date(now):
        return SymbolDaily(symbol, "success", 0, latest, latest, "Daily bars are already current.")
    start = datetime(start_day.year, start_day.month, start_day.day, tzinfo=timezone.utc)
    try:
        fetched = client.daily_bars_from(symbol, start, now)
    except Exception as exc:
        return SymbolDaily(symbol, "error", 0, latest, latest, _public_error(exc))
    if not fetched.bars:
        return SymbolDaily(symbol, "success", 0, latest, latest, "No new finished daily bar.")
    try:
        with get_connection(db_path) as connection:
            stored = insert_daily_prices(connection, [bar.as_row() for bar in fetched.bars])
    except Exception as exc:
        return SymbolDaily(symbol, "error", 0, latest, latest, _public_error(exc))
    earliest = min(bar.trading_date for bar in fetched.bars)
    newest = max(bar.trading_date for bar in fetched.bars)
    return SymbolDaily(
        symbol,
        "success",
        stored,
        earliest,
        newest,
        f"Saved {stored} daily bars, {earliest} through {newest}.",
    )


def _latest_daily_date(db_path: Path, symbol: str) -> str | None:
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT MAX(date) AS latest FROM daily_prices WHERE ticker = ?",
            (symbol,),
        ).fetchone()
    if row is None:
        return None
    return row["latest"]


def _sync_one(db_path: Path, client: DailySource, symbol: str, now: datetime) -> SymbolDaily:
    try:
        fetched = client.historical_daily_bars(symbol, now)
    except Exception as exc:
        message = _public_error(exc)
        return SymbolDaily(symbol, "error", 0, None, None, message)
    if not fetched.bars:
        if fetched.raw_bar_count == 0:
            message = "Alpaca returned no daily bars for this window."
        else:
            message = "Alpaca returned bars, but none were completed trading days."
        return SymbolDaily(symbol, "empty", 0, None, None, message)
    try:
        with get_connection(db_path) as connection:
            stored = insert_daily_prices(connection, [bar.as_row() for bar in fetched.bars])
    except Exception as exc:
        message = _public_error(exc)
        return SymbolDaily(symbol, "error", 0, None, None, message)
    earliest = min(bar.trading_date for bar in fetched.bars)
    latest = max(bar.trading_date for bar in fetched.bars)
    message = f"Saved {stored} daily bars, {earliest} through {latest}."
    return SymbolDaily(symbol, "success", stored, earliest, latest, message)


def main() -> int:
    outcome = run_daily_sync(DEFAULT_DB_PATH, DEFAULT_ENV_PATH)
    print(f"Daily bars use {outcome.database_path}")
    if outcome.missing_keys:
        print("No download was started. Alpaca API keys are missing.")
        print("Put APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env. Do not commit .env.")
        for name in outcome.missing_keys:
            print(f"  {name}")
        return 2
    saved = sum(item.stored for item in outcome.results)
    print(f"Wrote {saved} daily rows into daily_prices. Minute bars were not changed.")
    for item in outcome.results:
        print(f"{item.symbol}: {item.status} ({item.message})")
    with get_connection(outcome.database_path) as connection:
        summary = connection.execute(
            "SELECT COUNT(*) AS n, MIN(date) AS first_day, MAX(date) AS last_day FROM daily_prices"
        ).fetchone()
    print(
        f"daily_prices has {summary['n']} rows, "
        f"{summary['first_day']} through {summary['last_day']}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
