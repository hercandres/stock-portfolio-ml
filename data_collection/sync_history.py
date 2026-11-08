"""Seed the universe, then download historical minute bars when keys exist.

A missing key stops the process before any HTTP call. One ticker's failure
is recorded on ``sync_status`` and does not stop the remaining tickers.
Bars are stored only in SQLite.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from database.connection import (
    UTC_TIMESTAMP_FORMAT,
    ensure_utc_timestamp,
    get_connection,
    insert_historical_prices_ignoring_duplicates,
    transaction,
)
from database.initialize import DEFAULT_DB_PATH, initialize_database

from .alpaca_bars import AlpacaBarsClient, BarFetch
from .credentials import REQUIRED_ALPACA_KEYS, resolve_alpaca_keys
from .seed import seed_monitored_stocks
from .universe import MONITORED_STOCKS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


class BarsSource(Protocol):
    """Anything that returns completed bars for one symbol. Tests supply fakes."""

    def historical_minute_bars(self, symbol: str, now: datetime) -> BarFetch:
        """Return the bars to store for ``symbol`` as of ``now``."""
        ...


@dataclass(frozen=True)
class SymbolSync:
    symbol: str
    status: str
    inserted: int
    latest_timestamp: str | None
    message: str


@dataclass(frozen=True)
class CollectionRun:
    database_path: Path
    seeded_symbols: int
    missing_keys: tuple[str, ...]
    results: tuple[SymbolSync, ...]

    @property
    def downloaded(self) -> bool:
        return not self.missing_keys


def run_collection(
    db_path: str | Path,
    env_path: str | Path,
    *,
    environ: dict[str, str] | None = None,
    client: BarsSource | None = None,
    now: datetime | None = None,
    symbols: tuple[tuple[str, str], ...] | None = None,
) -> CollectionRun:
    """Seed stocks, then sync bars for ``symbols`` when credentials exist.

    ``client`` is set by tests. The command-line entry point leaves it empty
    and builds a client only after both Alpaca keys are present.
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
        return CollectionRun(path, seeded, tuple(missing), ())
    if client is None:
        assert resolved is not None
        client = AlpacaBarsClient(
            resolved[REQUIRED_ALPACA_KEYS[0]],
            resolved[REQUIRED_ALPACA_KEYS[1]],
        )
    results: list[SymbolSync] = []
    for symbol, _company_name in universe:
        results.append(_sync_one(path, client, symbol, moment))
    return CollectionRun(path, seeded, (), tuple(results))


def _sync_one(
    db_path: Path,
    client: BarsSource,
    symbol: str,
    now: datetime,
) -> SymbolSync:
    try:
        fetched = client.historical_minute_bars(symbol, now)
    except Exception as exc:
        message = _public_error(exc)
        _write_status(db_path, symbol, "error", None, message, now)
        return SymbolSync(symbol, "error", 0, None, message)
    if not fetched.bars:
        if fetched.raw_bar_count == 0:
            message = "Alpaca returned no 1-minute bars for this window."
        else:
            message = "Alpaca returned bars, but none were completed regular-session minutes."
        _write_status(db_path, symbol, "empty", None, message, now)
        return SymbolSync(symbol, "empty", 0, None, message)
    try:
        with get_connection(db_path) as connection:
            inserted = insert_historical_prices_ignoring_duplicates(
                connection,
                [bar.as_row() for bar in fetched.bars],
            )
    except Exception as exc:
        message = _public_error(exc)
        _write_status(db_path, symbol, "error", None, message, now)
        return SymbolSync(symbol, "error", 0, None, message)
    latest = max(bar.timestamp for bar in fetched.bars)
    message = f"Saved {inserted} new 1-minute bars."
    _write_status(db_path, symbol, "success", latest, None, now)
    return SymbolSync(symbol, "success", inserted, latest, message)


def _write_status(
    db_path: Path,
    symbol: str,
    status: str,
    latest_success: str | None,
    error_text: str | None,
    now: datetime,
) -> None:
    if latest_success is not None:
        latest_success = ensure_utc_timestamp(latest_success)
    attempted = now.astimezone(timezone.utc).strftime(UTC_TIMESTAMP_FORMAT)
    with get_connection(db_path) as connection:
        with transaction(connection):
            connection.execute(
                """
                INSERT INTO sync_status (
                    symbol, latest_success_timestamp, last_attempt_timestamp,
                    status, error_text
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    latest_success_timestamp = COALESCE(
                        excluded.latest_success_timestamp,
                        sync_status.latest_success_timestamp
                    ),
                    last_attempt_timestamp = excluded.last_attempt_timestamp,
                    status = excluded.status,
                    error_text = excluded.error_text
                """,
                (symbol, latest_success, attempted, status, error_text),
            )


def _seed_explicit(connection, symbols: tuple[tuple[str, str], ...], now: datetime) -> int:
    """Seed a caller-supplied list. Tests use this with synthetic tickers."""
    created_at = now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with transaction(connection):
        for symbol, company_name in symbols:
            connection.execute(
                """
                INSERT INTO stocks (
                    symbol, company_name, is_monitored, created_at
                ) VALUES (?, ?, 1, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    company_name = excluded.company_name,
                    is_monitored = 1
                """,
                (symbol, company_name, created_at),
            )
    return len(symbols)


def _public_error(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    if len(text) > 300:
        text = text[:300] + "..."
    return text or exc.__class__.__name__


def main() -> int:
    outcome = run_collection(DEFAULT_DB_PATH, DEFAULT_ENV_PATH)
    print(f"Seeded {outcome.seeded_symbols} monitored stocks in {outcome.database_path}")
    if outcome.missing_keys:
        print("No download was started. Alpaca API keys are missing.")
        print("In the Alpaca dashboard, open API Keys on the right sidebar and generate a key.")
        print("https://app.alpaca.markets/")
        print("Put these names in .env. The secret is shown once. Do not commit .env.")
        for name in outcome.missing_keys:
            print(f"  {name}")
        return 2
    saved = sum(item.inserted for item in outcome.results)
    print(f"Inserted {saved} new rows into historical_prices.")
    for item in outcome.results:
        detail = item.message
        print(f"{item.symbol}: {item.status} ({detail})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
