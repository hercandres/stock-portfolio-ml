"""Insert the monitored universe into ``stocks``.

Every seeded row is watched (``is_monitored`` = 1) and uses the regular US
session. Extended hours stay off. Running this again updates those fields
and keeps the original ``created_at``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from database.connection import UTC_TIMESTAMP_FORMAT, get_connection, transaction
from database.initialize import DEFAULT_DB_PATH, initialize_database

from .universe import MONITORED_STOCKS


def seed_monitored_stocks(
    connection,
    now: datetime | None = None,
) -> int:
    """Upsert the universe. Returns how many symbols are in the list."""
    moment = now if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    created_at = moment.astimezone(timezone.utc).strftime(UTC_TIMESTAMP_FORMAT)
    with transaction(connection):
        for symbol, company_name in MONITORED_STOCKS:
            connection.execute(
                """
                INSERT INTO stocks (
                    symbol, company_name, is_monitored, timezone,
                    regular_session_open, regular_session_close,
                    extended_hours_enabled, created_at
                ) VALUES (?, ?, 1, 'America/New_York', '09:30', '16:00', 0, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    company_name = excluded.company_name,
                    is_monitored = 1,
                    timezone = 'America/New_York',
                    regular_session_open = '09:30',
                    regular_session_close = '16:00',
                    extended_hours_enabled = 0
                """,
                (symbol, company_name, created_at),
            )
    return len(MONITORED_STOCKS)


def main() -> None:
    path = initialize_database(DEFAULT_DB_PATH)
    with get_connection(path) as connection:
        count = seed_monitored_stocks(connection)
    print(f"Seeded {count} monitored stocks in {path}")


if __name__ == "__main__":
    main()
