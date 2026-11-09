"""Delete minute rows older than 365 days while this process is running.

``historical_prices`` is the rolling year of 1-minute bars. ``daily_prices``
is a separate table and is kept. Nothing here registers a Windows service,
a startup task, or a scheduler. The incremental sync and the live session
call this before they return or while their process is still alive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from database.connection import (
    UTC_TIMESTAMP_FORMAT,
    database_file_size,
    delete_historical_prices_before,
    get_connection,
)

logger = logging.getLogger("data_collection.retention")

RETENTION_DAYS = 365
SIZE_WARNING_BYTES = 4 * 1024 * 1024 * 1024
SIZE_CAP_BYTES = 5 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class RetentionResult:
    deleted_rows: int
    database_bytes: int
    cutoff: str


def apply_retention(db_path: str | Path, now: datetime) -> RetentionResult:
    """Delete 1-minute bars strictly older than 365 calendar days.

    A minute at the cutoff stays. Newer minutes stay. Daily bars are not
    deleted. The main database file size is logged. A file larger than 4 GB
    is a warning. The storage cap target is 5 GB; this function does not
    delete rows inside the 365-day window to chase that number.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    cutoff = (now.astimezone(timezone.utc) - timedelta(days=RETENTION_DAYS)).strftime(
        UTC_TIMESTAMP_FORMAT
    )
    with get_connection(db_path) as connection:
        deleted = delete_historical_prices_before(connection, cutoff)
    size = database_file_size(db_path)
    logger.info(
        "database file size is %s bytes; retention deleted %s historical_prices rows older than %s",
        size,
        deleted,
        cutoff,
    )
    if size > SIZE_WARNING_BYTES:
        logger.warning(
            "database file size %s bytes exceeds 4 GB; the cap target is %s bytes (5 GB)",
            size,
            SIZE_CAP_BYTES,
        )
    return RetentionResult(deleted, size, cutoff)
