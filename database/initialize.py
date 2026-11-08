"""Create the local SQLite database from ``schema.sql``.

Running this module creates ``database/stock_market.db`` when no path is
given. That file stays on this machine and is gitignored.
"""

from __future__ import annotations

from pathlib import Path

from .connection import get_connection

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "stock_market.db"


def initialize_database(db_path: str | Path | None = None) -> Path:
    """Apply the Stage 1 schema and return the database path.

    Existing tables and rows are kept. Missing tables are created.
    """
    path = DEFAULT_DB_PATH if db_path is None else Path(db_path)
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_connection(path) as connection:
        connection.executescript(schema_sql)
    return path


def main() -> None:
    path = initialize_database()
    print(f"Initialized {path}")


if __name__ == "__main__":
    main()
