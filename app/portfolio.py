"""Read and write portfolio rows. Holdings stay off the watchlist.

Nothing in this module downloads prices or replaces stored bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data_collection.alpaca_bars import NEW_YORK, is_regular_us_session
from data_collection.retention import SIZE_CAP_BYTES, SIZE_WARNING_BYTES
from data_collection.universe import MONITORED_STOCKS
from database.connection import (
    UTC_TIMESTAMP_FORMAT,
    database_file_size,
    ensure_utc_timestamp,
    get_connection,
    transaction,
)

from .indicators import PriceBar, fetch_start

_SYMBOL_ORDER = {symbol: index for index, (symbol, _name) in enumerate(MONITORED_STOCKS)}


@dataclass(frozen=True)
class Position:
    symbol: str
    company_name: str
    quantity: float
    purchase_price: float
    latest_close: float | None
    latest_timestamp: str | None

    @property
    def invested(self) -> float:
        return self.quantity * self.purchase_price

    @property
    def market_value(self) -> float | None:
        if self.latest_close is None:
            return None
        return self.quantity * self.latest_close

    @property
    def unrealized(self) -> float | None:
        value = self.market_value
        if value is None:
            return None
        return value - self.invested

    def weight(self, total_value: float) -> float | None:
        value = self.market_value
        if value is None or total_value <= 0:
            return None
        return value / total_value


@dataclass(frozen=True)
class PortfolioTotals:
    invested: float
    market_value: float
    unrealized: float
    missing_prices: tuple[str, ...]


@dataclass(frozen=True)
class StockSnapshot:
    symbol: str
    latest_timestamp: str | None
    latest_close: float | None
    previous_close: float | None
    daily_change: float | None
    latest_volume: int | None
    session_volume: int | None


@dataclass(frozen=True)
class DatabaseOverview:
    monitored: int
    price_rows: int
    daily_rows: int
    latest_sync: str | None
    last_attempt: str | None
    success_count: int
    error_count: int
    error_symbols: tuple[str, ...]
    database_bytes: int
    cap_bytes: int
    warning_bytes: int
    market: str


def market_state(now: datetime) -> str:
    """``open`` during the regular US session, otherwise ``closed``."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if is_regular_us_session(now):
        return "open"
    return "closed"


def market_banner(now: datetime, latest_bar: str | None) -> tuple[str, str]:
    """Short status line. A closed market still points at stored bars."""
    state = market_state(now)
    if state == "closed":
        when = ""
        if latest_bar:
            when = f" Latest stored bar: {format_et(latest_bar)}."
        text = (
            "Market is closed."
            + when
            + " Charts and prices stay on the latest stored IEX bars."
        )
        return state, text
    return state, (
        "Market is open. Prices are stored IEX bars. The free historical feed "
        "is about 15 minutes behind. Completed minute bars update while Start Portfolio is open."
    )


def format_et(timestamp: str) -> str:
    parsed = datetime.strptime(ensure_utc_timestamp(timestamp), UTC_TIMESTAMP_FORMAT)
    local = parsed.replace(tzinfo=timezone.utc).astimezone(NEW_YORK)
    return local.strftime("%Y-%m-%d %H:%M ET")


def format_money(value: float) -> str:
    return f"${value:,.2f}"


def format_shares(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def format_percent(value: float) -> str:
    return f"{value:+.2%}"


def save_position(
    db_path: str | Path,
    symbol: str,
    quantity: float,
    purchase_price: float,
    now: datetime,
) -> str:
    """Replace the saved position for one symbol.

    Zero shares removes the holding. The watchlist row is left alone.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if quantity < 0 or purchase_price < 0:
        raise ValueError("shares and purchase price cannot be negative")
    stamp = now.astimezone(timezone.utc).strftime(UTC_TIMESTAMP_FORMAT)
    with get_connection(db_path) as connection:
        stock = connection.execute(
            "SELECT is_monitored FROM stocks WHERE symbol = ?",
            (symbol,),
        ).fetchone()
        if stock is None:
            raise ValueError("that symbol is not in the watchlist")
        with transaction(connection):
            connection.execute("DELETE FROM portfolio_holdings WHERE symbol = ?", (symbol,))
            if quantity == 0:
                return "cleared"
            connection.execute(
                """
                INSERT INTO portfolio_holdings (
                    symbol, quantity, purchase_price, purchase_time, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (symbol, quantity, purchase_price, stamp, stamp),
            )
    return "saved"


def list_positions(db_path: str | Path) -> list[Position]:
    """One row per monitored stock. Symbols without a holding have quantity 0."""
    with get_connection(db_path) as connection:
        stocks = connection.execute(
            """
            SELECT symbol, company_name
            FROM stocks
            WHERE is_monitored = 1
            """
        ).fetchall()
        holdings = connection.execute(
            """
            SELECT symbol, quantity, purchase_price
            FROM portfolio_holdings
            """
        ).fetchall()
        prices = connection.execute(
            """
            SELECT ticker, timestamp, close
            FROM historical_prices
            WHERE interval = '1m'
              AND (ticker, timestamp) IN (
                  SELECT ticker, MAX(timestamp)
                  FROM historical_prices
                  WHERE interval = '1m'
                  GROUP BY ticker
              )
            """
        ).fetchall()
    quantities: dict[str, list[tuple[float, float]]] = {}
    for row in holdings:
        quantities.setdefault(row["symbol"], []).append((row["quantity"], row["purchase_price"]))
    closes = {row["ticker"]: (row["close"], row["timestamp"]) for row in prices}
    positions: list[Position] = []
    for stock in stocks:
        symbol = stock["symbol"]
        lots = quantities.get(symbol, [])
        quantity = sum(lot[0] for lot in lots)
        if quantity > 0:
            purchase = sum(lot[0] * lot[1] for lot in lots) / quantity
        else:
            purchase = 0.0
        close = closes.get(symbol)
        positions.append(
            Position(
                symbol=symbol,
                company_name=stock["company_name"],
                quantity=quantity,
                purchase_price=purchase,
                latest_close=None if close is None else float(close[0]),
                latest_timestamp=None if close is None else close[1],
            )
        )
    positions.sort(key=lambda item: (_SYMBOL_ORDER.get(item.symbol, 10_000), item.symbol))
    return positions


def portfolio_totals(positions: list[Position]) -> PortfolioTotals:
    """Value uses the latest stored close. Invested uses the saved purchase price."""
    invested = 0.0
    market_value = 0.0
    invested_priced = 0.0
    missing: list[str] = []
    for position in positions:
        if position.quantity <= 0:
            continue
        invested += position.invested
        if position.market_value is None:
            missing.append(position.symbol)
            continue
        market_value += position.market_value
        invested_priced += position.invested
    return PortfolioTotals(
        invested=invested,
        market_value=market_value,
        unrealized=market_value - invested_priced,
        missing_prices=tuple(missing),
    )


def stock_snapshot(symbol: str, bars: list[PriceBar]) -> StockSnapshot:
    """Latest close, prior session close, and volume from stored bars only."""
    ordered = sorted(bars, key=lambda bar: bar.started_at)
    if not ordered:
        return StockSnapshot(symbol, None, None, None, None, None, None)
    latest = ordered[-1]
    by_day: dict[object, PriceBar] = {}
    session_volume = 0
    latest_day = latest.started_at.astimezone(NEW_YORK).date()
    for bar in ordered:
        by_day[bar.started_at.astimezone(NEW_YORK).date()] = bar
        if bar.started_at.astimezone(NEW_YORK).date() == latest_day:
            session_volume += bar.volume
    days = sorted(by_day)
    previous = by_day[days[-2]].close if len(days) >= 2 else None
    change = None
    if previous not in (None, 0):
        change = (latest.close - previous) / previous
    return StockSnapshot(
        symbol=symbol,
        latest_timestamp=latest.started_at.strftime(UTC_TIMESTAMP_FORMAT),
        latest_close=latest.close,
        previous_close=previous,
        daily_change=change,
        latest_volume=latest.volume,
        session_volume=session_volume,
    )


def load_bars(
    db_path: str | Path,
    symbol: str,
    *,
    start: datetime | None = None,
) -> list[PriceBar]:
    """Stored 1-minute bars for one symbol, oldest first."""
    sql = """
        SELECT timestamp, open, high, low, close, volume
        FROM historical_prices
        WHERE ticker = ? AND interval = '1m'
    """
    parameters: list[object] = [symbol]
    if start is not None:
        sql += " AND timestamp >= ?"
        parameters.append(start.astimezone(timezone.utc).strftime(UTC_TIMESTAMP_FORMAT))
    sql += " ORDER BY timestamp"
    with get_connection(db_path) as connection:
        rows = connection.execute(sql, parameters).fetchall()
    bars: list[PriceBar] = []
    for row in rows:
        stamp = datetime.strptime(row["timestamp"], UTC_TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
        bars.append(
            PriceBar(
                started_at=stamp,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )
        )
    return bars


def load_chart_bars(db_path: str | Path, symbol: str, range_key: str) -> list[PriceBar]:
    """Bars from the warmup start through the latest stored minute."""
    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT MAX(timestamp) AS latest
            FROM historical_prices
            WHERE ticker = ? AND interval = '1m'
            """,
            (symbol,),
        ).fetchone()
    if row is None or row["latest"] is None:
        return []
    latest = datetime.strptime(row["latest"], UTC_TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
    return load_bars(db_path, symbol, start=fetch_start(latest, range_key))


def load_daily_chart_bars(db_path: str | Path, symbol: str) -> list[PriceBar]:
    """Daily candles for one symbol, oldest first. Reads ``daily_prices`` only."""
    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT date, open, high, low, close, volume
            FROM daily_prices
            WHERE ticker = ?
            ORDER BY date
            """,
            (symbol,),
        ).fetchall()
    bars: list[PriceBar] = []
    for row in rows:
        year, month, day = (int(part) for part in str(row["date"]).split("-"))
        # 09:30 New York keeps the candle on the stored session date.
        started = datetime(year, month, day, 9, 30, tzinfo=NEW_YORK).astimezone(timezone.utc)
        bars.append(
            PriceBar(
                started_at=started,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )
        )
    return bars


def load_snapshot_bars(db_path: str | Path, symbol: str) -> list[PriceBar]:
    """Recent sessions, enough to compare the last two stored trading days."""
    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT MAX(timestamp) AS latest
            FROM historical_prices
            WHERE ticker = ? AND interval = '1m'
            """,
            (symbol,),
        ).fetchone()
    if row is None or row["latest"] is None:
        return []
    latest = datetime.strptime(row["latest"], UTC_TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
    return load_bars(db_path, symbol, start=latest - timedelta(days=12))


def latest_stored_timestamp(db_path: str | Path) -> str | None:
    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT MAX(timestamp) AS latest FROM historical_prices WHERE interval = '1m'"
        ).fetchone()
    if row is None:
        return None
    return row["latest"]


def database_overview(db_path: str | Path, now: datetime) -> DatabaseOverview:
    """Counts, sync cursor, file size, and whether the regular session is open."""
    path = Path(db_path)
    with get_connection(path) as connection:
        monitored = connection.execute(
            "SELECT COUNT(*) AS n FROM stocks WHERE is_monitored = 1"
        ).fetchone()["n"]
        price_rows = connection.execute(
            "SELECT COUNT(*) AS n FROM historical_prices"
        ).fetchone()["n"]
        daily_rows = connection.execute(
            "SELECT COUNT(*) AS n FROM daily_prices"
        ).fetchone()["n"]
        sync = connection.execute(
            """
            SELECT
                MAX(latest_success_timestamp) AS latest_sync,
                MAX(last_attempt_timestamp) AS last_attempt
            FROM sync_status
            """
        ).fetchone()
        statuses = connection.execute(
            "SELECT symbol, status FROM sync_status"
        ).fetchall()
    errors = tuple(sorted(row["symbol"] for row in statuses if row["status"] == "error"))
    success_count = sum(1 for row in statuses if row["status"] == "success")
    return DatabaseOverview(
        monitored=int(monitored),
        price_rows=int(price_rows),
        daily_rows=int(daily_rows),
        latest_sync=None if sync is None else sync["latest_sync"],
        last_attempt=None if sync is None else sync["last_attempt"],
        success_count=success_count,
        error_count=len(errors),
        error_symbols=errors,
        database_bytes=database_file_size(path),
        cap_bytes=SIZE_CAP_BYTES,
        warning_bytes=SIZE_WARNING_BYTES,
        market=market_state(now),
    )


def describe_catch_up(
    *,
    inserted: int,
    missing_keys: tuple[str, ...],
    errors: list[tuple[str, str]],
    database_bytes: int,
) -> str:
    """Text for the UPDATE MARKET DATA button. It does not include secrets."""
    if missing_keys:
        names = ", ".join(missing_keys)
        return f"No request was sent. These Alpaca names are missing from .env: {names}"
    lines = [f"Inserted {inserted} new rows."]
    if errors:
        failed = ", ".join(f"{symbol}: {message}" for symbol, message in errors)
        lines.append(f"Some symbols failed. {failed}")
    elif inserted == 0:
        lines.append("No new regular-session bars. A closed market is a normal result.")
    lines.append(f"Database file size is {database_bytes:,} bytes.")
    lines.append("This catch-up requests only missing minutes. It does not download the full year again.")
    return " ".join(lines)
