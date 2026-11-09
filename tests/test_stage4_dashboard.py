"""Stage 4 dashboard helpers.

Fixtures use TEST tickers and placeholder prices. They are not market data
and they are not written to the real database.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from app.charts import build_stock_figure
from app.indicators import (
    PriceBar,
    aggregate_bars,
    bollinger,
    prepare_chart,
    rsi,
    sma,
    vwap_series,
)
from app.launcher import launch_commands, stop_processes
from app.portfolio import (
    database_overview,
    describe_catch_up,
    list_positions,
    market_banner,
    portfolio_totals,
    save_position,
    stock_snapshot,
)
from database.connection import get_connection
from database.initialize import initialize_database

NY = ZoneInfo("America/New_York")
SUNDAY = datetime(2026, 11, 8, 16, 0, tzinfo=timezone.utc)
WEDNESDAY_OPEN = datetime(2026, 11, 4, 15, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


def _at(year: int, month: int, day: int, hour: int, minute: int, price: float, volume: int = 1) -> PriceBar:
    started = datetime(year, month, day, hour, minute, tzinfo=NY).astimezone(timezone.utc)
    return PriceBar(started, price, price, price, price, volume)


def _seed(db_path: Path, symbol: str = "TEST1", company: str = "Synthetic Fixture") -> None:
    initialize_database(db_path)
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO stocks (symbol, company_name, is_monitored, created_at)
            VALUES (?, ?, 1, ?)
            """,
            (symbol, company, "2026-11-04T15:00:00Z"),
        )


def _insert_bar(db_path: Path, symbol: str, bar: PriceBar) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO historical_prices (
                ticker, timestamp, open, high, low, close, volume, interval
            ) VALUES (?, ?, ?, ?, ?, ?, ?, '1m')
            """,
            (
                symbol,
                bar.started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
            ),
        )


def test_market_closed_banner_keeps_friday_wording() -> None:
    state, text = market_banner(SUNDAY, "2026-11-06T20:59:00Z")
    assert state == "closed"
    assert "Market is closed." in text
    assert "2026-11-06 15:59 ET" in text
    open_state, open_text = market_banner(WEDNESDAY_OPEN, "2026-11-04T15:00:00Z")
    assert open_state == "open"
    assert "Market is closed." not in open_text


def test_closed_session_chart_still_shows_friday() -> None:
    friday = [
        _at(2026, 11, 6, 9, 30, 10),
        _at(2026, 11, 6, 15, 59, 12),
    ]
    thursday = [_at(2026, 11, 5, 15, 59, 9)]
    points = prepare_chart(thursday + friday, "1D")
    assert points
    assert all(point.started_at.astimezone(NY).date().isoformat() == "2026-11-06" for point in points)
    assert points[-1].close == 12


def test_snapshot_uses_stored_bars_for_daily_change() -> None:
    bars = [
        _at(2026, 11, 4, 10, 0, 10, volume=2),
        _at(2026, 11, 4, 15, 59, 12, volume=3),
        _at(2026, 11, 5, 10, 0, 13, volume=4),
        _at(2026, 11, 5, 15, 59, 15, volume=5),
    ]
    snapshot = stock_snapshot("TEST1", bars)
    assert snapshot.latest_close == 15
    assert snapshot.previous_close == 12
    assert snapshot.daily_change == (15 - 12) / 12
    assert snapshot.latest_volume == 5
    assert snapshot.session_volume == 9
    assert snapshot.latest_timestamp == "2026-11-05T20:59:00Z"


def test_indicators_and_session_vwap() -> None:
    assert sma([1, 2, 3, 4, 5], 3) == [None, None, 2, 3, 4]
    rising = [float(value) for value in range(1, 17)]
    assert rsi(rising)[14] == 100
    falling = [float(value) for value in range(16, 0, -1)]
    assert rsi(falling)[14] == 0
    middle, upper, lower = bollinger([5.0] * 20)
    assert middle[-1] == 5
    assert upper[-1] == 5
    assert lower[-1] == 5
    first_session = [
        _at(2026, 11, 4, 10, 0, 10, volume=1),
        _at(2026, 11, 4, 10, 1, 20, volume=1),
    ]
    second_session = [_at(2026, 11, 5, 10, 0, 4, volume=1)]
    series = vwap_series(first_session + second_session, reset_each_session=True)
    assert series[1] == 15
    assert series[2] == 4


def test_five_minute_aggregation_keeps_ohlcv() -> None:
    bars = [
        PriceBar(_at(2026, 11, 4, 10, 0, 1).started_at, 1, 2, 1, 1.5, 10),
        PriceBar(_at(2026, 11, 4, 10, 1, 1).started_at, 1.5, 3, 1.2, 2, 4),
        PriceBar(_at(2026, 11, 4, 10, 5, 1).started_at, 2, 2, 2, 2, 7),
    ]
    candles = aggregate_bars(bars, "5m")
    assert len(candles) == 2
    assert candles[0].open == 1
    assert candles[0].high == 3
    assert candles[0].low == 1
    assert candles[0].close == 2
    assert candles[0].volume == 14
    assert candles[1].volume == 7


def test_holdings_stay_separate_and_drive_totals(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_stage4.db"
    _seed(db_path, "TEST1", "Synthetic One")
    _seed_extra(db_path, "TEST2", "Synthetic Two")
    _insert_bar(db_path, "TEST1", _at(2026, 11, 6, 15, 59, 12))
    _insert_bar(db_path, "TEST2", _at(2026, 11, 6, 15, 59, 4))
    assert save_position(db_path, "TEST1", 2, 10, SUNDAY) == "saved"
    assert save_position(db_path, "TEST2", 1, 8, SUNDAY) == "saved"
    positions = list_positions(db_path)
    by_symbol = {position.symbol: position for position in positions}
    totals = portfolio_totals(positions)
    assert totals.market_value == 28
    assert totals.invested == 28
    assert totals.unrealized == 0
    assert by_symbol["TEST1"].weight(totals.market_value) == 24 / 28
    assert by_symbol["TEST2"].weight(totals.market_value) == 4 / 28
    with get_connection(db_path) as connection:
        monitored = connection.execute(
            "SELECT is_monitored FROM stocks WHERE symbol = 'TEST1'"
        ).fetchone()["is_monitored"]
        holding_rows = connection.execute(
            "SELECT COUNT(*) AS n FROM portfolio_holdings WHERE symbol = 'TEST1'"
        ).fetchone()["n"]
    assert monitored == 1
    assert holding_rows == 1
    assert save_position(db_path, "TEST1", 0, 10, SUNDAY) == "cleared"
    cleared = {item.symbol: item.quantity for item in list_positions(db_path)}
    assert cleared["TEST1"] == 0


def test_save_replaces_one_position_and_rejects_a_negative(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_stage4.db"
    _seed(db_path)
    save_position(db_path, "TEST1", 1, 5, SUNDAY)
    save_position(db_path, "TEST1", 3, 7, SUNDAY)
    with get_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT quantity, purchase_price FROM portfolio_holdings WHERE symbol = 'TEST1'"
        ).fetchall()
    assert [(row["quantity"], row["purchase_price"]) for row in rows] == [(3, 7)]
    try:
        save_position(db_path, "TEST1", -1, 7, SUNDAY)
    except ValueError:
        pass
    else:
        raise AssertionError("negative shares were accepted")


def test_database_overview_reports_closed_market_and_size(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_stage4.db"
    _seed(db_path)
    _insert_bar(db_path, "TEST1", _at(2026, 11, 6, 15, 59, 12))
    overview = database_overview(db_path, SUNDAY)
    assert overview.monitored == 1
    assert overview.price_rows == 1
    assert overview.market == "closed"
    assert overview.database_bytes > 0
    assert overview.cap_bytes == 5 * 1024 * 1024 * 1024


def test_catch_up_text_does_not_request_a_full_year() -> None:
    text = describe_catch_up(inserted=0, missing_keys=(), errors=[], database_bytes=10)
    assert "Inserted 0 new rows." in text
    assert "closed market" in text
    assert "does not download the full year" in text
    missing = describe_catch_up(
        inserted=0,
        missing_keys=("APCA_API_KEY_ID",),
        errors=[],
        database_bytes=10,
    )
    assert "APCA_API_KEY_ID" in missing
    assert "No request was sent." in missing


def test_stock_figure_uses_stored_candles() -> None:
    points = prepare_chart(
        [_at(2026, 11, 6, 9, 30, 10, volume=2), _at(2026, 11, 6, 9, 31, 11, volume=3)],
        "1D",
    )
    figure = build_stock_figure("TEST1", points)
    kinds = [trace.type for trace in figure.data]
    assert "candlestick" in kinds
    assert "bar" in kinds


def test_stock_page_explains_each_chart_indicator() -> None:
    text = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert "How to read this chart" in text
    assert "Daily since 2016" in text
    assert "Minutes, last year" in text
    assert "load_daily_chart_bars" in text
    assert "load_chart_bars" in text
    assert "daily_prices" in text
    assert "historical_prices" in text
    for name in (
        "Price",
        "Volume",
        "SMA 20",
        "SMA 50",
        "Bollinger upper",
        "Bollinger lower",
        "VWAP",
        "RSI",
        "MACD",
        "Signal",
        "Histogram",
    ):
        assert name in text


def test_launcher_commands_are_the_session_and_dashboard_only() -> None:
    data, dashboard = launch_commands(Path("python"), ROOT)
    assert data[-2:] == ["-m", "data_collection.live_session"]
    assert "streamlit" in dashboard
    assert dashboard[dashboard.index("run") + 1].endswith("dashboard.py")
    joined = " ".join(data + dashboard)
    assert "sync_history" not in joined
    bat = (ROOT / "Start Portfolio.bat").read_text(encoding="utf-8")
    assert "app.launcher" in bat
    assert "schtasks" not in bat.lower()
    assert "pyinstaller" not in bat.lower()


def test_stop_processes_leaves_no_child() -> None:
    children = [
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]),
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]),
    ]
    try:
        stop_processes(children)
        for child in children:
            assert child.wait(timeout=5) is not None
    finally:
        stop_processes(children)


def _seed_extra(db_path: Path, symbol: str, company: str) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO stocks (symbol, company_name, is_monitored, created_at)
            VALUES (?, ?, 1, ?)
            """,
            (symbol, company, "2026-11-04T15:00:00Z"),
        )
