"""Streamlit portfolio dashboard.

Run with ``python -m streamlit run app/dashboard.py``. The Start Portfolio
window owns the live IEX session. This process only reads SQLite and, when
asked, runs the incremental catch-up. It does not train models.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from data_collection.incremental import run_incremental
from data_collection.sync_history import DEFAULT_ENV_PATH
from database.initialize import DEFAULT_DB_PATH

from app.charts import build_allocation_figure, build_stock_figure
from app.indicators import RANGE_KEYS, prepare_chart
from app.portfolio import (
    database_overview,
    describe_catch_up,
    format_et,
    format_money,
    format_percent,
    format_shares,
    latest_stored_timestamp,
    list_positions,
    load_chart_bars,
    load_snapshot_bars,
    market_banner,
    portfolio_totals,
    save_position,
    stock_snapshot,
)

PAGE_PORTFOLIO = "Portfolio"
PAGE_STOCK = "Stock"
PAGE_DATABASE = "Database"


def database_path() -> Path:
    override = os.environ.get("PORTFOLIO_DB_PATH", "").strip()
    if override:
        return Path(override)
    return Path(DEFAULT_DB_PATH)


def render(db_path: Path) -> None:
    st.set_page_config(page_title="Portfolio", layout="wide", initial_sidebar_state="expanded")
    _style()
    if not db_path.is_file():
        st.error("The SQLite database is not on this PC yet. Run python -m database.initialize first.")
        st.stop()
    now = datetime.now(timezone.utc)
    latest = latest_stored_timestamp(db_path)
    state, banner = market_banner(now, latest)
    banner_class = "market-closed" if state == "closed" else "market-open"
    st.markdown(f'<div class="market-banner {banner_class}">{banner}</div>', unsafe_allow_html=True)
    page = st.sidebar.radio("Section", [PAGE_PORTFOLIO, PAGE_STOCK, PAGE_DATABASE])
    st.sidebar.caption(
        "IEX free tier. Historical catch-up skips about the latest 15 minutes. "
        "Trades and quotes are not stored."
    )
    _catch_up_button(db_path)
    if "catch_up_result" in st.session_state:
        st.info(st.session_state["catch_up_result"])
    if page == PAGE_PORTFOLIO:
        _portfolio_page(db_path)
    elif page == PAGE_STOCK:
        _stock_page(db_path)
    else:
        _database_page(db_path, now)


def _catch_up_button(db_path: Path) -> None:
    st.sidebar.markdown("**Market data**")
    if st.sidebar.button("UPDATE MARKET DATA", type="primary"):
        with st.spinner("Requesting missing IEX minutes..."):
            try:
                outcome = run_incremental(db_path, DEFAULT_ENV_PATH)
            except Exception:
                st.session_state["catch_up_result"] = (
                    "Catch-up failed before it could finish. No secret was printed."
                )
            else:
                errors = [
                    (item.symbol, item.message)
                    for item in outcome.results
                    if item.status == "error"
                ]
                st.session_state["catch_up_result"] = describe_catch_up(
                    inserted=outcome.inserted,
                    missing_keys=outcome.missing_keys,
                    errors=errors,
                    database_bytes=outcome.database_bytes,
                )
        st.rerun()


def _portfolio_page(db_path: Path) -> None:
    st.header("Portfolio")
    st.caption("Holdings are saved in portfolio_holdings. The watchlist is unchanged.")
    positions = list_positions(db_path)
    if not positions:
        st.warning("No monitored stocks are in the database yet.")
        return
    totals = portfolio_totals(positions)
    value_column, invested_column, pl_column, count_column = st.columns(4)
    value_column.metric("Total value", format_money(totals.market_value))
    invested_column.metric("Total invested", format_money(totals.invested))
    pl_column.metric("Unrealized P/L", format_money(totals.unrealized))
    held = sum(1 for position in positions if position.quantity > 0)
    count_column.metric("Positions", str(held))
    if totals.missing_prices:
        st.caption(
            "No stored close yet for "
            + ", ".join(totals.missing_prices)
            + ". Those symbols are in invested cost and out of total value."
        )
    editor, chart = st.columns([1, 1])
    with editor:
        st.subheader("Position")
        labels = {f"{position.symbol} — {position.company_name}": position for position in positions}
        choice = st.selectbox("Stock", list(labels))
        selected = labels[choice]
        share_key = f"shares-{selected.symbol}"
        price_key = f"price-{selected.symbol}"
        if share_key not in st.session_state:
            st.session_state[share_key] = float(selected.quantity)
        if price_key not in st.session_state:
            st.session_state[price_key] = float(selected.purchase_price)
        shares = st.number_input("Shares", min_value=0.0, step=0.01, key=share_key)
        price = st.number_input("Purchase price", min_value=0.0, step=0.01, key=price_key)
        st.caption("Enter 0 shares to remove the position. This does not change the watchlist.")
        if st.button("Save holding"):
            save_position(db_path, selected.symbol, float(shares), float(price), datetime.now(timezone.utc))
            st.session_state.pop(f"shares-{selected.symbol}", None)
            st.session_state.pop(f"price-{selected.symbol}", None)
            st.rerun()
    with chart:
        allocation = build_allocation_figure(positions, totals.market_value)
        if allocation is None:
            st.info("Save a share count to see allocation weights.")
        else:
            st.plotly_chart(allocation, use_container_width=True)
    _holdings_table(positions, totals.market_value)


def _holdings_table(positions: list, total_value: float) -> None:
    header = (
        "| Symbol | Company | Shares | Purchase | Last close | Value | P/L | Weight |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    lines = [header]
    for position in positions:
        weight = position.weight(total_value)
        lines.append(
            "| {symbol} | {company} | {shares} | {buy} | {last} | {value} | {pl} | {weight} |\n".format(
                symbol=position.symbol,
                company=position.company_name,
                shares=format_shares(position.quantity) if position.quantity else "—",
                buy=format_money(position.purchase_price) if position.quantity else "—",
                last=format_money(position.latest_close) if position.latest_close is not None else "—",
                value=format_money(position.market_value) if position.market_value is not None else "—",
                pl=format_money(position.unrealized) if position.unrealized is not None else "—",
                weight=f"{weight:.1%}" if weight is not None else "—",
            )
        )
    st.subheader("Holdings")
    st.markdown("".join(lines))


_CHART_NOTES: tuple[tuple[str, str], ...] = (
    (
        "Price",
        "Each candlestick on the top panel is one slice of time. "
        "The thick body runs from the first price in that slice to the last price: green if it closed higher than it opened, red if it closed lower. "
        "The thin lines above and below are the highest and lowest prices reached. "
        "A tall candle means the price moved a lot. A short one means it barely moved.",
    ),
    (
        "Volume",
        "Volume, in the panel of that name, is how many shares traded during that same slice. "
        "A tall bar means more shares changed hands. A short bar means trading was quiet. "
        "Green bars line up with a candle that closed higher, and red bars line up with one that closed lower. "
        "A large price jump on a tall bar had more trading behind it than the same jump on a short bar.",
    ),
    (
        "SMA 20 and SMA 50",
        "A moving average is the average closing price over a run of candles, drawn as a smooth line on the price panel. "
        "SMA 20 uses the last 20 candles, so it follows the price more quickly. SMA 50 uses the last 50, so it moves more slowly. "
        "Price above both lines means the recent stretch has been rising. Price below both means it has been falling. "
        "When the quicker SMA 20 crosses the slower SMA 50, the shorter trend has changed direction against the longer one.",
    ),
    (
        "Bollinger upper and Bollinger lower",
        "These dotted lines on the price panel mark a typical high and a typical low for the recent candles. "
        "Bollinger upper is the high side of that range, and Bollinger lower is the low side. "
        "Wide apart means the price has been swinging. Close together means it has been calm. "
        "A candle on the upper line is high compared with recent prices, and one on the lower line is low. That only describes where it sits, not the next move.",
    ),
    (
        "VWAP",
        "VWAP is the average price people paid, giving more weight to candles where more shares traded. "
        "It is the line labeled VWAP on the price panel. On the intraday ranges it starts over each trading session. On the daily ranges it runs across the days in view. "
        "Price above VWAP means the latest price is higher than that average. Price below it means the latest price is lower.",
    ),
    (
        "RSI",
        "RSI, in the panel labeled RSI 14, is a score from 0 to 100 for whether recent candles were mostly gains or mostly losses. The 14 means it looks at the last 14 candles. "
        "Up near the dotted line at 70, recent candles were mostly rising. Down near the dotted line at 30, they were mostly falling. "
        "Near the middle, around 50, gains and losses have been about even. The score describes that recent stretch, not the next price.",
    ),
    (
        "MACD, Signal, and Histogram",
        "MACD compares a faster average of the price with a slower one. The MACD line is the gap between them, and Signal is a smoothed copy of that line. "
        "MACD above Signal means the faster average is pulling ahead, which usually goes with prices pushing up. MACD below Signal means it is lagging, which usually goes with prices pushing down. "
        "The Histogram bars draw that same gap: above zero when MACD is above Signal, below zero when it is not. Taller bars mean a wider gap.",
    ),
)


def _stock_page(db_path: Path) -> None:
    st.header("Stock")
    positions = list_positions(db_path)
    if not positions:
        st.warning("No monitored stocks are in the database yet.")
        return
    labels = [f"{position.symbol} — {position.company_name}" for position in positions]
    columns = st.columns([2, 1])
    choice = columns[0].selectbox("Stock", labels)
    range_key = columns[1].selectbox("Range", RANGE_KEYS, index=0)
    symbol = positions[labels.index(choice)].symbol
    snapshot = stock_snapshot(symbol, load_snapshot_bars(db_path, symbol))
    price_column, change_column, volume_column, session_column = st.columns(4)
    price_column.metric(
        "Latest price",
        format_money(snapshot.latest_close) if snapshot.latest_close is not None else "—",
    )
    change_column.metric(
        "Daily change",
        format_percent(snapshot.daily_change) if snapshot.daily_change is not None else "—",
    )
    volume_column.metric(
        "Last bar volume",
        f"{snapshot.latest_volume:,}" if snapshot.latest_volume is not None else "—",
    )
    session_column.metric(
        "Session volume",
        f"{snapshot.session_volume:,}" if snapshot.session_volume is not None else "—",
    )
    if snapshot.latest_timestamp:
        st.caption(f"Latest stored bar {format_et(snapshot.latest_timestamp)}.")
    points = prepare_chart(load_chart_bars(db_path, symbol, range_key), range_key)
    st.plotly_chart(build_stock_figure(symbol, points), use_container_width=True)
    st.caption(
        "SMA 20, SMA 50, Bollinger Bands, and VWAP are on the price panel. "
        "RSI and MACD use the same candles. Daily ranges use one candle per session."
    )
    st.subheader("How to read this chart")
    for title, body in _CHART_NOTES:
        st.markdown(f"**{title}.** {body}")


def _database_page(db_path: Path, now: datetime) -> None:
    st.header("Database")
    overview = database_overview(db_path, now)
    used = overview.database_bytes / overview.cap_bytes if overview.cap_bytes else 0
    monitored_column, rows_column, size_column, use_column = st.columns(4)
    monitored_column.metric("Stocks monitored", str(overview.monitored))
    rows_column.metric("Price rows", f"{overview.price_rows:,}")
    size_column.metric("Database size", f"{overview.database_bytes:,} bytes")
    use_column.metric("Storage use", f"{used:.1%} of 5 GB")
    if overview.database_bytes > overview.warning_bytes:
        st.warning("The database file is above 4 GB. The cap target is 5 GB.")
    sync_text = format_et(overview.latest_sync) if overview.latest_sync else "No successful sync yet"
    attempt_text = format_et(overview.last_attempt) if overview.last_attempt else "No attempt yet"
    st.write(f"**Latest sync timestamp:** {sync_text}")
    st.write(f"**Last attempt:** {attempt_text}")
    if overview.error_count:
        st.write(
            f"**Collection status:** {overview.error_count} symbol(s) failed: "
            + ", ".join(overview.error_symbols)
        )
    elif overview.success_count:
        st.write(f"**Collection status:** last catch-up succeeded for {overview.success_count} symbols.")
    else:
        st.write("**Collection status:** no sync has been recorded.")
    market_text = "open" if overview.market == "open" else "closed"
    st.write(f"**Regular US session:** {market_text}.")
    st.write("**Feed:** IEX, the free Alpaca tier. It is not the consolidated SIP feed.")
    st.write(
        "Historical catch-up is delayed by about 15 minutes. "
        "Completed 1-minute bars update only while the Start Portfolio window is open. "
        "Trades and quotes are not written to the database."
    )


def _style() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        .market-banner {
            border-radius: 12px;
            padding: 0.85rem 1rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
        }
        .market-closed {background: #fff6e8; color: #8a4b08; border: 1px solid #f3d7a1;}
        .market-open {background: #e8f6ef; color: #0f7b4c; border: 1px solid #b7e4c7;}
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 12px;
            padding: 0.55rem 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


render(database_path())
