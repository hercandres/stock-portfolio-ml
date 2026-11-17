"""Streamlit portfolio dashboard.

Run with ``python -m streamlit run app/dashboard.py``. The Start Portfolio
window owns the live IEX session. While that window is open, minute bars
append about once a minute and a finished day is written to daily prices.
This process only reads SQLite. Random forest, XGBoost, linear regression,
K-Means, and PCA each run once when the dashboard session starts, and again
only from their own buttons.
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

from database.initialize import DEFAULT_DB_PATH

from app.charts import build_allocation_figure, build_stock_figure
from app.indicators import AGGREGATION, RANGE_KEYS, prepare_chart, prepare_daily_chart
from app.portfolio import (
    database_overview,
    format_et,
    format_money,
    format_percent,
    format_shares,
    latest_stored_timestamp,
    list_positions,
    load_chart_bars,
    load_daily_chart_bars,
    load_snapshot_bars,
    market_banner,
    portfolio_totals,
    save_position,
    stock_snapshot,
)

PAGE_PORTFOLIO = "Portfolio"
PAGE_STOCK = "Stock"
PAGE_DATABASE = "Database"
PAGE_FOREST = "Random Forest"
PAGE_XGBOOST = "XGBoost"
PAGE_LINEAR = "Linear Regression"
PAGE_KMEANS = "K-Means"
PAGE_PCA = "PCA"
# The first three stay at the top of the sidebar. A new model is added to
# ML_PAGES, so it shows under ML Methods and does not move those three.
MAIN_PAGES = (PAGE_PORTFOLIO, PAGE_STOCK, PAGE_DATABASE)
ML_PAGES = (PAGE_FOREST, PAGE_XGBOOST, PAGE_LINEAR, PAGE_KMEANS, PAGE_PCA)

FOREST_EXPORT = ROOT / "machine_learning" / "exports" / "random_forest_features.csv"
FOREST_PREDICTIONS = ROOT / "machine_learning" / "outputs" / "predictions.csv"
FOREST_BACKTEST = ROOT / "machine_learning" / "outputs" / "backtest.csv"
FOREST_COMPARISON = ROOT / "machine_learning" / "outputs" / "model_comparison.csv"
# XGBoost reads FOREST_EXPORT. Its scores live in their own folder.
XGB_PREDICTIONS = ROOT / "machine_learning" / "outputs" / "xgboost" / "predictions.csv"
XGB_BACKTEST = ROOT / "machine_learning" / "outputs" / "xgboost" / "backtest.csv"
# Linear regression reads FOREST_EXPORT too. Its scores live in their own folder.
LINEAR_OUTPUT = ROOT / "machine_learning" / "outputs" / "linear_regression"
LINEAR_PREDICTIONS = LINEAR_OUTPUT / "predictions.csv"
LINEAR_BACKTEST = LINEAR_OUTPUT / "backtest.csv"
# K-Means reads FOREST_EXPORT too. Labels and centers live in their own folder.
KMEANS_OUTPUT = ROOT / "machine_learning" / "outputs" / "kmeans"
# PCA reads FOREST_EXPORT too. Variance and loadings live in their own folder.
PCA_OUTPUT = ROOT / "machine_learning" / "outputs" / "pca"


def _sidebar_page() -> str:
    """Portfolio, Stock, and Database stay above the model list.

    The highlighted name is the open window. Later models are names in
    ``ML_PAGES`` and appear under ML Methods.
    """
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = PAGE_PORTFOLIO
    current = st.session_state["nav_page"]
    if current not in MAIN_PAGES and current not in ML_PAGES:
        current = PAGE_PORTFOLIO
        st.session_state["nav_page"] = current
    for name in MAIN_PAGES:
        _nav_button(name, current)
    st.sidebar.subheader("ML Methods")
    for name in ML_PAGES:
        _nav_button(name, current)
    return st.session_state["nav_page"]


def _nav_button(name: str, current: str) -> None:
    """One sidebar window. The open window is the filled button."""
    if st.sidebar.button(
        name,
        key=f"nav-{name}",
        type="primary" if name == current else "secondary",
        use_container_width=True,
    ):
        st.session_state["nav_page"] = name
        st.rerun()


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
    page = _sidebar_page()
    st.sidebar.caption(
        "While this window is open, minute bars append about once a minute "
        "and a finished day is saved with the daily bars. "
        "The left chart button is daily prices. The right button is the last year of minutes. "
        "Closed nights, weekends, and holidays are left off both charts. "
        "Random forest, XGBoost, and linear regression train on the daily bars. "
        "K-Means groups those same days. "
        "PCA summarizes those same days."
    )
    _train_forest_once(db_path)
    _train_xgboost_once(db_path)
    _train_linear_once(db_path)
    _train_kmeans_once(db_path)
    _train_pca_once(db_path)
    if page == PAGE_PORTFOLIO:
        _portfolio_page(db_path)
    elif page == PAGE_STOCK:
        _stock_page(db_path)
    elif page == PAGE_FOREST:
        _random_forest_page(db_path)
    elif page == PAGE_XGBOOST:
        _xgboost_page(db_path)
    elif page == PAGE_LINEAR:
        _linear_regression_page(db_path)
    elif page == PAGE_KMEANS:
        _kmeans_page(db_path)
    elif page == PAGE_PCA:
        _pca_page(db_path)
    else:
        _database_page(db_path, now)


@st.fragment(run_every=60)
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


def _chart_notes(mode: str) -> tuple[tuple[str, str], ...]:
    """The shared notes, with the first one saying which prices are on screen."""
    notes = list(_CHART_NOTES)
    title, body = notes[0]
    if mode == "daily":
        showing = (
            "This chart is daily candles from daily_prices, from 2016 or this stock's first stored day, through the latest session. "
            "Each candle is one trading day. Nights, weekends, and holidays are skipped, so sessions sit back to back. "
            "No price is invented for those closed times. Drag or scroll to zoom, the same way as the minute chart. "
            "The button on the right switches to the last year of 1-minute candles. "
        )
    else:
        showing = (
            "This chart is the last year of 1-minute candles from historical_prices. "
            "Nights, weekends, and holidays are skipped, so one session continues into the next. "
            "A break is left only when a minute is missing from the data. Missing minutes are not filled with zero. "
            "The button on the left switches to daily candles. "
            "Models train on those daily bars, not on these minutes. "
        )
    notes[0] = (title, showing + body)
    return tuple(notes)


@st.fragment(run_every=60)
def _stock_page(db_path: Path) -> None:
    st.header("Stock")
    positions = list_positions(db_path)
    if not positions:
        st.warning("No monitored stocks are in the database yet.")
        return
    if "price_chart" not in st.session_state:
        st.session_state["price_chart"] = "minutes"
    labels = [f"{position.symbol} — {position.company_name}" for position in positions]
    choice = st.selectbox("Stock", labels)
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
        st.caption(f"Latest stored minute bar {format_et(snapshot.latest_timestamp)}.")
    mode = st.session_state["price_chart"]
    left, right = st.columns(2)
    if left.button(
        "Daily since 2016",
        type="primary" if mode == "daily" else "secondary",
        use_container_width=True,
    ):
        st.session_state["price_chart"] = "daily"
        st.rerun()
    if right.button(
        "Minutes, last year",
        type="primary" if mode == "minutes" else "secondary",
        use_container_width=True,
    ):
        st.session_state["price_chart"] = "minutes"
        st.rerun()
    if mode == "daily":
        points = prepare_daily_chart(load_daily_chart_bars(db_path, symbol))
        step = "1d"
        st.caption(
            "Showing daily candles from daily_prices. "
            "Closed days are skipped. Drag or scroll to zoom, the same way as the minute chart."
        )
    else:
        range_key = st.selectbox("Range", RANGE_KEYS, index=0)
        points = prepare_chart(load_chart_bars(db_path, symbol, range_key), range_key)
        step = AGGREGATION[range_key]
        st.caption(
            "Showing the last year of minute candles from historical_prices. "
            "Closed hours are skipped. A missing minute stays a break and is not filled with zero. "
            "SMA 20, SMA 50, Bollinger Bands, and VWAP are on the price panel. "
            "RSI and MACD use the same candles."
        )
    st.plotly_chart(build_stock_figure(symbol, points, step=step), use_container_width=True)
    st.subheader("How to read this chart")
    for title, body in _chart_notes(mode):
        st.markdown(f"**{title}.** {body}")


def _database_page(db_path: Path, now: datetime) -> None:
    st.header("Database")
    overview = database_overview(db_path, now)
    used = overview.database_bytes / overview.cap_bytes if overview.cap_bytes else 0
    monitored_column, minute_column, daily_column, size_column, use_column = st.columns(5)
    monitored_column.metric("Stocks monitored", str(overview.monitored))
    minute_column.metric("Minute rows", f"{overview.price_rows:,}")
    daily_column.metric("Daily rows", f"{overview.daily_rows:,}")
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
        "Minute bars in historical_prices are the last year. The chart's right button and the live session use them. "
        "Daily bars in daily_prices go back to 2016, or the first day a stock has, and the random forest trains on them. "
        "The 365-day delete does not remove daily bars."
    )
    st.write(
        "Historical catch-up is delayed by about 15 minutes. "
        "New minute bars append about once a minute while the Start Portfolio window is open. "
        "A finished day is added to daily prices then too. "
        "Trades and quotes are not written to the database."
    )


def _read_table(path: Path):
    import pandas as pd

    if not path.is_file():
        return None
    return pd.read_csv(path)


def _forest_symbols(db_path: Path) -> list[str]:
    table = _read_table(FOREST_EXPORT)
    if table is not None and "ticker" in table.columns:
        symbols = sorted(str(symbol) for symbol in table["ticker"].dropna().unique())
        if symbols:
            return symbols
    return [position.symbol for position in list_positions(db_path)]


def _train_forest_once(db_path: Path) -> None:
    """Train on daily prices the first time this dashboard session starts."""
    if st.session_state.get("forest_ran"):
        return
    st.session_state["forest_ran"] = True
    _train_forest(db_path)


def _train_forest(db_path: Path) -> None:
    """Read daily_prices with pandas, then fit the three scikit-learn forests."""
    with st.spinner("Reading daily_prices and training the three forests..."):
        from machine_learning.supervised.random_forest.features import build_and_save
        from machine_learning.supervised.random_forest.train import train_and_save

        try:
            build_and_save(db_path)
            train_and_save(db_path)
        except Exception:
            st.error("Random forest did not finish. No secret was printed.")


def _train_xgboost_once(db_path: Path) -> None:
    """Train XGBoost the first time this dashboard session starts."""
    if st.session_state.get("xgboost_ran"):
        return
    st.session_state["xgboost_ran"] = True
    _train_xgboost(db_path)


def _train_xgboost(db_path: Path) -> None:
    """Fit three XGBRegressor models on the shared feature CSV.

    This does not write a second feature file and it does not replace
    random_forest rows in the comparison file.
    """
    with st.spinner("Training XGBoost on the shared daily feature file..."):
        from machine_learning.supervised.xgboost.train import train_and_save

        try:
            train_and_save(db_path)
        except Exception:
            st.error("XGBoost did not finish. No secret was printed.")


def _forest_explanation() -> None:
    st.subheader("How the rows are built")
    st.markdown(
        "The model reads **daily_prices**: one row per stock per trading day, "
        "from 2016 or that stock's first day, through the latest session. "
        "The open, high, low, close, and volume on that row are the day's prices. "
        "Charts can show those daily candles or the last year of minute bars in "
        "**historical_prices**. Training uses the daily rows, not the minutes."
    )
    st.markdown(
        "Each number describes a day that has already finished. "
        "**return_1d**, **return_5d**, and **return_20d** are the percent change "
        "versus 1, 5, and 20 trading days ago. "
        "**dist_sma_20_pct** and **dist_sma_50_pct** say how far today's close "
        "sits from the 20-day and 50-day average, in percent. Above the average "
        "is positive. "
        "**rsi_14** compares the average up day with the average down day over "
        "14 days, on a 0 to 100 scale. Higher means the recent days were mostly up. "
        "**volatility_20d** is how widely the one-day percent changes have varied "
        "over 20 days. "
        "**volume_vs_20d** is today's volume divided by the average volume of the "
        "last 20 days. 1.0 is a typical day. "
        "**range_pct** is the day's high minus the low, divided by the close, in percent."
    )
    st.markdown(
        "The three targets are the percent change 1, 5, and 20 trading days "
        "in the future. Those are the answers. The forest is not given them as "
        "inputs. The ticker and the date only name the row. The numbers are not "
        "rescaled. A day missing one of the features is left out, which is why "
        "the first 50 trading days drop: there is no 50-day average yet. A recent "
        "day that does not yet have a 20-day future stays in the file, so the "
        "forest can still make a forecast from it."
    )
    st.subheader("How the forest learns")
    st.markdown(
        "A random forest is a crowd of small decision trees. Each tree looks at "
        "the features and guesses a percent change. The forest's guess is the "
        "average of those trees. There are three forests, because the next day, "
        "the next 5 days, and the next 20 days are different questions. Earlier "
        "days teach the forest. Later days of the same stock are the score. The "
        "rows are not shuffled. A shuffle would let the model study a day that "
        "had not happened yet."
    )
    st.markdown(
        "On the chart, **Actual** is the percent change that really happened on "
        "the later days held out of training. **Predicted** is what that forest "
        "guessed from the features known on each of those days. MAE is the average "
        "size of the miss, in percentage points. RMSE is a similar miss that counts "
        "a large error more heavily. Directional accuracy is how often the guess "
        "and the real move had the same sign. All scores every held-out day for "
        "the horizon you picked. One ticker scores only that stock. The comparison "
        "file at the bottom does not follow the dropdown."
    )


def _held_out_rows(symbol: str, horizon: str, backtest_path: Path = FOREST_BACKTEST):
    """Held-out rows for one horizon. All keeps every ticker."""
    table = _read_table(backtest_path)
    if table is None or "ticker" not in table.columns or "horizon" not in table.columns:
        return None
    rows = table[table["horizon"].astype(str) == str(horizon)]
    if symbol != "All":
        rows = rows[rows["ticker"].astype(str) == symbol]
    if rows.empty or "date" not in rows.columns:
        return rows
    sort_columns = ["date"] if symbol != "All" else ["ticker", "date"]
    return rows.sort_values(sort_columns)


def _forest_chart(symbol: str, horizon: str, backtest_path: Path = FOREST_BACKTEST) -> None:
    import plotly.graph_objects as go

    rows = _held_out_rows(symbol, horizon, backtest_path)
    if rows is None:
        st.info("The model has not scored the held-out days yet.")
        return
    if rows.empty:
        st.info("This stock has no held-out days for that horizon.")
        return
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=rows["date"],
            y=rows["actual_percent"],
            name="Actual",
            mode="lines+markers",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=rows["date"],
            y=rows["predicted_percent"],
            name="Predicted",
            mode="lines+markers",
        )
    )
    figure.update_layout(
        title=f"{symbol}: predicted vs actual percent, {horizon} trading days",
        xaxis_title="Date",
        yaxis_title="Percent",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(figure, width="stretch")


def _format_score(value: float, spread: float | None, as_percent: bool) -> str:
    if as_percent:
        text = f"{value:.1%}"
        if spread is not None:
            text += f" ± {spread:.1%}"
        return text
    text = f"{value:.3f}"
    if spread is not None:
        text += f" ± {spread:.3f}"
    return text


def _show_three_scores(scores: dict[str, float], spread: dict[str, float] | None) -> None:
    mae_spread = None if spread is None else spread["mae"]
    rmse_spread = None if spread is None else spread["rmse"]
    direction_spread = None if spread is None else spread["directional_accuracy"]
    mae_column, rmse_column, direction_column = st.columns(3)
    mae_column.metric("MAE", _format_score(scores["mae"], mae_spread, False))
    rmse_column.metric("RMSE", _format_score(scores["rmse"], rmse_spread, False))
    direction_column.metric(
        "Directional accuracy",
        _format_score(scores["directional_accuracy"], direction_spread, True),
    )


def _forest_all_metrics(horizon: str, backtest_path: Path = FOREST_BACKTEST) -> None:
    """Pooled scores for this horizon, plus how much the stocks differ."""
    from machine_learning.supervised.random_forest.train import (
        per_stock_scores,
        score_held_out,
        score_spread,
    )

    st.subheader("All stocks, this horizon")
    st.caption(
        f"Every held-out day for the {horizon}-day horizon is included. "
        "The ± value is the standard deviation of that score from stock to stock."
    )
    rows = _held_out_rows("All", horizon, backtest_path)
    if rows is None or rows.empty:
        st.info("There are no held-out days for that horizon.")
        return
    scores = score_held_out(rows)
    if scores is None:
        st.info("There are no held-out days for that horizon.")
        return
    _show_three_scores(scores, score_spread(per_stock_scores(rows)))


def _forest_normal_chart(horizon: str, backtest_path: Path = FOREST_BACKTEST) -> None:
    """Normal curves of the per-stock scores. Only the All choice uses this."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from machine_learning.supervised.random_forest.train import per_stock_scores

    rows = _held_out_rows("All", horizon, backtest_path)
    if rows is None or rows.empty:
        return
    per_stock = per_stock_scores(rows)
    if per_stock.empty:
        return
    st.caption(
        "Each bar counts stocks. The curve is a normal distribution with the "
        "same average and spread. This chart is only shown for All."
    )
    titles = ("MAE", "RMSE", "Directional accuracy")
    keys = ("mae", "rmse", "directional_accuracy")
    figure = make_subplots(rows=1, cols=3, subplot_titles=titles)
    for column, key in enumerate(keys, start=1):
        values = per_stock[key].to_numpy(dtype=float)
        figure.add_trace(
            go.Histogram(
                x=values,
                histnorm="probability density",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        if len(values) < 2:
            continue
        center = float(np.mean(values))
        spread = float(np.std(values, ddof=1))
        if spread <= 0:
            continue
        xs = np.linspace(center - 3 * spread, center + 3 * spread, 80)
        ys = np.exp(-0.5 * ((xs - center) / spread) ** 2) / (spread * np.sqrt(2 * np.pi))
        figure.add_trace(
            go.Scatter(x=xs, y=ys, mode="lines", name="Normal", showlegend=column == 1),
            row=1,
            col=column,
        )
    figure.update_layout(
        title=f"Per-stock scores, {horizon} trading days",
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(figure, width="stretch")


def _forest_metrics(symbol: str, horizon: str, backtest_path: Path = FOREST_BACKTEST) -> None:
    """Three scores for this stock only. They follow the selectors."""
    from machine_learning.supervised.random_forest.train import score_held_out

    st.subheader(f"This stock: {symbol}")
    st.caption(
        f"MAE, RMSE, and directional accuracy for {symbol} only, "
        f"over {horizon} trading days."
    )
    rows = _held_out_rows(symbol, horizon, backtest_path)
    if rows is None or rows.empty:
        st.info("This stock has no held-out days for that horizon.")
        return
    scores = score_held_out(rows)
    if scores is None:
        st.info("This stock has no held-out days for that horizon.")
        return
    _show_three_scores(scores, None)


def _forest_forecast(symbol: str, horizon: str, predictions_path: Path = FOREST_PREDICTIONS) -> None:
    predictions = _read_table(predictions_path)
    if predictions is None:
        st.write("This is a model estimate, not a trade.")
        return
    match = predictions[
        (predictions["ticker"].astype(str) == symbol)
        & (predictions["horizon"].astype(str) == str(horizon))
    ]
    if match.empty:
        st.write("This is a model estimate, not a trade.")
        return
    row = match.iloc[0]
    st.write(
        f"Latest {horizon}-day forecast for {symbol}: "
        f"{float(row['predicted percent']):.2f}% ({row['recommendation']}). "
        "This is a model estimate, not a trade."
    )


def _forest_score_notes() -> None:
    st.write(
        "Lower MAE and RMSE are closer prices, and higher directional accuracy "
        "means the up/down call was right more often."
    )
    st.write(
        "The recommendation window will rank models from this file once the others exist."
    )


def _forest_comparison_table(model_name: str = "random_forest") -> None:
    """Saved scores from model_comparison.csv. This does not follow the dropdown.

    Only ``model_name`` is shown. Other models' rows stay in the file.
    """
    st.subheader("Comparison file")
    st.caption(
        "Saved scores for every stock and every horizon. "
        "This table does not follow the stock dropdown."
    )
    table = _read_table(FOREST_COMPARISON)
    if table is None or table.empty or "model" not in table.columns:
        st.info("The comparison file is written when you run the model.")
        _forest_score_notes()
        return
    chosen = table[table["model"].astype(str) == model_name].copy()
    if chosen.empty:
        st.info(f"{model_name} has no rows in the comparison file yet.")
        _forest_score_notes()
        return
    chosen["horizon_days"] = chosen["horizon_days"].astype(int)
    chosen = chosen.sort_values("horizon_days")
    show = {
        "Horizon (days)": chosen["horizon_days"].tolist(),
        "MAE": [f"{float(value):.3f}" for value in chosen["mae"]],
        "RMSE": [f"{float(value):.3f}" for value in chosen["rmse"]],
        "Directional accuracy": [
            f"{float(value):.1%}" for value in chosen["directional_accuracy"]
        ],
        "Cutoff": [str(value) for value in chosen["cutoff"]],
    }
    import pandas as pd

    st.table(pd.DataFrame(show))
    _forest_score_notes()


def _random_forest_page(db_path: Path) -> None:
    st.header("Random Forest")
    st.caption("Daily prices are read with pandas. The forest is scikit-learn. It already ran when the app opened.")
    symbols = _forest_symbols(db_path)
    symbol = st.selectbox("Stock", ["All", *symbols])
    st.caption("All uses every stock. A ticker uses only that stock.")
    horizon = st.selectbox(
        "Horizon",
        ["1", "5", "20"],
        format_func=lambda value: f"{value} trading days",
    )
    if st.button("Run model"):
        _train_forest(db_path)
        st.rerun()
    _forest_explanation()
    if symbol == "All":
        _forest_all_metrics(horizon)
        _forest_normal_chart(horizon)
    else:
        _forest_chart(symbol, horizon)
        _forest_metrics(symbol, horizon)
        _forest_forecast(symbol, horizon)
    _forest_comparison_table()


def _xgboost_explanation() -> None:
    st.subheader("How the rows are built")
    st.markdown(
        "XGBoost reads the same file as the random forest: "
        "**random_forest_features.csv**. That file is built from **daily_prices**, "
        "one row per stock per trading day. There is no second feature file. "
        "The nine inputs are **return_1d**, **return_5d**, **return_20d**, "
        "**dist_sma_20_pct**, **dist_sma_50_pct**, **rsi_14**, **volatility_20d**, "
        "**volume_vs_20d**, and **range_pct**. They describe a day that has "
        "already finished. The numbers are not rescaled."
    )
    st.markdown(
        "The targets are **target_1d**, **target_5d**, and **target_20d**: the "
        "percent change 1, 5, and 20 trading days ahead. Those are the answers. "
        "The booster is not given them as inputs. The ticker and the date only "
        "name the row. A day missing a feature is left out. A recent day that "
        "does not yet have a 20-day future stays in the file, so the booster "
        "can still forecast from it. That row is not used as an answer."
    )
    st.subheader("How the booster learns")
    st.markdown(
        "XGBoost adds one small tree at a time. Each new tree tries to correct "
        "the mistakes of the trees already added. The library model is "
        "**XGBRegressor**. There are three boosters, because the next day, the "
        "next 5 days, and the next 20 days are different questions. Earlier days "
        "teach the booster. Later days of the same stock are the score. The rows "
        "are not shuffled. A shuffle would let the model study a day that had "
        "not happened yet."
    )
    st.markdown(
        "On the chart, **Actual** is the percent change that really happened on "
        "the later days held out of training. **Predicted** is what that booster "
        "guessed from the features known on each of those days. MAE is the average "
        "size of the miss, in percentage points. RMSE is a similar miss that counts "
        "a large error more heavily. Directional accuracy is how often the guess "
        "and the real move had the same sign. All scores every held-out day for "
        "the horizon you picked. One ticker scores only that stock. The ± value "
        "and the normal chart are only for All. The comparison file at the bottom "
        "shows the xgboost rows and does not follow the dropdown."
    )


def _xgboost_page(db_path: Path) -> None:
    st.header("XGBoost")
    st.caption(
        "The same daily feature file as the random forest. "
        "The booster is XGBRegressor. It already ran when the app opened."
    )
    symbols = _forest_symbols(db_path)
    symbol = st.selectbox("Stock", ["All", *symbols], key="xgb-stock")
    st.caption("All uses every stock. A ticker uses only that stock.")
    horizon = st.selectbox(
        "Horizon",
        ["1", "5", "20"],
        format_func=lambda value: f"{value} trading days",
        key="xgb-horizon",
    )
    if st.button("Run XGBoost"):
        _train_xgboost(db_path)
        st.rerun()
    _xgboost_explanation()
    if symbol == "All":
        _forest_all_metrics(horizon, XGB_BACKTEST)
        _forest_normal_chart(horizon, XGB_BACKTEST)
    else:
        _forest_chart(symbol, horizon, XGB_BACKTEST)
        _forest_metrics(symbol, horizon, XGB_BACKTEST)
        _forest_forecast(symbol, horizon, XGB_PREDICTIONS)
    _forest_comparison_table("xgboost")


def _train_linear_once(db_path: Path) -> None:
    """Fit the three lines the first time this dashboard session starts."""
    if st.session_state.get("linear_ran"):
        return
    st.session_state["linear_ran"] = True
    _train_linear(db_path)


def _train_linear(db_path: Path) -> None:
    """Fit three LinearRegression models on the shared feature CSV.

    This does not write a second feature file and it does not replace
    other models' rows in the comparison file. The inputs are not standardized.
    """
    with st.spinner("Fitting linear regression on the shared daily feature file..."):
        from machine_learning.supervised.linear_regression.train import train_and_save

        try:
            train_and_save(db_path)
        except Exception:
            st.error("Linear regression did not finish. No secret was printed.")


def _linear_explanation() -> None:
    st.subheader("How the rows are built")
    st.markdown(
        "Linear regression reads the same file as the random forest: "
        "**random_forest_features.csv**. That file is built from **daily_prices**, "
        "one row per stock per trading day. There is no second feature file. "
        "The nine inputs are **return_1d**, **return_5d**, **return_20d**, "
        "**dist_sma_20_pct**, **dist_sma_50_pct**, **rsi_14**, **volatility_20d**, "
        "**volume_vs_20d**, and **range_pct**. They describe a day that has "
        "already finished. The numbers are not rescaled."
    )
    st.markdown(
        "The targets are **target_1d**, **target_5d**, and **target_20d**: the "
        "percent change 1, 5, and 20 trading days ahead. Those are the answers. "
        "The line is not given them as inputs. The ticker and the date only "
        "name the row. A day missing a feature is left out. A recent day that "
        "does not yet have a 20-day future stays in the file, so the line "
        "can still forecast from it. That row is not used as an answer."
    )
    st.subheader("How the line learns")
    st.markdown(
        "Linear regression draws one straight line through the inputs. "
        "Each input gets a coefficient, which is its weight in that line. "
        "The library model is **LinearRegression**. There are three lines, "
        "because the next day, the next 5 days, and the next 20 days are "
        "different questions. Earlier days teach the line. Later days of the "
        "same stock are the score. The rows are not shuffled. A shuffle would "
        "let the model study a day that had not happened yet."
    )
    st.markdown(
        "The first chart sorts the held-out guesses and cuts them into up to "
        "10 bins with the same number of days. Each point is that bin's average "
        "guess and its average actual move. A straight line runs through those "
        "averages. **y = x** would be a perfect match. One stock is used when "
        "it has at least 30 held-out days. Otherwise the bins use every stock, "
        "and the caption says so. The chart under that is one stock across the "
        "held-out dates: the actual percent and the predicted percent. All "
        "shows the first ticker in the held-out set and names it. "
        "Pearson correlation says how each input moved with the target on the "
        "training rows. The coefficient is the fitted weight. Those two columns "
        "are different numbers. The table puts the strongest absolute correlation "
        "first. MAE is the average size of the miss, in percentage points. RMSE "
        "counts a large error more heavily. Directional accuracy is how often "
        "the guess and the real move had the same sign. All scores every held-out "
        "day and shows the mean, the standard deviation across stocks, and the "
        "normal chart. The comparison file at the bottom shows the "
        "linear_regression rows and does not follow the dropdown."
    )


def _linear_calibration_bins(rows, bins: int = 10):
    """Sort by the guess and cut into equal-count bins.

    Each returned row is one bin's average guess and average actual move.
    Ten is the most bins. Fewer held-out days means fewer bins.
    """
    import pandas as pd

    empty = pd.DataFrame(columns=["mean_predicted", "mean_actual", "rows"])
    if rows is None or len(rows) == 0:
        return empty
    ordered = rows.sort_values("predicted_percent", kind="mergesort").reset_index(drop=True)
    count = min(int(bins), len(ordered))
    base, extra = divmod(len(ordered), count)
    records = []
    start = 0
    for index in range(count):
        # The first leftover days go one each into the earliest bins.
        size = base + (1 if index < extra else 0)
        group = ordered.iloc[start:start + size]
        start += size
        records.append(
            {
                "mean_predicted": float(group["predicted_percent"].mean()),
                "mean_actual": float(group["actual_percent"].mean()),
                "rows": int(len(group)),
            }
        )
    return pd.DataFrame(records)


def _linear_calibration_choice(symbol: str, everyone):
    """Rows for the bins. A stock is used only when it has at least 30 days."""
    if symbol == "All" or everyone is None or len(everyone) == 0:
        return everyone, ""
    one = everyone[everyone["ticker"].astype(str) == str(symbol)]
    if len(one) >= 30:
        return one, ""
    note = (
        f"{symbol} has {len(one)} held-out days, fewer than 30, "
        "so these bins use every stock."
    )
    return everyone, note


def _linear_time_choice(symbol: str, everyone):
    """One ticker's held-out days. All uses the first ticker in the set."""
    if everyone is None or len(everyone) == 0:
        return everyone, symbol
    ordered = everyone.sort_values(["ticker", "date"])
    if symbol == "All":
        chosen = str(ordered["ticker"].iloc[0])
    else:
        chosen = str(symbol)
    rows = ordered[ordered["ticker"].astype(str) == chosen]
    return rows, chosen


def _linear_calibration_chart(symbol: str, horizon: str) -> None:
    """Bin averages of held-out guesses against what those days actually did."""
    import numpy as np
    import plotly.graph_objects as go

    everyone = _held_out_rows("All", horizon, LINEAR_BACKTEST)
    st.subheader("Calibration")
    if everyone is None:
        st.info("The model has not scored the held-out days yet.")
        return
    if everyone.empty:
        st.info("There are no held-out days for that horizon.")
        return
    rows, note = _linear_calibration_choice(symbol, everyone)
    means = _linear_calibration_bins(rows)
    if note:
        st.caption(note)
    elif symbol == "All":
        st.caption(
            f"Held-out days for the {horizon}-day horizon, every stock. "
            "Each point is a bin average."
        )
    else:
        st.caption(
            f"Held-out days for {symbol}, {horizon} trading days. "
            "Each point is a bin average."
        )
    predicted = means["mean_predicted"]
    actual = means["mean_actual"]
    low = float(min(predicted.min(), actual.min()))
    high = float(max(predicted.max(), actual.max()))
    if low == high:
        low -= 1.0
        high += 1.0
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=predicted,
            y=actual,
            mode="markers",
            name="Bin average",
        )
    )
    # One straight line through the bin averages, plus the perfect y = x line.
    if len(means) >= 2:
        slope, intercept = np.polyfit(
            predicted.to_numpy(dtype=float),
            actual.to_numpy(dtype=float),
            1,
        )
        x0 = float(predicted.min())
        x1 = float(predicted.max())
        figure.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[float(slope * x0 + intercept), float(slope * x1 + intercept)],
                mode="lines",
                name="Line through bins",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="y = x",
            line=dict(color="#1f4e79", width=2),
        )
    )
    figure.update_layout(
        title=f"Bin averages: mean predicted vs mean actual, {horizon} trading days",
        xaxis_title="Mean predicted percent",
        yaxis_title="Mean actual percent",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(figure, width="stretch")


def _linear_time_chart(symbol: str, horizon: str) -> None:
    """Held-out actual and predicted percent for one stock, by date."""
    import plotly.graph_objects as go

    everyone = _held_out_rows("All", horizon, LINEAR_BACKTEST)
    st.subheader("Held-out days")
    if everyone is None:
        st.info("The model has not scored the held-out days yet.")
        return
    if everyone.empty:
        st.info("There are no held-out days for that horizon.")
        return
    rows, chosen = _linear_time_choice(symbol, everyone)
    if rows is None or rows.empty:
        st.info("This stock has no held-out days for that horizon.")
        return
    if symbol == "All":
        st.caption(
            f"All is selected, so this chart uses {chosen}, "
            "the first ticker in the held-out set."
        )
    else:
        st.caption(f"Held-out days for {chosen}.")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=rows["date"],
            y=rows["actual_percent"],
            name="Actual",
            mode="lines",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=rows["date"],
            y=rows["predicted_percent"],
            name="Predicted",
            mode="lines",
        )
    )
    figure.update_layout(
        title=f"{chosen}: held-out actual and predicted percent, {horizon} trading days",
        xaxis_title="Date",
        yaxis_title="Percent",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(figure, width="stretch")


def _linear_number(value: object) -> str:
    """Three decimals, or n/a when a constant input has no correlation."""
    import pandas as pd

    if pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def _linear_relationship_table(horizon: str) -> None:
    """Each input's Pearson correlation beside its fitted coefficient."""
    import pandas as pd

    table = _read_table(LINEAR_OUTPUT / "coefficients.csv")
    st.subheader("Inputs and the line")
    st.caption(
        "Pearson correlation is how each input moved with this horizon's target "
        "on the training rows. The coefficient is that input's weight in the "
        "fitted line. They are different numbers. The strongest absolute "
        "correlation is first."
    )
    if table is None or table.empty or "pearson_correlation" not in table.columns:
        st.info("Linear regression has not saved coefficients yet.")
        return
    chosen = table[table["horizon"].astype(str) == str(horizon)].copy()
    if chosen.empty:
        st.info("Linear regression has not saved coefficients yet.")
        return
    chosen["abs_correlation"] = chosen["pearson_correlation"].abs()
    chosen = chosen.sort_values(
        ["abs_correlation", "feature"],
        ascending=[False, True],
        na_position="last",
    )
    show = {
        "Feature": [str(value) for value in chosen["feature"]],
        "Pearson correlation": [
            _linear_number(value) for value in chosen["pearson_correlation"]
        ],
        "Coefficient": [_linear_number(value) for value in chosen["coefficient"]],
    }
    st.table(pd.DataFrame(show))


def _linear_regression_page(db_path: Path) -> None:
    st.header("Linear Regression")
    st.caption(
        "The same daily feature file as the random forest. "
        "The line is LinearRegression. The inputs are not standardized. "
        "It already ran when the app opened."
    )
    symbols = _forest_symbols(db_path)
    symbol = st.selectbox("Stock", ["All", *symbols], key="linear-stock")
    st.caption("All uses every stock. A ticker uses only that stock.")
    horizon = st.selectbox(
        "Horizon",
        ["1", "5", "20"],
        format_func=lambda value: f"{value} trading days",
        key="linear-horizon",
    )
    if st.button("Run Linear Regression"):
        _train_linear(db_path)
        st.rerun()
    _linear_explanation()
    _linear_calibration_chart(symbol, horizon)
    _linear_time_chart(symbol, horizon)
    if symbol == "All":
        _forest_all_metrics(horizon, LINEAR_BACKTEST)
        _forest_normal_chart(horizon, LINEAR_BACKTEST)
    else:
        _forest_metrics(symbol, horizon, LINEAR_BACKTEST)
        _forest_forecast(symbol, horizon, LINEAR_PREDICTIONS)
    _linear_relationship_table(horizon)
    _forest_comparison_table("linear_regression")


def _train_kmeans_once(db_path: Path) -> None:
    """Group the daily feature rows the first time this dashboard session starts."""
    if st.session_state.get("kmeans_ran"):
        return
    st.session_state["kmeans_ran"] = True
    _train_kmeans(db_path)


def _train_kmeans(db_path: Path) -> None:
    """Standardize the shared feature file and keep the best k.

    This does not write a second feature file and it does not add a
    comparison row.
    """
    with st.spinner("Standardizing the daily features and choosing k..."):
        from machine_learning.unsupervised.kmeans.train import train_and_save

        try:
            train_and_save(db_path)
        except Exception:
            st.error("K-Means did not finish. No secret was printed.")


def _kmeans_explanation() -> None:
    st.subheader("How the rows are built")
    st.markdown(
        "K-Means reads the same file as the random forest: "
        "**random_forest_features.csv**. That file is built from **daily_prices**, "
        "one row per stock per trading day. There is no second feature file. "
        "The nine inputs are **return_1d**, **return_5d**, **return_20d**, "
        "**dist_sma_20_pct**, **dist_sma_50_pct**, **rsi_14**, **volatility_20d**, "
        "**volume_vs_20d**, and **range_pct**. The ticker and the date only name "
        "the row."
    )
    st.markdown(
        "Those nine numbers are not on the same scale. **rsi_14** sits between "
        "0 and 100. A one-day return is a percent, often near zero. K-Means "
        "groups days by distance, so a wide scale would dominate the groups. "
        "**StandardScaler** subtracts each column's average and divides by its "
        "spread before the groups are found. The random forest and XGBoost are "
        "trees. They stay unscaled. Linear regression is a straight line on "
        "those same unscaled numbers."
    )
    st.markdown(
        "**target_1d**, **target_5d**, and **target_20d** are future percent "
        "changes. They are not inputs. A day missing one of the nine inputs is "
        "left out."
    )
    st.subheader("How the groups are chosen")
    st.markdown(
        "K-Means places each day into the nearest group. **k** is how many "
        "groups to make. The run tries every k from 2 through 8. The silhouette "
        "score is higher when a day is close to its own group and far from the "
        "others. The highest score is kept. The library model is **KMeans**, "
        "and the seed is 42 so a rerun on the same file stays put. Past 2,000 "
        "rows, that score uses a fixed sample of 2,000 rows, with at least one "
        "row from every group. The groups themselves still use every row."
    )
    st.markdown(
        "The scatter draws each scaled day on the first two principal components. "
        "Those two axes are only the chart. They are not a separate model. "
        "The center table converts each group's center back to the original "
        "units, such as RSI and percent return. The average future 1-day return "
        "describes the days in a group. It is not a prediction score. This page "
        "does not show a forecast error, and it does not add a row to the "
        "comparison file."
    )


def _kmeans_silhouette_chart() -> None:
    """Bar for each k. The kept k is the darker bar."""
    import plotly.graph_objects as go

    scores = _read_table(KMEANS_OUTPUT / "silhouette.csv")
    summary = _read_table(KMEANS_OUTPUT / "summary.csv")
    st.subheader("Silhouette scores")
    st.caption(
        "Higher means the groups were easier to tell apart. "
        "The run keeps the highest score from k = 2 through 8."
    )
    if scores is None or scores.empty:
        st.info("K-Means has not scored k yet.")
        return
    chosen = None
    if summary is not None and not summary.empty and "chosen_k" in summary.columns:
        chosen = int(summary["chosen_k"].iloc[0])
        st.write(
            f"Kept k = {chosen}. "
            f"Silhouette = {float(summary['silhouette'].iloc[0]):.3f}."
        )
    colors = []
    for k in scores["k"]:
        if chosen is not None and int(k) == chosen:
            colors.append("#1f4e79")
        else:
            colors.append("#9fb3c8")
    figure = go.Figure(
        go.Bar(
            x=scores["k"],
            y=scores["silhouette"],
            marker_color=colors,
            name="Silhouette",
        )
    )
    figure.update_layout(
        title="Silhouette score for k = 2 through 8",
        xaxis_title="k",
        yaxis_title="Silhouette",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
    )
    st.plotly_chart(figure, width="stretch")


def _kmeans_scatter() -> None:
    """Scaled rows on two chart axes, colored by the group they joined."""
    import plotly.graph_objects as go

    rows = _read_table(KMEANS_OUTPUT / "assignments.csv")
    st.subheader("Days by cluster")
    if rows is None or rows.empty:
        st.info("K-Means has not grouped the days yet.")
        return
    figure = go.Figure()
    for cluster, group in rows.groupby("cluster", sort=True):
        figure.add_trace(
            go.Scattergl(
                x=group["pc1"],
                y=group["pc2"],
                mode="markers",
                name=f"Cluster {int(cluster)}",
                marker=dict(size=4, opacity=0.55),
            )
        )
    figure.update_layout(
        title="Scaled days, colored by cluster",
        xaxis_title="First principal component",
        yaxis_title="Second principal component",
        height=520,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "The first two principal components are only the chart axes, not a separate model."
    )


def _kmeans_size_table() -> None:
    import pandas as pd

    summary = _read_table(KMEANS_OUTPUT / "summary.csv")
    st.subheader("Rows in each cluster")
    if summary is None or summary.empty:
        st.info("K-Means has not counted the clusters yet.")
        return
    show = {
        "Cluster": [int(value) for value in summary["cluster"]],
        "Rows": [int(value) for value in summary["rows"]],
    }
    st.table(pd.DataFrame(show))


def _kmeans_center_table() -> None:
    import pandas as pd

    centers = _read_table(KMEANS_OUTPUT / "centers.csv")
    st.subheader("Cluster centers")
    st.caption("Each center is converted back to the original feature units.")
    if centers is None or centers.empty:
        st.info("K-Means has not saved centers yet.")
        return
    show = {"Cluster": [int(value) for value in centers["cluster"]]}
    for column in centers.columns:
        if column == "cluster":
            continue
        show[column] = [f"{float(value):.3f}" for value in centers[column]]
    st.table(pd.DataFrame(show))


def _kmeans_return_table() -> None:
    import pandas as pd

    summary = _read_table(KMEANS_OUTPUT / "summary.csv")
    st.subheader("Average future 1-day return")
    st.caption(
        "This is the average of target_1d for the days in that cluster. "
        "It describes those days. It is not a prediction score."
    )
    if summary is None or summary.empty or "mean_future_return_1d" not in summary.columns:
        st.info("There is no future 1-day return to describe yet.")
        return
    values = []
    for value in summary["mean_future_return_1d"]:
        if pd.isna(value):
            values.append("n/a")
        else:
            values.append(f"{float(value):.3f}")
    show = {
        "Cluster": [int(value) for value in summary["cluster"]],
        "Average future 1-day return": values,
    }
    st.table(pd.DataFrame(show))


def _kmeans_page(db_path: Path) -> None:
    st.header("K-Means")
    st.caption(
        "The same daily feature file as the random forest. "
        "The inputs are standardized, then grouped with KMeans. "
        "It already ran when the app opened."
    )
    if st.button("Run K-Means"):
        _train_kmeans(db_path)
        st.rerun()
    _kmeans_explanation()
    _kmeans_silhouette_chart()
    _kmeans_scatter()
    _kmeans_size_table()
    _kmeans_center_table()
    _kmeans_return_table()


def _train_pca_once(db_path: Path) -> None:
    """Fit PCA on the daily feature rows the first time this dashboard session starts."""
    if st.session_state.get("pca_ran"):
        return
    st.session_state["pca_ran"] = True
    _train_pca(db_path)


def _train_pca(db_path: Path) -> None:
    """Standardize the shared feature file and keep every component.

    This does not write a second feature file and it does not add a
    comparison row.
    """
    with st.spinner("Standardizing the daily features and fitting PCA..."):
        from machine_learning.unsupervised.pca.train import train_and_save

        try:
            train_and_save(db_path)
        except Exception:
            st.error("PCA did not finish. No secret was printed.")


def _pca_explanation() -> None:
    st.subheader("How the rows are built")
    st.markdown(
        "PCA reads the same file as the random forest: "
        "**random_forest_features.csv**. That file is built from **daily_prices**, "
        "one row per stock per trading day. There is no second feature file. "
        "The nine inputs are **return_1d**, **return_5d**, **return_20d**, "
        "**dist_sma_20_pct**, **dist_sma_50_pct**, **rsi_14**, **volatility_20d**, "
        "**volume_vs_20d**, and **range_pct**. The ticker and the date only name "
        "the row."
    )
    st.markdown(
        "Those nine numbers are not on the same scale. **rsi_14** sits between "
        "0 and 100. A one-day return is a percent, often near zero. "
        "**StandardScaler** subtracts each column's average and divides by its "
        "spread. PCA then looks for new directions in those scaled numbers. "
        "The random forest, XGBoost, and linear regression stay unscaled."
    )
    st.markdown(
        "**target_1d**, **target_5d**, and **target_20d** are future percent "
        "changes. They are not inputs. A day missing one of the nine inputs is "
        "left out."
    )
    st.subheader("How the components are found")
    st.markdown(
        "A component is a mix of the nine inputs. The first component is the "
        "mix that carries the most variance. The next one carries the most of "
        "what is left, and it is perpendicular to the first. The run keeps "
        "every component, one for each input. The library model is **PCA**, "
        "and the seed is 42 so a rerun on the same file stays put."
    )
    st.markdown(
        "The bar shows how much of the variance each component carries. The "
        "line adds those shares from the first component onward. The loadings "
        "table is each original feature's weight on the first three components. "
        "This page does not predict a future return. It does not show a "
        "forecast error, and it does not add a row to the comparison file."
    )


def _pca_variance_bar() -> None:
    """One bar per component. Taller means that component carries more variance."""
    import plotly.graph_objects as go

    table = _read_table(PCA_OUTPUT / "explained_variance.csv")
    st.subheader("Explained variance")
    st.caption("Each bar is the share of the standardized features that one component carries.")
    if table is None or table.empty:
        st.info("PCA has not measured the components yet.")
        return
    labels = [f"PC{int(value)}" for value in table["component"]]
    percents = [float(value) * 100 for value in table["explained_variance_ratio"]]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=percents,
            marker_color="#1f4e79",
            name="Explained variance",
        )
    )
    figure.update_layout(
        title="Explained variance for each component",
        xaxis_title="Component",
        yaxis_title="Explained variance (%)",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
    )
    st.plotly_chart(figure, width="stretch")
    if len(table) >= 2:
        first_two = float(table["cumulative_explained_variance"].iloc[1]) * 100
        st.write(f"The first two components explain {first_two:.1f}% of the variance.")


def _pca_cumulative_line() -> None:
    """Running total of the variance shares, from the first component onward."""
    import plotly.graph_objects as go

    table = _read_table(PCA_OUTPUT / "explained_variance.csv")
    st.subheader("Cumulative explained variance")
    st.caption("Each point adds that component's share to the components before it.")
    if table is None or table.empty:
        st.info("PCA has not added up the variance yet.")
        return
    labels = [f"PC{int(value)}" for value in table["component"]]
    percents = [float(value) * 100 for value in table["cumulative_explained_variance"]]
    figure = go.Figure(
        go.Scatter(
            x=labels,
            y=percents,
            mode="lines+markers",
            line=dict(color="#1f4e79", width=2),
            name="Cumulative explained variance",
        )
    )
    figure.update_layout(
        title="Cumulative explained variance",
        xaxis_title="Component",
        yaxis_title="Cumulative explained variance (%)",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
    )
    st.plotly_chart(figure, width="stretch")


def _pca_loadings_table() -> None:
    """Each original feature's weight on the first three components."""
    import pandas as pd

    loadings = _read_table(PCA_OUTPUT / "loadings.csv")
    st.subheader("Loadings")
    st.caption("A loading near 1 or -1 means that feature moves with that component.")
    if loadings is None or loadings.empty:
        st.info("PCA has not saved loadings yet.")
        return
    show = {"Feature": [str(value) for value in loadings["feature"]]}
    for column, title in (("pc1", "PC1"), ("pc2", "PC2"), ("pc3", "PC3")):
        if column not in loadings.columns:
            continue
        show[title] = [f"{float(value):.3f}" for value in loadings[column]]
    st.table(pd.DataFrame(show))


def _pca_page(db_path: Path) -> None:
    st.header("PCA")
    st.caption(
        "The same daily feature file as the random forest. "
        "The inputs are standardized, then summarized with PCA. "
        "It already ran when the app opened."
    )
    if st.button("Run PCA"):
        _train_pca(db_path)
        st.rerun()
    _pca_explanation()
    _pca_variance_bar()
    _pca_cumulative_line()
    _pca_loadings_table()


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
