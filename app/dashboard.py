"""Streamlit portfolio dashboard.

Run with ``python -m streamlit run app/dashboard.py``. The Start Portfolio
window owns the live IEX session. While that window is open, minute bars
append about once a minute and a finished day is written to daily prices.
This process only reads SQLite. The random forest trains once when the
dashboard session starts, and again only from Run model.
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
# The first three stay at the top of the sidebar. A new model is added to
# ML_PAGES, so it shows under ML Methods and does not move those three.
MAIN_PAGES = (PAGE_PORTFOLIO, PAGE_STOCK, PAGE_DATABASE)
ML_PAGES = (PAGE_FOREST,)

FOREST_EXPORT = ROOT / "machine_learning" / "exports" / "random_forest_features.csv"
FOREST_PREDICTIONS = ROOT / "machine_learning" / "outputs" / "predictions.csv"
FOREST_BACKTEST = ROOT / "machine_learning" / "outputs" / "backtest.csv"
FOREST_COMPARISON = ROOT / "machine_learning" / "outputs" / "model_comparison.csv"


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
        "The random forest trains on the daily bars."
    )
    _train_forest_once(db_path)
    if page == PAGE_PORTFOLIO:
        _portfolio_page(db_path)
    elif page == PAGE_STOCK:
        _stock_page(db_path)
    elif page == PAGE_FOREST:
        _random_forest_page(db_path)
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


def _held_out_rows(symbol: str, horizon: str):
    """Held-out rows for one horizon. All keeps every ticker."""
    table = _read_table(FOREST_BACKTEST)
    if table is None or "ticker" not in table.columns or "horizon" not in table.columns:
        return None
    rows = table[table["horizon"].astype(str) == str(horizon)]
    if symbol != "All":
        rows = rows[rows["ticker"].astype(str) == symbol]
    if rows.empty or "date" not in rows.columns:
        return rows
    sort_columns = ["date"] if symbol != "All" else ["ticker", "date"]
    return rows.sort_values(sort_columns)


def _forest_chart(symbol: str, horizon: str) -> None:
    import plotly.graph_objects as go

    rows = _held_out_rows(symbol, horizon)
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


def _forest_all_metrics(horizon: str) -> None:
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
    rows = _held_out_rows("All", horizon)
    if rows is None or rows.empty:
        st.info("There are no held-out days for that horizon.")
        return
    scores = score_held_out(rows)
    if scores is None:
        st.info("There are no held-out days for that horizon.")
        return
    _show_three_scores(scores, score_spread(per_stock_scores(rows)))


def _forest_normal_chart(horizon: str) -> None:
    """Normal curves of the per-stock scores. Only the All choice uses this."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from machine_learning.supervised.random_forest.train import per_stock_scores

    rows = _held_out_rows("All", horizon)
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


def _forest_metrics(symbol: str, horizon: str) -> None:
    """Three scores for this stock only. They follow the selectors."""
    from machine_learning.supervised.random_forest.train import score_held_out

    st.subheader(f"This stock: {symbol}")
    st.caption(
        f"MAE, RMSE, and directional accuracy for {symbol} only, "
        f"over {horizon} trading days."
    )
    rows = _held_out_rows(symbol, horizon)
    if rows is None or rows.empty:
        st.info("This stock has no held-out days for that horizon.")
        return
    scores = score_held_out(rows)
    if scores is None:
        st.info("This stock has no held-out days for that horizon.")
        return
    _show_three_scores(scores, None)


def _forest_forecast(symbol: str, horizon: str) -> None:
    predictions = _read_table(FOREST_PREDICTIONS)
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


def _forest_comparison_table() -> None:
    """Saved scores from model_comparison.csv. This does not follow the dropdown."""
    st.subheader("Comparison file")
    st.caption(
        "Saved scores for every stock and every horizon. "
        "This table does not follow the stock dropdown."
    )
    table = _read_table(FOREST_COMPARISON)
    if table is None or table.empty or "model" not in table.columns:
        st.info("The comparison file is written when you run the forest.")
        _forest_score_notes()
        return
    forest = table[table["model"].astype(str) == "random_forest"].copy()
    if forest.empty:
        st.info("Random forest has no rows in the comparison file yet.")
        _forest_score_notes()
        return
    forest["horizon_days"] = forest["horizon_days"].astype(int)
    forest = forest.sort_values("horizon_days")
    show = {
        "Horizon (days)": forest["horizon_days"].tolist(),
        "MAE": [f"{float(value):.3f}" for value in forest["mae"]],
        "RMSE": [f"{float(value):.3f}" for value in forest["rmse"]],
        "Directional accuracy": [
            f"{float(value):.1%}" for value in forest["directional_accuracy"]
        ],
        "Cutoff": [str(value) for value in forest["cutoff"]],
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
