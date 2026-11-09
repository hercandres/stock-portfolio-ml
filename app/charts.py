"""Plotly figures for the dashboard. Prices come from the caller.

Closed market time is removed from the axis. Nights, weekends, and holidays
are not drawn, and no bar is invented for them. A hole is left only when a
session bar is missing from the source. The candles on either side stay.
"""

from __future__ import annotations

from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_collection.alpaca_bars import NEW_YORK
from data_collection.market_calendar import holiday_dates, next_session_mark, session_has_a_hole

from .indicators import ChartPoint
from .portfolio import Position

UP = "#1b7f4e"
DOWN = "#b42318"
NAVY = "#1f4e79"
AMBER = "#b86e00"
VIOLET = "#6d28d9"


def build_stock_figure(symbol: str, points: list[ChartPoint], *, step: str = "1m") -> go.Figure:
    """Candles, volume, RSI, and MACD for one symbol.

    ``step`` is how far apart these candles are: 1m, 5m, 30m, or 1d.
    Sessions are drawn back to back. Closed time is not a gap in the line.
    """
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.52, 0.16, 0.16, 0.16],
        subplot_titles=(f"{symbol}", "Volume", "RSI 14", "MACD"),
    )
    if not points:
        figure.update_layout(title=f"{symbol} has no stored bars in this range")
        return figure
    columns = _plot_columns(points, step)
    when = columns["when"]
    colors = [
        UP if close is not None and open is not None and close >= open else DOWN
        for open, close in zip(columns["open"], columns["close"], strict=True)
    ]
    figure.add_trace(
        go.Candlestick(
            x=when,
            open=columns["open"],
            high=columns["high"],
            low=columns["low"],
            close=columns["close"],
            name="Price",
            increasing_line_color=UP,
            decreasing_line_color=DOWN,
        ),
        row=1,
        col=1,
    )
    _add_line(figure, when, columns["sma_fast"], "SMA 20", NAVY, 1)
    _add_line(figure, when, columns["sma_slow"], "SMA 50", AMBER, 1)
    _add_line(figure, when, columns["bb_upper"], "Bollinger upper", "#7aa2c4", 1, dash="dot")
    _add_line(figure, when, columns["bb_lower"], "Bollinger lower", "#7aa2c4", 1, dash="dot")
    _add_line(figure, when, columns["vwap"], "VWAP", VIOLET, 1)
    figure.add_trace(
        go.Bar(
            x=when,
            y=columns["volume"],
            name="Volume",
            marker_color=colors,
        ),
        row=2,
        col=1,
    )
    _add_line(figure, when, columns["rsi"], "RSI", NAVY, 3)
    figure.add_hline(y=70, line_dash="dot", line_color="#98a2b3", row=3, col=1)
    figure.add_hline(y=30, line_dash="dot", line_color="#98a2b3", row=3, col=1)
    _add_line(figure, when, columns["macd"], "MACD", NAVY, 4)
    _add_line(figure, when, columns["macd_signal"], "Signal", AMBER, 4)
    figure.add_trace(
        go.Bar(
            x=when,
            y=columns["macd_hist"],
            name="Histogram",
            marker_color=[
                UP if value is not None and value >= 0 else DOWN for value in columns["macd_hist"]
            ],
        ),
        row=4,
        col=1,
    )
    figure.update_layout(
        height=860,
        margin=dict(l=48, r=24, t=48, b=32),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        # Same symbol and step keep the user's zoom when a new minute arrives.
        uirevision=f"{symbol}-{step}",
    )
    figure.update_xaxes(rangebreaks=_closed_market_breaks(step))
    figure.update_yaxes(title_text="Price", row=1, col=1)
    figure.update_yaxes(title_text="Shares", row=2, col=1)
    figure.update_yaxes(range=[0, 100], row=3, col=1)
    return figure


def build_allocation_figure(positions: list[Position], total_value: float) -> go.Figure | None:
    """Weights of positions that have a stored close."""
    labels: list[str] = []
    values: list[float] = []
    for position in positions:
        value = position.market_value
        if position.quantity <= 0 or value is None or value <= 0 or total_value <= 0:
            continue
        labels.append(position.symbol)
        values.append(value)
    if not values:
        return None
    figure = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                sort=False,
                textinfo="label+percent",
            )
        ]
    )
    figure.update_layout(
        height=420,
        margin=dict(l=16, r=16, t=32, b=16),
        title="Allocation by latest stored value",
        showlegend=False,
    )
    return figure


def _add_line(
    figure: go.Figure,
    when: list,
    values: list[float | None],
    name: str,
    color: str,
    row: int,
    dash: str | None = None,
) -> None:
    if all(value is None for value in values):
        return
    line: dict[str, object] = {"color": color, "width": 1.6}
    if dash:
        line["dash"] = dash
    # A None is only a missing session bar. Closed time has no None, so the
    # line continues from one session into the next instead of jumping.
    figure.add_trace(
        go.Scatter(x=when, y=values, name=name, mode="lines", line=line, connectgaps=False),
        row=row,
        col=1,
    )


def _plot_columns(points: list[ChartPoint], step: str) -> dict[str, list]:
    """One list per series. A missing session bar becomes an empty slot.

    The empty slot has no price. The stored candle before it and the stored
    candle after it both stay in the lists.
    """
    names = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_fast",
        "sma_slow",
        "bb_upper",
        "bb_lower",
        "vwap",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
    )
    columns: dict[str, list] = {name: [] for name in names}
    columns["when"] = []
    previous: datetime | None = None
    for point in points:
        if previous is not None and session_has_a_hole(previous, point.started_at, step):
            columns["when"].append(_wall_time(next_session_mark(previous, step)))
            for name in names:
                columns[name].append(None)
        columns["when"].append(_wall_time(point.started_at))
        for name in names:
            columns[name].append(getattr(point, name))
        previous = point.started_at
    return columns


def _closed_market_breaks(step: str) -> list[dict[str, object]]:
    """Axis pieces Plotly should not draw. This does not add prices."""
    holidays = [day.isoformat() for day in sorted(holiday_dates())]
    breaks: list[dict[str, object]] = [
        {"bounds": ["sat", "mon"]},
        {"values": holidays},
    ]
    if step != "1d":
        # 16:00 through 09:30. The 09:30 open stays on the chart.
        breaks.append({"bounds": [16, 9.5], "pattern": "hour"})
    return breaks


def _wall_time(when: datetime) -> datetime:
    """New York clock time without a timezone, which is what the axis breaks expect."""
    local = when.astimezone(NEW_YORK).replace(tzinfo=None)
    return local
