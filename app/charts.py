"""Plotly figures for the dashboard. Prices come from the caller."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_collection.alpaca_bars import NEW_YORK

from .indicators import ChartPoint
from .portfolio import Position

UP = "#1b7f4e"
DOWN = "#b42318"
NAVY = "#1f4e79"
AMBER = "#b86e00"
VIOLET = "#6d28d9"


def build_stock_figure(symbol: str, points: list[ChartPoint]) -> go.Figure:
    """Candles, volume, RSI, and MACD for one symbol."""
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
    when = [point.started_at.astimezone(NEW_YORK) for point in points]
    colors = [UP if point.close >= point.open else DOWN for point in points]
    figure.add_trace(
        go.Candlestick(
            x=when,
            open=[point.open for point in points],
            high=[point.high for point in points],
            low=[point.low for point in points],
            close=[point.close for point in points],
            name="Price",
            increasing_line_color=UP,
            decreasing_line_color=DOWN,
        ),
        row=1,
        col=1,
    )
    _add_line(figure, when, points, "sma_fast", "SMA 20", NAVY, 1)
    _add_line(figure, when, points, "sma_slow", "SMA 50", AMBER, 1)
    _add_line(figure, when, points, "bb_upper", "Bollinger upper", "#7aa2c4", 1, dash="dot")
    _add_line(figure, when, points, "bb_lower", "Bollinger lower", "#7aa2c4", 1, dash="dot")
    _add_line(figure, when, points, "vwap", "VWAP", VIOLET, 1)
    figure.add_trace(
        go.Bar(
            x=when,
            y=[point.volume for point in points],
            name="Volume",
            marker_color=colors,
        ),
        row=2,
        col=1,
    )
    _add_line(figure, when, points, "rsi", "RSI", NAVY, 3)
    figure.add_hline(y=70, line_dash="dot", line_color="#98a2b3", row=3, col=1)
    figure.add_hline(y=30, line_dash="dot", line_color="#98a2b3", row=3, col=1)
    _add_line(figure, when, points, "macd", "MACD", NAVY, 4)
    _add_line(figure, when, points, "macd_signal", "Signal", AMBER, 4)
    histogram = [point.macd_hist for point in points]
    figure.add_trace(
        go.Bar(
            x=when,
            y=histogram,
            name="Histogram",
            marker_color=[UP if value is not None and value >= 0 else DOWN for value in histogram],
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
    )
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
    points: list[ChartPoint],
    field: str,
    name: str,
    color: str,
    row: int,
    dash: str | None = None,
) -> None:
    values = [getattr(point, field) for point in points]
    if all(value is None for value in values):
        return
    line: dict[str, object] = {"color": color, "width": 1.6}
    if dash:
        line["dash"] = dash
    figure.add_trace(
        go.Scatter(x=when, y=values, name=name, mode="lines", line=line, connectgaps=False),
        row=row,
        col=1,
    )
