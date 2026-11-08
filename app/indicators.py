"""Price math for the dashboard charts.

These functions read numbers the caller already loaded. They do not download
market data and they do not fill missing minutes with invented prices.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")

RANGE_KEYS = ("1D", "5D", "1M", "3M", "6M", "1Y")
AGGREGATION = {"1D": "1m", "5D": "5m", "1M": "30m", "3M": "1d", "6M": "1d", "1Y": "1d"}
_WARMUP = {
    "1D": timedelta(hours=6),
    "5D": timedelta(days=2),
    "1M": timedelta(days=5),
    "3M": timedelta(days=90),
    "6M": timedelta(days=90),
    "1Y": timedelta(days=90),
}


@dataclass(frozen=True)
class PriceBar:
    """One OHLCV bar. ``started_at`` is timezone-aware UTC."""

    started_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class ChartPoint:
    started_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    sma_fast: float | None
    sma_slow: float | None
    bb_mid: float | None
    bb_upper: float | None
    bb_lower: float | None
    vwap: float | None
    rsi: float | None
    macd: float | None
    macd_signal: float | None
    macd_hist: float | None


def sma(values: list[float], period: int) -> list[float | None]:
    """Simple moving average. Leading values are None until the window fills."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= period:
            running -= values[index - period]
        if index + 1 >= period:
            out.append(running / period)
        else:
            out.append(None)
    return out


def ema(values: list[float], period: int) -> list[float | None]:
    """Exponential moving average seeded with a simple average."""
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period:
        return [None] * len(values)
    multiplier = 2 / (period + 1)
    out: list[float | None] = [None] * (period - 1)
    seed = sum(values[:period]) / period
    out.append(seed)
    previous = seed
    for value in values[period:]:
        previous = value * multiplier + previous * (1 - multiplier)
        out.append(previous)
    return out


def rsi(closes: list[float], period: int = 14) -> list[float | None]:
    """Wilder RSI. A stretch of gains is 100 and a stretch of losses is 0."""
    if period <= 0:
        raise ValueError("period must be positive")
    if len(closes) <= period:
        return [None] * len(closes)
    changes = [closes[index] - closes[index - 1] for index in range(1, len(closes))]
    gains = [max(change, 0.0) for change in changes]
    losses = [max(-change, 0.0) for change in changes]
    average_gain = sum(gains[:period]) / period
    average_loss = sum(losses[:period]) / period
    out: list[float | None] = [None] * period
    out.append(_rsi_value(average_gain, average_loss))
    for index in range(period, len(changes)):
        average_gain = (average_gain * (period - 1) + gains[index]) / period
        average_loss = (average_loss * (period - 1) + losses[index]) / period
        out.append(_rsi_value(average_gain, average_loss))
    return out


def macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """MACD line, signal line, and histogram. Slow EMA uses the longer period."""
    fast_line = ema(closes, fast)
    slow_line = ema(closes, slow)
    macd_line: list[float | None] = []
    for fast_value, slow_value in zip(fast_line, slow_line, strict=True):
        if fast_value is None or slow_value is None:
            macd_line.append(None)
        else:
            macd_line.append(fast_value - slow_value)
    present = [value for value in macd_line if value is not None]
    signal_present = ema(present, signal)
    signal_line: list[float | None] = []
    histogram: list[float | None] = []
    cursor = 0
    for value in macd_line:
        if value is None:
            signal_line.append(None)
            histogram.append(None)
            continue
        signal_value = signal_present[cursor]
        cursor += 1
        signal_line.append(signal_value)
        if signal_value is None:
            histogram.append(None)
        else:
            histogram.append(value - signal_value)
    return macd_line, signal_line, histogram


def bollinger(
    closes: list[float],
    period: int = 20,
    width: float = 2.0,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Middle, upper, and lower Bollinger bands. Width is in standard deviations."""
    middle = sma(closes, period)
    upper: list[float | None] = []
    lower: list[float | None] = []
    for index, mid in enumerate(middle):
        if mid is None:
            upper.append(None)
            lower.append(None)
            continue
        window = closes[index - period + 1 : index + 1]
        variance = sum((value - mid) ** 2 for value in window) / period
        deviation = variance**0.5
        upper.append(mid + width * deviation)
        lower.append(mid - width * deviation)
    return middle, upper, lower


def vwap_series(bars: list[PriceBar], *, reset_each_session: bool) -> list[float | None]:
    """Volume-weighted average of typical price.

    Intraday charts restart each New York session. Daily charts accumulate
    across the bars they are given.
    """
    cumulative_price_volume = 0.0
    cumulative_volume = 0
    previous_day = None
    out: list[float | None] = []
    for bar in bars:
        day = bar.started_at.astimezone(NEW_YORK).date()
        if reset_each_session and previous_day is not None and day != previous_day:
            cumulative_price_volume = 0.0
            cumulative_volume = 0
        previous_day = day
        typical = (bar.high + bar.low + bar.close) / 3
        cumulative_price_volume += typical * bar.volume
        cumulative_volume += bar.volume
        if cumulative_volume == 0:
            out.append(None)
        else:
            out.append(cumulative_price_volume / cumulative_volume)
    return out


def aggregate_bars(bars: list[PriceBar], rule: str) -> list[PriceBar]:
    """Collapse minute bars into 5-minute, 30-minute, or daily candles."""
    if rule == "1m":
        return list(bars)
    grouped: dict[datetime, list[PriceBar]] = {}
    for bar in bars:
        key = _bucket_start(bar.started_at, rule)
        grouped.setdefault(key, []).append(bar)
    candles: list[PriceBar] = []
    for key in sorted(grouped):
        group = grouped[key]
        candles.append(
            PriceBar(
                started_at=key,
                open=group[0].open,
                high=max(item.high for item in group),
                low=min(item.low for item in group),
                close=group[-1].close,
                volume=sum(item.volume for item in group),
            )
        )
    return candles


def display_start(latest: datetime, range_key: str) -> datetime:
    """First instant included in the chart, anchored to the latest stored bar.

    Anchoring to the stored bar keeps Friday's session on screen when the
    clock is on a weekend.
    """
    if range_key not in AGGREGATION:
        raise ValueError(f"unknown range {range_key}")
    local = latest.astimezone(NEW_YORK)
    if range_key == "1D":
        start_local = local.replace(hour=0, minute=0, second=0, microsecond=0)
    elif range_key == "5D":
        start_local = (local - timedelta(days=4)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif range_key == "1M":
        start_local = local - timedelta(days=31)
    elif range_key == "3M":
        start_local = local - timedelta(days=93)
    elif range_key == "6M":
        start_local = local - timedelta(days=186)
    else:
        start_local = local - timedelta(days=366)
    return start_local.astimezone(timezone.utc)


def fetch_start(latest: datetime, range_key: str) -> datetime:
    """How far back to read so moving averages have bars before the window."""
    return display_start(latest, range_key) - _WARMUP[range_key]


def prepare_chart(bars: list[PriceBar], range_key: str) -> list[ChartPoint]:
    """Aggregate stored bars and attach indicators for the visible window."""
    if range_key not in AGGREGATION:
        raise ValueError(f"unknown range {range_key}")
    ordered = sorted(bars, key=lambda bar: bar.started_at)
    if not ordered:
        return []
    rule = AGGREGATION[range_key]
    candles = aggregate_bars(ordered, rule)
    closes = [bar.close for bar in candles]
    fast = sma(closes, 20)
    slow = sma(closes, 50)
    middle, upper, lower = bollinger(closes)
    vwap = vwap_series(candles, reset_each_session=rule != "1d")
    momentum = rsi(closes)
    macd_line, signal_line, histogram = macd(closes)
    visible_from = display_start(ordered[-1].started_at, range_key)
    points: list[ChartPoint] = []
    for index, bar in enumerate(candles):
        if bar.started_at < visible_from:
            continue
        points.append(
            ChartPoint(
                started_at=bar.started_at,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                sma_fast=fast[index],
                sma_slow=slow[index],
                bb_mid=middle[index],
                bb_upper=upper[index],
                bb_lower=lower[index],
                vwap=vwap[index],
                rsi=momentum[index],
                macd=macd_line[index],
                macd_signal=signal_line[index],
                macd_hist=histogram[index],
            )
        )
    return points


def _rsi_value(average_gain: float, average_loss: float) -> float:
    if average_loss == 0:
        return 100.0
    if average_gain == 0:
        return 0.0
    relative = average_gain / average_loss
    return 100 - (100 / (1 + relative))


def _bucket_start(started_at: datetime, rule: str) -> datetime:
    local = started_at.astimezone(NEW_YORK).replace(second=0, microsecond=0)
    if rule == "5m":
        local = local.replace(minute=(local.minute // 5) * 5)
    elif rule == "30m":
        local = local.replace(minute=(local.minute // 30) * 30)
    elif rule == "1d":
        local = local.replace(hour=0, minute=0)
    else:
        raise ValueError(f"unknown aggregation {rule}")
    return local.astimezone(timezone.utc)
