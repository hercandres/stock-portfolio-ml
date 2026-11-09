"""Alpaca historical daily bars.

These bars are stored in ``daily_prices``, one row per ticker per trading
day. Minute bars stay in ``historical_prices``. This module does not write
minute rows and does not delete them.

The request matches the minute loader: free IEX, split-adjusted, no trades
and no quotes. The start date is 2016-01-01. Alpaca returns the first day
it actually has. On this free feed that is 2020-07-27 for long-listed names.
A later listing, such as SPCX, starts on its first day (2026-06-12).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from database.connection import UTC_TIMESTAMP_FORMAT

# Price checks are shared with the minute loader so a bad bar is rejected the same way.
from .alpaca_bars import (
    ADJUSTMENT,
    DATA_BASE_URL,
    FEED,
    MIN_REQUEST_INTERVAL_SECONDS,
    NEW_YORK,
    REGULAR_CLOSE_MINUTES,
    AlpacaRequestError,
    AlpacaResponseError,
    urllib_transport,
    _price,
    _short_message,
    _volume,
)

BARS_PATH = "/v2/stocks/{symbol}/bars"
TIMEFRAME = "1Day"
# First day we ask for. The feed may begin later than this.
DAILY_START = date(2016, 1, 1)
PAGE_LIMIT = 10000
MAX_PAGES = 5

Transport = Callable[[str, dict[str, str]], tuple[int, dict[str, str], str]]
Pause = Callable[[float], None]


@dataclass(frozen=True)
class DailyBar:
    """One completed trading day. The date is YYYY-MM-DD."""

    ticker: str
    trading_date: str
    open: float
    high: float
    low: float
    close: float
    volume: int

    def as_row(self) -> tuple[str, str, float, float, float, float, int]:
        return (
            self.ticker,
            self.trading_date,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
        )


@dataclass(frozen=True)
class DailyFetch:
    bars: tuple[DailyBar, ...]
    raw_bar_count: int


def latest_session_date(now: datetime) -> date:
    """Last weekday whose regular session is already over.

    A weekday before 16:00 ET is still open, so that date is left out.
    Saturday and Sunday step back to Friday. Holiday calendars are not
    listed here; Alpaca has no bar on those days.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(NEW_YORK)
    day = local.date()
    minute_of_day = local.hour * 60 + local.minute
    session_done = local.weekday() < 5 and minute_of_day >= REGULAR_CLOSE_MINUTES
    if not session_done:
        day = day - timedelta(days=1)
    while day.weekday() >= 5:
        day = day - timedelta(days=1)
    return day


def trading_date(started_at: datetime) -> str:
    """Turn a daily-bar timestamp into YYYY-MM-DD.

    Alpaca sometimes marks the day at 00:00 UTC. That date is the UTC date.
    A bar at midnight New York is 04:00 or 05:00 UTC, and the New York date
    is the trading day. Both forms land on the session we want.
    """
    if started_at.tzinfo is None:
        raise ValueError("started_at must be timezone-aware")
    utc = started_at.astimezone(timezone.utc)
    if utc.hour == 0 and utc.minute == 0 and utc.second == 0:
        day = utc.date()
    else:
        day = utc.astimezone(NEW_YORK).date()
    return day.isoformat()


def keep_trading_date(trading_day: str, now: datetime) -> bool:
    """True for a weekday from 2016-01-01 through the latest finished session."""
    day = date.fromisoformat(trading_day)
    if day < DAILY_START or day.weekday() >= 5:
        return False
    return day <= latest_session_date(now)


class AlpacaDailyClient:
    """Download split-adjusted IEX daily bars for one symbol at a time."""

    def __init__(
        self,
        api_key_id: str,
        api_secret_key: str,
        *,
        transport: Transport | None = None,
        pause: Pause | None = None,
        feed: str = FEED,
    ) -> None:
        if feed != FEED:
            raise ValueError("daily history requests the IEX feed only")
        if not api_key_id.strip() or not api_secret_key.strip():
            raise ValueError("Alpaca key id and secret are required")
        self._api_key_id = api_key_id
        self._api_secret_key = api_secret_key
        self._transport = transport or urllib_transport
        self._pause = pause or time.sleep
        self._next_request_at = 0.0

    def historical_daily_bars(self, symbol: str, now: datetime) -> DailyFetch:
        """Bars from 2016-01-01, or the first day Alpaca has, through the last session."""
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        start = datetime(DAILY_START.year, DAILY_START.month, DAILY_START.day, tzinfo=timezone.utc)
        end = now.astimezone(timezone.utc).replace(microsecond=0)
        page_token: str | None = None
        stored: list[DailyBar] = []
        raw_bar_count = 0
        for page_number in range(1, MAX_PAGES + 1):
            body = self._get_page(symbol, start, end, page_token)
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as exc:
                raise AlpacaResponseError(f"{symbol}: bars response was not JSON") from exc
            if not isinstance(payload, dict):
                raise AlpacaResponseError(f"{symbol}: bars response was not an object")
            raw_bars = payload.get("bars") or []
            if not isinstance(raw_bars, list):
                raise AlpacaResponseError(f"{symbol}: bars field was not a list")
            raw_bar_count += len(raw_bars)
            for item in raw_bars:
                parsed = parse_daily_bar(symbol, item)
                if keep_trading_date(parsed.trading_date, now):
                    stored.append(parsed)
            token = payload.get("next_page_token")
            if not token:
                return DailyFetch(bars=tuple(stored), raw_bar_count=raw_bar_count)
            if not isinstance(token, str):
                raise AlpacaResponseError(f"{symbol}: next_page_token was not text")
            page_token = token
        raise AlpacaResponseError(
            f"{symbol}: stopped after {page_number} pages with more bars remaining"
        )

    def _get_page(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        page_token: str | None,
    ) -> str:
        url = _bars_url(symbol, start, end, page_token)
        headers = {
            "APCA-API-KEY-ID": self._api_key_id,
            "APCA-API-SECRET-KEY": self._api_secret_key,
            "Accept": "application/json",
        }
        last_error: AlpacaRequestError | None = None
        for attempt in range(1, 4):
            self._wait_turn()
            try:
                status, _response_headers, body = self._transport(url, headers)
            except (OSError, urllib.error.URLError) as exc:
                last_error = AlpacaRequestError(symbol, 0, type(exc).__name__)
                self._pause(float(attempt))
                continue
            if status == 200:
                return body
            message = _short_message(body)
            if status == 429 or status >= 500:
                last_error = AlpacaRequestError(symbol, status, message)
                self._pause(float(attempt))
                continue
            raise AlpacaRequestError(symbol, status, message)
        assert last_error is not None
        raise last_error

    def _wait_turn(self) -> None:
        wait = self._next_request_at - time.monotonic()
        if wait > 0:
            self._pause(wait)
        self._next_request_at = time.monotonic() + MIN_REQUEST_INTERVAL_SECONDS


def parse_daily_bar(symbol: str, item: object) -> DailyBar:
    """Read one daily bar. Trade count and VWAP are ignored."""
    if not isinstance(item, dict):
        raise AlpacaResponseError(f"{symbol}: a bar was not an object")
    started_at = _parse_bar_time(symbol, item.get("t"))
    open_ = _price(symbol, item.get("o"), "open")
    high = _price(symbol, item.get("h"), "high")
    low = _price(symbol, item.get("l"), "low")
    close = _price(symbol, item.get("c"), "close")
    volume = _volume(symbol, item.get("v"))
    if high < low or open_ < low or open_ > high or close < low or close > high:
        raise AlpacaResponseError(f"{symbol}: bar at {started_at.isoformat()} has invalid OHLC")
    return DailyBar(
        ticker=symbol,
        trading_date=trading_date(started_at),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _bars_url(symbol: str, start: datetime, end: datetime, page_token: str | None) -> str:
    encoded_symbol = urllib.parse.quote(symbol, safe=".")
    query = {
        "timeframe": TIMEFRAME,
        "start": start.strftime(UTC_TIMESTAMP_FORMAT),
        "end": end.strftime(UTC_TIMESTAMP_FORMAT),
        "limit": str(PAGE_LIMIT),
        "feed": FEED,
        "adjustment": ADJUSTMENT,
    }
    if page_token:
        query["page_token"] = page_token
    return f"{DATA_BASE_URL}{BARS_PATH.format(symbol=encoded_symbol)}?{urllib.parse.urlencode(query)}"


def _parse_bar_time(symbol: str, value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AlpacaResponseError(f"{symbol}: bar timestamp must be UTC text ending in Z")
    text = value[:-1]
    if "." in text:
        text = text.split(".", 1)[0]
    try:
        parsed = datetime.fromisoformat(text + "+00:00")
    except ValueError as exc:
        raise AlpacaResponseError(f"{symbol}: bar timestamp is not ISO-8601") from exc
    return parsed.astimezone(timezone.utc)
