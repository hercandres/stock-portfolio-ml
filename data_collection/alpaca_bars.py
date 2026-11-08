"""Alpaca historical 1-minute bars.

Stage 2 calls ``GET /v2/stocks/{symbol}/bars`` only. It does not open a
WebSocket and it does not request trades or quotes.

The free Basic plan's real-time equity coverage is IEX, not the
consolidated SIP feed. This client always sends ``feed=iex``. Historical
minute data on that plan goes back to 2016, with the latest 15 minutes
restricted, so each request stops 16 minutes before ``now`` and asks for
the 365 days before that. See Alpaca's market-data docs:
https://docs.alpaca.markets/us/docs/about-market-data-api
https://docs.alpaca.markets/us/docs/historical-stock-data-1
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from database.connection import UTC_TIMESTAMP_FORMAT

DATA_BASE_URL = "https://data.alpaca.markets"
BARS_PATH = "/v2/stocks/{symbol}/bars"
FEED = "iex"
TIMEFRAME = "1Min"
STORED_INTERVAL = "1m"
ADJUSTMENT = "split"
PAGE_LIMIT = 10000
MAX_PAGES = 80
MIN_REQUEST_INTERVAL_SECONDS = 0.35
# Basic-plan historical calls cannot use the latest 15 minutes. The bar we
# request must also have finished, so the newest bar starts 16 minutes ago.
RECENT_MINUTES_EXCLUDED = 16
HISTORY_DAYS = 365
NEW_YORK = ZoneInfo("America/New_York")
REGULAR_OPEN_MINUTES = 9 * 60 + 30
REGULAR_CLOSE_MINUTES = 16 * 60

Transport = Callable[[str, dict[str, str]], tuple[int, dict[str, str], str]]
Pause = Callable[[float], None]


class AlpacaRequestError(RuntimeError):
    """A historical bars request failed. The message omits credentials."""

    def __init__(self, symbol: str, status: int, message: str) -> None:
        self.symbol = symbol
        self.status = status
        super().__init__(f"{symbol}: Alpaca HTTP {status}: {message}")


class AlpacaResponseError(RuntimeError):
    """The bars payload could not be stored as completed OHLCV rows."""


@dataclass(frozen=True)
class MinuteBar:
    """One completed regular-session OHLCV bar. Trade count and VWAP are not kept."""

    ticker: str
    started_at: datetime
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int

    def as_row(self) -> tuple[str, str, float, float, float, float, int, str]:
        return (
            self.ticker,
            self.timestamp,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            STORED_INTERVAL,
        )


@dataclass(frozen=True)
class BarFetch:
    bars: tuple[MinuteBar, ...]
    raw_bar_count: int


def history_window(now: datetime) -> tuple[datetime, datetime]:
    """Return the inclusive UTC start and end sent to Alpaca."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    end = _floor_minute(now.astimezone(timezone.utc) - timedelta(minutes=RECENT_MINUTES_EXCLUDED))
    start = end - timedelta(days=HISTORY_DAYS)
    return start, end


def is_regular_us_session(started_at: datetime) -> bool:
    """True for a weekday bar that opens at or after 09:30 and before 16:00 ET."""
    local = started_at.astimezone(NEW_YORK)
    if local.weekday() >= 5:
        return False
    minute_of_day = local.hour * 60 + local.minute
    return REGULAR_OPEN_MINUTES <= minute_of_day < REGULAR_CLOSE_MINUTES


def bar_is_storable(started_at: datetime, now: datetime) -> bool:
    """Keep a bar only when it is complete, outside the recent window, and in session."""
    if started_at.tzinfo is None or now.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    started = started_at.astimezone(timezone.utc).replace(microsecond=0)
    current = now.astimezone(timezone.utc)
    if started.second != 0:
        return False
    if started + timedelta(minutes=1) > current:
        return False
    _start, end = history_window(current)
    if started > end:
        return False
    return is_regular_us_session(started)


class AlpacaBarsClient:
    """Download historical minute bars for one symbol at a time."""

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
            raise ValueError("Stage 2 requests the IEX feed only")
        if not api_key_id.strip() or not api_secret_key.strip():
            raise ValueError("Alpaca key id and secret are required")
        self._api_key_id = api_key_id
        self._api_secret_key = api_secret_key
        self._transport = transport or urllib_transport
        self._pause = pause or time.sleep
        self._next_request_at = 0.0

    def historical_minute_bars(self, symbol: str, now: datetime) -> BarFetch:
        start, end = history_window(now)
        page_token: str | None = None
        stored: list[MinuteBar] = []
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
                parsed = parse_bar(symbol, item)
                if bar_is_storable(parsed.started_at, now):
                    stored.append(parsed)
            token = payload.get("next_page_token")
            if not token:
                return BarFetch(bars=tuple(stored), raw_bar_count=raw_bar_count)
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


def parse_bar(symbol: str, item: object) -> MinuteBar:
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
    return MinuteBar(
        ticker=symbol,
        started_at=started_at,
        timestamp=started_at.strftime(UTC_TIMESTAMP_FORMAT),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def urllib_transport(url: str, headers: dict[str, str]) -> tuple[int, dict[str, str], str]:
    """GET one Alpaca URL. HTTP errors come back as status codes, not traces of the secret."""
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
            return response.status, dict(response.headers.items()), body
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        return error.code, dict(error.headers.items()), body


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


def _floor_minute(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(second=0, microsecond=0)


def _parse_bar_time(symbol: str, value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AlpacaResponseError(f"{symbol}: bar timestamp must be UTC text ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AlpacaResponseError(f"{symbol}: bar timestamp is not ISO-8601") from exc
    if parsed.second != 0 or parsed.microsecond != 0:
        raise AlpacaResponseError(f"{symbol}: minute bar timestamp is not aligned to a minute")
    return parsed.astimezone(timezone.utc)


def _price(symbol: str, value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlpacaResponseError(f"{symbol}: {field} is missing")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise AlpacaResponseError(f"{symbol}: {field} is not finite")
    return number


def _volume(symbol: str, value: object) -> int:
    if isinstance(value, bool) or value is None:
        raise AlpacaResponseError(f"{symbol}: volume is missing")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    else:
        raise AlpacaResponseError(f"{symbol}: volume is not a whole number")
    if number < 0:
        raise AlpacaResponseError(f"{symbol}: volume is negative")
    return number


def _short_message(body: str) -> str:
    text = body.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key in ("message", "error"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
    text = " ".join(text.split())
    if len(text) > 240:
        return text[:240] + "..."
    return text or "empty response"
