"""Foreground IEX stream. It runs only while this process is alive.

Opening the future app starts a session: incremental catch-up, then the
free Alpaca IEX WebSocket until ``stop`` or process exit. There is no
Windows service, startup task, or scheduler. Closing the process closes
the socket. Stage 4 will put a button on this start/stop pair.

The stream subscribes to minute bars and quotes for the monitored symbols.
Only a completed minute bar (``T`` of ``b``) is written. Quotes stay in
memory for the screen. Trades are not subscribed and are not stored.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from database.connection import get_connection, insert_completed_minute_bars
from database.initialize import DEFAULT_DB_PATH

from .alpaca_bars import (
    AlpacaResponseError,
    is_contiguous_regular_bar,
    live_bar_is_storable,
    parse_bar,
)
from .credentials import REQUIRED_ALPACA_KEYS, resolve_alpaca_keys
from .incremental import run_incremental, stored_timestamp
from .sync_history import DEFAULT_ENV_PATH, record_sync_status
from .universe import MONITORED_STOCKS

logger = logging.getLogger("data_collection.live_session")

IEX_STREAM_URL = "wss://stream.data.alpaca.markets/v2/iex"
FREE_STREAM_SYMBOL_LIMIT = 30
# Authentication and entitlement failures should stop, not reconnect in a loop.
AUTH_STOP_CODES = frozenset({401, 402, 403, 404, 406, 409, 410})
CatchUp = Callable[[bool], None]
Clock = Callable[[], datetime]
Backoff = Callable[[int], float]
SocketFactory = Callable[[], "StreamSocket"]


class StreamTimeout(TimeoutError):
    """The socket was quiet. The session can catch up, then keep reading."""


class StreamAuthError(RuntimeError):
    """Alpaca rejected the stream. The message does not include the secret."""

    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(f"Alpaca stream authentication failed ({code})")


class StreamSocket(Protocol):
    """One WebSocket connection. Tests pass a fake."""

    def send(self, data: str) -> None:
        """Send one text frame."""
        ...

    def recv(self) -> str:
        """Return one text frame, or raise when the connection drops."""
        ...

    def close(self) -> None:
        """Close the connection."""
        ...


@dataclass(frozen=True)
class QuoteSnapshot:
    """Latest quote kept in memory. It is never written to SQLite."""

    symbol: str
    bid: float | None
    ask: float | None
    timestamp: str


def auth_message(key_id: str, secret: str) -> dict[str, str]:
    return {"action": "auth", "key": key_id, "secret": secret}


def subscription_message(symbols: tuple[str, ...]) -> dict[str, object]:
    """Bars and quotes only. Trades are intentionally absent."""
    names = list(symbols)
    return {"action": "subscribe", "bars": names, "quotes": names}


def open_iex_socket(url: str = IEX_STREAM_URL) -> StreamSocket:
    """Open the free IEX stream. Any other URL is refused before connecting."""
    if url != IEX_STREAM_URL:
        raise ValueError("live session uses the free IEX stream only")
    import websocket

    return _WebsocketConnection(websocket.create_connection(url, timeout=60))


class _WebsocketConnection:
    def __init__(self, socket: object) -> None:
        self._socket = socket

    def send(self, data: str) -> None:
        self._socket.send(data)  # type: ignore[attr-defined]

    def recv(self) -> str:
        import websocket

        try:
            message = self._socket.recv()  # type: ignore[attr-defined]
        except websocket.WebSocketTimeoutException as exc:
            raise StreamTimeout("IEX stream was quiet") from exc
        if message is None or message == "":
            raise ConnectionError("stream closed")
        if isinstance(message, bytes):
            return message.decode("utf-8")
        return str(message)

    def close(self) -> None:
        self._socket.close()  # type: ignore[attr-defined]


class MarketDataSession:
    """Catch up missing minutes, then stream until ``stop``."""

    def __init__(
        self,
        db_path: str | Path,
        env_path: str | Path,
        *,
        symbols: tuple[str, ...] | None = None,
        socket_factory: SocketFactory | None = None,
        catch_up: CatchUp | None = None,
        clock: Clock | None = None,
        backoff: Backoff | None = None,
        api_key_id: str | None = None,
        api_secret_key: str | None = None,
        environ: dict[str, str] | None = None,
        catch_up_interval: float = 300.0,
        join_timeout: float = 5.0,
    ) -> None:
        names = tuple(symbol for symbol, _company in MONITORED_STOCKS) if symbols is None else symbols
        if not names:
            raise ValueError("live session needs at least one symbol")
        if len(names) > FREE_STREAM_SYMBOL_LIMIT:
            raise ValueError("free IEX stream allows 30 symbols")
        self.db_path = Path(db_path)
        self.env_path = Path(env_path)
        self.symbols = names
        self.latest_quotes: dict[str, QuoteSnapshot] = {}
        self._symbol_set = set(names)
        self._socket_factory = socket_factory or open_iex_socket
        self._catch_up_override = catch_up
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._backoff = backoff or _default_backoff
        self._api_key_id = api_key_id
        self._api_secret_key = api_secret_key
        self._environ = environ
        self._catch_up_interval = catch_up_interval
        self._join_timeout = join_timeout
        self._stop = threading.Event()
        self._quote_lock = threading.Lock()
        self._socket_lock = threading.Lock()
        self._socket: StreamSocket | None = None
        self._thread: threading.Thread | None = None
        self._last_catch_up = 0.0

    def quote_for(self, symbol: str) -> QuoteSnapshot | None:
        with self._quote_lock:
            return self.latest_quotes.get(symbol)

    def start(self) -> None:
        """Run ``run`` on a daemon thread. The thread dies with this process."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("market data session is already running")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run,
            name="market-data-session",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Close the socket and wait until the session thread is gone."""
        self._stop.set()
        self._close_socket()
        thread = self._thread
        if thread is None or thread is threading.current_thread() or not thread.is_alive():
            return
        thread.join(self._join_timeout)
        if thread.is_alive():
            raise RuntimeError("market data session did not stop")

    def run(self) -> None:
        """Foreground loop. Returns when the session is stopped or auth fails."""
        try:
            self._loop()
        except KeyboardInterrupt:
            self._stop.set()
        finally:
            self._close_socket()

    def _loop(self) -> None:
        self._catch_up(skip_when_closed=False)
        if self._stream_keys() is None:
            logger.error("Alpaca API keys are missing. Live session stopped before opening a socket.")
            return
        failures = 0
        while not self._stop.is_set():
            try:
                self._stream_until_drop()
                failures = 0
            except StreamAuthError as exc:
                logger.error("%s. Session stopped.", exc)
                break
            except Exception:
                if self._stop.is_set():
                    break
                failures += 1
                logger.warning("IEX stream dropped; reconnecting")
                self._catch_up(skip_when_closed=False)
                if self._stop.wait(self._backoff(failures)):
                    break

    def _catch_up(self, skip_when_closed: bool) -> None:
        try:
            if self._catch_up_override is not None:
                self._catch_up_override(skip_when_closed)
            else:
                pairs = tuple((symbol, symbol) for symbol in self.symbols)
                run_incremental(
                    self.db_path,
                    self.env_path,
                    environ=self._environ,
                    now=self._clock(),
                    symbols=pairs if self.symbols != tuple(s for s, _n in MONITORED_STOCKS) else None,
                    skip_when_closed=skip_when_closed,
                )
        except Exception:
            logger.warning("incremental sync failed; the session keeps running")
        self._last_catch_up = time.monotonic()

    def _stream_keys(self) -> tuple[str, str] | None:
        if self._api_key_id and self._api_secret_key:
            return self._api_key_id, self._api_secret_key
        env = self._environ
        if env is None:
            import os

            env = dict(os.environ)
        resolved, missing = resolve_alpaca_keys(self.env_path, env)
        if missing or resolved is None:
            return None
        return resolved[REQUIRED_ALPACA_KEYS[0]], resolved[REQUIRED_ALPACA_KEYS[1]]

    def _stream_until_drop(self) -> None:
        socket = self._socket_factory()
        with self._socket_lock:
            self._socket = socket
        try:
            self._wait_for(socket, "connected")
            key_id, secret = self._stream_keys() or ("", "")
            self._send(socket, auth_message(key_id, secret))
            self._wait_for(socket, "authenticated")
            self._send(socket, subscription_message(self.symbols))
            while not self._stop.is_set():
                try:
                    messages = self._read_messages(socket)
                except StreamTimeout:
                    self._catch_up(skip_when_closed=True)
                    continue
                self._handle_messages(messages)
                if time.monotonic() - self._last_catch_up >= self._catch_up_interval:
                    self._catch_up(skip_when_closed=True)
        finally:
            self._close_socket()

    def _wait_for(self, socket: StreamSocket, fragment: str) -> None:
        for _ in range(5):
            messages = self._read_messages(socket)
            self._raise_for_control(messages)
            if any(
                item.get("T") == "success" and fragment in str(item.get("msg", ""))
                for item in messages
            ):
                return
        raise ConnectionError(f"stream did not confirm {fragment}")

    def _read_messages(self, socket: StreamSocket) -> list[dict[str, object]]:
        raw = socket.recv()
        if raw is None or raw == "":
            raise ConnectionError("stream closed")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConnectionError("stream payload was not JSON") from exc
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            raise ConnectionError("stream payload was not a list")
        return [item for item in payload if isinstance(item, dict)]

    def _raise_for_control(self, messages: list[dict[str, object]]) -> None:
        for item in messages:
            if item.get("T") != "error":
                continue
            code = _error_code(item.get("code"))
            if code in AUTH_STOP_CODES:
                raise StreamAuthError(code)
            raise ConnectionError(f"stream error {code}")

    def _handle_messages(self, messages: list[dict[str, object]]) -> None:
        self._raise_for_control(messages)
        for item in messages:
            kind = item.get("T")
            if kind == "b":
                self._store_completed_bar(item)
            elif kind == "q":
                self._remember_quote(item)

    def _store_completed_bar(self, item: dict[str, object]) -> None:
        symbol = item.get("S")
        if not isinstance(symbol, str) or symbol not in self._symbol_set:
            return
        try:
            bar = parse_bar(symbol, item)
        except AlpacaResponseError:
            logger.warning("ignored an unusable bar message for %s", symbol)
            return
        if not live_bar_is_storable(bar.started_at, self._clock()):
            return
        try:
            with get_connection(self.db_path) as connection:
                # A new minute also drops this ticker's minute from one year ago.
                insert_completed_minute_bars(connection, [bar.as_row()])
        except Exception:
            logger.warning("could not store a completed bar for %s at %s", symbol, bar.timestamp)
            return
        self._advance_cursor(symbol, bar.timestamp, bar.started_at)

    def _advance_cursor(self, symbol: str, timestamp: str, started_at: datetime) -> None:
        with get_connection(self.db_path) as connection:
            row = connection.execute(
                "SELECT latest_success_timestamp FROM sync_status WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        cursor_text = None if row is None else row["latest_success_timestamp"]
        cursor = stored_timestamp(cursor_text) if cursor_text else None
        if not is_contiguous_regular_bar(cursor, started_at):
            return
        record_sync_status(self.db_path, symbol, "success", timestamp, None, self._clock())

    def _remember_quote(self, item: dict[str, object]) -> None:
        symbol = item.get("S")
        if not isinstance(symbol, str) or symbol not in self._symbol_set:
            return
        timestamp = item.get("t")
        snapshot = QuoteSnapshot(
            symbol=symbol,
            bid=_optional_price(item.get("bp")),
            ask=_optional_price(item.get("ap")),
            timestamp=timestamp if isinstance(timestamp, str) else "",
        )
        with self._quote_lock:
            self.latest_quotes[symbol] = snapshot

    def _send(self, socket: StreamSocket, payload: dict[str, object]) -> None:
        socket.send(json.dumps(payload))

    def _close_socket(self) -> None:
        with self._socket_lock:
            socket = self._socket
            self._socket = None
        if socket is None:
            return
        try:
            socket.close()
        except Exception:
            logger.warning("closing the IEX stream failed")


def _default_backoff(attempt: int) -> float:
    return min(30.0, float(2 ** min(attempt, 5)))


def _error_code(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _optional_price(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    session = MarketDataSession(DEFAULT_DB_PATH, DEFAULT_ENV_PATH)
    print(
        "Market data session is running in this process. "
        "Press Ctrl+C or close this window to stop collection."
    )
    try:
        session.run()
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
    print("Market data session stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
