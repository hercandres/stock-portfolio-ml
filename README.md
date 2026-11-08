# Stock Portfolio ML

Stage 4 opens a Streamlit dashboard and, while that Start Portfolio window is open, runs the IEX catch-up and live updates. Machine learning is not started. Wait for approval before any model stage.

The database file stays on this PC. It is not committed. Secrets stay out of git. `.env.example` lists Alpaca variable names only.

Private repository: https://github.com/hercandres/stock-portfolio-ml

## Monitored stocks

The editable list is `data_collection\universe.py`. It is the Finhacker ranking of the largest U.S. companies by market cap, data updated November 6, 2026, 4:00 PM ET:

https://www.finhacker.cz/en/largest-us-companies-by-market-cap/

Seeding sets `is_monitored` to 1 and the regular US session (`America/New_York`, 09:30–16:00) with extended hours off. `portfolio_holdings` stays separate from that watchlist.

## Storage

SQLite only, in `database\stock_market.db`. `historical_prices` keeps one row per completed OHLCV bar. The interval stored for these downloads is `1m`. There is no CSV store and no CSV export.

The collection target is about 25 symbols, 365 days of 1-minute bars, and a cap around 5 GB. Retention deletes `historical_prices` rows older than 365 calendar days when an incremental sync or a live session runs. It logs the database file size and warns if that file exceeds 4 GB. It does not delete newer rows to chase the cap.

## Alpaca

Stage 2 calls the historical bars endpoint only:

`GET https://data.alpaca.markets/v2/stocks/{symbol}/bars`

The request uses `timeframe=1Min`, `feed=iex`, and `adjustment=split`. Free Basic access is IEX, not consolidated real-time SIP. Historical requests leave out the latest 15 minutes and stay under 200 calls per minute. They do not ask for SIP, OTC, trades, or quotes.

`python -m data_collection.sync_history` is the full 365-day download. `python -m data_collection.incremental` asks only for minutes after `sync_status.latest_success_timestamp`. If that cursor is empty but bars are already stored, the next request starts after the newest stored bar. A weekend, holiday, or closed market is success with nothing new. One ticker's failure does not stop the others.

The live session uses `wss://stream.data.alpaca.markets/v2/iex` for the 25 symbols. It subscribes to completed minute bars and quotes. Only a completed bar is written. Quotes stay in memory. Trades are not subscribed and are not stored. A dropped socket reconnects, then the incremental sync fills minutes the stream missed. Authentication failure stops the session instead of retrying.

Collection runs only inside the Start Portfolio window. There is no Windows service, startup task, or scheduler. `python -m data_collection.live_session` still blocks in the foreground if you start it by itself. The normal way to open the app is to double-click `Start Portfolio.bat`.

Docs: https://docs.alpaca.markets/us/docs/about-market-data-api

Create a key in the Alpaca dashboard (right sidebar, API Keys), then put these names in `.env`:

- `APCA_API_KEY_ID` — the key id
- `APCA_API_SECRET_KEY` — the secret shown once when the key is created

Paper and live keys use those same names. Do not commit `.env`.

## Windows commands

PowerShell:

```powershell
cd $env:USERPROFILE\Desktop\Portfolio_ML
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m database.initialize
.\.venv\Scripts\python.exe -m data_collection.seed
.\.venv\Scripts\python.exe -m data_collection.sync_history
.\.venv\Scripts\python.exe -m data_collection.incremental
.\.venv\Scripts\python.exe -m data_collection.live_session
.\.venv\Scripts\python.exe -m pytest
```

`python -m data_collection.live_session` keeps running until Ctrl+C. Do not leave it open after a test. Pytest uses mocks and does not start it.

Command Prompt:

```bat
cd /d %USERPROFILE%\Desktop\Portfolio_ML
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m database.initialize
.venv\Scripts\python.exe -m data_collection.seed
.venv\Scripts\python.exe -m data_collection.sync_history
.venv\Scripts\python.exe -m data_collection.incremental
.venv\Scripts\python.exe -m data_collection.live_session
.venv\Scripts\python.exe -m pytest
```

`python -m database.initialize` creates `database\stock_market.db` and applies `database\schema.sql`. Running it again keeps existing rows.

`python -m data_collection.seed` upserts the 25 symbols.

`python -m data_collection.sync_history` seeds, then downloads a full year only when both Alpaca keys are present. If either key is missing it prints the missing names and does not call Alpaca. A failure or an empty response for one ticker is written to `sync_status` and the other tickers continue.

`python -m data_collection.incremental` is the catch-up command. It does not repeat a full year when a ticker already has a success cursor or stored bars. It prints how many new rows were inserted, how many old rows retention deleted, and the database file size. It then exits.

`python -m data_collection.live_session` runs that catch-up and then the IEX stream in the foreground. Stop it with Ctrl+C. The process should not still be running afterward.

## Dashboard

Double-click `Start Portfolio.bat` in this folder, or the copy on the Desktop named `Start Portfolio.bat`. The window starts two processes: the Stage 3 live session (incremental catch-up, then the IEX stream) and the Streamlit dashboard. Closing that window stops both. It does not install a Windows service or a startup task, and it does not build an executable.

The dashboard has three sections:

- Portfolio: share count and purchase price for each monitored stock, saved in `portfolio_holdings`. Total value uses the latest stored close. Total invested, unrealized P/L, and allocation weights are shown there. Holdings stay separate from the watchlist.
- Stock: latest stored price, daily percent change, volume, and a chart with candlesticks, volume, moving averages, RSI, MACD, Bollinger Bands, and VWAP. Choose the stock and the time range.
- Database: monitored count, row count, latest sync timestamp, collection status, file size, and use against the 5 GB cap. The feed is labeled IEX and about 15 minutes delayed on historical catch-up.

UPDATE MARKET DATA runs only the incremental catch-up. It does not download the full year again. A closed market still shows the latest stored bars, including Friday's session.

These commands were run on this PC with Python 3.14. They call the virtual environment's Python directly.

## Database

- File: `database\stock_market.db`
- Journal mode: WAL
- Foreign keys: on for every connection opened by `database.connection`
- Timestamps: UTC ISO-8601 text, `YYYY-MM-DDTHH:MM:SSZ`
- Writes use `get_connection`, a helper for one writer in this process
- Batch price inserts use one transaction and `executemany`
- Repeat downloads skip rows that already have the same ticker, timestamp, and interval
- `backup_database` copies the file with the SQLite backup API
- `database_file_size` returns the main database file size in bytes
- `delete_historical_prices_before` deletes `historical_prices` rows strictly older than a cutoff. Incremental sync and the live session call it for a 365-day cutoff. Nothing outside the process schedules it.

Git ignores `database\*.db`, `database\*.db-wal`, `database\*.db-shm`, `.env`, `.venv\`, `__pycache__\`, and `.pytest_cache\`.

## Tests

Pytest uses synthetic fixtures and mocked responses. Tickers in those fixtures look like `TEST1`. Those rows are not market data, and the tests do not write them into `database\stock_market.db`.

## Not in this stage

No machine learning, no model training, and no predictions screen. `requirements.txt` installs pytest, tzdata, websocket-client, streamlit, and plotly. The dashboard code does not import pandas or NumPy; Streamlit's installer brings those libraries in for its own widgets. scikit-learn, XGBoost, and TensorFlow are not installed. Wait for approval before any ML stage.
