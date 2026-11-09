# Stock Portfolio ML

Stage 4 opens a Streamlit dashboard and, while that Start Portfolio window is open, runs the IEX catch-up and live updates. Machine learning is not started. Wait for approval before any model stage.

The database file stays on this PC. It is not committed. Secrets stay out of git. `.env.example` lists Alpaca variable names only.

Private repository: https://github.com/hercandres/stock-portfolio-ml

## Monitored stocks

The editable list is `data_collection\universe.py`. It is the Finhacker ranking of the largest U.S. companies by market cap, data updated November 6, 2026, 4:00 PM ET:

https://www.finhacker.cz/en/largest-us-companies-by-market-cap/

Seeding sets `is_monitored` to 1 and the regular US session (`America/New_York`, 09:30–16:00) with extended hours off. `portfolio_holdings` stays separate from that watchlist.

## Storage

SQLite only, in `database\stock_market.db`. Two price tables stay separate.

`historical_prices` keeps completed 1-minute bars for about the last year. The interval is `1m`. When a new minute is stored, that ticker's minute from 365 days earlier is deleted. Retention also deletes minute rows older than 365 calendar days when an incremental sync or a live session runs.

`daily_prices` keeps one split-adjusted IEX bar per ticker per trading day. The download asks for 2016-01-01 through the latest finished session. The free IEX feed currently begins on 2020-07-27 for names listed before that, and on the first available day otherwise (PLTR 2020-09-30, SPCX 2026-06-12). Those rows are not deleted on the 365-day rule. The random forest trains on them.

The collection target is about 25 symbols, a year of 1-minute bars, daily bars back to 2016, and a cap around 5 GB. Retention logs the database file size and warns if that file exceeds 4 GB. It does not delete newer minute rows, or any daily rows, to chase the cap.

## Alpaca

Stage 2 calls the historical bars endpoint only:

`GET https://data.alpaca.markets/v2/stocks/{symbol}/bars`

Minute requests use `timeframe=1Min`, `feed=iex`, and `adjustment=split`. Daily requests use `timeframe=1Day` with the same feed and adjustment. Free Basic access is IEX, not consolidated real-time SIP. Historical minute requests leave out the latest 15 minutes and stay under 200 calls per minute. They do not ask for SIP, OTC, trades, or quotes.

`python -m data_collection.sync_history` is the full 365-day minute download. `python -m data_collection.sync_daily` downloads daily bars into `daily_prices` and does not change minute rows. `python -m data_collection.incremental` asks only for minutes after `sync_status.latest_success_timestamp`. If that cursor is empty but bars are already stored, the next request starts after the newest stored bar. A weekend, holiday, or closed market is success with nothing new. One ticker's failure does not stop the others.

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
.\.venv\Scripts\python.exe -m data_collection.sync_daily
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
.venv\Scripts\python.exe -m data_collection.sync_daily
.venv\Scripts\python.exe -m data_collection.incremental
.venv\Scripts\python.exe -m data_collection.live_session
.venv\Scripts\python.exe -m pytest
```

`python -m database.initialize` creates `database\stock_market.db` and applies `database\schema.sql`. Running it again keeps existing rows.

`python -m data_collection.seed` upserts the 25 symbols.

`python -m data_collection.sync_history` seeds, then downloads a full year of minutes only when both Alpaca keys are present. If either key is missing it prints the missing names and does not call Alpaca. A failure or an empty response for one ticker is written to `sync_status` and the other tickers continue.

`python -m data_collection.sync_daily` downloads daily bars from 2016-01-01 into `daily_prices`. It does not write `sync_status` and does not delete minute bars.

`python -m data_collection.incremental` is the catch-up command. It does not repeat a full year when a ticker already has a success cursor or stored bars. It prints how many new rows were inserted, how many old rows retention deleted, and the database file size. It then exits.

`python -m data_collection.live_session` runs that catch-up and then the IEX stream in the foreground. Stop it with Ctrl+C. The process should not still be running afterward.

## Dashboard

Double-click `Start Portfolio.bat` in this folder, or the copy on the Desktop named `Start Portfolio.bat`. The window starts two processes: the Stage 3 live session (incremental catch-up, then the IEX stream) and the Streamlit dashboard. Closing that window stops both. It does not install a Windows service or a startup task, and it does not build an executable.

The dashboard has three sections:

- Portfolio: share count and purchase price for each monitored stock, saved in `portfolio_holdings`. Total value uses the latest stored close. Total invested, unrealized P/L, and allocation weights are shown there. Holdings stay separate from the watchlist.
- Stock: latest stored price, daily percent change, volume, and a chart with candlesticks, volume, moving averages, RSI, MACD, Bollinger Bands, and VWAP. The left button shows daily candles back to 2016. The right button shows the last year of minute candles. Drag or scroll to zoom either chart.
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
- `delete_historical_prices_before` deletes 1-minute `historical_prices` rows strictly older than a cutoff. Incremental sync and the live session call it for a 365-day cutoff. `daily_prices` is not deleted. Nothing outside the process schedules it.
- Daily dates are `YYYY-MM-DD`. Minute timestamps stay `YYYY-MM-DDTHH:MM:SSZ`.

Git ignores `database\*.db`, `database\*.db-wal`, `database\*.db-shm`, `.env`, `.venv\`, `__pycache__\`, and `.pytest_cache\`.

## Tests

Pytest uses synthetic fixtures and mocked responses. Tickers in those fixtures look like `TEST1`. Those rows are not market data, and the tests do not write them into `database\stock_market.db`.

## Random forest

This is the only model. It does not run when the dashboard opens. On the Random Forest page, **RUN RANDOM FOREST** reads `daily_prices` with pandas, builds one row per stock per trading day back to 2016, then fits three scikit-learn forests for 1, 5, and 20 trading days. Minute bars are not the training rows.

```text
python -m machine_learning.supervised.random_forest.features
python -m machine_learning.supervised.random_forest.train
```

Feature rows go to `machine_learning\exports\random_forest_features.csv`. Scores and forecasts go to `machine_learning\outputs` (`metrics.csv`, `predictions.csv`, and the shared `model_comparison.csv`). Those CSV files are gitignored. Training also inserts `model_runs` and `ml_predictions`.

`requirements.txt` installs pandas, numpy, and scikit-learn for this model. XGBoost and TensorFlow are not installed.

## Not built yet

No XGBoost, LSTM, K-Means, PCA, or Isolation Forest. The recommendation window that will rank models has not been built. `model_comparison.csv` is the file it will read.
