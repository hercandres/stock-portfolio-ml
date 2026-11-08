# Stock Portfolio ML

Stage 2 stores a local SQLite database and can download completed historical 1-minute bars for 25 monitored stocks. The dashboard, streaming, retention scheduler, and machine learning are not started. Wait for approval before Stage 3.

The database file stays on this PC. It is not committed. Secrets stay out of git. `.env.example` lists Alpaca variable names only.

Private repository: https://github.com/hercandres/stock-portfolio-ml

## Monitored stocks

The editable list is `data_collection\universe.py`. It is the Finhacker ranking of the largest U.S. companies by market cap, data updated November 6, 2026, 4:00 PM ET:

https://www.finhacker.cz/en/largest-us-companies-by-market-cap/

Seeding sets `is_monitored` to 1 and the regular US session (`America/New_York`, 09:30–16:00) with extended hours off. `portfolio_holdings` stays separate from that watchlist.

## Storage

SQLite only, in `database\stock_market.db`. `historical_prices` keeps one row per completed OHLCV bar. The interval stored for these downloads is `1m`. There is no CSV store and no CSV export.

A later collection target is still about 25 symbols, 365 days of 1-minute bars, and a cap around 5 GB. Stage 2 requests that 365-day window from Alpaca but does not schedule retention.

## Alpaca

Stage 2 calls the historical bars endpoint only:

`GET https://data.alpaca.markets/v2/stocks/{symbol}/bars`

The request uses `timeframe=1Min`, `feed=iex`, and `adjustment=split`. Free Basic access is IEX, not consolidated real-time SIP. The client does not open a WebSocket and does not download trades or quotes. The latest 15 minutes are left out of each request. Basic historical minute data goes back to 2016, so a 365-day window is inside the published history, limited to minutes IEX actually printed.

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
.\.venv\Scripts\python.exe -m pytest
```

Command Prompt:

```bat
cd /d %USERPROFILE%\Desktop\Portfolio_ML
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m database.initialize
.venv\Scripts\python.exe -m data_collection.seed
.venv\Scripts\python.exe -m data_collection.sync_history
.venv\Scripts\python.exe -m pytest
```

`python -m database.initialize` creates `database\stock_market.db` and applies `database\schema.sql`. Running it again keeps existing rows.

`python -m data_collection.seed` upserts the 25 symbols.

`python -m data_collection.sync_history` seeds, then downloads only when both Alpaca keys are present. If either key is missing it prints the missing names and does not call Alpaca. A failure or an empty response for one ticker is written to `sync_status` and the other tickers continue.

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
- `delete_historical_prices_before` can delete old `historical_prices` rows. Nothing schedules it.

Git ignores `database\*.db`, `database\*.db-wal`, `database\*.db-shm`, `.env`, `.venv\`, `__pycache__\`, and `.pytest_cache\`.

## Tests

Pytest uses synthetic fixtures and mocked HTTP responses. Tickers in those fixtures look like `TEST1`. Those rows are not market data.

## Not in this stage

No Streamlit dashboard, no WebSocket stream, no retention scheduler, and no machine learning. `requirements.txt` installs pytest and tzdata (the New York time zone database used to keep the regular US session). Pandas, NumPy, scikit-learn, XGBoost, TensorFlow, Streamlit, and Plotly are named for later stages and are not installed.
