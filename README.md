# Stock Portfolio ML

Stage 1 only. This repository is a local SQLite database and the tests for it. Stages 2–7 are not started. There is no market API, no Alpaca client, no live or historical download, no Streamlit dashboard, and no machine learning code.

The database file stays on this PC. It is not committed. Secrets stay out of git. `.env.example` has empty placeholders only, and Stage 1 does not read them.

Private repository: https://github.com/hercandres/stock-portfolio-ml

## Storage design note

A later collection stage is expected to hold about 25 symbols and 365 days of 1-minute bars, with a storage cap around 5 GB. Stage 1 does not collect that data. It creates the tables and a retention function that can delete `historical_prices` rows older than a cutoff.

## Layout

- `database/` — schema, single-writer connection helper, and initialization
- `data_collection/` — empty placeholder for a later stage
- `machine_learning/` — empty placeholders for a later stage
- `app/` — empty placeholders for a later stage
- `tests/` — pytest with synthetic rows only

## Windows commands

PowerShell:

```powershell
cd $env:USERPROFILE\Desktop\Portfolio_ML
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m database.initialize
.\.venv\Scripts\python.exe -m pytest
```

Command Prompt:

```bat
cd /d %USERPROFILE%\Desktop\Portfolio_ML
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m database.initialize
.venv\Scripts\python.exe -m pytest
```

`python -m database.initialize` creates `database\stock_market.db` and applies `database\schema.sql`. Running it again creates any missing tables and keeps existing rows.

These commands were run on this PC with Python 3.14. They call the virtual environment's Python directly, so PowerShell execution policy does not have to allow `Activate.ps1`.

## Database

- File: `database\stock_market.db`
- Journal mode: WAL
- Foreign keys: on for every connection opened by `database.connection`
- Timestamps: UTC ISO-8601 text, `YYYY-MM-DDTHH:MM:SSZ`
- Writes use `get_connection`, a helper for one writer in this process
- Batch price inserts use one transaction and `executemany`
- `backup_database` copies the file with the SQLite backup API to a path the caller chooses
- `database_file_size` returns the main database file size in bytes
- `delete_historical_prices_before` deletes `historical_prices` strictly older than a cutoff and leaves rows at or after that cutoff in place

`stocks` is the watchlist. A new stock defaults to the regular US session (`America/New_York`, 09:30–16:00) with extended hours off. `portfolio_holdings` is separate from that watchlist.

Git ignores `database\*.db`, `database\*.db-wal`, `database\*.db-shm`, `.env`, `.venv\`, `__pycache__\`, and `.pytest_cache\`.

## Tests

Pytest uses synthetic fixtures only. Tickers look like `TEST1`. Those rows are not market data.

## Not installed in Stage 1

`requirements.txt` installs pytest only. Pandas, NumPy, scikit-learn, XGBoost, TensorFlow, Streamlit, and Plotly are named in a comment for later stages and are not installed.
