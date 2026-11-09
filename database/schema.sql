-- Stage 1 local schema.
-- Minute bars use UTC timestamps: YYYY-MM-DDTHH:MM:SSZ.
-- Daily bars use a calendar date: YYYY-MM-DD. They are a separate table.
-- The connection helper enables WAL and foreign keys. This script does not
-- download data or start a scheduler.

CREATE TABLE IF NOT EXISTS stocks (
    symbol TEXT PRIMARY KEY CHECK (length(symbol) > 0),
    company_name TEXT NOT NULL CHECK (length(company_name) > 0),
    is_monitored INTEGER NOT NULL DEFAULT 0 CHECK (is_monitored IN (0, 1)),
    timezone TEXT NOT NULL DEFAULT 'America/New_York',
    regular_session_open TEXT NOT NULL DEFAULT '09:30'
        CHECK (regular_session_open GLOB '[0-9][0-9]:[0-9][0-9]'),
    regular_session_close TEXT NOT NULL DEFAULT '16:00'
        CHECK (regular_session_close GLOB '[0-9][0-9]:[0-9][0-9]'),
    extended_hours_enabled INTEGER NOT NULL DEFAULT 0
        CHECK (extended_hours_enabled IN (0, 1)),
    created_at TEXT NOT NULL
        CHECK (
            created_at GLOB
            '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'
        )
);

CREATE TABLE IF NOT EXISTS historical_prices (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL CHECK (length(ticker) > 0),
    timestamp TEXT NOT NULL
        CHECK (
            timestamp GLOB
            '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'
        ),
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL CHECK (volume >= 0),
    interval TEXT NOT NULL CHECK (length(interval) > 0),
    UNIQUE (ticker, timestamp, interval),
    CHECK (high >= low AND open >= low AND open <= high AND close >= low AND close <= high),
    FOREIGN KEY (ticker) REFERENCES stocks (symbol)
);

CREATE INDEX IF NOT EXISTS idx_historical_prices_ticker_timestamp
    ON historical_prices (ticker, timestamp);

CREATE INDEX IF NOT EXISTS idx_historical_prices_interval
    ON historical_prices (interval);

-- One row per ticker per trading day. Models train on this table.
-- The 365-day delete applies to historical_prices only, not to these rows.
CREATE TABLE IF NOT EXISTS daily_prices (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL CHECK (length(ticker) > 0),
    date TEXT NOT NULL
        CHECK (date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL CHECK (volume >= 0),
    UNIQUE (ticker, date),
    CHECK (high >= low AND open >= low AND open <= high AND close >= low AND close <= high),
    FOREIGN KEY (ticker) REFERENCES stocks (symbol)
);

CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker_date
    ON daily_prices (ticker, date);

CREATE TABLE IF NOT EXISTS portfolio_holdings (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL CHECK (quantity > 0),
    purchase_price REAL NOT NULL CHECK (purchase_price >= 0),
    purchase_time TEXT NOT NULL
        CHECK (
            purchase_time GLOB
            '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'
        ),
    notes TEXT,
    created_at TEXT NOT NULL
        CHECK (
            created_at GLOB
            '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'
        ),
    FOREIGN KEY (symbol) REFERENCES stocks (symbol)
);

CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_symbol
    ON portfolio_holdings (symbol);

CREATE TABLE IF NOT EXISTS sync_status (
    symbol TEXT PRIMARY KEY,
    latest_success_timestamp TEXT
        CHECK (
            latest_success_timestamp IS NULL
            OR latest_success_timestamp GLOB
            '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'
        ),
    last_attempt_timestamp TEXT
        CHECK (
            last_attempt_timestamp IS NULL
            OR last_attempt_timestamp GLOB
            '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'
        ),
    status TEXT NOT NULL CHECK (length(status) > 0),
    error_text TEXT,
    FOREIGN KEY (symbol) REFERENCES stocks (symbol)
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY,
    model_name TEXT NOT NULL CHECK (length(model_name) > 0),
    started_at TEXT NOT NULL
        CHECK (
            started_at GLOB
            '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'
        ),
    finished_at TEXT
        CHECK (
            finished_at IS NULL
            OR finished_at GLOB
            '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'
        ),
    cutoff_timestamp TEXT NOT NULL
        CHECK (
            cutoff_timestamp GLOB
            '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'
        ),
    metrics TEXT,
    status TEXT NOT NULL CHECK (length(status) > 0)
);

CREATE TABLE IF NOT EXISTS ml_predictions (
    id INTEGER PRIMARY KEY,
    model_run_id INTEGER NOT NULL,
    ticker TEXT NOT NULL CHECK (length(ticker) > 0),
    prediction_value REAL NOT NULL,
    horizon TEXT NOT NULL CHECK (length(horizon) > 0),
    predicted_at TEXT NOT NULL
        CHECK (
            predicted_at GLOB
            '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'
        ),
    created_at TEXT NOT NULL
        CHECK (
            created_at GLOB
            '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'
        ),
    FOREIGN KEY (model_run_id) REFERENCES model_runs (id),
    FOREIGN KEY (ticker) REFERENCES stocks (symbol)
);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_ticker
    ON ml_predictions (ticker);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_model_run
    ON ml_predictions (model_run_id);
