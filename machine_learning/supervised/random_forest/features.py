"""One row per stock per trading day for the random forest.

Minute bars are the raw history. This file rolls them up to a day because
the question we ask is a daily percent change, not a one-minute tick.
The future targets are written beside the features so training can find
them. They are not inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from database.connection import connect
from database.initialize import DEFAULT_DB_PATH

# parents[0] is random_forest, [1] supervised, [2] machine_learning, [3] project.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPORT = PROJECT_ROOT / "machine_learning" / "exports" / "random_forest_features.csv"

# These describe the day that already happened. The forest may see them.
# They are left in their own units. Nothing here is standardized.
FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "dist_sma_20_pct",
    "dist_sma_50_pct",
    "rsi_14",
    "volatility_20d",
    "volume_vs_20d",
    "range_pct",
]

# Future percent changes. Kept for training and for the chart.
# A row can lack these and still be a valid forecast row.
TARGET_COLUMNS = ["target_1d", "target_5d", "target_20d"]

COLUMNS = ["ticker", "date", *FEATURE_COLUMNS, *TARGET_COLUMNS]


def collapse_to_days(minutes: pd.DataFrame) -> pd.DataFrame:
    """One bar per ticker per New York trading day.

    The table stores every minute. Treating 9:31 and 9:32 as two lessons
    would teach the forest the same day over and over.
    """
    empty = pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])
    if minutes.empty:
        return empty
    frame = minutes.copy()
    stamps = pd.to_datetime(frame["timestamp"], utc=True)
    # The session is defined in New York. Regular-session bars fall on that date.
    frame["date"] = stamps.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    frame = frame.sort_values(["ticker", "timestamp"])
    return frame.groupby(["ticker", "date"], as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI from the average up day and the average down day.

    A simple 14-day average is enough to explain. The 0-100 number is
    not rescaled before it goes into the forest.
    """
    change = close.diff()
    gain = change.clip(lower=0)
    loss = (-change).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # A window of only up days has no loss, so the score is the top of the scale.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    return rsi


def _one_ticker(group: pd.DataFrame) -> pd.DataFrame:
    """Features and future targets for a single stock, in date order."""
    group = group.sort_values("date").copy()
    group["date"] = group["date"].astype(str).str.slice(0, 10)
    close = group["close"].replace(0, np.nan)
    # Percent points, so 1.5 means plus 1.5 percent. Same unit as the targets.
    group["return_1d"] = close.pct_change(1) * 100
    group["return_5d"] = close.pct_change(5) * 100
    group["return_20d"] = close.pct_change(20) * 100
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    group["dist_sma_20_pct"] = (close / sma_20 - 1) * 100
    group["dist_sma_50_pct"] = (close / sma_50 - 1) * 100
    group["rsi_14"] = _rsi(close)
    # Sample standard deviation of the last 20 one-day percent changes.
    group["volatility_20d"] = group["return_1d"].rolling(20).std()
    average_volume = group["volume"].rolling(20).mean().replace(0, np.nan)
    # 1.0 means today's volume matches the last 20 days. Not a z-score.
    group["volume_vs_20d"] = group["volume"] / average_volume
    group["range_pct"] = (group["high"] - group["low"]) / close * 100
    # Negative shift looks forward. These columns are answers, not inputs.
    group["target_1d"] = (close.shift(-1) / close - 1) * 100
    group["target_5d"] = (close.shift(-5) / close - 1) * 100
    group["target_20d"] = (close.shift(-20) / close - 1) * 100
    return group


def engineer(daily: pd.DataFrame) -> pd.DataFrame:
    """Build the model table from daily bars.

    A row missing any feature is dropped. The early days have no 50-day
    average yet, so they cannot be used. A row missing a future target
    stays, because that is the row we can still forecast from.
    """
    if daily.empty:
        return pd.DataFrame(columns=COLUMNS)
    pieces = [_one_ticker(group) for _, group in daily.groupby("ticker", sort=False)]
    built = pd.concat(pieces, ignore_index=True)
    built = built.replace([np.inf, -np.inf], np.nan)
    built = built.dropna(subset=FEATURE_COLUMNS)
    return built[COLUMNS]


def load_daily_bars(db_path: str | Path) -> pd.DataFrame:
    """Read 1-minute bars and collapse them. This does not take the writer lock."""
    query = """
        SELECT ticker, timestamp, open, high, low, close, volume
        FROM historical_prices
        WHERE interval = '1m'
        ORDER BY ticker, timestamp
    """
    connection = connect(db_path)
    try:
        minutes = pd.read_sql_query(query, connection)
    finally:
        connection.close()
    return collapse_to_days(minutes)


def build_feature_frame(db_path: str | Path) -> pd.DataFrame:
    """Daily feature rows for every ticker stored in this database."""
    return engineer(load_daily_bars(db_path))


def build_and_save(db_path: str | Path, export_path: str | Path | None = None) -> Path:
    """Write the feature CSV. Returns the path that was written."""
    path = Path(export_path) if export_path is not None else DEFAULT_EXPORT
    frame = build_feature_frame(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def main() -> None:
    path = build_and_save(DEFAULT_DB_PATH)
    row_count = sum(1 for _ in path.open(encoding="utf-8")) - 1
    print(f"Wrote {row_count} rows to {path}")


if __name__ == "__main__":
    main()
