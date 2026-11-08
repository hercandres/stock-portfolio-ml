"""Random forest on a tiny synthetic frame.

These rows are not market data. Nothing here reads or writes
database/stock_market.db.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from database.connection import get_connection, transaction
from database.initialize import initialize_database
from machine_learning.supervised.random_forest.features import (
    COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_and_save,
    collapse_to_days,
    engineer,
)
from machine_learning.supervised.random_forest.train import (
    FEATURES,
    chronological_split,
    fit_horizons,
    per_stock_scores,
    recommendation,
    replace_model_rows,
    score_held_out,
    score_spread,
    train_and_save,
)

NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]


def _prices(days: int, ticker: str = "TEST1", start: float = 50.0) -> pd.DataFrame:
    dates = pd.bdate_range("2025-06-02", periods=days)
    close = start + np.linspace(0, 8, days) + np.sin(np.linspace(0, 10, days))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates.strftime("%Y-%m-%d"),
            "open": close - 0.4,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": 1000 + np.arange(days),
        }
    )


def test_minutes_collapse_to_one_row_per_day() -> None:
    minutes = pd.DataFrame(
        {
            "ticker": ["TEST1", "TEST1", "TEST1"],
            "timestamp": [
                "2026-11-04T14:30:00Z",
                "2026-11-04T14:31:00Z",
                "2026-11-05T14:30:00Z",
            ],
            "open": [10.0, 11.0, 20.0],
            "high": [10.5, 11.5, 20.5],
            "low": [9.5, 10.5, 19.5],
            "close": [10.2, 11.2, 20.2],
            "volume": [1, 2, 4],
        }
    )
    daily = collapse_to_days(minutes)
    assert list(daily["date"]) == ["2026-11-04", "2026-11-05"]
    first = daily.iloc[0]
    assert first["open"] == 10.0
    assert first["close"] == 11.2
    assert first["high"] == 11.5
    assert first["low"] == 9.5
    assert first["volume"] == 3


def test_engineer_drops_incomplete_features_and_keeps_open_targets() -> None:
    daily = _prices(120)
    built = engineer(daily)
    assert list(built.columns) == COLUMNS
    assert built[FEATURE_COLUMNS].notna().all().all()
    assert built["target_20d"].isna().any()
    assert built["date"].iloc[-1] == daily["date"].iloc[-1]
    # A missing volume removes that day. It does not invent a fill.
    punched = daily.copy()
    punched.loc[punched.index[-1], "volume"] = np.nan
    without = engineer(punched)
    assert daily["date"].iloc[-1] not in set(without["date"])


def test_forest_inputs_are_the_engineered_columns_only() -> None:
    assert FEATURES == FEATURE_COLUMNS
    for banned in ("ticker", "date", *TARGET_COLUMNS):
        assert banned not in FEATURES
    source = (ROOT / "machine_learning" / "supervised" / "random_forest" / "train.py").read_text(
        encoding="utf-8"
    )
    assert "StandardScaler" not in source
    assert "xgboost" not in source.lower()
    assert "tensorflow" not in source.lower()
    assert "train_test_split" not in source
    assert "shuffle=True" not in source


def test_split_is_chronological_per_ticker() -> None:
    frames = [_prices(120, "TEST1", 50), _prices(120, "TEST2", 80)]
    built = engineer(pd.concat(frames, ignore_index=True))
    train, test, cutoff = chronological_split(built, "target_5d")
    assert train["date"].max() < cutoff
    assert test["date"].min() >= cutoff
    assert set(train["date"]).isdisjoint(set(test["date"]))
    for ticker, group in built.groupby("ticker"):
        labeled = group.dropna(subset=["target_5d"])
        assert labeled.loc[labeled["date"] < cutoff, "date"].max() < cutoff
        assert labeled.loc[labeled["date"] >= cutoff, "date"].min() >= cutoff
    results = fit_horizons(built)
    assert [item["horizon"] for item in results] == ["1", "5", "20"]
    for item in results:
        assert item["points"]["date"].min() >= item["cutoff"]
        assert set(item["points"]["ticker"]) <= {"TEST1", "TEST2"}


def test_recommendation_band() -> None:
    assert recommendation(0.16) == "up"
    assert recommendation(-0.16) == "down"
    assert recommendation(0.15) == "flat"
    assert recommendation(0.0) == "flat"


def test_comparison_replaces_only_random_forest_rows(tmp_path: Path) -> None:
    path = tmp_path / "model_comparison.csv"
    pd.DataFrame(
        [
            {
                "model": "other_model",
                "horizon_days": 1,
                "mae": 9.0,
                "rmse": 9.0,
                "directional_accuracy": 0.1,
                "cutoff": "2026-01-01",
            }
        ]
    ).to_csv(path, index=False)
    first = [
        {
            "model": "random_forest",
            "horizon_days": horizon,
            "mae": 1.0,
            "rmse": 1.5,
            "directional_accuracy": 0.5,
            "cutoff": "2026-04-01",
        }
        for horizon in (1, 5, 20)
    ]
    replace_model_rows(path, first)
    replace_model_rows(
        path,
        [
            {
                "model": "random_forest",
                "horizon_days": horizon,
                "mae": 0.4,
                "rmse": 0.6,
                "directional_accuracy": 0.7,
                "cutoff": "2026-05-01",
            }
            for horizon in (1, 5, 20)
        ],
    )
    saved = pd.read_csv(path)
    other = saved[saved["model"] == "other_model"]
    forest = saved[saved["model"] == "random_forest"]
    assert len(other) == 1
    assert float(other.iloc[0]["mae"]) == 9.0
    assert len(forest) == 3
    assert set(forest["horizon_days"]) == {1, 5, 20}
    assert set(forest["mae"]) == {0.4}


def _seed_minutes(db_path: Path) -> None:
    initialize_database(db_path)
    dates = pd.bdate_range("2025-06-02", periods=120)
    rows = []
    for offset, ticker in ((0, "TEST1"), (15, "TEST2")):
        for index, day in enumerate(dates):
            close = 40 + offset + index * 0.15 + (index % 5)
            stamp = datetime(day.year, day.month, day.day, 15, 0, tzinfo=NY).astimezone(
                timezone.utc
            )
            rows.append(
                (
                    ticker,
                    stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    close - 0.4,
                    close + 0.8,
                    close - 0.8,
                    close,
                    1000 + index,
                )
            )
    with get_connection(db_path) as connection:
        with transaction(connection):
            for ticker in ("TEST1", "TEST2"):
                connection.execute(
                    """
                    INSERT INTO stocks (symbol, company_name, is_monitored, created_at)
                    VALUES (?, 'Synthetic Fixture', 1, '2025-06-02T14:00:00Z')
                    """,
                    (ticker,),
                )
            connection.executemany(
                """
                INSERT INTO historical_prices (
                    ticker, timestamp, open, high, low, close, volume, interval
                ) VALUES (?, ?, ?, ?, ?, ?, ?, '1m')
                """,
                rows,
            )


def test_train_writes_files_and_database_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_forest.db"
    _seed_minutes(db_path)
    export = tmp_path / "random_forest_features.csv"
    output = tmp_path / "outputs"
    build_and_save(db_path, export)
    results = train_and_save(db_path, export, output)
    assert len(results) == 3
    metrics = pd.read_csv(output / "metrics.csv")
    comparison = pd.read_csv(output / "model_comparison.csv")
    predictions = pd.read_csv(output / "predictions.csv")
    assert list(metrics["horizon"].astype(str)) == ["1", "5", "20"]
    assert list(comparison["model"]) == ["random_forest", "random_forest", "random_forest"]
    assert list(comparison["horizon_days"]) == [1, 5, 20]
    assert set(predictions["recommendation"]) <= {"up", "down", "flat"}
    assert set(predictions.columns) == {
        "ticker",
        "horizon",
        "cutoff",
        "predicted percent",
        "recommendation",
    }
    with get_connection(db_path) as connection:
        runs = connection.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0]
        stored = connection.execute("SELECT COUNT(*) FROM ml_predictions").fetchone()[0]
        cutoff = connection.execute(
            "SELECT cutoff_timestamp FROM model_runs LIMIT 1"
        ).fetchone()[0]
    assert runs == 3
    assert stored == 6
    assert cutoff.endswith("T21:00:00Z")


def test_held_out_scores_follow_the_selected_stock() -> None:
    rows = pd.DataFrame(
        {
            "ticker": ["TEST1", "TEST1", "TEST2", "TEST2"],
            "horizon": ["1", "1", "1", "5"],
            "predicted_percent": [1.0, 1.0, 5.0, 1.0],
            "actual_percent": [1.0, 3.0, 0.0, 1.0],
        }
    )
    one_day = score_held_out(rows[(rows["ticker"] == "TEST1") & (rows["horizon"] == "1")])
    other_stock = score_held_out(rows[(rows["ticker"] == "TEST2") & (rows["horizon"] == "1")])
    other_horizon = score_held_out(rows[(rows["ticker"] == "TEST2") & (rows["horizon"] == "5")])
    assert one_day is not None and other_stock is not None and other_horizon is not None
    assert one_day["mae"] == 1.0
    assert other_stock["mae"] == 5.0
    assert other_horizon["mae"] == 0.0
    assert one_day["directional_accuracy"] == 1.0
    assert other_stock["directional_accuracy"] == 0.0


def test_all_scores_pool_every_row_and_measure_stock_spread() -> None:
    rows = pd.DataFrame(
        {
            "ticker": ["TEST1", "TEST1", "TEST2", "TEST2"],
            "predicted_percent": [1.0, 1.0, 5.0, 5.0],
            "actual_percent": [1.0, 3.0, 0.0, 0.0],
        }
    )
    pooled = score_held_out(rows)
    per_stock = per_stock_scores(rows)
    spread = score_spread(per_stock)
    assert pooled is not None and spread is not None
    assert pooled["mae"] == 3.0
    assert list(per_stock["ticker"]) == ["TEST1", "TEST2"]
    assert spread["mae"] == float(np.std([1.0, 5.0], ddof=1))
    assert score_spread(per_stock.iloc[:1]) is None


def test_dashboard_explains_the_forest_and_does_not_rank_missing_models() -> None:
    text = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert 'st.button("RUN RANDOM FOREST")' in text
    assert "This is a model estimate, not a trade." in text
    assert '["All", *symbols]' in text
    assert 'st.subheader(f"This stock: {symbol}")' in text
    assert 'st.subheader("All stocks, this horizon")' in text
    assert 'st.subheader("Comparison file")' in text
    all_branch, single_branch = text.split('if symbol == "All":', 1)[1].split("else:", 1)
    assert "_forest_normal_chart" in all_branch
    assert "_forest_normal_chart" not in single_branch.split("def ", 1)[0]
    assert "score_held_out" in text
    assert "Lower MAE and RMSE are closer prices" in text
    assert "rank models from this file once the others exist." in text
    assert "random_forest" in text
    # Absent models are not listed, scored, or ranked on this page.
    for name in ("XGBoost", "LSTM", "K-Means", "Isolation Forest"):
        assert name not in text
