"""XGBoost on a tiny synthetic frame.

These rows are not market data. Nothing here reads or writes
database/stock_market.db or the real feature CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from database.connection import get_connection, transaction
from database.initialize import initialize_database
from machine_learning.supervised.random_forest.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    engineer,
)
from machine_learning.supervised.xgboost.train import (
    FEATURES,
    MODEL_NAME,
    fit_horizons,
    train_and_save,
)

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


def _seed_daily(db_path: Path) -> None:
    initialize_database(db_path)
    dates = pd.bdate_range("2025-06-02", periods=120)
    rows = []
    for offset, ticker in ((0, "TEST1"), (15, "TEST2")):
        for index, day in enumerate(dates):
            close = 40 + offset + index * 0.15 + (index % 5)
            rows.append(
                (
                    ticker,
                    day.strftime("%Y-%m-%d"),
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
                INSERT INTO daily_prices (
                    ticker, date, open, high, low, close, volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )


def test_xgboost_reads_the_shared_feature_file_and_not_a_second_csv() -> None:
    source = (ROOT / "machine_learning" / "supervised" / "xgboost" / "train.py").read_text(
        encoding="utf-8"
    )
    assert "random_forest_features.csv" in source
    assert "xgboost_features.csv" not in source
    assert "XGBRegressor" in source
    assert "StandardScaler" not in source
    assert "train_test_split" not in source
    assert "shuffle=True" not in source
    assert FEATURES == FEATURE_COLUMNS
    for banned in ("ticker", "date", *TARGET_COLUMNS):
        assert banned not in FEATURES
    dashboard = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert 'PAGE_XGBOOST = "XGBoost"' in dashboard
    assert "ML_PAGES = (PAGE_FOREST, PAGE_XGBOOST)" in dashboard
    assert 'st.button("Run XGBoost")' in dashboard
    assert "_train_xgboost_once" in dashboard
    assert "xgboost_features.csv" not in dashboard
    assert 'st.sidebar.subheader("ML Methods")' in dashboard


def test_open_target_still_gets_a_forecast() -> None:
    built = engineer(pd.concat([_prices(120, "TEST1", 50), _prices(120, "TEST2", 80)], ignore_index=True))
    assert built["target_20d"].isna().any()
    results = fit_horizons(built)
    assert [item["horizon"] for item in results] == ["1", "5", "20"]
    last_day = built["date"].max()
    for item in results:
        assert item["points"]["date"].min() >= item["cutoff"]
        forecast_days = {row["cutoff"] for row in item["forecasts"]}
        assert last_day in forecast_days
        assert {row["ticker"] for row in item["forecasts"]} == {"TEST1", "TEST2"}


def test_training_keeps_random_forest_comparison_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_xgboost.db"
    _seed_daily(db_path)
    built = engineer(
        pd.concat([_prices(120, "TEST1", 40), _prices(120, "TEST2", 55)], ignore_index=True)
    )
    feature_path = tmp_path / "random_forest_features.csv"
    built.to_csv(feature_path, index=False)
    output = tmp_path / "outputs"
    forest_metrics = output / "metrics.csv"
    output.mkdir()
    forest_metrics.write_text("horizon,MAE\n1,1.0\n", encoding="utf-8")
    comparison = output / "model_comparison.csv"
    pd.DataFrame(
        [
            {
                "model": "random_forest",
                "horizon_days": horizon,
                "mae": 3.5,
                "rmse": 4.5,
                "directional_accuracy": 0.55,
                "cutoff": "2026-01-02",
            }
            for horizon in (1, 5, 20)
        ]
    ).to_csv(comparison, index=False)
    results = train_and_save(
        db_path,
        feature_path,
        output / "xgboost",
        comparison,
    )
    assert len(results) == 3
    assert MODEL_NAME == "xgboost"
    saved = pd.read_csv(comparison)
    forest = saved[saved["model"] == "random_forest"]
    booster = saved[saved["model"] == "xgboost"]
    assert len(forest) == 3
    assert set(forest["mae"]) == {3.5}
    assert set(forest["cutoff"]) == {"2026-01-02"}
    assert list(booster["horizon_days"]) == [1, 5, 20]
    assert list(saved.columns[:6]) == [
        "model",
        "horizon_days",
        "mae",
        "rmse",
        "directional_accuracy",
        "cutoff",
    ]
    assert forest_metrics.read_text(encoding="utf-8") == "horizon,MAE\n1,1.0\n"
    assert (output / "xgboost" / "metrics.csv").is_file()
    assert (output / "xgboost" / "predictions.csv").is_file()
    assert (output / "xgboost" / "backtest.csv").is_file()
    assert not (tmp_path / "xgboost_features.csv").exists()
    with get_connection(db_path) as connection:
        names = {
            row["model_name"]
            for row in connection.execute("SELECT model_name FROM model_runs")
        }
    assert names == {"xgboost"}
