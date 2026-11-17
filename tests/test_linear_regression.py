"""Linear regression on a tiny synthetic frame.

These rows are not market data. Nothing here reads or writes
database/stock_market.db or the real feature CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from database.connection import get_connection, transaction
from database.initialize import initialize_database
from machine_learning.supervised.linear_regression.train import (
    FEATURES,
    MODEL_NAME,
    fit_horizons,
    train_and_save,
)
from machine_learning.supervised.random_forest.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
)
from machine_learning.supervised.random_forest.train import chronological_split

ROOT = Path(__file__).resolve().parents[1]


def _frame() -> pd.DataFrame:
    """Three inputs each explain one horizon. The other inputs stay at zero.

    The last day has no future percent, so it can be a forecast and not an answer.
    """
    days = pd.bdate_range("2024-01-02", periods=40)
    rows = []
    for ticker in ("TEST1", "TEST2"):
        for index, day in enumerate(days):
            return_1d = float(index - 20)
            return_5d = float((index % 5) - 2)
            return_20d = float((index % 3) - 1)
            open_target = index == len(days) - 1
            row = {
                "ticker": ticker,
                "date": day.strftime("%Y-%m-%d"),
                "return_1d": return_1d,
                "return_5d": return_5d,
                "return_20d": return_20d,
                "dist_sma_20_pct": 0.0,
                "dist_sma_50_pct": 0.0,
                "rsi_14": 0.0,
                "volatility_20d": 0.0,
                "volume_vs_20d": 0.0,
                "range_pct": 0.0,
                "target_1d": np.nan if open_target else 2.0 * return_1d + 0.5,
                "target_5d": np.nan if open_target else -3.0 * return_5d,
                "target_20d": np.nan if open_target else 4.0 * return_20d + 1.0,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _seed_stocks(db_path: Path) -> None:
    initialize_database(db_path)
    with get_connection(db_path) as connection:
        with transaction(connection):
            for ticker in ("TEST1", "TEST2"):
                connection.execute(
                    """
                    INSERT INTO stocks (symbol, company_name, is_monitored, created_at)
                    VALUES (?, 'Synthetic Fixture', 1, '2024-01-02T14:00:00Z')
                    """,
                    (ticker,),
                )


def test_linear_regression_reads_the_shared_feature_file_and_does_not_scale() -> None:
    source = (
        ROOT / "machine_learning" / "supervised" / "linear_regression" / "train.py"
    ).read_text(encoding="utf-8")
    assert "random_forest_features.csv" in source
    assert source.count("features.csv") == source.count("random_forest_features.csv")
    assert "linear_regression_features.csv" not in source
    assert "LinearRegression" in source
    assert "StandardScaler" not in source
    assert "train_test_split" not in source
    assert "shuffle=True" not in source
    assert 'method="pearson"' in source
    assert FEATURES == FEATURE_COLUMNS
    assert FEATURES == [
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
    for banned in ("ticker", "date", *TARGET_COLUMNS):
        assert banned not in FEATURES
    dashboard = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert 'PAGE_LINEAR = "Linear Regression"' in dashboard
    assert "ML_PAGES = (PAGE_FOREST, PAGE_XGBOOST, PAGE_LINEAR, PAGE_KMEANS, PAGE_PCA)" in dashboard
    assert 'st.header("Linear Regression")' in dashboard
    assert 'st.button("Run Linear Regression")' in dashboard
    assert "_train_linear_once" in dashboard
    assert "linear_regression_features.csv" not in dashboard
    assert "Pearson correlation" in dashboard
    assert "Coefficient" in dashboard
    assert 'name="y = x"' in dashboard
    assert "Bin averages:" in dashboard
    assert "Mean predicted percent" in dashboard
    assert "Mean actual percent" in dashboard
    assert "_linear_scatter" not in dashboard
    page = dashboard.split("def _linear_regression_page", 1)[1].split("def _train_kmeans_once", 1)[0]
    before_scores, rest = page.split('if symbol == "All":', 1)
    assert "_linear_calibration_chart(symbol, horizon)" in before_scores
    assert "_linear_time_chart(symbol, horizon)" in before_scores
    assert before_scores.index("_linear_calibration_chart") < before_scores.index("_linear_time_chart")
    all_branch, single_branch = rest.split("else:", 1)
    assert "_forest_all_metrics" in all_branch
    assert "_forest_normal_chart" in all_branch
    assert "_forest_metrics" in single_branch
    assert "_linear_calibration_chart" not in single_branch
    assert "_linear_time_chart" not in single_branch
    render = dashboard.split("def render", 1)[1].split("def _portfolio_page", 1)[0]
    assert render.index("_train_xgboost_once") < render.index("_train_linear_once")
    assert render.index("_train_linear_once") < render.index("_train_kmeans_once")


def test_each_horizon_keeps_its_own_line_and_sorts_by_absolute_correlation() -> None:
    frame = _frame()
    results = fit_horizons(frame)
    assert [item["horizon"] for item in results] == ["1", "5", "20"]
    expected = {
        "1": ("return_1d", 2.0),
        "5": ("return_5d", -3.0),
        "20": ("return_20d", 4.0),
    }
    last_day = frame["date"].max()
    for item in results:
        feature, weight = expected[item["horizon"]]
        target = {"1": "target_1d", "5": "target_5d", "20": "target_20d"}[item["horizon"]]
        train, _, cutoff = chronological_split(frame, target)
        relationships = item["relationships"]
        assert list(relationships.columns) == [
            "feature",
            "pearson_correlation",
            "coefficient",
        ]
        assert list(relationships["feature"])[:1] == [feature]
        absolute = relationships["pearson_correlation"].abs()
        known = absolute.dropna()
        assert list(known) == sorted(known, reverse=True)
        assert float(relationships.iloc[0]["pearson_correlation"]) == (
            1.0 if weight > 0 else -1.0
        )
        saved = relationships.set_index("feature")
        assert abs(float(saved.loc[feature, "coefficient"]) - weight) < 1e-8
        fresh = float(train[feature].corr(train[target], method="pearson"))
        assert abs(float(saved.loc[feature, "pearson_correlation"]) - fresh) < 1e-12
        assert item["points"]["date"].min() >= item["cutoff"]
        assert last_day not in set(item["points"]["date"])
        assert cutoff == item["cutoff"]
        assert {row["ticker"] for row in item["forecasts"]} == {"TEST1", "TEST2"}
        assert last_day in {row["cutoff"] for row in item["forecasts"]}


def test_ticker_labels_do_not_change_the_line() -> None:
    frame = _frame()
    renamed = frame.copy()
    renamed["ticker"] = renamed["ticker"].map({"TEST1": "ZZZ", "TEST2": "QQQ"})
    original = fit_horizons(frame)
    again = fit_horizons(renamed)
    for left, right in zip(original, again, strict=True):
        assert np.allclose(
            left["relationships"]["coefficient"],
            right["relationships"]["coefficient"],
            equal_nan=True,
        )
        assert left["mae"] == right["mae"]


def test_training_replaces_only_linear_regression_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_linear.db"
    _seed_stocks(db_path)
    feature_path = tmp_path / "random_forest_features.csv"
    _frame().to_csv(feature_path, index=False)
    output = tmp_path / "outputs"
    output.mkdir()
    forest_metrics = output / "metrics.csv"
    forest_metrics.write_text("horizon,MAE\n1,1.0\n", encoding="utf-8")
    comparison = output / "model_comparison.csv"
    kept = []
    for model_name in ("random_forest", "xgboost"):
        for horizon in (1, 5, 20):
            kept.append(
                {
                    "model": model_name,
                    "horizon_days": horizon,
                    "mae": 3.5,
                    "rmse": 4.5,
                    "directional_accuracy": 0.55,
                    "cutoff": "2026-01-02",
                }
            )
    kept.append(
        {
            "model": "linear_regression",
            "horizon_days": 99,
            "mae": 9.9,
            "rmse": 9.9,
            "directional_accuracy": 0.1,
            "cutoff": "1999-01-01",
        }
    )
    pd.DataFrame(kept).to_csv(comparison, index=False)
    results = train_and_save(
        db_path,
        feature_path,
        output / "linear_regression",
        comparison,
    )
    assert len(results) == 3
    assert MODEL_NAME == "linear_regression"
    saved = pd.read_csv(comparison)
    forest = saved[saved["model"] == "random_forest"]
    booster = saved[saved["model"] == "xgboost"]
    lines = saved[saved["model"] == "linear_regression"]
    assert len(forest) == 3
    assert len(booster) == 3
    assert set(forest["mae"]) == {3.5}
    assert set(booster["cutoff"]) == {"2026-01-02"}
    assert list(lines["horizon_days"]) == [1, 5, 20]
    assert 9.9 not in set(lines["mae"])
    assert 99 not in set(lines["horizon_days"])
    assert list(saved.columns[:6]) == [
        "model",
        "horizon_days",
        "mae",
        "rmse",
        "directional_accuracy",
        "cutoff",
    ]
    assert forest_metrics.read_text(encoding="utf-8") == "horizon,MAE\n1,1.0\n"
    folder = output / "linear_regression"
    assert (folder / "metrics.csv").is_file()
    assert (folder / "predictions.csv").is_file()
    assert (folder / "backtest.csv").is_file()
    coefficients = pd.read_csv(folder / "coefficients.csv")
    assert list(coefficients.columns) == [
        "horizon",
        "feature",
        "pearson_correlation",
        "coefficient",
    ]
    one_day = coefficients[coefficients["horizon"].astype(str) == "1"]
    assert list(one_day["feature"])[0] == "return_1d"
    absolute = one_day["pearson_correlation"].abs()
    known = absolute.dropna()
    assert list(known) == sorted(known, reverse=True)
    assert not (tmp_path / "linear_regression_features.csv").exists()
    with get_connection(db_path) as connection:
        names = {
            row["model_name"]
            for row in connection.execute("SELECT model_name FROM model_runs")
        }
    assert names == {"linear_regression"}


def _dashboard_function(name: str):
    """Load one dashboard helper without starting Streamlit."""
    import ast

    text = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            namespace: dict = {}
            exec(ast.get_source_segment(text, node), namespace)
            return namespace[name]
    raise AssertionError(name)


def test_calibration_bins_are_equal_count_and_a_thin_stock_uses_everyone() -> None:
    bins_of = _dashboard_function("_linear_calibration_bins")
    choose = _dashboard_function("_linear_calibration_choice")
    time_rows = _dashboard_function("_linear_time_choice")
    predicted = np.arange(25, dtype=float)
    rows = pd.DataFrame(
        {
            "ticker": ["AAA"] * 25,
            "date": pd.bdate_range("2024-01-02", periods=25).strftime("%Y-%m-%d"),
            "predicted_percent": predicted,
            "actual_percent": predicted * 2,
        }
    )
    means = bins_of(rows)
    assert len(means) == 10
    assert int(means["rows"].sum()) == 25
    assert means["rows"].max() - means["rows"].min() <= 1
    assert list(means["mean_predicted"]) == sorted(means["mean_predicted"])
    assert np.allclose(means["mean_actual"], means["mean_predicted"] * 2)
    short = bins_of(rows.iloc[:7])
    assert len(short) == 7
    assert list(short["rows"]) == [1, 1, 1, 1, 1, 1, 1]
    thick = pd.concat([rows.assign(ticker="BBB")] * 2, ignore_index=True)
    mixed = pd.concat([rows.assign(ticker="AAA"), thick], ignore_index=True)
    chosen, note = choose("BBB", mixed)
    assert note == ""
    assert set(chosen["ticker"]) == {"BBB"}
    assert len(chosen) == 50
    thin = rows.iloc[:12].assign(ticker="BBB")
    everyone = pd.concat([rows.assign(ticker="ZZZ"), thin], ignore_index=True)
    fallback, note = choose("BBB", everyone)
    assert "fewer than 30" in note
    assert "every stock" in note
    assert len(fallback) == len(everyone)
    wide, wide_note = choose("All", everyone)
    assert wide_note == ""
    assert len(wide) == len(everyone)
    timed, ticker = time_rows("All", everyone)
    assert ticker == "BBB"
    assert set(timed["ticker"]) == {"BBB"}
    one_stock, ticker = time_rows("ZZZ", everyone)
    assert ticker == "ZZZ"
    assert len(one_stock) == 25
