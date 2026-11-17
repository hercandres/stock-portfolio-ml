"""PCA on a tiny synthetic frame.

These rows are not market data. Nothing here reads or writes
database/stock_market.db or the real feature CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from machine_learning.supervised.random_forest.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
)
from machine_learning.unsupervised.pca.train import (
    DEFAULT_OUTPUT,
    FEATURES,
    fit_pca,
    train_and_save,
)

ROOT = Path(__file__).resolve().parents[1]


def _signal_frame() -> pd.DataFrame:
    """Eight inputs copy one signal. One input does not. Targets are not inputs.

    Scaling gives every column the same spread, so the shared signal is
    most of the variance and the leftover column is not.
    """
    rng = np.random.default_rng(0)
    count = 80
    signal = rng.normal(0.0, 3.0, size=count)
    days = pd.bdate_range("2024-01-02", periods=count)
    rows = []
    for index in range(count):
        values = np.empty(9)
        for feature_index in range(8):
            sign = 1.0 if feature_index % 2 == 0 else -1.0
            values[feature_index] = sign * signal[index] + rng.normal(0.0, 0.01)
        values[8] = rng.normal(0.0, 1.0)
        row = {
            "ticker": "TEST1" if index < 30 else "TEST2",
            "date": days[index].strftime("%Y-%m-%d"),
            "target_1d": 1.0,
            "target_5d": 2.0,
            "target_20d": 3.0,
        }
        for name, value in zip(FEATURES, values, strict=True):
            row[name] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def test_pca_reads_the_shared_feature_file_and_scales_inputs() -> None:
    source = (ROOT / "machine_learning" / "unsupervised" / "pca" / "train.py").read_text(
        encoding="utf-8"
    )
    assert "random_forest_features.csv" in source
    assert "pca_features.csv" not in source
    assert "StandardScaler" in source
    assert "PCA" in source
    assert "random_state=42" in source
    assert "n_components=len(FEATURES)" in source
    assert "model_comparison" in source
    assert "replace_model_rows" not in source
    assert FEATURES == FEATURE_COLUMNS
    for banned in ("ticker", "date", *TARGET_COLUMNS):
        assert banned not in FEATURES
    assert DEFAULT_OUTPUT.name == "pca"
    assert DEFAULT_OUTPUT.parent.name == "outputs"
    dashboard = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert 'PAGE_PCA = "PCA"' in dashboard
    assert "ML_PAGES = (PAGE_FOREST, PAGE_XGBOOST, PAGE_LINEAR, PAGE_KMEANS, PAGE_PCA)" in dashboard
    assert 'st.button("Run PCA")' in dashboard
    assert "_train_pca_once" in dashboard
    assert "pca_features.csv" not in dashboard
    section = dashboard.split("def _pca_explanation", 1)[1].split("def _style", 1)[0]
    assert "_show_three_scores" not in section
    assert "_forest_comparison_table" not in section
    assert "A loading near 1 or -1 means that feature moves with that component." in section
    for banned in ("MAE", "RMSE", "directional_accuracy", "directional accuracy"):
        assert banned not in section
    render = dashboard.split("def render", 1)[1].split("def _portfolio_page", 1)[0]
    assert render.index("_train_forest_once") < render.index("_train_kmeans_once")
    assert render.index("_train_kmeans_once") < render.index("_train_pca_once")


def test_components_use_only_the_nine_inputs() -> None:
    frame = _signal_frame()
    result = fit_pca(frame)
    variance = result["explained_variance"]
    loadings = result["loadings"]
    ratios = variance["explained_variance_ratio"].to_numpy(dtype=float)
    assert result["n_rows"] == len(frame)
    assert list(variance["component"]) == list(range(1, 10))
    assert list(loadings["feature"]) == FEATURES
    assert list(loadings.columns) == ["feature", *[f"pc{index}" for index in range(1, 10)]]
    assert abs(float(ratios.sum()) - 1.0) < 1e-8
    assert list(ratios) == sorted(ratios, reverse=True)
    cumulative = variance["cumulative_explained_variance"].to_numpy(dtype=float)
    assert abs(float(cumulative[1]) - float(ratios[:2].sum())) < 1e-8
    assert abs(float(cumulative[-1]) - 1.0) < 1e-8
    # Eight columns move together, so the first component carries most of the variance.
    assert float(ratios[0]) > 0.8
    weights = loadings.set_index("feature")["pc1"].astype(float)
    assert weights.abs().idxmin() == "range_pct"
    for column in (f"pc{index}" for index in range(1, 10)):
        assert abs(float(np.square(loadings[column].to_numpy(dtype=float)).sum()) - 1.0) < 1e-6
    dot = float(np.dot(loadings["pc1"], loadings["pc2"]))
    assert abs(dot) < 1e-6
    poisoned = frame.copy()
    poisoned["target_1d"] = np.linspace(-100, 100, len(frame))
    poisoned["target_5d"] = np.linspace(50, -50, len(frame))
    poisoned["target_20d"] = 7
    poisoned["ticker"] = "TEST9"
    poisoned["date"] = "1999-01-01"
    again = fit_pca(poisoned)
    assert again["explained_variance"]["explained_variance_ratio"].tolist() == (
        result["explained_variance"]["explained_variance_ratio"].tolist()
    )
    for column in loadings.columns:
        if column == "feature":
            continue
        left = result["loadings"][column].to_numpy(dtype=float)
        right = again["loadings"][column].to_numpy(dtype=float)
        assert np.allclose(left, right)


def test_training_writes_pca_files_and_leaves_the_comparison_file(tmp_path: Path) -> None:
    frame = _signal_frame()
    # A day missing one input is left out. It is not filled in.
    frame.loc[0, "rsi_14"] = np.nan
    feature_path = tmp_path / "random_forest_features.csv"
    frame.to_csv(feature_path, index=False)
    output = tmp_path / "outputs" / "pca"
    comparison = output / "model_comparison.csv"
    output.mkdir(parents=True)
    original = "model,mae\nrandom_forest,3.5\n"
    comparison.write_text(original, encoding="utf-8")
    result = train_and_save(tmp_path / "unused.db", feature_path, output)
    assert result["n_rows"] == len(frame) - 1
    assert comparison.read_text(encoding="utf-8") == original
    assert sorted(path.name for path in output.iterdir()) == [
        "explained_variance.csv",
        "loadings.csv",
        "model_comparison.csv",
    ]
    variance = pd.read_csv(output / "explained_variance.csv")
    loadings = pd.read_csv(output / "loadings.csv")
    assert list(variance.columns) == [
        "component",
        "explained_variance",
        "explained_variance_ratio",
        "cumulative_explained_variance",
    ]
    assert list(loadings["feature"]) == FEATURES
    assert "pc1" in loadings.columns and "pc2" in loadings.columns and "pc3" in loadings.columns
    for table in (variance, loadings):
        for banned in ("mae", "rmse", "directional_accuracy", "MAE", "RMSE"):
            assert banned not in table.columns
    assert not (tmp_path / "pca_features.csv").exists()
