"""K-Means on a tiny synthetic frame.

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
from machine_learning.unsupervised.kmeans.train import (
    DEFAULT_OUTPUT,
    FEATURES,
    fit_kmeans,
    train_and_save,
)

ROOT = Path(__file__).resolve().parents[1]


def _blobs() -> pd.DataFrame:
    """Three tight groups. The future columns are not inputs."""
    rng = np.random.default_rng(0)
    centers = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 1.0, 1.0, 1.0]),
        np.array([8.0, 8.0, 8.0, 8.0, 8.0, 55.0, 3.0, 2.0, 2.5]),
        np.array([16.0, 16.0, 16.0, 16.0, 16.0, 80.0, 5.0, 3.0, 4.0]),
    ]
    future = [1.5, -0.5, 3.0]
    days = pd.bdate_range("2024-01-02", periods=40)
    rows = []
    for group, (center, target) in enumerate(zip(centers, future, strict=True)):
        for index in range(40):
            values = center + rng.normal(0.0, 0.05, size=9)
            row = {
                "ticker": f"TEST{group + 1}",
                "date": days[index].strftime("%Y-%m-%d"),
                "target_1d": target,
                "target_5d": 100.0 * group,
                "target_20d": -40.0 * group,
            }
            for name, value in zip(FEATURES, values, strict=True):
                row[name] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def test_kmeans_reads_the_shared_feature_file_and_scales_inputs() -> None:
    source = (ROOT / "machine_learning" / "unsupervised" / "kmeans" / "train.py").read_text(
        encoding="utf-8"
    )
    assert "random_forest_features.csv" in source
    assert "kmeans_features.csv" not in source
    assert "StandardScaler" in source
    assert "KMeans" in source
    assert "random_state=42" in source
    assert "PCA(n_components=2)" in source
    assert "not a separate model" in source
    assert "model_comparison" in source
    assert "replace_model_rows" not in source
    assert FEATURES == FEATURE_COLUMNS
    for banned in ("ticker", "date", *TARGET_COLUMNS):
        assert banned not in FEATURES
    assert DEFAULT_OUTPUT.name == "kmeans"
    assert DEFAULT_OUTPUT.parent.name == "outputs"
    dashboard = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert 'PAGE_KMEANS = "K-Means"' in dashboard
    assert "ML_PAGES = (PAGE_FOREST, PAGE_XGBOOST, PAGE_LINEAR, PAGE_KMEANS, PAGE_PCA)" in dashboard
    assert 'st.button("Run K-Means")' in dashboard
    assert "_train_kmeans_once" in dashboard
    assert "kmeans_features.csv" not in dashboard
    section = dashboard.split("def _kmeans_explanation", 1)[1].split("def _style", 1)[0]
    assert "_show_three_scores" not in section
    assert "_forest_comparison_table" not in section
    assert "not a prediction score" in section
    assert "not a separate model" in section
    render = dashboard.split("def render", 1)[1].split("def _portfolio_page", 1)[0]
    assert render.index("_train_forest_once") < render.index("_train_kmeans_once")


def test_best_silhouette_k_is_kept_and_targets_are_not_inputs() -> None:
    frame = _blobs()
    result = fit_kmeans(frame)
    assert result["k"] == 3
    assert list(result["scores"]["k"]) == [2, 3, 4, 5, 6, 7, 8]
    assert int(result["scores"].loc[result["scores"]["silhouette"].idxmax(), "k"]) == 3
    assert set(result["assignments"]["cluster"]) == {0, 1, 2}
    assert len(result["assignments"]) == len(frame)
    assert list(result["centers"]["cluster"]) == [0, 1, 2]
    labeled = frame.copy()
    labeled["cluster"] = result["assignments"]["cluster"].to_numpy()
    for cluster, group in labeled.groupby("cluster"):
        center = result["centers"].loc[result["centers"]["cluster"] == cluster].iloc[0]
        for column in FEATURES:
            assert abs(float(center[column]) - float(group[column].mean())) < 1e-6
        described = result["summary"].loc[result["summary"]["cluster"] == cluster].iloc[0]
        assert abs(float(described["mean_future_return_1d"]) - float(group["target_1d"].mean())) < 1e-6
    poisoned = frame.copy()
    poisoned["target_1d"] = np.linspace(-100, 100, len(frame))
    poisoned["target_5d"] = np.linspace(50, -50, len(frame))
    poisoned["target_20d"] = 7
    poisoned["ticker"] = "TEST9"
    again = fit_kmeans(poisoned)
    assert again["assignments"]["cluster"].tolist() == result["assignments"]["cluster"].tolist()


def test_training_writes_kmeans_files_and_leaves_the_comparison_file(tmp_path: Path) -> None:
    frame = _blobs()
    # A day missing one input is left out. It is not filled in.
    frame.loc[0, "rsi_14"] = np.nan
    feature_path = tmp_path / "random_forest_features.csv"
    frame.to_csv(feature_path, index=False)
    output = tmp_path / "outputs" / "kmeans"
    comparison = output / "model_comparison.csv"
    output.mkdir(parents=True)
    original = "model,mae\nrandom_forest,3.5\n"
    comparison.write_text(original, encoding="utf-8")
    result = train_and_save(tmp_path / "unused.db", feature_path, output)
    assert result["k"] == 3
    assert comparison.read_text(encoding="utf-8") == original
    assert sorted(path.name for path in output.iterdir()) == [
        "assignments.csv",
        "centers.csv",
        "model_comparison.csv",
        "silhouette.csv",
        "summary.csv",
    ]
    assignments = pd.read_csv(output / "assignments.csv")
    centers = pd.read_csv(output / "centers.csv")
    scores = pd.read_csv(output / "silhouette.csv")
    summary = pd.read_csv(output / "summary.csv")
    assert list(assignments.columns) == ["ticker", "date", "cluster", "pc1", "pc2"]
    dropped = assignments[
        (assignments["ticker"].astype(str) == "TEST1")
        & (assignments["date"].astype(str) == "2024-01-02")
    ]
    assert dropped.empty
    assert list(scores["k"]) == [2, 3, 4, 5, 6, 7, 8]
    assert set(summary["chosen_k"]) == {3}
    assert int(summary["rows"].sum()) == len(frame) - 1
    assert list(centers.columns) == ["cluster", *FEATURES]
    for table in (assignments, centers, scores, summary):
        for banned in ("mae", "rmse", "directional_accuracy", "MAE", "RMSE"):
            assert banned not in table.columns
    assert not (tmp_path / "kmeans_features.csv").exists()


def test_large_file_sample_still_returns_a_k(monkeypatch) -> None:
    import machine_learning.unsupervised.kmeans.train as kmeans_train

    monkeypatch.setattr(kmeans_train, "SILHOUETTE_SAMPLE", 30)
    result = kmeans_train.fit_kmeans(_blobs())
    assert result["k"] in range(2, 9)
    assert len(result["scores"]) == 7
