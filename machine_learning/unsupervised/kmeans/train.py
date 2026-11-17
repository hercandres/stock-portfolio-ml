"""Group similar days with K-Means.

The rows come from the random forest feature file. There is no second
feature CSV. pandas reads that file. ``KMeans`` is the scikit-learn model.

The clusterer never sees the future percent, the ticker, or the date.
The ticker and the date only name the row. The nine inputs are
standardized with ``StandardScaler`` because K-Means uses distance.
The trees stay unscaled. This file does not change them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from machine_learning.supervised.random_forest.features import (
    DEFAULT_EXPORT,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
)

# Same nine inputs the shared feature file already built.
# Future targets are not in this list.
FEATURES = list(FEATURE_COLUMNS)

# Past this many rows, the silhouette score uses a fixed sample.
# The groups themselves still use every row. A full distance table for
# every daily row does not fit in memory.
SILHOUETTE_SAMPLE = 2000

# Scores, labels, and centers for this model only.
# Nothing here is added to model_comparison.csv.
DEFAULT_OUTPUT = PROJECT_ROOT / "machine_learning" / "outputs" / "kmeans"


def _complete_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows that have all nine inputs. A missing future target can stay."""
    needed = ["ticker", "date", *FEATURES]
    missing = [name for name in needed if name not in frame.columns]
    if missing:
        raise ValueError("feature file is missing columns: " + ", ".join(missing))
    usable = frame.dropna(subset=list(FEATURES)).copy()
    usable["date"] = usable["date"].astype(str).str.slice(0, 10)
    usable["ticker"] = usable["ticker"].astype(str)
    usable = usable.reset_index(drop=True)
    # k goes through 8, and each k needs more rows than groups.
    if len(usable) <= 8:
        raise ValueError("need more than 8 complete rows to try k from 2 through 8")
    return usable


def _silhouette(scaled: np.ndarray, labels: np.ndarray) -> float:
    """How cleanly each row sits in its own group. Higher is better.

    A large file uses 2,000 rows and a fixed seed, with at least one row
    from every group. A rerun on the same file picks the same k.
    """
    if len(scaled) <= SILHOUETTE_SAMPLE:
        return float(silhouette_score(scaled, labels))
    rng = np.random.default_rng(42)
    chosen = []
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        share = SILHOUETTE_SAMPLE * len(members) / len(labels)
        take = min(len(members), max(1, int(round(share))))
        chosen.append(rng.choice(members, size=take, replace=False))
    index = np.concatenate(chosen)
    return float(silhouette_score(scaled[index], labels[index]))


def _kmeans(k: int) -> KMeans:
    """One K-Means model. The seed keeps a rerun on the same CSV stable.

    Ten starts, and the start with the tighter groups is kept.
    """
    return KMeans(n_clusters=k, random_state=42, n_init=10)


def fit_kmeans(frame: pd.DataFrame) -> dict:
    """Standardize the nine inputs, try k from 2 through 8, and keep the best.

    The two principal components are saved only so the chart has x and y.
    They are not a separate model.
    """
    usable = _complete_rows(frame)
    scaler = StandardScaler()
    # FEATURES is the only matrix the clusterer is allowed to see.
    scaled = np.asarray(scaler.fit_transform(usable[FEATURES]), dtype=float)
    scores = []
    best_model = None
    best_labels = None
    best_score = None
    best_k = None
    for k in range(2, 9):
        model = _kmeans(k)
        labels = model.fit_predict(scaled)
        score = _silhouette(scaled, labels)
        scores.append({"k": k, "silhouette": score})
        # A tie keeps the smaller k, because a later tie does not replace it.
        if best_score is None or score > best_score:
            best_score = score
            best_k = k
            best_model = model
            best_labels = labels
    assert best_model is not None and best_labels is not None and best_k is not None
    # Chart axes only. This fit is not saved as its own model.
    axes = np.asarray(PCA(n_components=2).fit_transform(scaled), dtype=float)
    original = scaler.inverse_transform(best_model.cluster_centers_)
    centers = pd.DataFrame(original, columns=FEATURES)
    centers.insert(0, "cluster", list(range(best_k)))
    assignments = usable[["ticker", "date"]].copy()
    assignments["cluster"] = best_labels.astype(int)
    assignments["pc1"] = axes[:, 0]
    assignments["pc2"] = axes[:, 1]
    has_future = "target_1d" in usable.columns
    summary_rows = []
    for cluster in range(best_k):
        member_index = np.flatnonzero(best_labels == cluster)
        if has_future and len(member_index):
            future = usable.iloc[member_index]["target_1d"]
            # Mean skips a day that does not have a future yet.
            # This number is a description. It was not an input.
            future_mean = float(future.mean()) if future.notna().any() else np.nan
        else:
            future_mean = np.nan
        summary_rows.append(
            {
                "chosen_k": int(best_k),
                "silhouette": float(best_score),
                "cluster": int(cluster),
                "rows": int(len(member_index)),
                "mean_future_return_1d": future_mean,
            }
        )
    return {
        "k": int(best_k),
        "silhouette": float(best_score),
        "scores": pd.DataFrame(scores),
        "assignments": assignments,
        "centers": centers,
        "summary": pd.DataFrame(summary_rows),
    }


def write_outputs(result: dict, output_dir: Path) -> None:
    """Write labels, centers, and the silhouette scores for k = 2..8.

    This folder is the only place those files go. The shared comparison
    file is left untouched.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result["assignments"].to_csv(output_dir / "assignments.csv", index=False)
    result["centers"].to_csv(output_dir / "centers.csv", index=False)
    result["scores"].to_csv(output_dir / "silhouette.csv", index=False)
    result["summary"].to_csv(output_dir / "summary.csv", index=False)


def train_and_save(
    db_path: str | Path | None = None,
    feature_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Read the shared feature CSV, choose k, and save labels and centers.

    ``db_path`` is accepted so the dashboard can call this the same way as
    the other models. The database is not opened. No comparison row is written.
    """
    del db_path
    features = Path(feature_path) if feature_path is not None else DEFAULT_EXPORT
    if not features.is_file():
        raise FileNotFoundError(
            "The shared feature file is missing: random_forest_features.csv."
        )
    folder = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    frame = pd.read_csv(features)
    result = fit_kmeans(frame)
    write_outputs(result, folder)
    return result


def main() -> None:
    result = train_and_save(None)
    print(f"k={result['k']} silhouette={result['silhouette']:.4f}")


if __name__ == "__main__":
    main()
