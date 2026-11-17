"""Summarize the daily features with PCA.

The rows come from the random forest feature file. There is no second
feature CSV. pandas reads that file. ``PCA`` is the scikit-learn model.

The model never sees the future percent, the ticker, or the date.
The ticker and the date only name the row. The nine inputs are
standardized with ``StandardScaler`` because they are not on one scale.
PCA then keeps every component. This file does not change the trees.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from machine_learning.supervised.random_forest.features import (
    DEFAULT_EXPORT,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
)

# Same nine inputs the shared feature file already built.
# Future targets are not in this list.
FEATURES = list(FEATURE_COLUMNS)

# Explained variance and loadings for this model only.
# Nothing here is added to model_comparison.csv.
DEFAULT_OUTPUT = PROJECT_ROOT / "machine_learning" / "outputs" / "pca"


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
    # One component per input, so there must be at least one row per input.
    if len(usable) < len(FEATURES):
        raise ValueError("need at least 9 complete rows for one component per input")
    return usable


def fit_pca(frame: pd.DataFrame) -> dict:
    """Standardize the nine inputs, then keep every principal component.

    The loadings are each feature's weight on a component. A weight near
    1 or -1 means that feature moves with that component.
    """
    usable = _complete_rows(frame)
    scaler = StandardScaler()
    # FEATURES is the only matrix PCA is allowed to see.
    scaled = np.asarray(scaler.fit_transform(usable[FEATURES]), dtype=float)
    # One component per input. The seed keeps a rerun on the same CSV stable.
    model = PCA(n_components=len(FEATURES), random_state=42)
    model.fit(scaled)
    ratios = np.asarray(model.explained_variance_ratio_, dtype=float)
    # Running share, so the second row is the first two components together.
    cumulative = np.cumsum(ratios)
    explained = pd.DataFrame(
        {
            "component": list(range(1, len(ratios) + 1)),
            "explained_variance": np.asarray(model.explained_variance_, dtype=float),
            "explained_variance_ratio": ratios,
            "cumulative_explained_variance": cumulative,
        }
    )
    # components_ is one row per component. Flip it so each column is a component.
    weights = np.asarray(model.components_, dtype=float).T
    loadings = pd.DataFrame(
        {f"pc{index + 1}": weights[:, index] for index in range(weights.shape[1])}
    )
    loadings.insert(0, "feature", list(FEATURES))
    return {
        "explained_variance": explained,
        "loadings": loadings,
        "n_rows": int(len(usable)),
    }


def write_outputs(result: dict, output_dir: Path) -> None:
    """Write the variance table and the loadings.

    This folder is the only place those files go. The shared comparison
    file is left untouched.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result["explained_variance"].to_csv(output_dir / "explained_variance.csv", index=False)
    result["loadings"].to_csv(output_dir / "loadings.csv", index=False)


def train_and_save(
    db_path: str | Path | None = None,
    feature_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Read the shared feature CSV, fit PCA, and save variance and loadings.

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
    result = fit_pca(frame)
    write_outputs(result, folder)
    return result


def main() -> None:
    result = train_and_save(None)
    variance = result["explained_variance"]
    first_two = float(variance["cumulative_explained_variance"].iloc[1])
    print(f"components={len(variance)} first_two={first_two:.4f}")


if __name__ == "__main__":
    main()
