"""Train one linear regression for each horizon.

The rows come from the random forest feature file. There is no second
feature CSV. pandas reads that file. ``LinearRegression`` draws the line.

The line never sees the future percent, the ticker, or the date.
Numbers are not standardized. The rows are not shuffled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from database.connection import get_connection, transaction, utc_now_iso
from database.initialize import DEFAULT_DB_PATH
from machine_learning.supervised.random_forest.features import (
    DEFAULT_EXPORT,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
)
from machine_learning.supervised.random_forest.train import (
    HORIZONS,
    chronological_split,
    recommendation,
    replace_model_rows,
    score_held_out,
)

# Same nine inputs the shared feature file already built.
FEATURES = list(FEATURE_COLUMNS)

MODEL_NAME = "linear_regression"

# Scores and forecasts for this model only. The comparison file stays shared.
DEFAULT_OUTPUT = PROJECT_ROOT / "machine_learning" / "outputs" / "linear_regression"
COMPARISON_PATH = PROJECT_ROOT / "machine_learning" / "outputs" / "model_comparison.csv"


def _line() -> LinearRegression:
    """A straight line from scikit-learn. The same rows always give the same line.

    Nothing here rescales the inputs. The intercept is the line's starting height.
    """
    return LinearRegression()


def _relationships(train: pd.DataFrame, target: str, model: LinearRegression) -> pd.DataFrame:
    """Pearson correlation and the fitted weight for each input.

    Correlation is how the input and the target move together on the
    training rows. The coefficient is that input's weight in the line.
    A large correlation is not the same thing as a large coefficient.
    """
    rows = []
    for index, feature in enumerate(FEATURES):
        # Pearson is the default, named here so the column is not a mystery.
        correlation = train[feature].corr(train[target], method="pearson")
        rows.append(
            {
                "feature": feature,
                "pearson_correlation": float(correlation),
                "coefficient": float(model.coef_[index]),
            }
        )
    table = pd.DataFrame(rows)
    # The strongest link, up or down, is the first row.
    table["abs_correlation"] = table["pearson_correlation"].abs()
    table = table.sort_values(
        ["abs_correlation", "feature"],
        ascending=[False, True],
        na_position="last",
    )
    return table.drop(columns=["abs_correlation"]).reset_index(drop=True)


def fit_horizons(frame: pd.DataFrame) -> list[dict]:
    """Fit one line per horizon and score it on the later days.

    A row can miss ``target_20d`` and still be the forecast row. That row
    is not used as an answer. The line only learns rows that have the
    target for that horizon.
    """
    frame = frame.copy()
    frame["date"] = frame["date"].astype(str).str.slice(0, 10)
    results = []
    for horizon, target in HORIZONS:
        train, test, cutoff = chronological_split(frame, target)
        model = _line()
        # FEATURES is the only matrix the line is allowed to see.
        model.fit(train[FEATURES], train[target])
        predicted = np.asarray(model.predict(test[FEATURES]), dtype=float)
        actual = test[target].to_numpy(dtype=float)
        scores = score_held_out(
            pd.DataFrame({"predicted_percent": predicted, "actual_percent": actual})
        )
        assert scores is not None
        points = test[["ticker", "date"]].copy()
        points["horizon"] = horizon
        points["predicted_percent"] = predicted
        points["actual_percent"] = actual
        # Newest feature row of each ticker, even when its 20-day target is unknown.
        latest = frame.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
        forward = np.asarray(model.predict(latest[FEATURES]), dtype=float)
        forecasts = []
        for ticker, value, day in zip(latest["ticker"], forward, latest["date"], strict=True):
            forecasts.append(
                {
                    "ticker": str(ticker),
                    "horizon": horizon,
                    "cutoff": str(day),
                    "predicted percent": float(value),
                    "recommendation": recommendation(float(value)),
                }
            )
        results.append(
            {
                "horizon": horizon,
                "mae": scores["mae"],
                "rmse": scores["rmse"],
                "directional_accuracy": scores["directional_accuracy"],
                "cutoff": cutoff,
                "points": points,
                "forecasts": forecasts,
                "relationships": _relationships(train, target, model),
            }
        )
    return results


def write_outputs(
    results: list[dict],
    output_dir: Path,
    comparison_path: Path,
) -> None:
    """Write this model's files, and only this model's comparison rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(
        [
            {
                "horizon": item["horizon"],
                "MAE": item["mae"],
                "RMSE": item["rmse"],
                "directional accuracy": item["directional_accuracy"],
                "cutoff": item["cutoff"],
            }
            for item in results
        ]
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    forecasts = pd.DataFrame([row for item in results for row in item["forecasts"]])
    forecasts = forecasts[
        ["ticker", "horizon", "cutoff", "predicted percent", "recommendation"]
    ]
    forecasts.to_csv(output_dir / "predictions.csv", index=False)
    # The scatter needs the held-out guesses next to what happened.
    points = pd.concat([item["points"] for item in results], ignore_index=True)
    points.to_csv(output_dir / "backtest.csv", index=False)
    relationships = []
    for item in results:
        block = item["relationships"].copy()
        block.insert(0, "horizon", item["horizon"])
        relationships.append(block)
    pd.concat(relationships, ignore_index=True).to_csv(
        output_dir / "coefficients.csv",
        index=False,
    )
    replace_model_rows(
        comparison_path,
        [
            {
                "model": MODEL_NAME,
                "horizon_days": int(item["horizon"]),
                "mae": item["mae"],
                "rmse": item["rmse"],
                "directional_accuracy": item["directional_accuracy"],
                "cutoff": item["cutoff"],
            }
            for item in results
        ],
        model=MODEL_NAME,
    )


def record_runs(
    db_path: str | Path,
    results: list[dict],
    started_at: str,
    finished_at: str,
) -> None:
    """Insert one model_runs row per horizon and the forward predictions."""
    with get_connection(db_path) as connection:
        known = {row["symbol"] for row in connection.execute("SELECT symbol FROM stocks")}
        with transaction(connection):
            for item in results:
                metrics_text = (
                    f"MAE={item['mae']:.6f}; "
                    f"RMSE={item['rmse']:.6f}; "
                    f"directional_accuracy={item['directional_accuracy']:.6f}"
                )
                cursor = connection.execute(
                    """
                    INSERT INTO model_runs (
                        model_name, started_at, finished_at,
                        cutoff_timestamp, metrics, status
                    ) VALUES (?, ?, ?, ?, ?, 'success')
                    """,
                    (
                        MODEL_NAME,
                        started_at,
                        finished_at,
                        f"{item['cutoff']}T21:00:00Z",
                        metrics_text,
                    ),
                )
                run_id = cursor.lastrowid
                payload = []
                for forecast in item["forecasts"]:
                    ticker = str(forecast["ticker"])
                    if ticker not in known:
                        continue
                    payload.append(
                        (
                            run_id,
                            ticker,
                            float(forecast["predicted percent"]),
                            str(item["horizon"]),
                            finished_at,
                            finished_at,
                        )
                    )
                if payload:
                    connection.executemany(
                        """
                        INSERT INTO ml_predictions (
                            model_run_id, ticker, prediction_value, horizon,
                            predicted_at, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        payload,
                    )


def train_and_save(
    db_path: str | Path,
    feature_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    comparison_path: str | Path | None = None,
) -> list[dict]:
    """Read the shared feature CSV, fit three lines, and save the scores.

    The CSV is ``random_forest_features.csv``. This model does not write
    another feature file. ``LinearRegression`` learns the nine input columns.
    """
    features = Path(feature_path) if feature_path is not None else DEFAULT_EXPORT
    if not features.is_file():
        raise FileNotFoundError(
            "The shared feature file is missing: random_forest_features.csv."
        )
    folder = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    comparison = Path(comparison_path) if comparison_path is not None else COMPARISON_PATH
    frame = pd.read_csv(features)
    started_at = utc_now_iso()
    results = fit_horizons(frame)
    finished_at = utc_now_iso()
    write_outputs(results, folder, comparison)
    record_runs(db_path, results, started_at, finished_at)
    return results


def main() -> None:
    results = train_and_save(DEFAULT_DB_PATH)
    for item in results:
        print(
            f"horizon {item['horizon']}: "
            f"MAE={item['mae']:.4f} "
            f"RMSE={item['rmse']:.4f} "
            f"directional_accuracy={item['directional_accuracy']:.4f} "
            f"cutoff={item['cutoff']}"
        )


if __name__ == "__main__":
    main()
