"""Train one XGBoost model for each horizon.

The rows come from the random forest feature file. There is no second
feature CSV. pandas reads that file. ``XGBRegressor`` is the booster.

The booster never sees the future percent, the ticker, or the date.
Numbers are not standardized. The rows are not shuffled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

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

MODEL_NAME = "xgboost"

# Scores and forecasts for this model only. The comparison file stays shared.
DEFAULT_OUTPUT = PROJECT_ROOT / "machine_learning" / "outputs" / "xgboost"
COMPARISON_PATH = PROJECT_ROOT / "machine_learning" / "outputs" / "model_comparison.csv"


def _booster() -> XGBRegressor:
    """A modest booster from the xgboost library. The seed keeps a rerun stable.

    n_jobs stays at 1 so training does not leave extra worker processes.
    """
    return XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )


def fit_horizons(frame: pd.DataFrame) -> list[dict]:
    """Fit one booster per horizon and score it on the later days.

    A row can miss ``target_20d`` and still be the forecast row. That row
    is not used as an answer. The booster only learns rows that have the
    target for that horizon.
    """
    frame = frame.copy()
    frame["date"] = frame["date"].astype(str).str.slice(0, 10)
    results = []
    for horizon, target in HORIZONS:
        train, test, cutoff = chronological_split(frame, target)
        model = _booster()
        # FEATURES is the only matrix the booster is allowed to see.
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
    # The chart needs the held-out guesses next to what happened.
    points = pd.concat([item["points"] for item in results], ignore_index=True)
    points.to_csv(output_dir / "backtest.csv", index=False)
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
    """Read the shared feature CSV, fit three boosters, and save the scores.

    The CSV is ``random_forest_features.csv``. This model does not write
    another feature file. ``XGBRegressor`` learns the nine input columns.
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
