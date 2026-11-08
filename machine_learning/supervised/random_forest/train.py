"""Train one random forest for each horizon.

The forest never sees the future percent, the ticker, or the date.
The ticker and the date only name the row. The future percent is the answer.
Numbers are not standardized.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from database.connection import get_connection, transaction, utc_now_iso
from database.initialize import DEFAULT_DB_PATH

from .features import DEFAULT_EXPORT, FEATURE_COLUMNS, PROJECT_ROOT

# Same list the feature file builds. Targets are not in it.
FEATURES = list(FEATURE_COLUMNS)

HORIZONS = (
    ("1", "target_1d"),
    ("5", "target_5d"),
    ("20", "target_20d"),
)

# A forecast this small is not a useful up or down call.
FLAT_BAND = 0.15

MODEL_NAME = "random_forest"

DEFAULT_OUTPUT = PROJECT_ROOT / "machine_learning" / "outputs"

COMPARISON_COLUMNS = [
    "model",
    "horizon_days",
    "mae",
    "rmse",
    "directional_accuracy",
    "cutoff",
]


def recommendation(percent: float) -> str:
    """Turn a forecast into up, down, or flat."""
    if percent > FLAT_BAND:
        return "up"
    if percent < -FLAT_BAND:
        return "down"
    return "flat"


def _cutoff_date(dates: pd.Series) -> str:
    """First day of the held-out period. The last 20% of dates are the score.

    One date is used for every ticker. Earlier days teach the forest.
    Later days are scored. The rows are not shuffled.
    """
    ordered = sorted(set(dates.astype(str).str.slice(0, 10)))
    if len(ordered) < 5:
        raise ValueError("need at least 5 dated rows with a target to hold out a test period")
    index = int(len(ordered) * 0.8)
    index = min(max(index, 1), len(ordered) - 1)
    return ordered[index]


def chronological_split(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Split each ticker on one shared date. Train is before that date."""
    labeled = frame.dropna(subset=[target]).copy()
    labeled["date"] = labeled["date"].astype(str).str.slice(0, 10)
    cutoff = _cutoff_date(labeled["date"])
    train = labeled[labeled["date"] < cutoff]
    test = labeled[labeled["date"] >= cutoff]
    if train.empty or test.empty:
        raise ValueError(f"{target} has no chronological train and test rows")
    return train, test, cutoff


def _forest() -> RandomForestRegressor:
    """A modest forest. The seed keeps a rerun on the same CSV stable.

    n_jobs stays at 1 so training does not leave extra worker processes.
    """
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1,
    )


def _score(predicted: np.ndarray, actual: np.ndarray) -> tuple[float, float, float]:
    """MAE, RMSE, and the share of rows whose sign matches.

    The flat band is only for the recommendation label. This score still
    asks whether the guess and the real move pointed the same way.
    """
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    directional = float(np.mean(np.sign(predicted) == np.sign(actual)))
    return mae, rmse, directional


def fit_horizons(frame: pd.DataFrame) -> list[dict]:
    """Fit one forest per horizon and score it on the later days."""
    frame = frame.copy()
    frame["date"] = frame["date"].astype(str).str.slice(0, 10)
    results = []
    for horizon, target in HORIZONS:
        train, test, cutoff = chronological_split(frame, target)
        model = _forest()
        # FEATURES is the only matrix the forest is allowed to see.
        model.fit(train[FEATURES], train[target])
        predicted = np.asarray(model.predict(test[FEATURES]), dtype=float)
        actual = test[target].to_numpy(dtype=float)
        mae, rmse, directional = _score(predicted, actual)
        points = test[["ticker", "date"]].copy()
        points["horizon"] = horizon
        points["predicted_percent"] = predicted
        points["actual_percent"] = actual
        # The newest feature row of each ticker, even when its target is still unknown.
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
                "mae": mae,
                "rmse": rmse,
                "directional_accuracy": directional,
                "cutoff": cutoff,
                "points": points,
                "forecasts": forecasts,
            }
        )
    return results


def replace_model_rows(
    path: Path,
    rows: list[dict],
    model: str = MODEL_NAME,
) -> None:
    """Replace this model's rows and leave every other model untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        existing = pd.read_csv(path)
    else:
        existing = pd.DataFrame(columns=COMPARISON_COLUMNS)
    if "model" not in existing.columns:
        existing = pd.DataFrame(columns=COMPARISON_COLUMNS)
    others = existing[existing["model"].astype(str) != model]
    incoming = pd.DataFrame(rows)
    incoming["model"] = model
    combined = pd.concat([others, incoming], ignore_index=True)
    for column in COMPARISON_COLUMNS:
        if column not in combined.columns:
            combined[column] = pd.NA
    extras = [column for column in combined.columns if column not in COMPARISON_COLUMNS]
    combined = combined[COMPARISON_COLUMNS + extras]
    combined.to_csv(path, index=False)


def write_outputs(results: list[dict], output_dir: Path) -> None:
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
    forecasts = pd.DataFrame(
        [row for item in results for row in item["forecasts"]]
    )
    forecasts = forecasts[
        ["ticker", "horizon", "cutoff", "predicted percent", "recommendation"]
    ]
    forecasts.to_csv(output_dir / "predictions.csv", index=False)
    # The chart needs the held-out guesses next to what happened.
    # predictions.csv is only the latest forward call.
    points = pd.concat([item["points"] for item in results], ignore_index=True)
    points.to_csv(output_dir / "backtest.csv", index=False)
    replace_model_rows(
        output_dir / "model_comparison.csv",
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
    )


def record_runs(
    db_path: str | Path,
    results: list[dict],
    started_at: str,
    finished_at: str,
) -> None:
    """Insert one model_runs row per horizon and the forward predictions.

    The cutoff clock is 21:00 UTC so the date fits the stored timestamp
    format. The date itself is the first held-out trading day.
    """
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
) -> list[dict]:
    """Read the feature CSV, fit three forests, and save scores plus predictions."""
    features = Path(feature_path) if feature_path is not None else DEFAULT_EXPORT
    folder = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    frame = pd.read_csv(features)
    started_at = utc_now_iso()
    results = fit_horizons(frame)
    finished_at = utc_now_iso()
    write_outputs(results, folder)
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
