"""
Time-series backtesting for the 3-signal ensemble pipeline.

Evaluates direction accuracy, magnitude MAE, and investment performance
across chronological train/test splits.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.models.train import (
    compute_calibration,
    train_delta_regression,
    build_analog_index,
    train_ensemble_weights,
    calibrate_confidence_intervals,
    save_metrics,
)
from src.features.engineering import REGRESSION_FEATURES
from src.models.predict import (
    predict_attr_delta,
    aggregate_player_predictions,
)
from src.models.registry import normalize_attr_name


def time_series_splits(df: pd.DataFrame, n_splits: int = 3) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split by update_id chronologically."""
    updates = (
        df[["game_year", "update_id", "update_date"]]
        .drop_duplicates()
        .sort_values(["game_year", "update_id"])
    )
    splits = []
    n = len(updates)
    if n < n_splits + 1:
        mid = n // 2
        train_updates = updates.iloc[:mid]
        test_updates = updates.iloc[mid:]
        train = df.merge(train_updates, on=["game_year", "update_id"])
        test = df.merge(test_updates, on=["game_year", "update_id"])
        return [(train, test)]

    fold_size = n // (n_splits + 1)
    for i in range(1, n_splits + 1):
        cut = fold_size * i
        train_updates = updates.iloc[:cut]
        test_updates = updates.iloc[cut: cut + fold_size]
        if test_updates.empty:
            continue
        train = df.merge(train_updates, on=["game_year", "update_id"])
        test = df.merge(test_updates, on=["game_year", "update_id"])
        splits.append((train, test))
    return splits


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 20) -> float:
    if len(y_score) == 0:
        return 0.0
    k = min(k, len(y_score))
    top_idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[top_idx].mean())


def evaluate_fold(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Evaluate the 3-signal ensemble on one fold.

    Metrics:
      - direction_accuracy: sign(predicted) matches sign(delta)
      - magnitude_mae: |predicted_delta - actual_delta| for changes
      - ensemble_mae: overall error of blended prediction
      - confidence_calibration: fraction of actual deltas within confidence interval
      - investment_precision_at_20: top predicted deltas that actually changed
    """
    from sklearn.metrics import accuracy_score, mean_absolute_error

    metrics = {}

    # Train components on this fold
    calibration = compute_calibration(train)
    regression_models = train_delta_regression(train)
    analog_index = build_analog_index(train)
    ensemble_weights = train_ensemble_weights(train)
    confidence_buckets = calibrate_confidence_intervals(train)

    # Run prediction on test set
    test = test.copy()
    test["attr_norm"] = test["attribute_name"].apply(normalize_attr_name)
    test["is_hitter_flag"] = test["is_hitter"].astype(int)

    pred_deltas = []
    conf_lows = []
    conf_highs = []
    change_probs = []
    gaps = []

    for _, row in test.iterrows():
        d, prob, cl, ch, g = predict_attr_delta(row["attr_norm"], row)
        pred_deltas.append(d)
        change_probs.append(prob)
        conf_lows.append(cl)
        conf_highs.append(ch)
        gaps.append(g)

    test["predicted_delta"] = pred_deltas
    test["change_prob"] = change_probs
    test["confidence_low"] = conf_lows
    test["confidence_high"] = conf_highs
    test["gap_today"] = gaps
    test["pred_positive"] = (test["predicted_delta"] > 0.5).astype(int)
    test["delta_positive"] = (test["delta"] > 0).astype(int)
    test["changed"] = (test["delta"] != 0).astype(int)
    test["pred_changed"] = (test["predicted_delta"].abs() >= 0.5).astype(int)

    # Direction accuracy
    correct = test["pred_positive"] == test["delta_positive"]
    metrics["direction_accuracy"] = float(correct.mean())

    changed_only = test[test["changed"] == 1]
    if not changed_only.empty:
        metrics["direction_accuracy_changed"] = float(
            accuracy_score(
                changed_only["delta_positive"],
                (changed_only["predicted_delta"] > 0).astype(int)
            )
        )

    # Magnitude MAE on changed attributes
    if not changed_only.empty:
        metrics["magnitude_mae"] = float(
            mean_absolute_error(changed_only["delta"], changed_only["predicted_delta"])
        )

    # Ensemble overall MAE
    metrics["ensemble_mae"] = float(
        mean_absolute_error(test["delta"], test["predicted_delta"])
    )

    # Confidence calibration: fraction of actual deltas within CI
    within_ci = (
        (test["delta"] >= test["confidence_low"]) &
        (test["delta"] <= test["confidence_high"])
    )
    metrics["confidence_within_ci"] = float(within_ci.mean())

    # Change detection
    if test["changed"].sum() > 0:
        metrics["change_precision"] = float(
            (test["pred_changed"] & test["changed"]).sum() / max(test["pred_changed"].sum(), 1)
        )
        metrics["change_recall"] = float(
            (test["pred_changed"] & test["changed"]).sum() / max(test["changed"].sum(), 1)
        )
        metrics["change_f1"] = float(
            2 * metrics["change_precision"] * metrics["change_recall"] /
            max(metrics["change_precision"] + metrics["change_recall"], 0.001)
        )

    # Player-level OVR delta
    player_pred = (
        test.groupby(["game_year", "update_id", "card_uuid"])
        .agg(
            ovr_delta=("delta", "sum"),
            pred_delta_sum=("predicted_delta", "sum"),
            n_attrs=("predicted_delta", "count"),
        )
        .reset_index()
    )
    player_pred["pred_ovr_delta"] = (player_pred["pred_delta_sum"] / player_pred["n_attrs"].clip(lower=1) * 1.5).clip(-8.0, 8.0)
    if len(player_pred) > 5:
        metrics["ovr_delta_mae"] = float(
            mean_absolute_error(player_pred["ovr_delta"], player_pred["pred_ovr_delta"])
        )
        rmse = np.sqrt(np.mean((player_pred["ovr_delta"] - player_pred["pred_ovr_delta"]) ** 2))
        metrics["ovr_delta_rmse"] = float(rmse)

    # Precision at k (top predicted change magnitudes)
    y_true = (test["delta"] != 0).astype(int).values
    y_score = test["predicted_delta"].abs().values
    metrics["precision_at_20"] = precision_at_k(y_true, y_score, 20)
    metrics["precision_at_50"] = precision_at_k(y_true, y_score, 50)

    return metrics


def run_backtest(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        path = PROCESSED_DIR / "training_examples.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
            except (ImportError, ModuleNotFoundError):
                from src.features.engineering import build_training_dataset
                df = build_training_dataset()
        else:
            from src.features.engineering import build_training_dataset
            df = build_training_dataset()

    if df.empty:
        return {"error": "no training data"}

    splits = time_series_splits(df)
    all_metrics = []
    for i, (train, test) in enumerate(splits):
        fold_metrics = evaluate_fold(train, test)
        fold_metrics["fold"] = i
        all_metrics.append(fold_metrics)
        save_metrics({k: v for k, v in fold_metrics.items() if k != "fold"}, fold=f"fold_{i}")

    metric_names = [
        "direction_accuracy", "direction_accuracy_changed",
        "magnitude_mae", "ensemble_mae",
        "change_precision", "change_recall", "change_f1",
        "ovr_delta_mae", "ovr_delta_rmse",
        "precision_at_20", "precision_at_50",
        "confidence_within_ci",
    ]
    summary = {"folds": len(all_metrics)}
    for metric in metric_names:
        vals = [f.get(metric) for f in all_metrics if metric in f]
        if vals:
            summary[metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "values": [float(v) for v in vals],
            }

    summary_path = PROCESSED_DIR / "backtest_summary.json"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
