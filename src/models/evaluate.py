"""
Time-series backtesting for the unified prediction pipeline.

Evaluates direction accuracy, magnitude MAE, and change-event detection
across chronological train/test splits.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
    r2_score,
)

from src.config import PROCESSED_DIR
from src.models.train import (
    compute_calibration,
    train_change_classifier,
    save_metrics,
    CHANGE_FEATURES,
)
from src.models.predict import (
    predict_attr_delta,
    _load_calibration,
    _load_classifiers,
    _get_cal,
    _fallback_change_prob,
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
        test_updates = updates.iloc[cut : cut + fold_size]
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
    """Evaluate the unified pipeline on one fold.

    Metrics:
      - direction_accuracy: sign(gap) matches sign(delta)
      - magnitude_mae: |predicted_delta - actual_delta| for changes
      - change_precision/recall/f1: detecting delta != 0
      - change_roc_auc: classifier discrimination
      - ovr_delta_mae: player-level OVR delta error
      - precision_at_20: top 20 investment targets
    """
    metrics = {}

    # ── Train calibration + classifier on this fold ──────────────────────
    calibration = compute_calibration(train)
    classifiers = train_change_classifier(train)

    # ── Predict on test set ────────────────────────────────────────────────
    test = test.copy()
    test["attr_norm"] = test["attribute_name"].apply(normalize_attr_name)
    test["is_hitter_flag"] = test["is_hitter"].astype(int)

    # Run prediction for each row
    pred_deltas = []
    for _, row in test.iterrows():
        d, prob, gap = predict_attr_delta(row["attr_norm"], row, {})
        pred_deltas.append(d)

    test["predicted_delta"] = pred_deltas
    test["pred_positive"] = (test["predicted_delta"] > 0.5).astype(int)
    test["delta_positive"] = (test["delta"] > 0).astype(int)
    test["changed"] = (test["delta"] != 0).astype(int)
    test["pred_changed"] = (test["predicted_delta"].abs() >= 0.5).astype(int)

    # ── Direction accuracy (on all rows, including no-change) ─────────────
    gap_direction = (test["gap"] > 0).astype(int)
    actual_direction = (test["delta"] > 0).astype(int)
    metrics["direction_accuracy_gap"] = float(accuracy_score(actual_direction, gap_direction))

    correct = test["pred_positive"] == test["delta_positive"]
    metrics["direction_accuracy_model"] = float(correct.mean())

    # ── Direction accuracy on non-zero changes only ───────────────────────
    changed = test[test["changed"] == 1]
    if not changed.empty:
        metrics["direction_accuracy_changed"] = float(
            accuracy_score(changed["delta_positive"], (changed["predicted_delta"] > 0).astype(int))
        )
        metrics["gap_direction_accuracy_changed"] = float(
            accuracy_score(changed["delta_positive"], (changed["gap"] > 0).astype(int))
        )

    # ── Magnitude MAE on changed attributes ───────────────────────────────
    if not changed.empty:
        metrics["magnitude_mae"] = float(mean_absolute_error(changed["delta"], changed["predicted_delta"]))

    # Calibrated MAE (what predict.py would produce with calibration)
    cal_deltas = []
    for _, row in test.iterrows():
        cal = _get_cal(row["attr_norm"], int(row.get("game_year", 26)), int(row.get("ovr_before", 75)))
        g = row["gap"]
        if abs(g) >= cal["thresh"]:
            cd = g * cal["scale"]
            cd = max(-cal["max"], min(cal["max"], cd))
        else:
            cd = 0.0
        cal_deltas.append(cd)
    test["calibrated_delta"] = cal_deltas
    changed_cal = test[test["changed"] == 1]
    if not changed_cal.empty:
        metrics["magnitude_mae_calibrated"] = float(
            mean_absolute_error(changed_cal["delta"], changed_cal["calibrated_delta"])
        )

    # ── Change detection (precision/recall/f1) ────────────────────────────
    if test["changed"].sum() > 0:
        metrics["change_precision"] = float(
            (test["pred_changed"] & test["changed"]).sum() / max(test["pred_changed"].sum(), 1)
        )
        metrics["change_recall"] = float(
            (test["pred_changed"] & test["changed"]).sum() / max(test["changed"].sum(), 1)
        )
        metrics["change_f1"] = float(
            f1_score(test["changed"], test["pred_changed"], zero_division=0)
        )

    # ── Classifier ROC-AUC (if classifier available) ──────────────────────
    label = "hitter" if (test["is_hitter_flag"].mean() > 0.5) else "pitcher"
    if label in classifiers and hasattr(classifiers[label], "predict_proba"):
        try:
            X = test[CHANGE_FEATURES].fillna(0).values
            probs = classifiers[label].predict_proba(X)[:, 1]
            metrics["change_roc_auc"] = float(roc_auc_score(test["changed"], probs))
        except Exception:
            pass

    # ── Player-level OVR delta ────────────────────────────────────────────
    player_pred = (
        test.groupby(["game_year", "update_id", "card_uuid"])
        .agg(
            ovr_delta=("delta", "sum"),
            pred_delta_sum=("predicted_delta", "sum"),
            n_attrs=("predicted_delta", "count"),
        )
        .reset_index()
    )
    player_pred["pred_ovr_delta"] = player_pred["pred_delta_sum"] / player_pred["n_attrs"].clip(lower=1) * 1.5

    if len(player_pred) > 5:
        metrics["ovr_delta_mae"] = float(
            mean_absolute_error(player_pred["ovr_delta"], player_pred["pred_ovr_delta"])
        )
        metrics["ovr_delta_r2"] = float(
            r2_score(player_pred["ovr_delta"], player_pred["pred_ovr_delta"])
        )

    # ── Precision at k (top predicted deltas) ────────────────────────────
    metrics["precision_at_20"] = precision_at_k(
        (test["delta"] > 2).values,
        test["predicted_delta"].abs().values,
        20,
    )
    metrics["precision_at_50"] = precision_at_k(
        (test["delta"] > 2).values,
        test["predicted_delta"].abs().values,
        50,
    )

    return metrics


def run_backtest(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        path = PROCESSED_DIR / "training_examples.parquet"
        if path.exists():
            df = pd.read_parquet(path)
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
        "direction_accuracy_gap", "direction_accuracy_model",
        "direction_accuracy_changed", "gap_direction_accuracy_changed",
        "magnitude_mae", "magnitude_mae_calibrated",
        "change_precision", "change_recall", "change_f1",
        "change_roc_auc", "ovr_delta_mae", "ovr_delta_r2",
        "precision_at_20", "precision_at_50",
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
