from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None  # Graceful fallback on platforms without lightgbm


from src.config import HITTER_ATTRS, MODELS_DIR, PITCHER_ATTRS, TIER_ORDER, ALIAS_MAP
from src.db import ModelMetrics, init_db
from src.formulas.ratings import refit_and_save, project_attribute, LEAGUE_AVG
from src.models.registry import normalize_attr_name, attrs_for_position

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Component 1: Data-driven per-attribute calibration
# ═══════════════════════════════════════════════════════════════════════════

def compute_calibration(
    df: pd.DataFrame,
    min_samples: int = 10,
) -> dict:
    """Compute per-attribute calibration from historical deltas and gaps.

    For each (game_year, attribute_name):
      - threshold: p30(|gap|) for changes — ignore gaps below this
      - scale: median(|delta| / |gap|) for changes — maps gap to delta
      - max_delta: p95(|delta|) for changes — cap on movement

    Falls back to group-level calibration when an attribute has too few samples.
    Returns nested dict: {game_year: {attr: {thresh, scale, max}}}
    """
    cal: dict = {}
    changes = df[df["delta"] != 0].copy()
    changes["abs_delta"] = changes["delta"].abs()
    changes["abs_gap"] = changes["gap"].abs()
    changes["ratio"] = np.where(
        changes["abs_gap"] > 0.5,
        (changes["abs_delta"] / changes["abs_gap"]).clip(0, 2),
        np.nan,
    )

    for year, year_group in changes.groupby("game_year"):
        year = int(year)
        cal[year] = {}

        # Per-attribute calibration
        for attr, group in year_group.groupby("attribute_name"):
            attr = normalize_attr_name(attr)
            if attr in cal[year]:
                continue
            if len(group) < min_samples:
                continue

            ratios = group["ratio"].dropna()
            abs_gaps = group["abs_gap"].dropna()
            abs_deltas = group["abs_delta"].dropna()

            if ratios.empty or abs_gaps.empty:
                continue

            cal[year][attr] = {
                "thresh": float(np.percentile(abs_gaps, 30)),
                "scale": float(np.median(ratios)),
                "max": float(np.percentile(abs_deltas, 95)),
            }

        # Fallback: group-level calibration for attrs with too few samples
        from src.models.registry import stat_group
        group_pool: dict[str, list[dict]] = {}
        for attr, group in year_group.groupby("attribute_name"):
            attr_norm = normalize_attr_name(attr)
            if attr_norm in cal[year]:
                continue
            sg = stat_group(attr_norm)
            if sg not in group_pool:
                group_pool[sg] = []
            ratios = group["ratio"].dropna()
            abs_gaps = group["abs_gap"].dropna()
            abs_deltas = group["abs_delta"].dropna()
            if not ratios.empty:
                group_pool[sg].append({
                    "ratio_med": float(np.median(ratios)),
                    "max_delta": float(np.percentile(abs_deltas, 95)),
                    "thresh": float(np.percentile(abs_gaps, 30)),
                })

        for sg, entries in group_pool.items():
            if not entries:
                continue
            merged = {
                "thresh": float(np.median([e["thresh"] for e in entries])),
                "scale": float(np.median([e["ratio_med"] for e in entries])),
                "max": float(np.median([e["max_delta"] for e in entries])),
            }
            for attr in HITTER_ATTRS + PITCHER_ATTRS:
                if stat_group(attr) == sg and attr not in cal[year]:
                    cal[year][attr] = dict(merged)

    return cal


# ═══════════════════════════════════════════════════════════════════════════
#  Component 2: Change classifier (binary)
# ═══════════════════════════════════════════════════════════════════════════

CHANGE_FEATURES = [
    "gap", "gap_abs", "ovr_gap",
    "days_since_last_update",
    "ovr_distance_to_tier_boundary",
    "is_established_star",
    "sample_size_ok",
    "stat_k_pct", "stat_bb_pct", "stat_avg", "stat_iso",
    "stat_k9", "stat_bb9",
]


def train_change_classifier(df: pd.DataFrame) -> dict:
    """Train binary classifiers predicting P(change) for hitters and pitchers.

    Two models (hitter/pitcher) sharing signal across all attributes of that
    type. Uses gap-derived features + stat context.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["target"] = (df["delta"] != 0).astype(int)
    df["is_hitter_flag"] = df["is_hitter"].astype(int)

    models = {}
    for label, is_hitter in [("hitter", 1), ("pitcher", 0)]:
        sub = df[df["is_hitter_flag"] == is_hitter]
        if len(sub) < 100:
            logger.warning("Too few %s rows (%d); skipping classifier.", label, len(sub))
            continue

        X = sub[CHANGE_FEATURES].fillna(0).values
        y = sub["target"].values

        pos_ratio = y.mean()
        if pos_ratio < 0.01 or pos_ratio > 0.99:
            logger.warning("Skewed target for %s (%.1f%% positive); using dummy.", label, pos_ratio * 100)
            models[label] = {"dummy": pos_ratio}
            continue

        if LGBMClassifier is None:
            logger.warning("lightgbm not available; skipping %s classifier.", label)
            models[label] = {"dummy": pos_ratio}
            continue

        model = LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            class_weight="balanced",
            verbose=-1,
        )
        model.fit(X, y)

        # Feature importance
        paired = sorted(zip(CHANGE_FEATURES, model.feature_importances_), key=lambda x: -x[1])
        logger.info("── Change classifier (%s) feature importance ──", label)
        for fname, imp in paired[:10]:
            logger.info("  %-35s %6d", fname, imp)

        models[label] = model

    joblib.dump(models, MODELS_DIR / "change_classifiers.joblib")
    return models


# ═══════════════════════════════════════════════════════════════════════════
#  Component 3: Per-position OVR weights (from Ridge)
# ═══════════════════════════════════════════════════════════════════════════

def _fit_ovr_weights(df: pd.DataFrame) -> dict:
    """Estimate attribute contribution weights to OVR by position."""
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        logger.warning("sklearn not available; skipping OVR weight fitting.")
        return {}
    weights = {}
    for pos, group in df.groupby("position"):
        if len(group) < 20 or not pos:
            continue
        pivot = group.pivot_table(
            index=["game_year", "update_id", "card_uuid"],
            columns="attribute_name",
            values="delta",
            aggfunc="sum",
            fill_value=0,
        )
        if pivot.empty:
            continue
        ovr_deltas = group.groupby(["game_year", "update_id", "card_uuid"])["delta"].sum()
        common = pivot.index.intersection(ovr_deltas.index)
        if len(common) < 10:
            continue
        X = pivot.loc[common].values
        y = ovr_deltas.loc[common].values
        model = Ridge(alpha=10.0)
        model.fit(X, y)
        weights[pos] = {
            "attributes": pivot.columns.tolist(),
            "coef": model.coef_.tolist(),
            "intercept": float(model.intercept_),
        }
    return weights


# ═══════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def train_all(training_path: Path | None = None) -> dict:
    from src.config import PROCESSED_DIR

    path = training_path or PROCESSED_DIR / "training_examples.parquet"
    if not path.exists():
        from src.features.engineering import build_training_dataset
        df = build_training_dataset()
    else:
        try:
            df = pd.read_parquet(path)
        except (ImportError, ModuleNotFoundError):
            logger.warning("Parquet engine not available; rebuilding training dataset.")
            from src.features.engineering import build_training_dataset
            df = build_training_dataset()

    if df.empty:
        raise ValueError("No training data available. Run backfill first.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Refit formula coefficients
    coeffs = refit_and_save(df)

    # 2. Compute per-attribute / per-year calibration
    calibration = compute_calibration(df)
    cal_path = MODELS_DIR / "calibration.json"
    cal_path.write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Calibration computed for %d game years.", len(calibration))

    # 3. Train change classifiers (hitter / pitcher)
    classifiers = train_change_classifier(df)

    # 4. Fit OVR weights per position
    ovr_weights = _fit_ovr_weights(df)
    joblib.dump(ovr_weights, MODELS_DIR / "ovr_weights.joblib")
    logger.info("OVR weights fit for %d positions.", len(ovr_weights))

    return {
        "n_cal_years": len(calibration),
        "classifiers": list(classifiers.keys()),
        "ovr_positions": len(ovr_weights),
    }


def save_metrics(metrics: dict, fold: str = "test") -> None:
    Session = init_db()
    with Session() as session:
        for model_name, model_metrics in metrics.items():
            for metric_name, value in model_metrics.items():
                session.add(
                    ModelMetrics(
                        model_name=model_name,
                        metric_name=metric_name,
                        metric_value=float(value),
                        fold=fold,
                    )
                )
        session.commit()
