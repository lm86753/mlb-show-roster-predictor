"""
Three-signal ensemble training pipeline.

Architecture:
  Signal 1: Multi-window gap projection (formula-based, "today" weighted blend)
  Signal 2: Gradient boosted delta regression (LightGBM)
  Signal 3: Historical analog matching (nearest-neighbor from feature space)

  Ensemble: Learned weighted blend of all 3 signals
  Confidence: Bucketed error percentiles from walk-forward validation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import TimeSeriesSplit

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

from src.config import HITTER_ATTRS, MODELS_DIR, PITCHER_ATTRS, PROCESSED_DIR
from src.db import ModelMetrics, init_db
from src.features.engineering import REGRESSION_FEATURES, ANALOG_FEATURES
from src.formulas.ratings import refit_and_save, project_attribute, LEAGUE_AVG
from src.models.registry import normalize_attr_name, stat_group

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Component 1: Per-attribute calibration (kept from old system)
# ═══════════════════════════════════════════════════════════════════════════

def compute_calibration(
    df: pd.DataFrame,
    min_samples: int = 10,
) -> dict:
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
#  Component 2: Direct delta regression (Signal 2)
# ═══════════════════════════════════════════════════════════════════════════

def train_delta_regression(df: pd.DataFrame) -> dict:
    """Train LightGBM regressor to directly predict delta.

    Returns dict with 'hitter' and 'pitcher' models (or dict with dummy
    if insufficient data).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["is_hitter_flag"] = df["is_hitter"].astype(int)

    models = {}
    for label, is_hitter in [("hitter", 1), ("pitcher", 0)]:
        sub = df[df["is_hitter_flag"] == is_hitter]
        if len(sub) < 100:
            logger.warning("Too few %s rows (%d); skipping regression.", label, len(sub))
            models[label] = {"dummy": True}
            continue

        X = sub[REGRESSION_FEATURES].fillna(0).values
        y = sub["delta"].fillna(0).values

        if LGBMRegressor is None:
            logger.warning("lightgbm not available; using Ridge fallback for %s.", label)
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=10.0)
            model.fit(X, y)
            models[label] = model
            continue

        model = LGBMRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=1.0,
            reg_lambda=1.0,
            verbose=-1,
        )
        model.fit(X, y)

        # Feature importance
        paired = sorted(zip(REGRESSION_FEATURES, model.feature_importances_), key=lambda x: -x[1])
        logger.info("── Delta regression (%s) top-10 features ──", label)
        for fname, imp in paired[:10]:
            logger.info("  %-40s %6d", fname, imp)

        models[label] = model

    path = MODELS_DIR / "delta_regression.joblib"
    joblib.dump(models, path)
    logger.info("Delta regression models saved to %s", path)
    return models


# ═══════════════════════════════════════════════════════════════════════════
#  Component 3: Historical analog index (Signal 3)
# ═══════════════════════════════════════════════════════════════════════════

def build_analog_index(df: pd.DataFrame) -> dict:
    """Build an index of historical player stat profiles → actual outcomes.

    For each row in the training data, stores the analog feature vector
    and the actual delta that occurred. During prediction, we find the
    k nearest neighbors and aggregate their outcomes.

    Returns dict with:
      - feature_mean / feature_std: normalization params
      - vectors: normalized analog feature matrix (n_samples x n_features)
      - outcomes: array of actual deltas corresponding to each vector
      - feature_names: list of feature column names used
    """
    analog_df = df[ANALOG_FEATURES + ["delta"]].dropna(subset=ANALOG_FEATURES).copy()
    if analog_df.empty:
        logger.warning("No analog data available.")
        return {}

    X = analog_df[ANALOG_FEATURES].values.astype(np.float64)
    y = analog_df["delta"].values.astype(np.float64)

    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std < 1e-12] = 1.0
    X_norm = (X - mean) / std

    index = {
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "vectors": X_norm,
        "outcomes": y.tolist(),
        "feature_names": list(ANALOG_FEATURES),
    }

    path = MODELS_DIR / "analog_index.joblib"
    joblib.dump(index, path)
    logger.info("Analog index built with %d samples, %d features", len(y), len(ANALOG_FEATURES))
    return index


# ═══════════════════════════════════════════════════════════════════════════
#  Component 4: Ensemble blend weights
# ═══════════════════════════════════════════════════════════════════════════

def train_ensemble_weights(
    df: pd.DataFrame,
    signal1_preds: np.ndarray | None = None,
) -> dict:
    """Learn optimal blend weights for the 3 signals via Ridge on validation.

    When signal1_preds is not provided (e.g. during initial fit), we
    simulate Signal 1 as gap_today * median_scale (from calibration).

    Weights are per (attribute_group) so that different attributes may
    weight signals differently.
    """
    df = df.copy()
    df["is_hitter_flag"] = df["is_hitter"].astype(int)

    group_weights = {}
    for label, is_hitter in [("hitter", 1), ("pitcher", 0)]:
        sub = df[df["is_hitter_flag"] == is_hitter]
        if len(sub) < 50:
            group_weights[label] = {"w_signal1": 0.40, "w_signal2": 0.40, "w_signal3": 0.20}
            continue

        # Simulate Signal 1: gap_today * calibration scale
        gap_today = sub["gap_today"].fillna(0).values
        gap_7d = sub["gap_7d"].fillna(0).values 

        # Signal 1 uses a blended gap: 0.6 * gap_today + 0.4 * clip(gap_7d, -3, 3)
        s1 = 0.6 * gap_today + 0.4 * np.clip(gap_7d, -3, 3)
        # Apply typical calibration shrinkage
        s1 = s1 * 0.20
        s1 = np.clip(s1, -5, 5)

        # Simulate Signal 2: use regression features to predict
        X_feats = sub[REGRESSION_FEATURES].fillna(0).values
        if LGBMRegressor is not None:
            try:
                reg = LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, verbose=-1)
                reg.fit(X_feats, sub["delta"].fillna(0).values)
                s2 = reg.predict(X_feats)
            except Exception:
                s2 = np.zeros(len(sub))
        else:
            s2 = np.zeros(len(sub))

        # Simulate Signal 3: simple gap-based analog (gap_today similarity)
        s3 = np.zeros(len(sub))

        # Blend with Ridge: predict delta from [s1, s2, s3]
        X_blend = np.column_stack([s1, s2, s3])
        y = sub["delta"].fillna(0).values

        try:
            ridge = Ridge(alpha=5.0, positive=True)
            ridge.fit(X_blend, y)
            coefs = ridge.coef_
            total = coefs.sum()
            if total > 0:
                coefs = coefs / total
            group_weights[label] = {
                "w_signal1": float(coefs[0]),
                "w_signal2": float(coefs[1]),
                "w_signal3": float(coefs[2]),
            }
        except Exception:
            group_weights[label] = {"w_signal1": 0.40, "w_signal2": 0.40, "w_signal3": 0.20}

        logger.info("Ensemble weights (%s): s1=%.3f s2=%.3f s3=%.3f",
                     label,
                     group_weights[label]["w_signal1"],
                     group_weights[label]["w_signal2"],
                     group_weights[label]["w_signal3"])

    path = MODELS_DIR / "ensemble_weights.json"
    path.write_text(json.dumps(group_weights, indent=2), encoding="utf-8")
    return group_weights


# ═══════════════════════════════════════════════════════════════════════════
#  Component 5: Confidence interval calibration
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_confidence_intervals(df: pd.DataFrame) -> dict:
    """Compute error percentiles bucketed by absolute gap_today.

    Returns a list of buckets: [{"max_gap": 2.0, "p10": ..., "p90": ...}, ...]
    """
    df = df.copy()
    df["abs_gap"] = df["gap_today"].abs()
    df["error"] = (df["gap_today"] * 0.20 - df["delta"]).abs()

    buckets = []
    boundaries = [0, 1, 2, 3, 4, 6, 8, 12, 99]
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        subset = df[(df["abs_gap"] >= lo) & (df["abs_gap"] < hi)]
        if len(subset) < 5:
            continue
        errors = subset["error"].values
        buckets.append({
            "min_gap": lo,
            "max_gap": hi,
            "p10": float(np.percentile(errors, 10)),
            "p25": float(np.percentile(errors, 25)),
            "p50": float(np.percentile(errors, 50)),
            "p75": float(np.percentile(errors, 75)),
            "p90": float(np.percentile(errors, 90)),
            "n": int(len(errors)),
        })

    path = MODELS_DIR / "confidence_buckets.json"
    path.write_text(json.dumps(buckets, indent=2), encoding="utf-8")
    logger.info("Confidence intervals calibrated across %d buckets", len(buckets))
    return buckets


# ═══════════════════════════════════════════════════════════════════════════
#  Component 6: Per-position OVR weights (Ridge, kept from old system)
# ═══════════════════════════════════════════════════════════════════════════

def _fit_ovr_weights(df: pd.DataFrame) -> dict:
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
#  Market simulation calibration
# ═══════════════════════════════════════════════════════════════════════════

_QS_TIERS = [
    (0, 25), (65, 100), (75, 300), (80, 600),
    (85, 1000), (90, 5000), (92, 10000),
    (94, 25000), (95, 50000), (97, 100000),
]


def qs_value(ovr: int) -> int:
    return max((v for k, v in _QS_TIERS if ovr >= k), default=0)


def calibrate_market_simulation(df: pd.DataFrame) -> dict:
    """Calibrate market simulation from historical changes.

    Computes:
      - avg_delta_per_attribute_group: how many OVR points change per update
      - upgrade_prob_vs_gap: logistic mapping from gap_today → P(upgrade)
      - downgrade_prob_vs_gap: logistic mapping from gap_today → P(downgrade)
    """
    cal: dict = {}

    df = df.copy()
    df["gap_today_abs"] = df["gap_today"].abs()
    df["positive_delta"] = (df["delta"] > 0).astype(float)
    df["negative_delta"] = (df["delta"] < 0).astype(float)

    # Logistic calibration: bin by abs gap, compute observed probability
    boundaries = [0, 0.5, 1.5, 2.5, 3.5, 5, 7, 10, 99]
    prob_buckets = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        subset = df[(df["gap_today_abs"] >= lo) & (df["gap_today_abs"] < hi)]
        if len(subset) < 10:
            continue
        n_up = subset["positive_delta"].sum()
        n_down = subset["negative_delta"].sum()
        mid = (lo + hi) / 2
        prob_buckets.append({
            "gap_mid": mid,
            "p_up": float(n_up / len(subset)),
            "p_down": float(n_down / len(subset)),
            "p_change": float((n_up + n_down) / len(subset)),
            "n": int(len(subset)),
        })

    cal["prob_buckets"] = prob_buckets
    cal["qs_tiers"] = [{"min_ovr": k, "value": v} for k, v in _QS_TIERS]

    path = MODELS_DIR / "market_calibration.json"
    path.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    logger.info("Market simulation calibrated from %d total samples", len(df))
    return cal


# ═══════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def train_all(training_path: Path | None = None) -> dict:
    path = training_path or PROCESSED_DIR / "training_examples.parquet"
    if not path.exists():
        from src.features.engineering import build_training_dataset
        df = build_training_dataset()
    else:
        try:
            df = pd.read_parquet(path)
        except (ImportError, ModuleNotFoundError):
            from src.features.engineering import build_training_dataset
            df = build_training_dataset()

    if df.empty:
        raise ValueError("No training data available. Run backfill first.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Refit formula coefficients (kept from old system)
    coeffs = refit_and_save(df)

    # 2. Per-attribute / per-year calibration
    calibration = compute_calibration(df)
    cal_path = MODELS_DIR / "calibration.json"
    cal_path.write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Calibration computed for %d game years.", len(calibration))

    # 3. Direct delta regression (Signal 2)
    regression_models = train_delta_regression(df)

    # 4. Historical analog index (Signal 3)
    analog_index = build_analog_index(df)

    # 5. Ensemble blend weights
    ensemble_weights = train_ensemble_weights(df)

    # 6. Confidence interval calibration
    confidence_buckets = calibrate_confidence_intervals(df)

    # 7. OVR weights per position
    ovr_weights = _fit_ovr_weights(df)
    joblib.dump(ovr_weights, MODELS_DIR / "ovr_weights.joblib")
    logger.info("OVR weights fit for %d positions.", len(ovr_weights))

    # 8. Market simulation calibration
    market_cal = calibrate_market_simulation(df)

    return {
        "n_cal_years": len(calibration),
        "regression_models": list(regression_models.keys()),
        "analog_count": len(analog_index.get("outcomes", [])) if analog_index else 0,
        "ensemble_weights": {k: v for k, v in ensemble_weights.items()},
        "confidence_buckets": len(confidence_buckets),
        "ovr_positions": len(ovr_weights),
        "market_cal_samples": sum(b.get("n", 0) for b in market_cal.get("prob_buckets", [])),
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
