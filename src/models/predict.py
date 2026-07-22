"""
Three-signal ensemble prediction pipeline.

Architecture:
  Signal 1: Multi-window gap projection → calibrated delta (as-if-update-today)
  Signal 2: Gradient boosted regression → direct delta prediction
  Signal 3: Historical analog matching → k-NN weighted outcome

  Ensemble: Learned weighted blend → final delta
  Confidence: Bucketed error percentiles → interval
  Market Sim: Expected stub profit from calibrated probabilities

All signals are computed per-attribute, then aggregated to player-level OVR.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DB_PATH, MODELS_DIR, HITTER_ATTRS, PITCHER_ATTRS
from src.db import AttributeChange, PlayerStatWindow, Prediction, init_db, dumps
from src.formulas.ratings import project_attribute, LEAGUE_AVG
from src.models.registry import normalize_attr_name, attrs_for_position, stat_group
from src.features.engineering import (
    compute_window_projections, compute_window_gaps,
    sample_size_ok, ovr_distance_to_tier_boundary,
    WINDOW_PRIORITY, WINDOW_WEIGHTS,
    REGRESSION_FEATURES,
)

logger = logging.getLogger(__name__)

# ── Static model cache ─────────────────────────────────────────────────────
_CALIBRATION = None
_REGRESSION = None
_ANALOG_INDEX = None
_ENSEMBLE_WEIGHTS = None
_CONFIDENCE_BUCKETS = None
_OVR_WEIGHTS = None
_MARKET_CAL = None


# ── Model loaders (lazy) ───────────────────────────────────────────────────

def _load_calibration() -> dict:
    global _CALIBRATION
    if _CALIBRATION is not None:
        return _CALIBRATION
    path = MODELS_DIR / "calibration.json"
    _CALIBRATION = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _CALIBRATION


def _load_regression() -> dict:
    global _REGRESSION
    if _REGRESSION is not None:
        return _REGRESSION
    path = MODELS_DIR / "delta_regression.joblib"
    _REGRESSION = joblib.load(path) if path.exists() else {}
    return _REGRESSION


def _load_analog_index() -> dict:
    global _ANALOG_INDEX
    if _ANALOG_INDEX is not None:
        return _ANALOG_INDEX
    path = MODELS_DIR / "analog_index.joblib"
    _ANALOG_INDEX = joblib.load(path) if path.exists() else {}
    return _ANALOG_INDEX


def _load_ensemble_weights() -> dict:
    global _ENSEMBLE_WEIGHTS
    if _ENSEMBLE_WEIGHTS is not None:
        return _ENSEMBLE_WEIGHTS
    path = MODELS_DIR / "ensemble_weights.json"
    _ENSEMBLE_WEIGHTS = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _ENSEMBLE_WEIGHTS


def _load_confidence_buckets() -> list:
    global _CONFIDENCE_BUCKETS
    if _CONFIDENCE_BUCKETS is not None:
        return _CONFIDENCE_BUCKETS
    path = MODELS_DIR / "confidence_buckets.json"
    _CONFIDENCE_BUCKETS = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    return _CONFIDENCE_BUCKETS


def _load_ovr_weights() -> dict:
    global _OVR_WEIGHTS
    if _OVR_WEIGHTS is not None:
        return _OVR_WEIGHTS
    path = MODELS_DIR / "ovr_weights.joblib"
    _OVR_WEIGHTS = joblib.load(path) if path.exists() else {}
    return _OVR_WEIGHTS


def _load_market_cal() -> dict:
    global _MARKET_CAL
    if _MARKET_CAL is not None:
        return _MARKET_CAL
    path = MODELS_DIR / "market_calibration.json"
    _MARKET_CAL = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _MARKET_CAL


# ── Attribute → OVR contribution weights ──────────────────────────────────
# Same defaults as before, loaded from trained when available

_DEFAULT_HITTER_OVR = {
    "contact_left": 0.16, "contact_right": 0.16,
    "power_left": 0.16, "power_right": 0.16,
    "plate_vision": 0.10, "plate_discipline": 0.05,
    "batting_clutch": 0.05, "speed": 0.05,
    "fielding_ability": 0.04, "arm_strength": 0.03,
    "arm_accuracy": 0.02, "reaction_time": 0.02,
}

_DEFAULT_PITCHER_OVR = {
    "pitch_velocity": 0.18, "pitch_control": 0.18, "pitch_movement": 0.18,
    "pitching_clutch": 0.04, "stamina": 0.04,
    "k_per_9": 0.08, "k_per_9_r": 0.05, "k_per_9_l": 0.05,
    "h_per_9": 0.06, "h_per_9_r": 0.04, "hr_per_9": 0.04, "bb_per_9": 0.04,
}


def _position_is_hitter(pos: str) -> bool:
    return pos not in ("SP", "RP", "CP", "P")


def _load_merged_ovr_weights() -> tuple[dict[str, float], dict[str, float]]:
    trained = _load_ovr_weights()
    BLEND = 0.7

    def _avg_coefs(position_groups: list[str], default: dict) -> dict:
        merged = dict(default)
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for pos in position_groups:
            entry = trained.get(pos)
            if entry is None:
                continue
            for attr, coef in zip(entry.get("attributes", []), entry.get("coef", [])):
                sums[attr] = sums.get(attr, 0.0) + coef
                counts[attr] = counts.get(attr, 0) + 1
        if not sums:
            return merged
        avg = {a: sums[a] / counts[a] for a in sums}
        total = sum(abs(v) for v in avg.values())
        if total < 0.001:
            return merged
        normed = {a: abs(v) / total for a, v in avg.items()}
        for attr, default_w in default.items():
            trained_w = normed.get(attr, 0.0)
            merged[attr] = default_w * (1 - BLEND) + trained_w * BLEND
        total_w = sum(merged.values())
        if total_w > 0:
            merged = {a: v / total_w for a, v in merged.items()}
        return merged

    hitter_pos = [p for p in trained if _position_is_hitter(p)]
    pitcher_pos = [p for p in trained if not _position_is_hitter(p)]
    return _avg_coefs(hitter_pos, _DEFAULT_HITTER_OVR), _avg_coefs(pitcher_pos, _DEFAULT_PITCHER_OVR)


_MERGED_HITTER_OVR: dict | None = None
_MERGED_PITCHER_OVR: dict | None = None


def _ovr_weight(attr: str, is_hitter: bool) -> float:
    global _MERGED_HITTER_OVR, _MERGED_PITCHER_OVR
    if _MERGED_HITTER_OVR is None:
        _MERGED_HITTER_OVR, _MERGED_PITCHER_OVR = _load_merged_ovr_weights()
    weights = _MERGED_HITTER_OVR if is_hitter else _MERGED_PITCHER_OVR
    return weights.get(attr, 0.02)


# ── Default calibration fallback ───────────────────────────────────────────
_DEFAULT_CAL = {"thresh": 2.0, "scale": 0.20, "max": 4.0}

_ATTR_DEFAULTS = {
    "contact_left": {"thresh": 2.0, "scale": 0.20, "max": 4.0},
    "contact_right": {"thresh": 2.0, "scale": 0.20, "max": 4.0},
    "power_left": {"thresh": 2.0, "scale": 0.20, "max": 4.0},
    "power_right": {"thresh": 2.0, "scale": 0.20, "max": 4.0},
    "plate_vision": {"thresh": 1.5, "scale": 0.25, "max": 5.0},
    "plate_discipline": {"thresh": 2.0, "scale": 0.20, "max": 4.0},
    "batting_clutch": {"thresh": 2.0, "scale": 0.20, "max": 5.0},
    "speed": {"thresh": 2.0, "scale": 0.20, "max": 4.0},
    "fielding_ability": {"thresh": 3.0, "scale": 0.15, "max": 3.0},
    "arm_strength": {"thresh": 3.0, "scale": 0.15, "max": 3.0},
    "arm_accuracy": {"thresh": 3.0, "scale": 0.15, "max": 3.0},
    "reaction_time": {"thresh": 3.0, "scale": 0.15, "max": 3.0},
    "pitch_velocity": {"thresh": 1.5, "scale": 0.25, "max": 5.0},
    "pitch_control": {"thresh": 2.0, "scale": 0.22, "max": 5.0},
    "pitch_movement": {"thresh": 2.0, "scale": 0.22, "max": 5.0},
    "pitching_clutch": {"thresh": 1.5, "scale": 0.25, "max": 5.0},
    "stamina": {"thresh": 3.0, "scale": 0.15, "max": 3.0},
    "k_per_9": {"thresh": 1.5, "scale": 0.22, "max": 5.0},
    "hr_per_9": {"thresh": 1.5, "scale": 0.22, "max": 5.0},
    "k_per_9_r": {"thresh": 1.5, "scale": 0.22, "max": 5.0},
    "k_per_9_l": {"thresh": 1.5, "scale": 0.22, "max": 5.0},
    "h_per_9_r": {"thresh": 1.5, "scale": 0.25, "max": 5.0},
    "h_per_9": {"thresh": 1.5, "scale": 0.25, "max": 5.0},
    "bb_per_9": {"thresh": 1.5, "scale": 0.22, "max": 5.0},
}


def _get_cal(attr: str, game_year: int = 26, ovr: int = 75) -> dict:
    cal = _load_calibration()
    year_cal = cal.get(str(game_year), cal.get(game_year, {}))
    if attr in year_cal:
        c = year_cal[attr]
    elif attr in _ATTR_DEFAULTS:
        c = _ATTR_DEFAULTS[attr]
    else:
        c = _DEFAULT_CAL
    thresh_factor = 1.0 + (ovr - 75) / 75 * 0.4
    factor = 1.0 + max(0, (99 - ovr) / 99) * 0.75
    return {
        "thresh": c["thresh"] * thresh_factor,
        "scale": c["scale"],
        "max": c["max"] * factor,
    }


def _ovr_factor(ovr: int) -> float:
    return 1.0 + max(0, (99 - ovr) / 99) * 0.75


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_historical_deltas() -> dict[str, list[float]]:
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute('''
        SELECT attribute_name, delta
        FROM attribute_changes
        WHERE delta IS NOT NULL AND delta != 0
          AND update_date >= DATE('now', '-45 DAYS')
    ''')
    deltas = defaultdict(list)
    for attr, delta in c.fetchall():
        if -15 <= delta <= 15:
            attr_norm = normalize_attr_name(attr)
            deltas[attr_norm].append(delta)
    conn.close()
    return dict(deltas)


def _get_player_trend(mlb_id: int, attr: str) -> float:
    Session = init_db()
    with Session() as session:
        rows = (
            session.query(AttributeChange.delta)
            .filter(
                AttributeChange.mlb_player_id == mlb_id,
                AttributeChange.attribute_name == attr,
                AttributeChange.delta.isnot(None))
            .order_by(AttributeChange.update_date.desc())
            .limit(3)
            .all()
        )
    deltas = [r[0] for r in rows if r[0] is not None]
    if not deltas:
        return 0.0
    weights = [0.5, 0.3, 0.2][:len(deltas)]
    return sum(d * w for d, w in zip(deltas, weights)) / sum(weights)


def _get_stat_windows(mlb_id: int) -> dict:
    Session = init_db()
    with Session() as s:
        rows = (
            s.query(PlayerStatWindow)
            .filter(PlayerStatWindow.mlb_player_id == mlb_id)
            .order_by(PlayerStatWindow.as_of_date.desc())
            .all()
        )
    windows = {}
    for r in rows:
        if r.window not in windows and r.stats_json:
            windows[r.window] = json.loads(r.stats_json)
    return windows


def _extract_windows(row: pd.Series) -> tuple[dict, bool]:
    sw = None
    if "stat_windows_json" in row.index and row.get("stat_windows_json"):
        try:
            swj = row["stat_windows_json"]
            sw = json.loads(swj) if isinstance(swj, str) else swj
        except (json.JSONDecodeError, TypeError):
            pass
    if sw is None:
        mlb_id = row.get("mlb_player_id")
        if mlb_id:
            sw = _get_stat_windows(int(mlb_id))
    windows = sw or {}
    has_data = bool(windows.get("21d")) or bool(windows.get("ytd"))
    return windows, has_data


# ═══════════════════════════════════════════════════════════════════════════
#  Signal 1: Multi-window gap projection (project as-if-update-were-today)
# ═══════════════════════════════════════════════════════════════════════════

def signal1_predict(
    attr: str,
    row: pd.Series,
    windows: dict,
    has_data: bool,
) -> float:
    """Signal 1: calibrated multi-window gap projection.

    Always projects as if the update were today — uses blended gap_today
    from all available windows (35% 7d, 30% 14d, 20% 21d, 15% YTD).
    """
    attr = normalize_attr_name(attr)
    rating = float(row.get("rating_before", row.get("current_rating", 60)))
    ovr = int(row.get("current_ovr", row.get("ovr_before", 75)))
    is_hitter = bool(row.get("is_hitter", True))
    game_year = int(row.get("game_year", 26))

    # Compute multi-window projections → gaps
    projs = compute_window_projections(attr, windows, is_hitter)
    gaps = compute_window_gaps(projs, int(rating))

    # Use gap_today as primary signal (weighted blend of all windows)
    gap_today = gaps.get("gap_today", gaps.get("gap_21d", 0.0))

    # Fall back to gap_21d if today/blend not available
    if gap_today == 0.0 and abs(gaps.get("gap_21d", 0.0)) > 0:
        gap_today = gaps["gap_21d"]

    # Calibrated magnitude
    cal = _get_cal(attr, game_year, ovr)

    if attr == "stamina" and not has_data:
        return 0.0

    if has_data and abs(gap_today) >= cal["thresh"]:
        delta = gap_today * cal["scale"]
        return max(-cal["max"], min(cal["max"], delta))
    elif not has_data and abs(gap_today) >= 2.5:
        delta = gap_today * 0.10
        return max(-3.0, min(3.0, delta))
    else:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Signal 2: Direct regression (LightGBM → delta)
# ═══════════════════════════════════════════════════════════════════════════

def signal2_predict(
    attr: str,
    row: pd.Series,
) -> float:
    """Signal 2: gradient boosted regression directly predicting delta.

    Uses the trained LightGBM regressor with all engineered features.
    """
    is_hitter = bool(row.get("is_hitter", True))
    label = "hitter" if is_hitter else "pitcher"

    models = _load_regression()
    model = models.get(label)
    if model is None or (isinstance(model, dict) and model.get("dummy")):
        return 0.0

    try:
        X = np.array([row.get(f, 0.0) for f in REGRESSION_FEATURES]).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0)
        pred = float(model.predict(X)[0])
        return max(-8.0, min(8.0, pred))
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Signal 3: Historical analog matching (k-NN weighted outcome)
# ═══════════════════════════════════════════════════════════════════════════

def signal3_predict(
    attr: str,
    row: pd.Series,
    windows: dict,
    k: int = 20,
) -> float:
    """Signal 3: find k most similar historical players, weight their outcomes.

    Uses cosine similarity in the analog feature space.
    Returns weighted average of the k nearest neighbors' actual deltas.
    """
    index = _load_analog_index()
    if not index or "vectors" not in index:
        return 0.0

    feature_names = index["feature_names"]
    mean = np.array(index["feature_mean"])
    std = np.array(index["feature_std"])
    vectors = index["vectors"]
    outcomes = np.array(index["outcomes"])

    # Build current player's analog feature vector
    feat_vals = []
    for fname in feature_names:
        val = row.get(fname, 0.0)
        if pd.isna(val) or val is None:
            val = 0.0
        feat_vals.append(float(val))
    x = np.array(feat_vals, dtype=np.float64)
    x_norm = (x - mean) / std

    # Cosine similarity
    sims = cosine_similarity(x_norm.reshape(1, -1), vectors)[0]

    # Top-k
    k = min(k, len(sims))
    top_idx = np.argsort(sims)[::-1][:k]
    top_sims = sims[top_idx]
    top_outcomes = outcomes[top_idx]

    # Weight by similarity (only positive similarities)
    weights = np.maximum(top_sims, 0.0)
    total_w = weights.sum()
    if total_w < 0.01:
        return 0.0

    weighted_avg = float(np.dot(weights, top_outcomes) / total_w)
    return max(-8.0, min(8.0, weighted_avg))


# ═══════════════════════════════════════════════════════════════════════════
#  Confidence interval from bucketed errors
# ═══════════════════════════════════════════════════════════════════════════

def _compute_confidence_interval(
    gap_today: float,
    predicted_delta: float,
    has_data: bool,
) -> tuple[float, float]:
    """Return (low, high) confidence bounds around predicted_delta.

    Finds the gap bucket from calibration and applies the percentile errors.
    When no data or no bucket matches, returns ±2.
    """
    if not has_data:
        return (predicted_delta - 2.0, predicted_delta + 2.0)

    buckets = _load_confidence_buckets()
    abs_gap = abs(gap_today)
    for bucket in buckets:
        if bucket["min_gap"] <= abs_gap < bucket["max_gap"]:
            p10 = bucket["p10"]
            p90 = bucket["p90"]
            # Weight the interval: wider for larger predicted deltas
            return (predicted_delta - p90, predicted_delta + p90)

    return (predicted_delta - 2.0, predicted_delta + 2.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Per-attribute prediction (ensembles all 3 signals)
# ═══════════════════════════════════════════════════════════════════════════

def predict_attr_delta(
    attr: str,
    row: pd.Series,
    hist_deltas: dict[str, list[float]] | None = None,
) -> tuple[float, float, float, float, float]:
    """Predict delta for one attribute using ensemble of 3 signals.

    Returns:
      (predicted_delta, change_prob, confidence_low, confidence_high, gap_today)
    """
    attr = normalize_attr_name(attr)
    rating = float(row.get("rating_before", row.get("current_rating", 60)))
    ovr = int(row.get("current_ovr", row.get("ovr_before", 75)))
    mlb_raw = row.get("mlb_player_id")
    mlb_id = int(mlb_raw) if (mlb_raw is not None and pd.notna(mlb_raw)) else None
    is_hitter = bool(row.get("is_hitter", True))
    game_year = int(row.get("game_year", 26))

    windows, has_data = _extract_windows(row)

    # Compute gap_today for confidence & change_prob
    projs = compute_window_projections(attr, windows, is_hitter)
    gaps_dict = compute_window_gaps(projs, int(rating))
    gap_today = gaps_dict.get("gap_today", gaps_dict.get("gap_21d", 0.0))
    if gap_today == 0.0 and abs(gaps_dict.get("gap_21d", 0.0)) > 0:
        gap_today = gaps_dict["gap_21d"]

    # Signal 1: multi-window gap projection
    s1 = signal1_predict(attr, row, windows, has_data)

    # Signal 2: direct regression
    s2 = signal2_predict(attr, row)

    # Signal 3: analog matching
    s3 = signal3_predict(attr, row, windows)

    # Ensemble blend
    weights = _load_ensemble_weights()
    label = "hitter" if is_hitter else "pitcher"
    w = weights.get(label, {"w_signal1": 0.40, "w_signal2": 0.40, "w_signal3": 0.20})

    # Dynamic weight adjustment: when no stat data, rely less on s1
    if not has_data:
        w["w_signal1"] = 0.20
        w["w_signal2"] = 0.60
        w["w_signal3"] = 0.20
    # When no regression model, rely more on gap
    reg_models = _load_regression()
    if label not in reg_models or (isinstance(reg_models.get(label), dict) and reg_models[label].get("dummy")):
        w["w_signal1"] = 0.60
        w["w_signal2"] = 0.0
        w["w_signal3"] = 0.40

    total = w["w_signal1"] + w["w_signal2"] + w["w_signal3"]
    if total > 0:
        w = {k: v / total for k, v in w.items()}

    predicted = s1 * w["w_signal1"] + s2 * w["w_signal2"] + s3 * w["w_signal3"]

    # Simple trend overlay (from old system, still useful)
    if mlb_id and abs(predicted) > 0.5:
        trend = _get_player_trend(mlb_id, attr)
        if abs(trend) > 0.5 and np.sign(trend) == np.sign(predicted):
            predicted += trend * 0.15
            cal = _get_cal(attr, game_year, ovr)
            predicted = max(-cal["max"], min(cal["max"], predicted))

    # Clip
    predicted = max(-8.0, min(8.0, predicted))

    # Change probability: logistic from gap_today
    change_prob = 0.55 / (1.0 + np.exp(-0.5 * (abs(gap_today) - 4.0)))

    # Confidence interval
    conf_low, conf_high = _compute_confidence_interval(gap_today, predicted, has_data)

    return predicted, change_prob, conf_low, conf_high, round(gap_today, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  DataFrame pipeline
# ═══════════════════════════════════════════════════════════════════════════

def predict_attributes(df: pd.DataFrame) -> pd.DataFrame:
    hist = _get_historical_deltas()
    results = []

    for _, row in df.iterrows():
        attr = normalize_attr_name(str(row.get("attribute_name", "")))
        if not attr:
            continue

        rating = float(row.get("rating_before", row.get("current_rating", 60)))
        delta, prob, conf_low, conf_high, formula_gap = predict_attr_delta(attr, row, hist)
        new_rating = rating + delta

        delta_strength = min(1.0, abs(delta) / 2.0)
        p_dir = 0.5 + 0.5 * delta_strength
        if delta > 0:
            up_prob = prob * p_dir
            dn_prob = max(0.001, prob - up_prob)
        elif delta < 0:
            dn_prob = prob * p_dir
            up_prob = max(0.001, prob - dn_prob)
        else:
            split = 0.5 * prob
            up_prob = max(split, 0.001)
            dn_prob = max(split, 0.001)

        results.append({
            "card_uuid": row.get("card_uuid"),
            "player_name": row.get("player_name"),
            "mlb_player_id": row.get("mlb_player_id"),
            "attribute_name": attr,
            "rating_before": int(rating),
            "current_ovr": int(row.get("current_ovr", row.get("ovr_before", 75))),
            "current_rarity": row.get("current_rarity", ""),
            "is_hitter": int(bool(row.get("is_hitter", True))),
            "position": row.get("position", ""),
            "projected_rating": int(round(new_rating)),
            "predicted_delta": round(delta, 1),
            "gap": formula_gap,
            "gap_today": formula_gap,
            "change_prob": round(prob, 3),
            "confidence_low": round(conf_low, 1),
            "confidence_high": round(conf_high, 1),
            "upgrade_prob_attr": round(up_prob, 3),
            "downgrade_prob_attr": round(dn_prob, 3),
            "confidence_score": 30 if row.get("mlb_player_id") else 0,
            "has_stat_data": int(_extract_windows(row)[1]),
            "mismatch_score": 0.0,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
#  Player-level aggregation
# ═══════════════════════════════════════════════════════════════════════════

_TIER_BOUNDARIES = [65, 75, 85, 90, 95]

_QS_TIERS = [
    (0, 25), (65, 100), (75, 300), (80, 600),
    (85, 1000), (90, 5000), (92, 10000),
    (94, 25000), (95, 50000), (97, 100000),
]


def _qs_value(ovr: int) -> int:
    return max((v for k, v in _QS_TIERS if ovr >= k), default=0)


def _calibrated_prob(gap_today: float, direction: str = "up") -> float:
    """Get calibrated probability from market_calibration.json buckets."""
    market_cal = _load_market_cal()
    buckets = market_cal.get("prob_buckets", [])
    if not buckets:
        return 0.5
    abs_gap = abs(gap_today)
    # Find the bucket this gap falls into
    for i, bucket in enumerate(buckets):
        if i < len(buckets) - 1:
            next_mid = buckets[i + 1]["gap_mid"]
            if abs_gap >= bucket["gap_mid"] and abs_gap < next_mid:
                return bucket.get(f"p_{direction}", bucket.get("p_up" if direction == "up" else "p_down", 0.5))
    last = buckets[-1]
    return last.get(f"p_{direction}", last.get("p_up" if direction == "up" else "p_down", 0.5))


def aggregate_player_predictions(attr_df: pd.DataFrame) -> pd.DataFrame:
    """Roll per-attribute predictions up to player-level OVR delta w/ market sim."""
    attr_df = attr_df.copy()
    attr_df["is_hitter"] = attr_df["is_hitter"].astype(bool)
    attr_df["ovr_weight"] = attr_df.apply(
        lambda r: _ovr_weight(r["attribute_name"], r["is_hitter"]), axis=1
    )
    attr_df["weighted_delta"] = attr_df["predicted_delta"] * attr_df["ovr_weight"]
    attr_df["abs_delta"] = attr_df["predicted_delta"].abs()

    grouped = (
        attr_df.groupby(["card_uuid", "player_name", "mlb_player_id", "current_ovr", "current_rarity", "is_hitter"])
        .agg(
            n_attrs=("predicted_delta", "count"),
            weighted_sum=("weighted_delta", "sum"),
            n_up=("predicted_delta", lambda s: (s > 0.1).sum()),
            n_down=("predicted_delta", lambda s: (s < -0.1).sum()),
            weighted_up_sum=("weighted_delta", lambda s: s[s > 0].sum()),
            weighted_down_sum=("weighted_delta", lambda s: s[s < 0].sum()),
            change_prob_mean=("change_prob", "mean"),
            up_prob_mean=("upgrade_prob_attr", "mean"),
            dn_prob_mean=("downgrade_prob_attr", "mean"),
            avg_abs_delta=("abs_delta", "mean"),
            avg_gap_today=("gap_today", "mean"),
            avg_confidence_low=("confidence_low", "mean"),
            avg_confidence_high=("confidence_high", "mean"),
        )
        .reset_index()
    )

    # OVR delta from weighted sum of attribute deltas
    # Use trained OVR weights (already applied), no arbitrary multiplier
    grouped["predicted_ovr_delta"] = (grouped["weighted_sum"] * 1.5).clip(-8.0, 8.0)

    # Tier-aware boost near boundaries (small, data-driven)
    def _tier_boost(ovr):
        dist = min(abs(ovr - b) for b in _TIER_BOUNDARIES)
        if dist <= 2:
            return 0.15 * (2 - dist) / 2
        return 0.0
    boost = grouped["current_ovr"].apply(_tier_boost)
    grouped["predicted_ovr_delta"] = (grouped["predicted_ovr_delta"] * (1.0 + boost)).clip(-8.0, 8.0)

    # Directional probabilities from calibrated gap→prob mapping
    def _ovr_probs(row):
        gap = row["avg_gap_today"]
        p_up = _calibrated_prob(gap, "up")
        p_down = _calibrated_prob(gap, "down")
        # Scale by direction
        if row["predicted_ovr_delta"] > 0:
            p_up = min(0.99, p_up * 1.2)
            p_down = max(0.01, p_down * 0.8)
        elif row["predicted_ovr_delta"] < 0:
            p_down = min(0.99, p_down * 1.2)
            p_up = max(0.01, p_up * 0.8)
        return pd.Series({"upgrade_probability": p_up, "downgrade_probability": p_down})

    probs = grouped.apply(_ovr_probs, axis=1)
    grouped["upgrade_probability"] = probs["upgrade_probability"]
    grouped["downgrade_probability"] = probs["downgrade_probability"]

    # Tier-jump probability
    def _tier_jump(row):
        d = row["predicted_ovr_delta"]
        c = row["current_ovr"]
        best = 0.0
        if d > 0:
            for b in _TIER_BOUNDARIES:
                if c < b and b - c <= d:
                    prob = min(0.75, 0.15 + (d - (b - c)) * 0.12)
                    best = max(best, prob)
        elif d < 0:
            upper = [64, 74, 84, 89, 94]
            for ub in upper:
                if c > ub and c - ub <= -d:
                    prob = min(0.50, 0.10 + (-d - (c - ub)) * 0.10)
                    best = max(best, prob)
        return best
    grouped["tier_jump_probability"] = grouped.apply(_tier_jump, axis=1)

    # Direction consensus
    total_w = grouped["weighted_up_sum"].abs() + grouped["weighted_down_sum"].abs()
    no_movement = total_w < 0.001
    safe_total = total_w.clip(lower=0.01)
    pct_up_weighted = grouped["weighted_up_sum"].clip(lower=0) / safe_total
    grouped["direction_consensus"] = np.where(no_movement, 0.0, (2 * pct_up_weighted - 1).clip(-1, 1))

    # ── Market-simulated investment score ──────────────────────────────
    # Expected Value = P(up) * profit_up + P(down) * profit_down + P(hold) * 0
    # profit_up = QS(new_ovr) - QS(current_ovr)
    # profit_down = QS(new_ovr) - QS(current_ovr) (negative = loss)
    def _expected_value(row):
        ovr = row["current_ovr"]
        delta = row["predicted_ovr_delta"]
        p_up = row["upgrade_probability"]
        p_down = row["downgrade_probability"]

        # Project new OVR
        new_ovr_up = min(99, max(0, ovr + int(round(abs(delta)))))
        new_ovr_down = min(99, max(0, ovr - int(round(abs(delta)))))

        cur_qs = _qs_value(ovr)
        qs_up = _qs_value(new_ovr_up)
        qs_down = _qs_value(new_ovr_down)

        profit_up = qs_up - cur_qs
        profit_down = qs_down - cur_qs

        ev = p_up * profit_up + p_down * profit_down

        # Apply stack limit (max 20)
        total_ev = ev * 20

        # ROI if we buy at current QS
        if cur_qs > 0:
            roi = (ev / cur_qs) * 100
        else:
            roi = 0.0

        # Final score: blend EV and tier-jump potential
        score = ev + row["tier_jump_probability"] * 500

        return pd.Series({
            "investment_score": round(score, 0),
            "expected_value_per_card": round(ev, 0),
            "total_ev_20_stack": round(total_ev, 0),
            "roi_pct": round(roi, 1),
            "current_qs": cur_qs,
            "projected_qs_up": qs_up,
            "projected_qs_down": qs_down,
        })

    ev_metrics = grouped.apply(_expected_value, axis=1)
    grouped["investment_score"] = ev_metrics["investment_score"]
    grouped["expected_value_per_card"] = ev_metrics["expected_value_per_card"]
    grouped["total_ev_20_stack"] = ev_metrics["total_ev_20_stack"]
    grouped["roi_pct"] = ev_metrics["roi_pct"]
    grouped["current_qs"] = ev_metrics["current_qs"]
    grouped["projected_qs_up"] = ev_metrics["projected_qs_up"]
    grouped["projected_qs_down"] = ev_metrics["projected_qs_down"]

    return grouped.sort_values("investment_score", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
#  Top-level orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run_predictions(
    live_df: pd.DataFrame,
    horizon_days: int = 1,
    persist: bool = True,
) -> pd.DataFrame:
    if live_df.empty:
        return pd.DataFrame()

    attr_preds = predict_attributes(live_df)
    player_preds = aggregate_player_predictions(attr_preds)

    if persist:
        Session = init_db()
        with Session() as session:
            session.query(Prediction).filter_by(horizon_days=horizon_days).delete()
            for _, row in player_preds.iterrows():
                attrs = attr_preds[attr_preds["card_uuid"] == row["card_uuid"]]
                session.add(Prediction(
                    card_uuid=row["card_uuid"],
                    player_name=row["player_name"],
                    mlb_player_id=int(row["mlb_player_id"]) if pd.notna(row.get("mlb_player_id")) else None,
                    current_ovr=int(row["current_ovr"]),
                    current_rarity=row["current_rarity"],
                    predicted_ovr_delta=float(row["predicted_ovr_delta"]),
                    upgrade_probability=float(row["upgrade_probability"]),
                    downgrade_probability=float(row["downgrade_probability"]),
                    tier_jump_probability=float(row["tier_jump_probability"]),
                    sample_size_ok=1,
                    horizon_days=horizon_days,
                    attributes_json=dumps(attrs.to_dict(orient="records")),
                    avg_gap=float(row.get("avg_gap_today", 0.0)),
                    direction_consensus=float(row.get("direction_consensus", 0.5)),
                ))
            session.commit()

    return player_preds


# ═══════════════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════════

def is_roster_update_today() -> dict:
    from datetime import datetime, timedelta

    Session = init_db()
    with Session() as session:
        row = (
            session.query(AttributeChange.update_date)
            .order_by(AttributeChange.update_date.desc())
            .first()
        )
    latest = row[0] if row else None
    today = datetime.utcnow().date()
    days_since = None
    is_today = False
    if latest:
        try:
            latest_dt = datetime.strptime(str(latest), "%Y-%m-%d").date()
            days_since = (today - latest_dt).days
            is_today = days_since == 0
        except (ValueError, TypeError):
            latest_dt = None
    else:
        latest_dt = None

    next_update = (latest_dt + timedelta(days=7)) if latest_dt else None
    days_until = None
    if next_update:
        days_until = (next_update - today).days

    return {
        "is_update_today": bool(is_today),
        "latest_update_date": str(latest_dt) if latest_dt else None,
        "days_since_last_update": days_since,
        "next_expected_update": str(next_update) if next_update else None,
        "days_until_next_update": days_until,
    }


def expected_stub_profit(
    current_ovr: int, predicted_delta: float, buy_price: int = 0
) -> dict:
    new_ovr = int(current_ovr + round(predicted_delta))
    new_ovr = max(0, min(99, new_ovr))
    cur_qs = _qs_value(current_ovr)
    new_qs = _qs_value(new_ovr)
    cost = buy_price if buy_price else cur_qs
    ppc = max(0, new_qs - cost)
    return {
        "current_qs": cur_qs,
        "projected_qs": new_qs,
        "profit_per_card": ppc,
        "total_profit": ppc * 20,
        "roi_pct": round(ppc / cost * 100, 1) if cost else 0,
        "max_stack": 20,
    }
