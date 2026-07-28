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
import threading
from collections import defaultdict
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DB_PATH, MODELS_DIR, QS_TIERS, HITTER_ATTRS, PITCHER_ATTRS
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


# ── Constants ───────────────────────────────────────────────────────────────

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

_ATTR_DEFAULTS = {
    "contact_left":          {"thresh": 3.0, "scale": 0.18, "max": 8.0},
    "contact_right":         {"thresh": 3.0, "scale": 0.18, "max": 8.0},
    "power_left":            {"thresh": 3.0, "scale": 0.18, "max": 8.0},
    "power_right":           {"thresh": 3.0, "scale": 0.18, "max": 8.0},
    "plate_vision":          {"thresh": 3.0, "scale": 0.18, "max": 8.0},
    "plate_discipline":      {"thresh": 3.0, "scale": 0.20, "max": 8.0},
    "batting_clutch":        {"thresh": 3.0, "scale": 0.18, "max": 8.0},
    "speed":                 {"thresh": 4.0, "scale": 0.15, "max": 6.0},
    "fielding_ability":      {"thresh": 4.0, "scale": 0.12, "max": 5.0},
    "arm_strength":          {"thresh": 4.0, "scale": 0.12, "max": 5.0},
    "arm_accuracy":          {"thresh": 4.0, "scale": 0.12, "max": 5.0},
    "reaction_time":         {"thresh": 4.0, "scale": 0.12, "max": 5.0},
    "pitch_velocity":        {"thresh": 3.0, "scale": 0.18, "max": 8.0},
    "pitch_control":         {"thresh": 3.0, "scale": 0.18, "max": 8.0},
    "pitch_movement":        {"thresh": 3.0, "scale": 0.18, "max": 8.0},
    "pitching_clutch":       {"thresh": 3.0, "scale": 0.18, "max": 8.0},
    "stamina":               {"thresh": 5.0, "scale": 0.10, "max": 5.0},
    "k_per_9":               {"thresh": 3.0, "scale": 0.18, "max": 8.0},
    "hr_per_9":              {"thresh": 3.0, "scale": 0.18, "max": 8.0},
    "k_per_9_r":             {"thresh": 3.0, "scale": 0.18, "max": 8.0},
    "k_per_9_l":             {"thresh": 3.0, "scale": 0.18, "max": 8.0},
    "h_per_9_r":             {"thresh": 3.0, "scale": 0.18, "max": 8.0},
    "h_per_9":               {"thresh": 3.0, "scale": 0.18, "max": 8.0},
    "bb_per_9":              {"thresh": 3.0, "scale": 0.18, "max": 8.0},
}

_TIER_BOUNDARIES = [65, 75, 85, 90, 95]


# ── Helpers ────────────────────────────────────────────────────────────────

def _position_is_hitter(pos: str) -> bool:
    return pos not in ("SP", "RP", "CP", "P")


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


def _qs_value(ovr: int) -> int:
    return max((v for k, v in QS_TIERS if ovr >= k), default=0)


# ── PredictionEngine ────────────────────────────────────────────────────────

class PredictionEngine:
    """Lazy-loaded, thread-safe container for prediction models."""

    def __init__(self) -> None:
        self._calibration: dict | None = None
        self._regression: dict | None = None
        self._analog_index: dict | None = None
        self._ensemble_weights: dict | None = None
        self._confidence_buckets: list | None = None
        self._ovr_weights: dict | None = None
        self._market_cal: dict | None = None
        self._merged_hitter_ovr: dict | None = None
        self._merged_pitcher_ovr: dict | None = None
        self._analog_cache: dict[tuple, float] = {}

    @property
    def calibration(self) -> dict:
        if self._calibration is None:
            path = MODELS_DIR / "calibration.json"
            self._calibration = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return self._calibration

    @property
    def regression(self) -> dict:
        if self._regression is None:
            path = MODELS_DIR / "delta_regression.joblib"
            self._regression = joblib.load(path) if path.exists() else {}
        return self._regression

    @property
    def analog_index(self) -> dict:
        if self._analog_index is None:
            path = MODELS_DIR / "analog_index.joblib"
            self._analog_index = joblib.load(path) if path.exists() else {}
        return self._analog_index

    @property
    def ensemble_weights(self) -> dict:
        if self._ensemble_weights is None:
            path = MODELS_DIR / "ensemble_weights.json"
            self._ensemble_weights = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return self._ensemble_weights

    @property
    def confidence_buckets(self) -> list:
        if self._confidence_buckets is None:
            path = MODELS_DIR / "confidence_buckets.json"
            self._confidence_buckets = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        return self._confidence_buckets

    @property
    def ovr_weights(self) -> dict:
        if self._ovr_weights is None:
            path = MODELS_DIR / "ovr_weights.joblib"
            self._ovr_weights = joblib.load(path) if path.exists() else {}
        return self._ovr_weights

    @property
    def market_cal(self) -> dict:
        if self._market_cal is None:
            path = MODELS_DIR / "market_calibration.json"
            self._market_cal = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return self._market_cal

    def _load_merged_ovr_weights(self) -> tuple[dict[str, float], dict[str, float]]:
        trained = self.ovr_weights
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

    @property
    def merged_hitter_ovr(self) -> dict:
        if self._merged_hitter_ovr is None:
            self._merged_hitter_ovr, self._merged_pitcher_ovr = self._load_merged_ovr_weights()
        return self._merged_hitter_ovr

    @property
    def merged_pitcher_ovr(self) -> dict:
        if self._merged_pitcher_ovr is None:
            self._merged_hitter_ovr, self._merged_pitcher_ovr = self._load_merged_ovr_weights()
        return self._merged_pitcher_ovr

    def ovr_weight(self, attr: str, is_hitter: bool) -> float:
        weights = self.merged_hitter_ovr if is_hitter else self.merged_pitcher_ovr
        return weights.get(attr, 0.02)

    def get_cal(self, attr: str, game_year: int = 26, ovr: int = 75) -> dict:
        cal = self.calibration
        year_cal = cal.get(str(game_year), cal.get(game_year, {}))
        if attr in year_cal:
            c = dict(year_cal[attr])
            d = _ATTR_DEFAULTS.get(attr, {"thresh": 3.0, "scale": 0.15, "max": 5.0})
            c["thresh"] = max(c["thresh"], d["thresh"])
            c["scale"] = min(c["scale"], d["scale"])
            c["max"] = min(c["max"], d["max"])
        elif attr in _ATTR_DEFAULTS:
            c = _ATTR_DEFAULTS[attr]
        else:
            c = {"thresh": 3.0, "scale": 0.15, "max": 5.0}
        return {"thresh": c["thresh"], "scale": c["scale"], "max": c["max"]}

    def calibrated_prob(self, gap_today: float, direction: str = "up") -> float:
        market_cal = self.market_cal
        buckets = market_cal.get("prob_buckets", [])
        if not buckets:
            return 0.5
        abs_gap = abs(gap_today)
        for i, bucket in enumerate(buckets):
            if i < len(buckets) - 1:
                next_mid = buckets[i + 1]["gap_mid"]
                if abs_gap >= bucket["gap_mid"] and abs_gap < next_mid:
                    return bucket.get(
                        f"p_{direction}",
                        bucket.get("p_up" if direction == "up" else "p_down", 0.5),
                    )
        last = buckets[-1]
        return last.get(
            f"p_{direction}",
            last.get("p_up" if direction == "up" else "p_down", 0.5),
        )


# ── Module-level engine singleton ──────────────────────────────────────────

_engine: PredictionEngine | None = None
_engine_lock = threading.Lock()


def _get_engine() -> PredictionEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = PredictionEngine()
    return _engine


# ═══════════════════════════════════════════════════════════════════════════
#  Signal 1: Multi-window gap projection (project as-if-update-were-today)
# ═══════════════════════════════════════════════════════════════════════════

def signal1_predict(
    attr: str,
    row: pd.Series,
    windows: dict,
    has_data: bool,
    engine: PredictionEngine | None = None,
) -> float:
    """Signal 1: uses absolute gap_today (blended multi-window projection)."""
    engine = engine or _get_engine()
    attr = normalize_attr_name(attr)
    rating = float(row.get("rating_before", row.get("current_rating", 60)))
    ovr = int(row.get("current_ovr", row.get("ovr_before", 75)))
    is_hitter = bool(row.get("is_hitter", True))
    game_year = int(row.get("game_year", 26))

    if not has_data:
        return 0.0

    projs = compute_window_projections(attr, windows, is_hitter)
    gaps = compute_window_gaps(projs, int(rating))

    gap_today = gaps.get("gap_today", gaps.get("gap_21d", 0.0))
    if gap_today == 0.0 and abs(gaps.get("gap_21d", 0.0)) > 0:
        gap_today = gaps["gap_21d"]

    cal = engine.get_cal(attr, game_year, ovr)

    if abs(gap_today) >= cal["thresh"]:
        delta = gap_today * cal["scale"]
        return max(-cal["max"], min(cal["max"], delta))

    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Signal 2: Direct regression (LightGBM → delta)
# ═══════════════════════════════════════════════════════════════════════════

def signal2_predict(
    attr: str,
    row: pd.Series,
    engine: PredictionEngine | None = None,
) -> float:
    """Signal 2: gradient boosted regression directly predicting delta."""
    engine = engine or _get_engine()
    is_hitter = bool(row.get("is_hitter", True))
    label = "hitter" if is_hitter else "pitcher"

    models = engine.regression
    model = models.get(label)
    if model is None or (isinstance(model, dict) and model.get("dummy")):
        return 0.0

    try:
        X = pd.DataFrame([[row.get(f, 0.0) for f in REGRESSION_FEATURES]], columns=REGRESSION_FEATURES)
        X = X.fillna(0.0)
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
    engine: PredictionEngine | None = None,
) -> float:
    """Signal 3: find k most similar historical players, weight their outcomes."""
    engine = engine or _get_engine()
    index = engine.analog_index
    if not index or "vectors" not in index:
        return 0.0

    feature_names = index["feature_names"]
    mean = np.array(index["feature_mean"])
    std = np.array(index["feature_std"])
    vectors = index["vectors"]
    outcomes = np.array(index["outcomes"])

    feat_vals = []
    for fname in feature_names:
        val = row.get(fname, 0.0)
        if pd.isna(val) or val is None:
            val = 0.0
        feat_vals.append(float(val))

    mlb_id = row.get("mlb_player_id")
    is_hitter = bool(row.get("is_hitter", True))
    cache_key = (
        int(mlb_id) if mlb_id is not None and pd.notna(mlb_id) else None,
        is_hitter,
        tuple(round(v, 6) for v in feat_vals),
    )
    if cache_key in engine._analog_cache:
        return engine._analog_cache[cache_key]

    try:
        x = np.array(feat_vals, dtype=np.float64)
        x_norm = (x - mean) / std

        sims = cosine_similarity(x_norm.reshape(1, -1), vectors)[0]

        k = min(k, len(sims))
        top_idx = np.argsort(sims)[::-1][:k]
        top_sims = sims[top_idx]
        top_outcomes = outcomes[top_idx]

        weights = np.maximum(top_sims, 0.0)
        total_w = weights.sum()
        if total_w < 0.01:
            result = 0.0
        else:
            result = max(-8.0, min(8.0, float(np.dot(weights, top_outcomes) / total_w)))

        engine._analog_cache[cache_key] = result
        return result
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Confidence interval from bucketed errors
# ═══════════════════════════════════════════════════════════════════════════

def _compute_confidence_interval(
    gap_today: float,
    predicted_delta: float,
    has_data: bool,
    engine: PredictionEngine | None = None,
) -> tuple[float, float]:
    """Return (low, high) confidence bounds around predicted_delta.

    Uses bucketed error percentiles calibrated on both abs_gap and
    predicted delta magnitude. Widens the interval when the bucket has
    fewer samples.
    """
    engine = engine or _get_engine()
    if not has_data:
        return (predicted_delta - 3.0, predicted_delta + 3.0)

    buckets = engine.confidence_buckets
    abs_gap = abs(gap_today)
    predicted_delta_abs = abs(np.clip(gap_today, -20, 20) * 0.15)

    best_bucket = None
    best_dist = float("inf")
    for bucket in buckets:
        if bucket["min_gap"] <= abs_gap < bucket["max_gap"]:
            med = bucket.get("median_predicted_delta_abs", 0.5)
            dist = abs(predicted_delta_abs - med)
            if dist < best_dist:
                best_dist = dist
                best_bucket = bucket

    if best_bucket is None:
        return (predicted_delta - 2.0, predicted_delta + 2.0)

    n = best_bucket.get("n", 10)
    scale_factor = 1.0 + (1.0 / max(n, 1)) * 0.5
    p90 = best_bucket["p90"] * scale_factor

    return (predicted_delta - p90, predicted_delta + p90)


# ═══════════════════════════════════════════════════════════════════════════
#  Per-attribute prediction (ensembles all 3 signals)
# ═══════════════════════════════════════════════════════════════════════════

def predict_attr_delta(
    attr: str,
    row: pd.Series,
    hist_deltas: dict[str, list[float]] | None = None,
    engine: PredictionEngine | None = None,
) -> tuple[float, float, float, float, float]:
    """Predict delta for one attribute using ensemble of 3 signals.

    Returns:
      (predicted_delta, change_prob, confidence_low, confidence_high, gap_today)
    """
    engine = engine or _get_engine()
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
    s1 = signal1_predict(attr, row, windows, has_data, engine=engine)

    # Signal 2: direct regression
    s2 = signal2_predict(attr, row, engine=engine)

    # Signal 3: analog matching
    s3 = signal3_predict(attr, row, windows, engine=engine)

    # Ensemble blend
    weights = engine.ensemble_weights
    label = "hitter" if is_hitter else "pitcher"
    w = weights.get(label, {"w_signal1": 0.40, "w_signal2": 0.40, "w_signal3": 0.20})

    # Dynamic weight adjustment: when no stat data, rely less on s1
    if not has_data:
        w["w_signal1"] = 0.20
        w["w_signal2"] = 0.60
        w["w_signal3"] = 0.20
    # When no regression model, rely more on gap
    reg_models = engine.regression
    if label not in reg_models or (isinstance(reg_models.get(label), dict) and reg_models[label].get("dummy")):
        w["w_signal1"] = 0.60
        w["w_signal2"] = 0.0
        w["w_signal3"] = 0.40

    total = w["w_signal1"] + w["w_signal2"] + w["w_signal3"]
    if total > 0:
        w = {k: v / total for k, v in w.items()}

    predicted = s1 * w["w_signal1"] + s2 * w["w_signal2"] + s3 * w["w_signal3"]

    # Apply calibration cap to ensemble output.
    cal = engine.get_cal(attr, game_year, ovr)
    predicted = max(-cal["max"], min(cal["max"], predicted))

    # Simple trend overlay (from old system, still useful)
    if mlb_id and abs(predicted) > 0.5:
        trend = _get_player_trend(mlb_id, attr)
        if abs(trend) > 0.5 and np.sign(trend) == np.sign(predicted):
            predicted += trend * 0.15
            predicted = max(-cal["max"], min(cal["max"], predicted))

    # Hard safety clip
    predicted = max(-8.0, min(8.0, predicted))

    # Change probability: logistic from gap_today
    change_prob = 0.55 / (1.0 + np.exp(-0.5 * (abs(gap_today) - 4.0)))

    # Confidence interval
    conf_low, conf_high = _compute_confidence_interval(gap_today, predicted, has_data, engine=engine)

    return predicted, change_prob, conf_low, conf_high, round(gap_today, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  DataFrame pipeline
# ═══════════════════════════════════════════════════════════════════════════

def predict_attributes(df: pd.DataFrame, engine: PredictionEngine | None = None) -> pd.DataFrame:
    engine = engine or _get_engine()
    hist = _get_historical_deltas()
    results = []

    for _, row in df.iterrows():
        attr = normalize_attr_name(str(row.get("attribute_name", "")))
        if not attr:
            continue

        rating = float(row.get("rating_before", row.get("current_rating", 60)))
        delta, prob, conf_low, conf_high, formula_gap = predict_attr_delta(attr, row, hist, engine=engine)
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

def aggregate_player_predictions(attr_df: pd.DataFrame, engine: PredictionEngine | None = None) -> pd.DataFrame:
    """Roll per-attribute predictions up to player-level OVR delta w/ market sim."""
    engine = engine or _get_engine()
    attr_df = attr_df.copy()
    attr_df["is_hitter"] = attr_df["is_hitter"].astype(bool)
    attr_df["ovr_weight"] = attr_df.apply(
        lambda r: engine.ovr_weight(r["attribute_name"], r["is_hitter"]), axis=1
    )
    attr_df["weighted_delta"] = attr_df["predicted_delta"] * attr_df["ovr_weight"]
    attr_df["abs_delta"] = attr_df["predicted_delta"].abs()

    grouped = (
        attr_df.groupby(["card_uuid", "player_name", "mlb_player_id", "current_ovr", "current_rarity", "is_hitter"], dropna=False)
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

    grouped["predicted_ovr_delta"] = grouped["weighted_sum"].clip(-5.0, 5.0)

    def _ovr_probs(row):
        gap = row["avg_gap_today"]
        p_up = engine.calibrated_prob(gap, "up")
        p_down = engine.calibrated_prob(gap, "down")
        if row["predicted_ovr_delta"] > 0:
            p_up = min(0.99, p_up * 1.2)
            p_down = max(0.01, p_down * 0.8)
        elif row["predicted_ovr_delta"] < 0:
            p_down = min(0.99, p_down * 1.2)
            p_up = max(0.01, p_up * 0.8)
        return pd.Series({"upgrade_probability": p_up, "downgrade_probability": p_down})

    probs = grouped.apply(_ovr_probs, axis=1)
    if "upgrade_probability" in probs.columns:
        grouped["upgrade_probability"] = probs["upgrade_probability"]
        grouped["downgrade_probability"] = probs["downgrade_probability"]
    else:
        grouped["upgrade_probability"] = 0.5
        grouped["downgrade_probability"] = 0.5

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

    total_w = grouped["weighted_up_sum"].abs() + grouped["weighted_down_sum"].abs()
    no_movement = total_w < 0.001
    safe_total = total_w.clip(lower=0.01)
    pct_up_weighted = grouped["weighted_up_sum"].clip(lower=0) / safe_total
    grouped["direction_consensus"] = np.where(no_movement, 0.0, (2 * pct_up_weighted - 1).clip(-1, 1))

    def _expected_value(row):
        delta = row["predicted_ovr_delta"]
        p_up = row["upgrade_probability"]
        p_down = row["downgrade_probability"]
        tier_jump = row["tier_jump_probability"]

        direction = np.sign(delta) * min(1.0, abs(delta) / 3.0)
        confidence = p_up - p_down
        tier_bonus = np.sign(delta) * tier_jump * 20.0 if abs(delta) > 0.5 else 0.0

        score = (direction * 50 + confidence * 30 + tier_bonus)
        score = max(-100, min(100, score))

        ovr = row["current_ovr"]
        cur_qs = _qs_value(ovr)
        stub_ev = (p_up - p_down) * cur_qs
        stub_ev = max(-cur_qs, min(cur_qs, stub_ev))

        return pd.Series({
            "investment_score": round(score, 0),
            "expected_value_per_card": round(stub_ev, 0),
            "total_ev_20_stack": round(stub_ev * 20, 0),
            "roi_pct": round((delta / ovr) * 100, 1) if ovr else 0,
            "current_qs": cur_qs,
            "projected_qs_up": cur_qs + int(round(max(0, stub_ev))),
            "projected_qs_down": cur_qs - int(round(max(0, -stub_ev))),
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
    engine: PredictionEngine | None = None,
) -> pd.DataFrame:
    engine = engine or _get_engine()
    if live_df.empty:
        return pd.DataFrame()

    attr_preds = predict_attributes(live_df, engine=engine)
    player_preds = aggregate_player_predictions(attr_preds, engine=engine)

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

def is_roster_update_today(engine: PredictionEngine | None = None) -> dict:
    _ = engine or _get_engine()
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
    current_ovr: int, predicted_delta: float, buy_price: int = 0,
    engine: PredictionEngine | None = None,
) -> dict:
    _ = engine or _get_engine()
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


# ── Unchanged DB/cache helpers ─────────────────────────────────────────────

@lru_cache(maxsize=4096)
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
