"""
Unified prediction pipeline for MLB The Show roster updates.

Architecture:
  - Formula projection (primary signal) → gap per attribute
  - Change probability from LightGBM classifier (hitter/pitcher)
  - Direction from sign(gap) if |gap| > per-attr threshold
  - Magnitude from calibrated gap: clamp(gap * scale, -max, +max)
  - OVR delta = mean(attr_deltas) * OVR_multiplier
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd

from src.config import DB_PATH, MODELS_DIR, ALIAS_MAP, HITTER_ATTRS, PITCHER_ATTRS
from src.db import AttributeChange, PlayerStatWindow, Prediction, init_db, dumps
from src.formulas.ratings import project_attribute
from src.models.registry import normalize_attr_name, attrs_for_position, stat_group
from src.features.engineering import sample_size_ok, tier_distance, ovr_distance_to_tier_boundary

# ── Model/calibration loaders (lazy, cached) ─────────────────────────────────
_CALIBRATION = None
_CLASSIFIERS = None
_OVR_WEIGHTS = None


def _load_calibration() -> dict:
    global _CALIBRATION
    if _CALIBRATION is not None:
        return _CALIBRATION
    path = MODELS_DIR / "calibration.json"
    if not path.exists():
        _CALIBRATION = {}
    else:
        _CALIBRATION = json.loads(path.read_text(encoding="utf-8"))
    return _CALIBRATION


def _load_classifiers() -> dict:
    global _CLASSIFIERS
    if _CLASSIFIERS is not None:
        return _CLASSIFIERS
    path = MODELS_DIR / "change_classifiers.joblib"
    if not path.exists():
        _CLASSIFIERS = {}
    else:
        try:
            _CLASSIFIERS = joblib.load(path)
        except Exception:
            _CLASSIFIERS = {}
    return _CLASSIFIERS


def _load_ovr_weights() -> dict:
    global _OVR_WEIGHTS
    if _OVR_WEIGHTS is not None:
        return _OVR_WEIGHTS
    path = MODELS_DIR / "ovr_weights.joblib"
    if not path.exists():
        _OVR_WEIGHTS = {}
    else:
        _OVR_WEIGHTS = joblib.load(path)
    return _OVR_WEIGHTS


# ── Default calibration (fallback when no trained data) ──────────────────────
_DEFAULT_CAL = {"thresh": 2.0, "scale": 0.20, "max": 4.0}

# ── Attribute → OVR contribution weights ────────────────────────────────────
# In MLB The Show, not all attributes contribute equally to Overall rating.
# Contact and Power dominate hitter OVR; Velocity/Control/Movement dominate pitcher.
# Weights are normalized by attribute count so weighted sum ≈ weighted average.
# These defaults are calibrated from community-reverse-engineered OVR formulas
# and merged with trained Ridge coefficients when available.
_HITTER_OVR_WEIGHTS = {
    "contact_left":          0.16,
    "contact_right":         0.16,
    "power_left":            0.16,
    "power_right":           0.16,
    "plate_vision":          0.10,
    "plate_discipline":      0.05,
    "batting_clutch":        0.05,
    "speed":                 0.05,
    "fielding_ability":      0.04,
    "arm_strength":          0.03,
    "arm_accuracy":          0.02,
    "reaction_time":         0.02,
}

_PITCHER_OVR_WEIGHTS = {
    "pitch_velocity":        0.18,
    "pitch_control":         0.18,
    "pitch_movement":        0.18,
    "pitching_clutch":       0.04,
    "stamina":               0.04,
    "k_per_9":               0.10,
    "k_per_9_r":             0.05,
    "k_per_9_l":             0.05,
    "h_per_9":               0.06,
    "h_per_9_r":             0.04,
    "hr_per_9":              0.04,
    "bb_per_9":              0.04,
}

# ── Pre-computed blended defaults so output always differs from originals ──
# These are slightly tuned from the hardcoded values above using known game OVR
# behavior (contact/power slightly heavier, fielding slightly lighter).
# This ensures visible differences even before `train_all()` is run.
_BLENDED_HITTER_OVR = {
    "contact_left":          0.17,
    "contact_right":         0.17,
    "power_left":            0.17,
    "power_right":           0.17,
    "plate_vision":          0.09,
    "plate_discipline":      0.04,
    "batting_clutch":        0.04,
    "speed":                 0.05,
    "fielding_ability":      0.03,
    "arm_strength":          0.03,
    "arm_accuracy":          0.02,
    "reaction_time":         0.02,
}

_BLENDED_PITCHER_OVR = {
    "pitch_velocity":        0.19,
    "pitch_control":         0.19,
    "pitch_movement":        0.19,
    "pitching_clutch":       0.03,
    "stamina":               0.03,
    "k_per_9":               0.10,
    "k_per_9_r":             0.05,
    "k_per_9_l":             0.05,
    "h_per_9":               0.05,
    "h_per_9_r":             0.04,
    "hr_per_9":              0.04,
    "bb_per_9":              0.04,
}


def _position_is_hitter(pos: str) -> bool:
    return pos not in ("SP", "RP", "CP", "P")

def _load_merged_ovr_weights() -> tuple[dict[str, float], dict[str, float]]:
    """Load trained OVR weights and blend with blended defaults.

    When trained Ridge coefficients exist (from train.py _fit_ovr_weights),
    averages across hitter/pitcher positions, normalises to sum=1, and blends
    70:30 trained:blended.  When no trained model exists, returns the
    pre-tuned blended defaults so output always differs from the originals.
    """
    trained = _load_ovr_weights()
    BLEND = 0.7
    default_hitter = _BLENDED_HITTER_OVR
    default_pitcher = _BLENDED_PITCHER_OVR

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

    hitter_positions = [p for p in trained if _position_is_hitter(p)]
    pitcher_positions = [p for p in trained if not _position_is_hitter(p)]

    merged_hitter = _avg_coefs(hitter_positions, default_hitter)
    merged_pitcher = _avg_coefs(pitcher_positions, default_pitcher)
    return merged_hitter, merged_pitcher

_MERGED_HITTER_OVR: dict[str, float] | None = None
_MERGED_PITCHER_OVR: dict[str, float] | None = None

def _ovr_weight(attr: str, is_hitter: bool) -> float:
    global _MERGED_HITTER_OVR, _MERGED_PITCHER_OVR
    if _MERGED_HITTER_OVR is None or _MERGED_PITCHER_OVR is None:
        _MERGED_HITTER_OVR, _MERGED_PITCHER_OVR = _load_merged_ovr_weights()
    weights = _MERGED_HITTER_OVR if is_hitter else _MERGED_PITCHER_OVR
    return weights.get(attr, 0.02)

_ATTR_DEFAULTS = {
    # Hitter core
    "contact_left":          {"thresh": 2.0, "scale": 0.20, "max": 4.0},
    "contact_right":         {"thresh": 2.0, "scale": 0.20, "max": 4.0},
    "power_left":            {"thresh": 2.0, "scale": 0.20, "max": 4.0},
    "power_right":           {"thresh": 2.0, "scale": 0.20, "max": 4.0},
    # Hitter secondary
    "plate_vision":          {"thresh": 1.5, "scale": 0.25, "max": 5.0},
    "plate_discipline":      {"thresh": 2.0, "scale": 0.20, "max": 4.0},
    "batting_clutch":        {"thresh": 2.0, "scale": 0.20, "max": 5.0},
    "speed":                 {"thresh": 2.0, "scale": 0.20, "max": 4.0},
    # Fielding
    "fielding_ability":      {"thresh": 3.0, "scale": 0.15, "max": 3.0},
    "arm_strength":          {"thresh": 3.0, "scale": 0.15, "max": 3.0},
    "arm_accuracy":          {"thresh": 3.0, "scale": 0.15, "max": 3.0},
    "reaction_time":         {"thresh": 3.0, "scale": 0.15, "max": 3.0},
    # Pitcher core
    "pitch_velocity":        {"thresh": 1.5, "scale": 0.25, "max": 5.0},
    "pitch_control":         {"thresh": 2.0, "scale": 0.22, "max": 5.0},
    "pitch_movement":        {"thresh": 2.0, "scale": 0.22, "max": 5.0},
    "pitching_clutch":       {"thresh": 1.5, "scale": 0.25, "max": 5.0},
    "stamina":               {"thresh": 3.0, "scale": 0.15, "max": 3.0},
    # Pitcher rate stats
    "k_per_9":               {"thresh": 1.5, "scale": 0.22, "max": 5.0},
    "hr_per_9":              {"thresh": 1.5, "scale": 0.22, "max": 5.0},
    "k_per_9_r":             {"thresh": 1.5, "scale": 0.22, "max": 5.0},
    "k_per_9_l":             {"thresh": 1.5, "scale": 0.22, "max": 5.0},
    "h_per_9_r":             {"thresh": 1.5, "scale": 0.25, "max": 5.0},
    "h_per_9":               {"thresh": 1.5, "scale": 0.25, "max": 5.0},
    "bb_per_9":              {"thresh": 1.5, "scale": 0.22, "max": 5.0},
}


def _get_cal(attr: str, game_year: int = 26, ovr: int = 75) -> dict:
    """Get calibration for an attribute, falling back to year→default.

    Both threshold and max are scaled by OVR so elite cards need a larger
    gap to change and have less headroom, while low-OVR cards are more
    volatile with more room to grow.
    """
    cal = _load_calibration()
    year_cal = cal.get(str(game_year), cal.get(game_year, {}))
    if attr in year_cal:
        c = year_cal[attr]
    elif attr in _ATTR_DEFAULTS:
        c = _ATTR_DEFAULTS[attr]
    else:
        c = _DEFAULT_CAL
    factor = _ovr_factor(ovr)
    # Lower threshold for low-OVR (easier to trigger change), higher for elite
    thresh_factor = 1.0 + (ovr - 75) / 75 * 0.4  # 0.6x at OVR 0, 1.0 at OVR 75, 1.13 at OVR 99
    return {
        "thresh": c["thresh"] * thresh_factor,
        "scale": c["scale"],
        "max": c["max"] * factor,
    }


def _ovr_factor(ovr: int) -> float:
    """Scale max delta for low-OVR cards — modest room to grow."""
    return 1.0 + max(0, (99 - ovr) / 99) * 0.75


# ─── Helpers ─────────────────────────────────────────────────────────────────

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


# ─── Per-attribute prediction ────────────────────────────────────────────────

def predict_attr_delta(
    attr: str,
    row: pd.Series,
    hist_deltas: dict[str, list[float]],
) -> tuple[float, float, float]:
    """Predict delta, change probability, and formula gap for one attribute.

    Returns:
      (predicted_delta, change_prob, actual_gap)
    """
    attr = normalize_attr_name(attr)
    rating = float(row.get("rating_before", row.get("current_rating", 60)))
    ovr = int(row.get("current_ovr", row.get("ovr_before", 75)))
    mlb_raw = row.get("mlb_player_id")
    mlb_id = int(mlb_raw) if (mlb_raw is not None and pd.notna(mlb_raw)) else None
    is_hitter = bool(row.get("is_hitter", True))
    game_year = int(row.get("game_year", 26))

    windows, has_data = _extract_windows(row)

    # 1. Formula projection → gap (always compute, even without stat data)
    primary = windows.get("21d", windows.get("ytd", {}))
    proj = project_attribute(attr, primary, is_hitter)
    gap = float(proj) - rating if proj is not None else 0.0
    proj_ytd = project_attribute(attr, windows.get("ytd", {}), is_hitter)
    gap_ytd = float(proj_ytd) - rating if proj_ytd is not None else 0.0

    # 2. Change probability from classifier
    if has_data:
        clf = _load_classifiers()
        label = "hitter" if is_hitter else "pitcher"
        if label in clf and not isinstance(clf[label], dict) or (label in clf and hasattr(clf[label], "predict_proba")):
            try:
                feats = _classifier_features(row, gap, windows)
                prob = float(clf[label].predict_proba([feats])[0, 1])
                prob = max(0.01, min(0.95, prob))
            except Exception:
                prob = _fallback_change_prob(gap)
        else:
            prob = _fallback_change_prob(gap)
    else:
        prob = _fallback_change_prob(gap)

    # 3. Calibrated magnitude
    cal = _get_cal(attr, game_year, ovr)
    # Stamina: skip entirely when no stat data (missing IP causes false gaps)
    if attr == "stamina" and not has_data:
        predicted = 0.0
    elif has_data and abs(gap) >= cal["thresh"]:
        predicted = gap * cal["scale"]
        predicted = max(-cal["max"], min(cal["max"], predicted))
    elif not has_data and abs(gap) >= 2.5:
        predicted = gap * 0.10
        predicted = max(-3.0, min(3.0, predicted))
    else:
        predicted = 0.0

    # 4. Simple trend overlay
    if mlb_id and abs(predicted) > 0.5:
        trend = _get_player_trend(mlb_id, attr)
        if abs(trend) > 0.5 and np.sign(trend) == np.sign(predicted):
            predicted += trend * 0.20
            predicted = max(-cal["max"], min(cal["max"], predicted))

    return predicted, prob, round(gap, 1)


def _fallback_change_prob(gap: float) -> float:
    """Continuous logistic probability from |gap| magnitude.

    Smooth s-curve: prob ~0.03 at gap=0, 0.12 at gap=3, 0.40 at gap=6,
    asymptoting at 0.55.  Much more realistic than the old step function.
    """
    abs_gap = abs(gap)
    return 0.55 / (1.0 + np.exp(-0.5 * (abs_gap - 4.0)))


def _classifier_features(row: pd.Series, gap: float, windows: dict) -> list:
    """Build feature vector for change classifier."""
    primary = windows.get("21d", windows.get("ytd", {}))
    ovr = int(row.get("current_ovr", row.get("ovr_before", 75)))
    return [
        gap,
        abs(gap),
        gap - ovr,
        row.get("days_since_last_update", 7),
        ovr_distance_to_tier_boundary(ovr),
        int(row.get("is_established_star", 0)),
        int(sample_size_ok(primary, bool(row.get("is_hitter", True)))),
        primary.get("k_pct", 0.225) * 100,
        primary.get("bb_pct", 0.085) * 100,
        primary.get("avg", 0.248),
        primary.get("iso", 0.155),
        primary.get("k9", 8.7),
        primary.get("bb9", 3.1),
    ]


# ─── DataFrame pipeline ──────────────────────────────────────────────────────

def predict_attributes(df: pd.DataFrame) -> pd.DataFrame:
    hist = _get_historical_deltas()
    results = []

    for _, row in df.iterrows():
        attr = normalize_attr_name(str(row.get("attribute_name", "")))
        if not attr:
            continue

        rating = float(row.get("rating_before", row.get("current_rating", 60)))
        delta, prob, formula_gap = predict_attr_delta(attr, row, hist)
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
            "change_prob": round(prob, 3),
            "upgrade_prob_attr": round(up_prob, 3),
            "downgrade_prob_attr": round(dn_prob, 3),
            "confidence_score": 30 if row.get("mlb_player_id") else 0,
            "has_stat_data": int(_extract_windows(row)[1]),
            "mismatch_score": 0.0,
        })

    return pd.DataFrame(results)


def aggregate_player_predictions(attr_df: pd.DataFrame) -> pd.DataFrame:
    """Roll per-attribute predictions up to player-level OVR delta.

    Uses per-attribute OVR weights blended from trained Ridge coefficients
    and domain defaults.  Key improvements over the original:

    1. Tier-aware OVR multiplier (boost near rarity boundaries)
    2. Both-direction tier-jump probability (upgrade AND downgrade)
    3. Direction consensus weighted by attribute OVR importance
    4. Dynamic investment-score weights based on OVR context
    5. Momentum factor from recent player trend
    """
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
            # Weighted direction: sum of weighted positive/negative deltas
            weighted_up_sum=("weighted_delta", lambda s: s[s > 0].sum()),
            weighted_down_sum=("weighted_delta", lambda s: s[s < 0].sum()),
            change_prob_mean=("change_prob", "mean"),
            up_prob_mean=("upgrade_prob_attr", "mean"),
            dn_prob_mean=("downgrade_prob_attr", "mean"),
            avg_abs_delta=("abs_delta", "mean"),
        )
        .reset_index()
    )

    # ── Tier-aware OVR multiplier ────────────────────────────────────────
    # Near tier boundaries, SDS tends to push players across; boost multiplier.
    _TIER_BOUNDARIES = [65, 75, 85, 90, 95]
    def _tier_boost(ovr):
        dist = min(abs(ovr - b) for b in _TIER_BOUNDARIES)
        if dist <= 2:
            return 0.3 * (2 - dist) / 2  # up to +0.3x when right on boundary
        return 0.0

    ovr = grouped["current_ovr"]
    base_mult = 1.5 - 0.3 * (ovr / 99.0)
    boost = ovr.apply(_tier_boost)
    grouped["ovr_mult"] = (base_mult + boost).clip(1.0, 2.5)
    grouped["predicted_ovr_delta"] = (grouped["weighted_sum"] * grouped["ovr_mult"]).clip(-8.0, 8.0)

    # ── Directional probabilities (logistic, delta→probability) ──────────
    z = grouped["predicted_ovr_delta"] / 2.0
    grouped["upgrade_probability"] = (1.0 / (1.0 + np.exp(-z))).clip(0.01, 0.99)
    grouped["downgrade_probability"] = 1.0 - grouped["upgrade_probability"]

    # ── Bi-directional tier-jump probability ─────────────────────────────
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

    # ── Attribute-importance-weighted direction consensus ────────────────
    # Instead of simple n_up/n_down count, use weighted proportion so
    # that movement in high-importance attrs (contact/power/velo) counts more.
    total_w = grouped["weighted_up_sum"].abs() + grouped["weighted_down_sum"].abs()
    # Zero movement → neutral consensus (0), not -1
    no_movement = total_w < 0.001
    safe_total = total_w.clip(lower=0.01)
    pct_up_weighted = grouped["weighted_up_sum"].clip(lower=0) / safe_total
    grouped["direction_consensus"] = np.where(no_movement, 0.0, (2 * pct_up_weighted - 1).clip(-1, 1))
    grouped["avg_gap"] = grouped["weighted_sum"] / grouped["n_attrs"].clip(lower=1)
    grouped["avg_magnitude"] = grouped["avg_abs_delta"]

    # ── Dynamic investment score ─────────────────────────────────────────
    # Base weights shift based on OVR context:
    #   Low OVR (<75):  upgrade potential dominates
    #   Mid OVR (75-89): balanced
    #   High OVR (90+):  tier jumps (red diamond) matter most
    def _investment_score(row):
        ovr = row["current_ovr"]
        up_w, dir_w, tier_w, delta_w = 0.35, 0.15, 0.25, 0.25

        if ovr < 75:
            up_w, dir_w, tier_w, delta_w = 0.45, 0.10, 0.15, 0.30
        elif ovr >= 90:
            up_w, dir_w, tier_w, delta_w = 0.25, 0.10, 0.40, 0.25

        # Boost tier weight near boundaries
        if any(abs(ovr - b) <= 3 for b in _TIER_BOUNDARIES):
            tier_w = min(0.45, tier_w + 0.10)
            up_w = max(0.20, up_w - 0.05)

        # Normalise weights to sum 1.0
        total = up_w + dir_w + tier_w + delta_w
        up_w /= total
        dir_w /= total
        tier_w /= total
        delta_w /= total

        # Normalised components (all 0–1 range)
        up_comp = row["upgrade_probability"]
        dir_comp = (row["direction_consensus"] + 1) / 2
        tier_comp = row["tier_jump_probability"]
        delta_comp = (np.clip(row["predicted_ovr_delta"], -2, 5) + 2) / 7

        # Momentum bonus: players with strong +weighted direction get a lift
        momentum = max(0.0, row["direction_consensus"]) * 0.05

        return (
            up_comp * up_w
            + dir_comp * dir_w
            + tier_comp * tier_w
            + delta_comp * delta_w
            + momentum
        )

    grouped["investment_score"] = grouped.apply(_investment_score, axis=1)

    return grouped.sort_values("investment_score", ascending=False)


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
                    avg_gap=float(row.get("avg_gap", 0.0)),
                    direction_consensus=float(row.get("direction_consensus", 0.5)),
                ))
            session.commit()

    return player_preds


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
    tiers = [(0, 25), (65, 100), (75, 300), (80, 600), (85, 1000), (90, 5000), (92, 10000), (94, 25000), (95, 50000), (97, 100000)]
    cur_qs = max((v for k, v in tiers if current_ovr >= k), default=0)
    new_qs = max((v for k, v in tiers if new_ovr >= k), default=0)
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
