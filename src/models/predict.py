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

from src.config import DB_PATH, MODELS_DIR, ALIAS_MAP
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
        _CLASSIFIERS = joblib.load(path)
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
_DEFAULT_CAL = {"thresh": 2.0, "scale": 0.30, "max": 8.0}

_ATTR_DEFAULTS = {
    "contact_left":          {"thresh": 1.5, "scale": 0.35, "max": 8.0},
    "contact_right":         {"thresh": 1.5, "scale": 0.35, "max": 8.0},
    "power_left":            {"thresh": 1.5, "scale": 0.33, "max": 8.0},
    "power_right":           {"thresh": 1.5, "scale": 0.33, "max": 8.0},
    "plate_vision":          {"thresh": 1.0, "scale": 0.40, "max": 10.0},
    "plate_discipline":      {"thresh": 1.5, "scale": 0.30, "max": 8.0},
    "batting_clutch":        {"thresh": 1.5, "scale": 0.35, "max": 12.0},
    "speed":                 {"thresh": 1.0, "scale": 0.40, "max": 12.0},
    "fielding_ability":      {"thresh": 2.0, "scale": 0.28, "max": 8.0},
    "arm_strength":          {"thresh": 2.0, "scale": 0.28, "max": 8.0},
    "arm_accuracy":          {"thresh": 2.0, "scale": 0.28, "max": 8.0},
    "reaction_time":         {"thresh": 2.0, "scale": 0.28, "max": 8.0},
    "pitch_velocity":        {"thresh": 1.0, "scale": 0.40, "max": 10.0},
    "pitch_control":         {"thresh": 1.5, "scale": 0.35, "max": 10.0},
    "pitch_movement":        {"thresh": 1.5, "scale": 0.33, "max": 10.0},
    "pitching_clutch":       {"thresh": 1.0, "scale": 0.40, "max": 14.0},
    "stamina":               {"thresh": 1.5, "scale": 0.45, "max": 18.0},
    "k_per_9":               {"thresh": 1.0, "scale": 0.35, "max": 10.0},
    "hr_per_9":              {"thresh": 1.0, "scale": 0.35, "max": 10.0},
    "k_per_9_r":             {"thresh": 1.0, "scale": 0.35, "max": 10.0},
    "k_per_9_l":             {"thresh": 1.0, "scale": 0.35, "max": 10.0},
    "h_per_9_r":             {"thresh": 1.0, "scale": 0.40, "max": 12.0},
    "h_per_9":               {"thresh": 1.0, "scale": 0.40, "max": 12.0},
    "bb_per_9":              {"thresh": 1.0, "scale": 0.35, "max": 10.0},
}


def _get_cal(attr: str, game_year: int = 26, ovr: int = 75) -> dict:
    """Get calibration for an attribute, falling back to year→default."""
    cal = _load_calibration()
    year_cal = cal.get(str(game_year), cal.get(game_year, {}))
    if attr in year_cal:
        c = year_cal[attr]
    elif attr in _ATTR_DEFAULTS:
        c = _ATTR_DEFAULTS[attr]
    else:
        c = _DEFAULT_CAL
    return {
        "thresh": c["thresh"],
        "scale": c["scale"],
        "max": c["max"] * _ovr_factor(ovr),
    }


def _ovr_factor(ovr: int) -> float:
    """Scale max delta for low-OVR cards (more room to grow)."""
    return 1.0 + max(0, (99 - ovr) / 99) * 2.0


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
    if has_data and abs(gap) >= cal["thresh"]:
        predicted = gap * cal["scale"]
        predicted = max(-cal["max"], min(cal["max"], predicted))
    elif not has_data and abs(gap) >= 2.0:
        predicted = gap * 0.12
        predicted = max(-5.0, min(5.0, predicted))
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
    """Probability from |gap| magnitude when classifier is unavailable."""
    abs_gap = abs(gap)
    if abs_gap >= 8.0:
        return 0.55
    if abs_gap >= 5.0:
        return 0.40
    if abs_gap >= 3.0:
        return 0.25
    if abs_gap >= 1.5:
        return 0.12
    if abs_gap >= 0.5:
        return 0.05
    return 0.02


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
    """Roll per-attribute predictions up to player-level OVR delta."""
    grouped = (
        attr_df.groupby(["card_uuid", "player_name", "mlb_player_id", "current_ovr", "current_rarity"])
        .agg(
            n_attrs=("predicted_delta", "count"),
            delta_sum=("predicted_delta", "sum"),
            n_up=("predicted_delta", lambda s: (s > 0.1).sum()),
            n_down=("predicted_delta", lambda s: (s < -0.1).sum()),
            change_prob_mean=("change_prob", "mean"),
            up_prob_mean=("upgrade_prob_attr", "mean"),
            dn_prob_mean=("downgrade_prob_attr", "mean"),
        )
        .reset_index()
    )

    mean_delta = grouped["delta_sum"] / grouped["n_attrs"].clip(lower=1)
    ovr_mult = 2.0 - 0.5 * (grouped["current_ovr"] / 99.0)
    grouped["predicted_ovr_delta"] = (mean_delta * ovr_mult).clip(-12.0, 12.0)

    # Convert predicted OVR delta to proper directional probabilities
    # Using a logistic function: delta=0 → p=0.50, delta=+2 → p≈0.73, delta=+3 → p≈0.82, delta=+5 → p≈0.92
    # This makes upgrade + downgrade sum to 1.0 and ties probability directly to signal strength
    z = grouped["predicted_ovr_delta"] / 2.0
    grouped["upgrade_probability"] = (1.0 / (1.0 + np.exp(-z))).clip(0.01, 0.99)
    grouped["downgrade_probability"] = 1.0 - grouped["upgrade_probability"]

    def _tier_jump(row):
        d = row["predicted_ovr_delta"]
        c = row["current_ovr"]
        if d <= 0:
            return 0.0
        for b in [65, 75, 85, 90, 95]:
            if c < b and b - c <= d:
                return min(0.7, 0.2 + (d - (b - c)) * 0.15)
        return 0.0

    grouped["tier_jump_probability"] = grouped.apply(_tier_jump, axis=1)
    grouped["direction_consensus"] = (2 * pct_up - 1).clip(-1, 1)
    grouped["avg_gap"] = grouped["delta_sum"] / grouped["n_attrs"].clip(lower=1)

    grouped["investment_score"] = (
        grouped["upgrade_probability"] * 0.40
        + (grouped["direction_consensus"] + 1) / 2 * 0.15
        + grouped["tier_jump_probability"] * 0.20
        + (grouped["predicted_ovr_delta"].clip(-2, 5) + 2) / 7 * 0.25
    )

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
    tiers = [(0, 25), (65, 100), (75, 300), (85, 1000), (90, 5000), (95, 10000)]
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
