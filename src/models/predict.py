"""
Live scoring for the attribute-update model (see ``train.py``).

Per attribute: stacked delta, 80% interval, P(up) / P(down).
Per card: OVR-change distribution → upgrade/downgrade/tier probabilities and
expected quicksell change, all from the same calibrated distribution.
"""

from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd

from src.config import CORE_ATTRS
from src.db import AttributeChange, Prediction, dumps, init_db
from src.models.train import MODEL_PATH, score_players, score_rows

logger = logging.getLogger(__name__)

_model: dict | None = None
_model_lock = threading.Lock()


def load_model() -> dict:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not MODEL_PATH.exists():
                    raise FileNotFoundError(f"No trained model at {MODEL_PATH}. Run scripts/train.py first.")
                _model = joblib.load(MODEL_PATH)
    return _model


def predict_live(live: pd.DataFrame, model: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score live rows. Returns (attribute_predictions, player_predictions)."""
    model = model or load_model()
    if live.empty:
        return pd.DataFrame(), pd.DataFrame()

    scored = []
    for group in CORE_ATTRS:
        rows = live[live["group"] == group]
        if rows.empty:
            continue
        sig = model["groups"][group].predict(rows)
        scored.append(score_rows(rows, sig, model["calibration"][group]))
    attrs = pd.concat(scored)

    players = score_players(attrs, model["calibration"]).reset_index()
    meta = live.groupby("card_uuid")[["player_name", "mlb_player_id", "team", "position", "rarity", "is_pitcher_card"]].first()
    stat_n = np.where(live["group"] == "hitting", live.get("ytd_pa", 0), live.get("ytd_bf", 0))
    has_stats = pd.Series(stat_n, index=live.index).fillna(0).groupby(live["card_uuid"]).max() >= 40
    gaps = attrs.groupby("card_uuid")["gap"].mean()
    players = players.join(meta, on="card_uuid").join(has_stats.rename("sample_size_ok"), on="card_uuid")
    players = players.join(gaps.rename("avg_gap"), on="card_uuid")
    players["direction_consensus"] = players["upgrade_probability"] - players["downgrade_probability"]
    players["roi_pct"] = 100 * players["expected_qs_change"] / players["current_qs"].clip(lower=1)
    return attrs, players.sort_values("expected_qs_change", ascending=False)


# Real MLB stats shown beside each card (season to date and the last 30 days).
DISPLAY_STATS = {
    "hitting": ["pa", "avg", "obp", "slg", "iso", "k_pct", "bb_pct", "hr_pct"],
    "pitching": ["bf", "era", "whip", "k_pct", "bb_pct", "hr_pct", "ip_per_g"],
}


def stat_records(live: pd.DataFrame) -> dict[str, dict]:
    """card_uuid -> {"group", "season": {...}, "last30": {...}} from the card's own stat group."""
    out: dict[str, dict] = {}
    # to_dict, not itertuples: itertuples renames columns that start with a digit ("30d_pa").
    for row in live.drop_duplicates(["card_uuid", "group"]).to_dict("records"):
        group = "pitching" if row["is_pitcher_card"] else "hitting"
        if row["group"] != group:
            continue

        def pick(window: str) -> dict:
            vals = {k: row.get(f"{window}_{k}") for k in DISPLAY_STATS[group]}
            return {k: (None if v is None or pd.isna(v) else round(float(v), 4)) for k, v in vals.items()}

        out[row["card_uuid"]] = {"group": group, "season": pick("ytd"), "last30": pick("30d"), "last_season": pick("prev")}
    return out


def attribute_records(attrs: pd.DataFrame, card_uuid: str) -> list[dict]:
    sub = attrs[attrs["card_uuid"] == card_uuid]
    out = []
    for r in sub.itertuples():
        delta = float(r.ens)
        out.append({
            "attribute_name": r.attr,
            "rating_before": int(r.rating_before),
            "projected_rating": int(round(r.rating_before + delta)),
            "stat_projection": None if pd.isna(r.proj) else round(float(r.proj), 1),
            "predicted_delta": round(delta, 2),
            "gap": None if pd.isna(r.gap) else round(float(r.gap), 1),
            "confidence_low": round(float(r.lo), 1),
            "confidence_high": round(float(r.hi), 1),
            "upgrade_prob_attr": round(float(r.p_up), 3),
            "downgrade_prob_attr": round(float(r.p_down), 3),
            "change_prob": round(float(r.p_up + r.p_down), 3),
            "last_delta": int(r.last_delta or 0),
        })
    return out


def run_predictions(live: pd.DataFrame, persist: bool = True) -> pd.DataFrame:
    attrs, players = predict_live(live)
    stats = stat_records(live)
    if persist and not players.empty:
        Session = init_db()
        with Session() as session:
            session.query(Prediction).delete()
            for p in players.itertuples():
                session.add(Prediction(
                    card_uuid=p.card_uuid,
                    player_name=p.player_name,
                    mlb_player_id=int(p.mlb_player_id) if pd.notna(p.mlb_player_id) else None,
                    current_ovr=int(p.ovr_before),
                    current_rarity=p.rarity,
                    predicted_ovr_delta=round(float(p.mu), 3),
                    ovr_delta_sd=round(float(p.sd), 3),
                    upgrade_probability=round(float(p.upgrade_probability), 4),
                    downgrade_probability=round(float(p.downgrade_probability), 4),
                    tier_jump_probability=round(float(p.tier_up_probability), 4),
                    tier_down_probability=round(float(p.tier_down_probability), 4),
                    gold_probability=round(float(p.gold_probability), 4),
                    diamond_probability=round(float(p.diamond_probability), 4),
                    ovr_move_probs_json=dumps([round(float(x), 4) for x in p.move_probs]),
                    sample_size_ok=int(bool(p.sample_size_ok)),
                    horizon_days=1,
                    attributes_json=dumps(attribute_records(attrs, p.card_uuid)),
                    stats_json=dumps(stats.get(p.card_uuid, {})),
                    avg_gap=None if pd.isna(p.avg_gap) else round(float(p.avg_gap), 2),
                    direction_consensus=round(float(p.direction_consensus), 4),
                    investment_score=round(100 * float(p.direction_consensus), 1),
                    expected_value_per_card=round(float(p.expected_qs_change), 1),
                    roi_pct=round(float(p.roi_pct), 1),
                ))
            session.commit()
    return players


def update_schedule(today: date | None = None) -> dict:
    """Last/next monthly attribute update, from the observed cadence."""
    from src.config import MAJOR_UPDATE_MIN_CARDS
    from sqlalchemy import func

    today = today or datetime.utcnow().date()
    Session = init_db()
    with Session() as session:
        rows = (
            session.query(AttributeChange.update_date, func.count(func.distinct(AttributeChange.card_uuid)))
            .filter(AttributeChange.update_date.isnot(None))
            .group_by(AttributeChange.update_date)
            .all()
        )
    all_dates = sorted(datetime.strptime(d[:10], "%Y-%m-%d").date() for d, _ in rows if d)
    majors = sorted(
        datetime.strptime(d[:10], "%Y-%m-%d").date() for d, n in rows if d and n >= MAJOR_UPDATE_MIN_CARDS
    )
    status = {
        "latest": str(all_dates[-1]) if all_dates else None,
        "days_since": (today - all_dates[-1]).days if all_dates else None,
        "is_update_today": bool(all_dates and all_dates[-1] == today),
        "last_attribute_update": str(majors[-1]) if majors else None,
        "next_expected": None,
        "days_until": None,
    }
    if majors:
        gaps = [(b - a).days for a, b in zip(majors, majors[1:]) if (b.year == a.year)]
        cadence = int(np.median(gaps)) if gaps else 35
        nxt = majors[-1] + timedelta(days=cadence)
        while nxt < today:
            nxt += timedelta(days=cadence)
        status.update({"next_expected": str(nxt), "days_until": (nxt - today).days, "cadence_days": cadence})
    return status
