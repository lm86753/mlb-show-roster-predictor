from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from src.config import quicksell_value
from src.db import AttributeChange, CardSnapshot, Prediction, safe_init_db
from src.models.registry import normalize_attr_name

app = FastAPI(
    title="MLB The Show 26 Roster Update Predictor",
    description="Predict Live Series attribute updates from MLB performance stats.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


class RefreshRequest(BaseModel):
    game_year: int = 26


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "static_predictions.json"
# Card art is served from The Show's CDN, which re-renders it after every update.
CARD_ART_URL = "https://cards.theshow.com/mlb26/{uuid}-baked-{size}.webp"


@lru_cache(maxsize=1)
def _static_predictions() -> dict | None:
    if not STATIC_PREDICTIONS_PATH.exists():
        return None
    try:
        return json.loads(STATIC_PREDICTIONS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _model_summary() -> dict:
    from src.models.train import load_summary

    s = load_summary()
    if not s:
        return {}
    return {
        "trained_on": s.get("trained_on", []),
        "metrics": s.get("summary", {}),
        "folds": s.get("folds", []),
        "ovr_weights": s.get("ovr_weights", {}),
        "stat_cutoff_days": s.get("stat_cutoff_days"),
        "last_update_review": s.get("last_update_review", {}),
    }


def _update_status() -> dict:
    try:
        from src.models.predict import update_schedule
        return update_schedule()
    except Exception as exc:
        return {"latest": None, "days_since": None, "days_until": None,
                "next_expected": None, "is_update_today": False, "error": str(exc)}


def _serialize(p: Prediction, snap: CardSnapshot | None = None, with_attrs: bool = True) -> dict:
    out = {
        "card_uuid": p.card_uuid,
        "player_name": p.player_name,
        "mlb_player_id": p.mlb_player_id,
        "current_ovr": p.current_ovr,
        "current_rarity": p.current_rarity,
        "current_qs": quicksell_value(p.current_ovr or 0),
        "predicted_ovr_delta": p.predicted_ovr_delta,
        "ovr_delta_sd": p.ovr_delta_sd,
        "upgrade_probability": p.upgrade_probability or 0.0,
        "downgrade_probability": p.downgrade_probability or 0.0,
        "tier_jump_probability": p.tier_jump_probability or 0.0,
        "tier_down_probability": p.tier_down_probability or 0.0,
        "sample_size_ok": bool(p.sample_size_ok),
        "avg_gap": p.avg_gap,
        "direction_consensus": p.direction_consensus,
        "investment_score": p.investment_score,
        "expected_value_per_card": p.expected_value_per_card,
        "roi_pct": p.roi_pct,
        "created_at": str(p.created_at),
    }
    if snap is not None or with_attrs:
        out.update({
            "team": snap.team if snap else None,
            "position": snap.position if snap else None,
            "is_hitter": snap.is_hitter if snap else None,
        })
    if with_attrs:
        out["attributes"] = json.loads(p.attributes_json or "[]")
        out["stats"] = json.loads(p.stats_json or "{}")
    return out


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/card-image/{card_uuid}")
def get_card_image(card_uuid: str, size: str = Query("sm", pattern="^(sm|lg)$")):
    return RedirectResponse(CARD_ART_URL.format(uuid=card_uuid, size=size), status_code=307)


@app.get("/dashboard")
def get_dashboard():
    Session = safe_init_db()
    if Session is None:
        return _static_predictions() or {"count": 0, "predictions": [], "update_status": {}, "model": _model_summary()}
    with Session() as session:
        predictions = session.query(Prediction).all()
        if not predictions:
            static = _static_predictions()
            if static is not None:
                return static
            return {"count": 0, "predictions": [], "update_status": _update_status(), "model": _model_summary()}
        snaps = {
            s.card_uuid: s
            for s in session.query(CardSnapshot).filter(CardSnapshot.card_uuid.in_([p.card_uuid for p in predictions]))
        }
        result = [_serialize(p, snaps.get(p.card_uuid)) for p in predictions]
    return {
        "count": len(result),
        "predictions": result,
        "update_status": _update_status(),
        "model": _model_summary(),
    }


@app.get("/update-status")
def update_status():
    return _update_status()


@app.get("/predictions")
def get_predictions(
    limit: int = Query(50, ge=1, le=500),
    sort: str = Query("expected_value_per_card", pattern="^(expected_value_per_card|predicted_ovr_delta|upgrade_probability|downgrade_probability)$"),
):
    Session = safe_init_db()
    if Session is None:
        return {"count": 0, "predictions": []}
    with Session() as session:
        rows = session.query(Prediction).order_by(getattr(Prediction, sort).desc()).limit(limit).all()
        out = [_serialize(p, with_attrs=False) for p in rows]
    return {"count": len(out), "predictions": out}


@app.get("/player/{card_uuid}")
def get_player(card_uuid: str):
    Session = safe_init_db()
    if Session is None:
        raise HTTPException(503, "Database not available")
    with Session() as session:
        p = session.query(Prediction).filter_by(card_uuid=card_uuid).first()
        if not p:
            raise HTTPException(404, "Player prediction not found")
        snap = session.query(CardSnapshot).filter_by(card_uuid=card_uuid).first()
        return _serialize(p, snap)


@app.get("/accuracy")
def get_accuracy():
    """Walk-forward backtest results for the current model."""
    return _model_summary()


@app.get("/player-search")
def player_search(q: str = Query(..., min_length=1, description="Player name search term")):
    Session = safe_init_db()
    if Session is None:
        return {"query": q, "count": 0, "results": []}
    with Session() as session:
        players = (
            session.query(Prediction)
            .filter(Prediction.player_name.ilike(f"%{q}%"))
            .order_by(Prediction.expected_value_per_card.desc())
            .limit(50)
            .all()
        )
        results = [_serialize(p, with_attrs=False) for p in players]
    return {"query": q, "count": len(results), "results": results}


@app.get("/player-history/{card_uuid}")
def player_history(card_uuid: str, limit: int = Query(8, ge=1, le=30)):
    """Past attribute changes for a card, grouped by update (most recent first)."""
    Session = safe_init_db()
    if Session is None:
        raise HTTPException(503, "Database not available")
    with Session() as session:
        changes = (
            session.query(AttributeChange)
            .filter(AttributeChange.card_uuid == card_uuid)
            .order_by(AttributeChange.update_date.desc(), AttributeChange.id)
            .all()
        )
    by_update: dict[str, dict] = {}
    for c in changes:
        u = by_update.setdefault(c.update_date, {
            "update_date": c.update_date, "update_name": c.update_name,
            "ovr_before": c.ovr_before, "ovr_after": c.ovr_after, "changes": [],
        })
        u["changes"].append({
            "attribute": normalize_attr_name(c.attribute_name or "", 26),
            "rating_before": c.rating_before, "rating_after": c.rating_after, "delta": c.delta,
        })
    return {"card_uuid": card_uuid, "updates": list(by_update.values())[:limit]}


@app.get("/player-trend/{mlb_id}")
def player_trend(mlb_id: int):
    """Last 5 rating changes per attribute for a player (by MLB ID)."""
    Session = safe_init_db()
    if Session is None:
        raise HTTPException(503, "Database not available")
    with Session() as session:
        changes = (
            session.query(AttributeChange)
            .filter(AttributeChange.mlb_player_id == mlb_id)
            .order_by(AttributeChange.update_date.desc(), AttributeChange.id.desc())
            .all()
        )
    if not changes:
        raise HTTPException(404, f"No historical data for MLB player ID {mlb_id}")
    trend: dict[str, list[dict]] = defaultdict(list)
    for c in changes:
        attr = normalize_attr_name(c.attribute_name or "", c.game_year)
        if len(trend[attr]) < 5:
            trend[attr].append({
                "attribute": attr,
                "rating_before": c.rating_before,
                "rating_after": c.rating_after,
                "delta": c.delta,
                "update_date": c.update_date,
                "update_name": c.update_name,
            })
    return {"mlb_player_id": mlb_id, "player_name": changes[0].player_name, "trend": dict(trend)}


@app.get("/history/changes")
def get_historical_changes(limit: int = Query(100, ge=1, le=1000)):
    Session = safe_init_db()
    if Session is None:
        return {"changes": []}
    with Session() as session:
        changes = session.query(AttributeChange).order_by(AttributeChange.id.desc()).limit(limit).all()
        return {
            "changes": [
                {
                    "player_name": c.player_name,
                    "attribute": normalize_attr_name(c.attribute_name or "", c.game_year),
                    "delta": c.delta,
                    "update": c.update_name,
                    "game_year": c.game_year,
                }
                for c in changes
            ]
        }


@app.post("/refresh/cards")
def refresh_cards(game_year: int = 26):
    from src.ingest.cards import fetch_live_series_cards, link_cards_to_mlb_ids

    stats = fetch_live_series_cards(game_year=game_year)
    linked = link_cards_to_mlb_ids(game_year=game_year)
    return {"cards": stats, "linked": linked}


@app.post("/refresh/predictions")
def refresh_predictions(req: RefreshRequest):
    from src.features.dataset import build_live_dataset
    from src.models.predict import run_predictions

    preds = run_predictions(build_live_dataset(req.game_year))
    return {"scored": len(preds)}


@app.post("/train")
def train_models(game_year: int = 26):
    from src.features.dataset import build_training_dataset
    from src.models.train import train_all

    summary = train_all(build_training_dataset(game_year))
    return {"trained_on": summary["trained_on"], "metrics": summary["summary"]}
