from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.db import AttributeChange, CardSnapshot, ModelMetrics, Prediction, init_db
from src.features.engineering import build_live_features
from src.ingest.cards import fetch_live_series_cards, link_cards_to_mlb_ids
from src.models.evaluate import run_backtest
from src.models.predict import is_roster_update_today, run_predictions
from src.models.train import train_all

app = FastAPI(
    title="MLB The Show 26 Roster Update Predictor",
    description="Predict Live Series rating changes from MLB performance stats.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RefreshRequest(BaseModel):
    game_year: int = 26
    horizon_days: int = 1


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CARD_IMG_DIRS = [
    PROJECT_ROOT / "data" / "card_images_real",   # clean CDN images (preferred)
    PROJECT_ROOT / "data" / "card_images",          # generated fallback
]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/card-image/{card_uuid}")
def get_card_image(card_uuid: str):
    for d in CARD_IMG_DIRS:
        if d.exists():
            f = d / f"{card_uuid}.png"
            if f.exists():
                from fastapi.responses import FileResponse
                return FileResponse(str(f), media_type="image/png")
    raise HTTPException(404, "Card image not found")


@app.get("/dashboard")
def get_dashboard(horizon_days: int = Query(1, ge=1, le=30)):
    Session = init_db()
    with Session() as session:
        predictions = (
            session.query(Prediction)
            .filter(Prediction.horizon_days == horizon_days)
            .all()
        )
        card_uuids = [p.card_uuid for p in predictions]
        snapshots = (
            session.query(CardSnapshot)
            .filter(CardSnapshot.card_uuid.in_(card_uuids))
            .all()
        )
        snap_map = {s.card_uuid: s for s in snapshots}

        result = []
        for p in predictions:
            s = snap_map.get(p.card_uuid)
            attrs = json.loads(p.attributes_json or "[]")
            has_img = any(
                (d / f"{p.card_uuid}.png").exists()
                for d in CARD_IMG_DIRS if d.exists()
            )
            result.append({
                "card_uuid": p.card_uuid,
                "player_name": p.player_name,
                "mlb_player_id": p.mlb_player_id,
                "current_ovr": p.current_ovr,
                "current_rarity": p.current_rarity,
                "predicted_ovr_delta": p.predicted_ovr_delta,
                "upgrade_probability": p.upgrade_probability,
                "downgrade_probability": p.downgrade_probability,
                "tier_jump_probability": p.tier_jump_probability,
                "sample_size_ok": bool(p.sample_size_ok),
                "avg_gap": p.avg_gap,
                "direction_consensus": p.direction_consensus,
                "team": s.team if s else None,
                "position": s.position if s else None,
                "is_hitter": s.is_hitter if s else None,
                "has_card_image": has_img,
                "attributes": attrs,
                "created_at": str(p.created_at),
            })

        # Update status
        status = {"latest": None, "days_since": None, "days_until": None}
        try:
            row = session.execute(
                "SELECT update_date FROM attribute_changes WHERE update_date IS NOT NULL ORDER BY update_date DESC LIMIT 1"
            ).fetchone()
            if row:
                latest = datetime.strptime(str(row[0]), "%Y-%m-%d").date()
                today = datetime.utcnow().date()
                days_since = (today - latest).days
                next_update = latest + timedelta(days=14)
                days_until = (next_update - today).days
                status = {
                    "latest": str(latest),
                    "days_since": days_since,
                    "days_until": days_until,
                    "next_expected": str(next_update),
                    "is_update_today": days_since == 0,
                }
        except Exception:
            pass

    return {"count": len(result), "predictions": result, "update_status": status}


@app.get("/update-status")
def update_status():
    """
    Returns whether today has a roster update based on the most recent
    attribute_changes.update_date. The dashboard uses this to show
    'UPDATE TODAY' or 'No Update Today'.
    """
    return is_roster_update_today()


@app.get("/predictions")
def get_predictions(
    horizon_days: int = Query(1, ge=1, le=30),
    limit: int = Query(50, ge=1, le=500),
    min_upgrade_prob: float = Query(0.0, ge=0.0, le=1.0),
):
    Session = init_db()
    with Session() as session:
        q = (
            session.query(Prediction)
            .filter(Prediction.horizon_days == horizon_days)
            .filter(Prediction.upgrade_probability >= min_upgrade_prob)
            .order_by(Prediction.upgrade_probability.desc())
            .limit(limit)
        )
        rows = [
            {
                "player_name": p.player_name,
                "card_uuid": p.card_uuid,
                "current_ovr": p.current_ovr,
                "current_rarity": p.current_rarity,
                "predicted_ovr_delta": p.predicted_ovr_delta,
                "upgrade_probability": p.upgrade_probability,
                "downgrade_probability": p.downgrade_probability,
                "tier_jump_probability": p.tier_jump_probability,
                "sample_size_ok": bool(p.sample_size_ok),
                "avg_gap": p.avg_gap,
                "direction_consensus": p.direction_consensus,
                "created_at": str(p.created_at),
            }
            for p in q.all()
        ]
    return {"count": len(rows), "predictions": rows}


@app.get("/player/{card_uuid}")
def get_player(card_uuid: str, horizon_days: int = 1):
    Session = init_db()
    with Session() as session:
        p = (
            session.query(Prediction)
            .filter_by(card_uuid=card_uuid, horizon_days=horizon_days)
            .first()
        )
        if not p:
            raise HTTPException(404, "Player prediction not found")
        attrs = json.loads(p.attributes_json or "[]")
        return {
            "player_name": p.player_name,
            "card_uuid": p.card_uuid,
            "current_ovr": p.current_ovr,
            "current_rarity": p.current_rarity,
            "predicted_ovr_delta": p.predicted_ovr_delta,
            "upgrade_probability": p.upgrade_probability,
            "downgrade_probability": p.downgrade_probability,
            "tier_jump_probability": p.tier_jump_probability,
            "avg_gap": p.avg_gap,
            "direction_consensus": p.direction_consensus,
            "attributes": attrs,
        }


@app.get("/accuracy")
def get_accuracy():
    Session = init_db()
    with Session() as session:
        metrics = session.query(ModelMetrics).order_by(ModelMetrics.created_at.desc()).limit(50).all()
        return {
            "metrics": [
                {
                    "model": m.model_name,
                    "metric": m.metric_name,
                    "value": m.metric_value,
                    "fold": m.fold,
                }
                for m in metrics
            ]
        }


@app.get("/player-search")
def player_search(q: str = Query(..., min_length=1, description="Player name search term")):
    """
    Search for players by name. Returns matching players with their
    latest prediction data (T-1 horizon by default).
    """
    Session = init_db()
    with Session() as session:
        # Find distinct players matching the name
        players = (
            session.query(Prediction)
            .filter(Prediction.player_name.ilike(f"%{q}%"))
            .filter(Prediction.horizon_days == 1)
            .order_by(Prediction.upgrade_probability.desc())
            .limit(50)
            .all()
        )
        results = [
            {
                "player_name": p.player_name,
                "card_uuid": p.card_uuid,
                "mlb_player_id": p.mlb_player_id,
                "current_ovr": p.current_ovr,
                "current_rarity": p.current_rarity,
                "predicted_ovr_delta": p.predicted_ovr_delta,
                "upgrade_probability": p.upgrade_probability,
                "downgrade_probability": p.downgrade_probability,
                "tier_jump_probability": p.tier_jump_probability,
                "avg_gap": p.avg_gap,
                "direction_consensus": p.direction_consensus,
                "created_at": str(p.created_at),
            }
            for p in players
        ]
    return {"query": q, "count": len(results), "results": results}


@app.get("/player-trend/{mlb_id}")
def player_trend(mlb_id: int):
    """
    Return the last 5 rating changes per attribute for a player (by MLB ID).
    Returns a dict keyed with attribute_name -> list of recent changes
    (most recent first), each with rating_before, rating_after, delta, update_date.
    """
    Session = init_db()
    with Session() as session:
        # Get all attribute changes for this player, ordered most recent first
        changes = (
            session.query(AttributeChange)
            .filter(AttributeChange.mlb_player_id == mlb_id)
            .order_by(AttributeChange.update_date.desc(), AttributeChange.id.desc())
            .all()
        )

        if not changes:
            raise HTTPException(404, f"No historical data for MLB player ID {mlb_id}")

        # Group by attribute, keep last 5 per attribute
        from collections import defaultdict
        trend: dict[str, list[dict]] = defaultdict(list)
        player_name = changes[0].player_name
        for c in changes:
            attr = c.attribute_name
            if len(trend[attr]) < 5:
                trend[attr].append({
                    "attribute": attr,
                    "rating_before": c.rating_before,
                    "rating_after": c.rating_after,
                    "delta": c.delta,
                    "update_date": c.update_date,
                    "update_name": c.update_name,
                })

    return {
        "mlb_player_id": mlb_id,
        "player_name": player_name,
        "trend": dict(trend),
    }


@app.get("/history/changes")
def get_historical_changes(limit: int = 100):
    Session = init_db()
    with Session() as session:
        changes = (
            session.query(AttributeChange)
            .order_by(AttributeChange.id.desc())
            .limit(limit)
            .all()
        )
        return {
            "changes": [
                {
                    "player_name": c.player_name,
                    "attribute": c.attribute_name,
                    "delta": c.delta,
                    "update": c.update_name,
                    "game_year": c.game_year,
                }
                for c in changes
            ]
        }


@app.post("/refresh/cards")
def refresh_cards(game_year: int = 26):
    stats = fetch_live_series_cards(game_year=game_year)
    linked = link_cards_to_mlb_ids(game_year=game_year)
    return {"cards": stats, "linked": linked}


@app.post("/refresh/predictions")
def refresh_predictions(req: RefreshRequest):
    live_df = build_live_features(game_year=req.game_year, horizon_days=req.horizon_days)
    preds = run_predictions(live_df, horizon_days=req.horizon_days)
    return {"scored": len(preds)}


@app.post("/train")
def train_models():
    result = train_all()
    backtest = run_backtest()
    return {"training": result, "backtest": backtest}
