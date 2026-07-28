"""Lightweight Vercel serverless entry point.

Serves precomputed predictions from data/static_predictions.json without
importing the ML stack. Keeps the function bundle small enough for Vercel's
225 MB limit.
"""
import json
import os
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route
from starlette.responses import JSONResponse, Response

_ROOT = Path(__file__).resolve().parent.parent
os.environ["VERCEL"] = "1"

_STATIC_PREDICTIONS_PATH = _ROOT / "data" / "static_predictions.json"
_CARD_IMG_DIRS = [
    _ROOT / "data" / "card_images_real",
    _ROOT / "data" / "card_images",
]


def _load_static_predictions() -> dict:
    if _STATIC_PREDICTIONS_PATH.exists():
        try:
            data = json.loads(_STATIC_PREDICTIONS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"count": 0, "predictions": [], "update_status": {"latest": None, "days_since": None, "days_until": None}}


_STATIC_CACHE = _load_static_predictions()


async def health(request):
    return JSONResponse({"status": "ok"})


async def dashboard(request):
    horizon_days = int(request.query_params.get("horizon_days", 1))
    result = dict(_STATIC_CACHE)
    result["update_status"] = {"latest": None, "days_since": None, "days_until": None}
    return JSONResponse(result)


async def predictions(request):
    horizon_days = int(request.query_params.get("horizon_days", 1))
    limit = int(request.query_params.get("limit", 50))
    min_upgrade_prob = float(request.query_params.get("min_upgrade_prob", 0.0))
    preds = _STATIC_CACHE.get("predictions", [])
    filtered = [p for p in preds if p.get("upgrade_probability", 0) >= min_upgrade_prob]
    filtered = sorted(filtered, key=lambda p: p.get("upgrade_probability", 0), reverse=True)[:limit]
    return JSONResponse({"count": len(filtered), "predictions": filtered})


async def player_search(request):
    q = request.query_params.get("q", "")
    preds = _STATIC_CACHE.get("predictions", [])
    results = [p for p in preds if q.lower() in p.get("player_name", "").lower()][:50]
    return JSONResponse({"query": q, "count": len(results), "results": results})


async def player(request):
    card_uuid = request.path_params.get("card_uuid", "")
    horizon_days = int(request.query_params.get("horizon_days", 1))
    preds = _STATIC_CACHE.get("predictions", [])
    for p in preds:
        if p.get("card_uuid") == card_uuid:
            return JSONResponse(p)
    return JSONResponse({"detail": "Player prediction not found"}, status_code=404)


async def accuracy(request):
    return JSONResponse({"metrics": []})


async def update_status(request):
    return JSONResponse({
        "is_update_today": False,
        "latest_update_date": None,
        "days_since_last_update": None,
        "next_expected_update": None,
        "days_until_next_update": None,
    })


async def card_image(request):
    card_uuid = request.path_params.get("card_uuid", "")
    for d in _CARD_IMG_DIRS:
        if d.exists():
            f = d / f"{card_uuid}.png"
            if f.exists():
                return Response(f.read_bytes(), media_type="image/png")
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="400" height="560">
      <rect width="400" height="560" fill="#1a1a2e" rx="12"/>
      <text x="200" y="280" text-anchor="middle" fill="#666" font-family="sans-serif" font-size="24">No Image</text>
    </svg>"""
    return Response(content=svg, media_type="image/svg+xml")


async def not_found(request):
    return JSONResponse({"detail": "Not found"}, status_code=404)


routes = [
    Route("/api/health", health),
    Route("/api/dashboard", dashboard),
    Route("/api/predictions", predictions),
    Route("/api/player-search", player_search),
    Route("/api/player/{card_uuid}", player),
    Route("/api/accuracy", accuracy),
    Route("/api/update-status", update_status),
    Route("/api/card-image/{card_uuid}", card_image),
]

app = Starlette(
    routes=routes,
    middleware=[
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
    ],
    default=not_found,
)
