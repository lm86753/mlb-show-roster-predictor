"""Lightweight Vercel serverless entry point.

Serves precomputed predictions from data/static_predictions.json without
any third-party dependencies. Keeps the function bundle small enough for
Vercel's 225 MB limit.
"""
import json
import os
import urllib.parse
from pathlib import Path

os.environ["VERCEL"] = "1"

_ROOT = Path(__file__).resolve().parent.parent
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


def _json_response(body: dict, status: int = 200) -> dict:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": payload.decode("utf-8"),
        "isBase64Encoded": False,
    }


def _cors_response(status_code: int, body) -> dict:
    headers = {
        "content-type": "application/json",
        "access-control-allow-origin": "*",
        "access-control-allow-methods": "GET, OPTIONS",
        "access-control-allow-headers": "*",
    }
    if isinstance(body, str):
        payload = body
    else:
        payload = json.dumps(body, ensure_ascii=False)
    return {
        "statusCode": status_code,
        "headers": headers,
        "body": payload,
        "isBase64Encoded": False,
    }


async def _not_found(scope, receive, send):
    resp = _cors_response(404, {"detail": "Not found"})
    await send({
        "type": "http.response.start",
        "status": resp["statusCode"],
        "headers": [(k.lower().encode("utf-8"), v.encode("utf-8")) for k, v in resp["headers"].items()],
    })
    await send({
        "type": "http.response.body",
        "body": resp["body"].encode("utf-8"),
        "more_body": False,
    })


class Router:
    def __init__(self) -> None:
        self.routes: list[tuple[str, str, callable]] = []

    def add_route(self, path: str, method: str, handler: callable) -> None:
        self.routes.append((path, method, handler))

    async def __call__(self, scope, receive, send):
        method = (scope.get("method") or "GET").upper()
        path = urllib.parse.unquote(scope.get("path", "/") or "/")

        if method == "OPTIONS":
            resp = _cors_response(204, "")
            await send({
                "type": "http.response.start",
                "status": resp["statusCode"],
                "headers": [(k.lower().encode("utf-8"), v.encode("utf-8")) for k, v in resp["headers"].items()],
            })
            await send({
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            })
            return

        for route_path, route_method, handler in self.routes:
            if route_method != method:
                continue
            params = _match_path(route_path, path)
            if params is not None:
                request = {"path_params": params, "query_params": _parse_qs(scope.get("query_string", b""))}
                await handler(request, receive, send)
                return

        await _not_found(scope, receive, send)


def _parse_qs(qs: bytes) -> dict:
    if not qs:
        return {}
    return dict(urllib.parse.parse_qsl(qs.decode("utf-8"), keep_blank_values=True))


def _match_path(route: str, actual: str) -> dict | None:
    route_parts = [p for p in route.split("/") if p]
    actual_parts = [p for p in actual.split("/") if p]
    if len(route_parts) != len(actual_parts):
        return None
    params = {}
    for r, a in zip(route_parts, actual_parts):
        if r.startswith("{") and r.endswith("}"):
            params[r[1:-1]] = urllib.parse.unquote(a)
        elif r != a:
            return None
    return params


router = Router()


async def _send_json(handler, request, receive, send, status=200):
    resp = _cors_response(status, handler(request))
    await send({
        "type": "http.response.start",
        "status": resp["statusCode"],
        "headers": [(k.lower().encode("utf-8"), v.encode("utf-8")) for k, v in resp["headers"].items()],
    })
    await send({
        "type": "http.response.body",
        "body": resp["body"].encode("utf-8"),
        "more_body": False,
    })


def _handler_health(request, receive, send):
    async def wrapped(request, receive, send):
        await _send_json(lambda r: {"status": "ok"}, request, receive, send)
    return wrapped


def _handler_dashboard(request, receive, send):
    async def wrapped(request, receive, send):
        result = dict(_STATIC_CACHE)
        result["update_status"] = {"latest": None, "days_since": None, "days_until": None}
        await _send_json(lambda r: result, request, receive, send)
    return wrapped


def _handler_predictions(request, receive, send):
    async def wrapped(request, receive, send):
        limit = int(request.get("query_params", {}).get("limit", 50))
        min_upgrade_prob = float(request.get("query_params", {}).get("min_upgrade_prob", 0.0))
        preds = _STATIC_CACHE.get("predictions", [])
        filtered = [p for p in preds if p.get("upgrade_probability", 0) >= min_upgrade_prob]
        filtered = sorted(filtered, key=lambda p: p.get("upgrade_probability", 0), reverse=True)[:limit]
        await _send_json(lambda r: {"count": len(filtered), "predictions": filtered}, request, receive, send)
    return wrapped


def _handler_player_search(request, receive, send):
    async def wrapped(request, receive, send):
        q = request.get("query_params", {}).get("q", "")
        preds = _STATIC_CACHE.get("predictions", [])
        results = [p for p in preds if q.lower() in p.get("player_name", "").lower()][:50]
        await _send_json(lambda r: {"query": q, "count": len(results), "results": results}, request, receive, send)
    return wrapped


def _handler_player(request, receive, send):
    async def wrapped(request, receive, send):
        card_uuid = request.get("path_params", {}).get("card_uuid", "")
        preds = _STATIC_CACHE.get("predictions", [])
        for p in preds:
            if p.get("card_uuid") == card_uuid:
                await _send_json(lambda r: p, request, receive, send)
                return
        resp = _cors_response(404, {"detail": "Player prediction not found"})
        await send({
            "type": "http.response.start",
            "status": resp["statusCode"],
            "headers": [(k.lower().encode("utf-8"), v.encode("utf-8")) for k, v in resp["headers"].items()],
        })
        await send({
            "type": "http.response.body",
            "body": resp["body"].encode("utf-8"),
            "more_body": False,
        })
    return wrapped


def _handler_accuracy(request, receive, send):
    async def wrapped(request, receive, send):
        await _send_json(lambda r: {"metrics": []}, request, receive, send)
    return wrapped


def _handler_update_status(request, receive, send):
    async def wrapped(request, receive, send):
        await _send_json(lambda r: {
            "is_update_today": False,
            "latest_update_date": None,
            "days_since_last_update": None,
            "next_expected_update": None,
            "days_until_next_update": None,
        }, request, receive, send)
    return wrapped


def _handler_card_image(request, receive, send):
    async def wrapped(request, receive, send):
        card_uuid = request.get("path_params", {}).get("card_uuid", "")
        for d in _CARD_IMG_DIRS:
            if d.exists():
                f = d / f"{card_uuid}.png"
                if f.exists():
                    headers = {
                        "content-type": "image/png",
                        "access-control-allow-origin": "*",
                        "access-control-allow-methods": "GET, OPTIONS",
                        "access-control-allow-headers": "*",
                    }
                    await send({
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(k.lower().encode("utf-8"), v.encode("utf-8")) for k, v in headers.items()],
                    })
                    await send({
                        "type": "http.response.body",
                        "body": f.read_bytes(),
                        "more_body": False,
                    })
                    return
        svg = """<svg xmlns="http://www.w3.org/2000/svg" width="400" height="560">
          <rect width="400" height="560" fill="#1a1a2e" rx="12"/>
          <text x="200" y="280" text-anchor="middle" fill="#666" font-family="sans-serif" font-size="24">No Image</text>
        </svg>"""
        headers = {
            "content-type": "image/svg+xml",
            "access-control-allow-origin": "*",
            "access-control-allow-methods": "GET, OPTIONS",
            "access-control-allow-headers": "*",
        }
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(k.lower().encode("utf-8"), v.encode("utf-8")) for k, v in headers.items()],
        })
        await send({
            "type": "http.response.body",
            "body": svg.encode("utf-8"),
            "more_body": False,
        })
    return wrapped


router.add_route("/api/health", "GET", _handler_health)
router.add_route("/api/dashboard", "GET", _handler_dashboard)
router.add_route("/api/predictions", "GET", _handler_predictions)
router.add_route("/api/player-search", "GET", _handler_player_search)
router.add_route("/api/player/{card_uuid}", "GET", _handler_player)
router.add_route("/api/accuracy", "GET", _handler_accuracy)
router.add_route("/api/update-status", "GET", _handler_update_status)
router.add_route("/api/card-image/{card_uuid}", "GET", _handler_card_image)

app = router
