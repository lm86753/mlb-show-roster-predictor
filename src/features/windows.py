"""Point-in-time stat windows built from per-game logs.

Training rows and live rows both go through ``windows_as_of`` so the model
sees identically-constructed features at fit and predict time. Only games
strictly before ``as_of`` are used, so historical rows never see the future.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

from src.config import CACHE_DIR
from src.features.momentum import GameLogFetcher, _parse_ip
from src.ingest.mlb_stats import MLBStatsClient

logger = logging.getLogger(__name__)

WINDOW_DAYS = {"7d": 7, "14d": 14, "30d": 30}

# Sample size at which a window's rate is trusted ~50% vs. the prior season
# when building the regressed "stab" window (classic stabilization points).
HIT_STAB_PA = 150
PIT_STAB_BF = 170

HIT_RATES = ["avg", "obp", "slg", "iso", "k_pct", "bb_pct", "hr_pct"]
PIT_RATES = ["k_pct", "bb_pct", "hr_pct", "h_pct", "whip", "era", "ip_per_g"]


def _log_path(mlb_id: int, season: int, group: str) -> Path:
    return CACHE_DIR / "game_logs_raw" / f"games_{mlb_id}_{season}_{group}.json"


def load_game_log(mlb_id: int, season: int, group: str, max_age_hours: float | None = None) -> list[dict]:
    """Cached game log; refetched when older than ``max_age_hours`` (current season only).

    The cache is only replaced after a successful fetch, so a network failure
    falls back to the last good copy.
    """
    path = _log_path(mlb_id, season, group)
    cached = None
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = None
        stale = (
            max_age_hours is not None
            and season >= date.today().year
            and (time.time() - path.stat().st_mtime) > max_age_hours * 3600
        )
        if cached is not None and not stale:
            return cached

    fetcher = GameLogFetcher(delay=0.0)
    try:
        data = fetcher._get(
            f"/people/{mlb_id}/stats", params={"stats": "gameLog", "group": group, "season": season}
        )
    except Exception as exc:
        logger.warning("Game log fetch failed for %s/%s/%s: %s", mlb_id, season, group, exc)
        return cached or []
    games = sorted(fetcher._extract_game_log(data), key=lambda g: g["date"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(games), encoding="utf-8")
    return games


def prefetch_game_logs(
    keys: list[tuple[int, int, str]],
    max_age_hours: float | None = None,
    workers: int = 8,
) -> int:
    """Fetch many (mlb_id, season, group) logs concurrently. Returns count."""
    keys = list(dict.fromkeys(keys))
    done = 0

    def _one(k):
        try:
            load_game_log(*k, max_age_hours=max_age_hours)
        except Exception as exc:  # network hiccups shouldn't kill a batch
            logger.warning("game log %s failed: %s", k, exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(_one, keys):
            done += 1
            if done % 250 == 0:
                logger.info("  game logs %d/%d", done, len(keys))
    return done


# ── Aggregation ─────────────────────────────────────────────────────────────

def _sum(games: list[dict], key: str) -> float:
    return float(sum(float(g["stat"].get(key, 0) or 0) for g in games))


def aggregate_hitting(games: list[dict]) -> dict:
    ab, h, bb = _sum(games, "atBats"), _sum(games, "hits"), _sum(games, "baseOnBalls")
    so, hr, tb = _sum(games, "strikeOuts"), _sum(games, "homeRuns"), _sum(games, "totalBases")
    hbp, sf = _sum(games, "hitByPitch"), _sum(games, "sacFlies")
    pa = _sum(games, "plateAppearances") or (ab + bb + hbp + sf)
    return _hit_rates(len(games), pa, ab, h, bb, so, hr, tb, hbp, sf)


def _hit_rates(g, pa, ab, h, bb, so, hr, tb, hbp, sf) -> dict:
    avg = h / ab if ab else None
    slg = tb / ab if ab else None
    obp_den = ab + bb + hbp + sf
    return {
        "g": g, "pa": pa,
        "avg": avg,
        "obp": (h + bb + hbp) / obp_den if obp_den else None,
        "slg": slg,
        "iso": (slg - avg) if ab else None,
        "k_pct": so / pa if pa else None,
        "bb_pct": bb / pa if pa else None,
        "hr_pct": hr / pa if pa else None,
    }


def aggregate_pitching(games: list[dict]) -> dict:
    ip = sum(_parse_ip(g["stat"].get("inningsPitched", "0")) for g in games)
    bf, so, bb = _sum(games, "battersFaced"), _sum(games, "strikeOuts"), _sum(games, "baseOnBalls")
    hr, h, er = _sum(games, "homeRuns"), _sum(games, "hits"), _sum(games, "earnedRuns")
    gs = _sum(games, "gamesStarted")
    return _pit_rates(len(games), ip, bf, so, bb, hr, h, er, gs)


def _pit_rates(g, ip, bf, so, bb, hr, h, er, gs) -> dict:
    return {
        "g": g, "bf": bf, "ip": ip, "gs_share": gs / g if g else None,
        "k_pct": so / bf if bf else None,
        "bb_pct": bb / bf if bf else None,
        "hr_pct": hr / bf if bf else None,
        "h_pct": h / bf if bf else None,
        "whip": (bb + h) / ip if ip else None,
        "era": er * 9 / ip if ip else None,
        "ip_per_g": ip / g if g else None,
    }


def season_totals_to_window(stats: dict, group: str) -> dict:
    """Convert an MLB season stat block into the same shape as a window."""
    if not stats:
        return {}
    f = lambda k: float(stats.get(k, 0) or 0)  # noqa: E731
    if group == "hitting":
        return _hit_rates(
            f("gamesPlayed"), f("plateAppearances"), f("atBats"), f("hits"), f("baseOnBalls"),
            f("strikeOuts"), f("homeRuns"), f("totalBases"), f("hitByPitch"), f("sacFlies"),
        )
    return _pit_rates(
        f("gamesPlayed") or f("gamesPitched"), _parse_ip(stats.get("inningsPitched", "0")),
        f("battersFaced"), f("strikeOuts"), f("baseOnBalls"), f("homeRuns"), f("hits"),
        f("earnedRuns"), f("gamesStarted"),
    )


_season_client: MLBStatsClient | None = None


def prior_season_window(mlb_id: int, season: int, group: str) -> dict:
    """Previous full season's stats (cached on disk by MLBStatsClient)."""
    global _season_client
    _season_client = _season_client or MLBStatsClient(delay=0.0)
    getter = (
        _season_client.get_season_hitting_stats if group == "hitting"
        else _season_client.get_season_pitching_stats
    )
    try:
        return season_totals_to_window(getter(mlb_id, season - 1), group)
    except Exception:
        return {}


def _stabilize(ytd: dict, prev: dict, group: str) -> dict:
    """Season-to-date rates regressed toward last season by sample size."""
    n_key, k, rates = ("pa", HIT_STAB_PA, HIT_RATES) if group == "hitting" else ("bf", PIT_STAB_BF, PIT_RATES)
    n = ytd.get(n_key) or 0
    out = {n_key: n}
    for r in rates:
        cur, base = ytd.get(r), prev.get(r)
        if cur is None:
            out[r] = base
        elif base is None:
            out[r] = cur
        else:
            out[r] = (cur * n + base * k) / (n + k)
    return out


def windows_as_of(games: list[dict], as_of: str | date, group: str, prev: dict | None = None) -> dict[str, dict]:
    """Rolling windows using only games played before ``as_of``."""
    as_of_d = as_of if isinstance(as_of, date) else datetime.strptime(str(as_of)[:10], "%Y-%m-%d").date()
    agg = aggregate_hitting if group == "hitting" else aggregate_pitching
    past = [g for g in games if g.get("date") and g["date"] < as_of_d.isoformat()]
    out = {}
    for name, days in WINDOW_DAYS.items():
        start = (as_of_d - timedelta(days=days)).isoformat()
        out[name] = agg([g for g in past if g["date"] >= start])
    out["ytd"] = agg(past)
    out["prev"] = prev or {}
    out["stab"] = _stabilize(out["ytd"], out["prev"], group)
    return out
