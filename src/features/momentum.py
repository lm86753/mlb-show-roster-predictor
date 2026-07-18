"""Advanced momentum features computed from individual game logs.

This module computes momentum features that capture the *direction* and
*stability* of a player's recent performance, going beyond simple aggregated
window stats. Features include:

- Trend (linear regression slope over last N games)
- Streak (consecutive games improving vs declining)
- Consistency (coefficient of variation)
- Volatility clustering (GARCH-like: are bad games followed by bad games?)

All features are computed from individual game-level data fetched from the
MLB Stats API and cached on disk.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import requests

from src.config import CACHE_DIR, MLB_STATS_API

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stat definitions per player type
# ---------------------------------------------------------------------------

HITTER_GAME_STATS = ["avg", "ops", "k_pct", "iso", "hr", "hits", "ab", "pa"]
PITCHER_GAME_STATS = ["era", "k9", "bb9", "whip", "k_pct", "ip"]

# For trend computation we use these specific stats
HITTER_TREND_STATS = ["avg", "ops", "k_pct"]
PITCHER_TREND_STATS = ["era", "k9", "bb9"]

# Higher-is-better vs lower-is-better (for streak direction)
HIGHER_IS_BETTER = {
    "avg": True, "ops": True, "k_pct": False, "iso": True, "hr": True,
    "hits": True, "ab": True, "pa": True,
    "era": False, "k9": True, "bb9": False, "whip": False, "ip": True,
}


# ---------------------------------------------------------------------------
# Raw game log fetching & caching
# ---------------------------------------------------------------------------

class GameLogFetcher:
    """Fetches and caches individual game logs from the MLB Stats API."""

    def __init__(self, delay: float = 0.15):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "mlb-show-roster-predictor/1.0"})

    def _cache_path(self, player_id: int, season: int, group: str) -> Path:
        return CACHE_DIR / "game_logs_raw" / f"games_{player_id}_{season}_{group}.json"

    def fetch_game_log(
        self, player_id: int, season: int, group: str = "hitting"
    ) -> list[dict]:
        """Fetch individual game logs for a player/season, cached on disk.

        Returns a list of dicts, each with 'date' and 'stat' keys, sorted
        chronologically.
        """
        cache_path = self._cache_path(player_id, season, group)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(raw, list) and raw:
                    return raw
            except (json.JSONDecodeError, OSError):
                cache_path.unlink(missing_ok=True)

        try:
            data = self._get(
                f"/people/{player_id}/stats",
                params={"stats": "gameLog", "group": group, "season": season},
            )
        except Exception as exc:
            logger.warning("Failed to fetch game log for %s/%s: %s", player_id, season, exc)
            return []

        games = self._extract_game_log(data)
        games.sort(key=lambda g: g["date"])

        try:
            cache_path.write_text(json.dumps(games, default=str), encoding="utf-8")
        except OSError:
            pass

        return games

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{MLB_STATS_API}{path}"
        resp = self.session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        time.sleep(self.delay)
        return resp.json()

    @staticmethod
    def _extract_game_log(data: dict) -> list[dict]:
        games = []
        for block in data.get("stats", []):
            for split in block.get("splits", []):
                stat = split.get("stat", {})
                date_str = split.get("date", "")[:10]
                if date_str:
                    games.append({"date": date_str, "stat": stat})
        return games


# ---------------------------------------------------------------------------
# Per-game stat derivation
# ---------------------------------------------------------------------------

def _parse_ip(ip_str: str) -> float:
    """Convert innings pitched string (e.g. '6.1') to float outs/3."""
    ip_str = str(ip_str or "0")
    if "." in ip_str:
        whole, frac = ip_str.split(".", 1)
        return int(whole) + int(frac) / 3
    return float(ip_str)


def derive_hitter_game_stats(stat: dict) -> dict:
    """Derive per-game hitter stats from raw API stat block."""
    ab = int(stat.get("atBats", 0))
    h = int(stat.get("hits", 0))
    bb = int(stat.get("baseOnBalls", 0))
    so = int(stat.get("strikeOuts", 0))
    hr = int(stat.get("homeRuns", 0))
    hbp = int(stat.get("hitByPitch", 0))
    sf = int(stat.get("sacFlies", 0))
    tb = int(stat.get("totalBases", 0))
    abb = int(stat.get("atBats", 0))
    pa = ab + bb + hbp + sf

    avg = h / ab if ab else 0.0
    obp = (h + bb + hbp) / pa if pa else 0.0
    slg = tb / abb if abb else 0.0
    ops = obp + slg
    k_pct = so / pa if pa else 0.0
    iso = slg - avg

    return {
        "avg": avg, "ops": ops, "k_pct": k_pct, "iso": iso,
        "hr": hr, "hits": h, "ab": ab, "pa": pa,
    }


def derive_pitcher_game_stats(stat: dict) -> dict:
    """Derive per-game pitcher stats from raw API stat block."""
    ip = _parse_ip(stat.get("inningsPitched", "0"))
    bb = int(stat.get("baseOnBalls", 0))
    so = int(stat.get("strikeOuts", 0))
    hr = int(stat.get("homeRuns", 0))
    h = int(stat.get("hits", 0))
    er = int(stat.get("earnedRuns", 0))
    bf = int(stat.get("battersFaced", 0))

    era = er * 9 / ip if ip else 0.0
    k9 = so * 9 / ip if ip else 0.0
    bb9 = bb * 9 / ip if ip else 0.0
    whip = (bb + h) / ip if ip else 0.0
    k_pct = so / bf if bf else 0.0

    return {
        "era": era, "k9": k9, "bb9": bb9, "whip": whip,
        "k_pct": k_pct, "ip": ip,
    }


# ---------------------------------------------------------------------------
# MomentumComputer
# ---------------------------------------------------------------------------

class MomentumComputer:
    """Computes advanced momentum features from gamelog data.

    Usage::

        computer = MomentumComputer()
        features = computer.compute_all(player_id=12345, season=2026, is_hitter=True)
        # Returns dict of momentum features

    The class caches raw game logs on disk and derived per-game stat arrays
    in-memory for the lifetime of the instance.
    """

    def __init__(
        self,
        fetcher: GameLogFetcher | None = None,
        trend_window: int = 20,
        streak_window: int = 10,
        consistency_window: int = 30,
    ):
        self.fetcher = fetcher or GameLogFetcher()
        self.trend_window = trend_window
        self.streak_window = streak_window
        self.consistency_window = consistency_window
        # In-memory cache: (player_id, season, is_hitter) -> stat arrays dict
        self._stat_cache: dict[tuple, dict[str, np.ndarray]] = {}
        self._games_cache: dict[tuple, list[dict]] = {}

    # ---- public API ----

    def compute_all(
        self,
        player_id: int,
        season: int,
        is_hitter: bool,
    ) -> dict[str, float]:
        """Compute all momentum features for a player.

        Returns a flat dict of feature_name -> float value.
        """
        games = self._get_games(player_id, season, is_hitter)
        if not games or len(games) < 2:
            return self._empty_features(is_hitter)

        stat_arrays = self._get_stat_arrays(player_id, season, is_hitter, games)

        features: dict[str, float] = {}

        # 1) Trend (linear regression slope)
        trend = self.compute_trend(games, stat_arrays, is_hitter)
        features.update(trend)

        # 2) Streak
        streak = self.compute_streak(stat_arrays, is_hitter)
        features.update(streak)

        # 3) Consistency
        consistency = self.compute_consistency(stat_arrays, is_hitter)
        features.update(consistency)

        # 4) Volatility clustering
        vol = self.compute_volatility_clustering(stat_arrays, is_hitter)
        features.update(vol)

        return features

    def compute_trend(
        self,
        games: list[dict],
        stat_arrays: dict[str, np.ndarray] | None = None,
        is_hitter: bool | None = None,
    ) -> dict[str, float]:
        """Linear regression slope over the last ``trend_window`` games.

        For hitters: AVG, OPS, K%
        For pitchers: ERA, K/9, BB/9

        Returns dict like ``{"trend_avg": 0.012, "trend_ops": -0.005, ...}``.
        """
        if stat_arrays is None:
            stat_arrays = {}
        trend_stats = HITTER_TREND_STATS if is_hitter else PITCHER_TREND_STATS
        n = self.trend_window

        result: dict[str, float] = {}
        for stat_name in trend_stats:
            arr = stat_arrays.get(stat_name)
            if arr is None or len(arr) < 3:
                result[f"trend_{stat_name}"] = 0.0
                continue
            recent = arr[-n:]
            slope = _linear_slope(recent)
            result[f"trend_{stat_name}"] = float(slope)

        return result

    def compute_streak(
        self,
        stat_arrays: dict[str, np.ndarray],
        is_hitter: bool,
        window: int | None = None,
    ) -> dict[str, float]:
        """Count consecutive games where the primary stat improved vs declined.

        Uses OPS for hitters, ERA for pitchers as the primary streak stat.
        Returns positive count for improving streak, negative for declining.

        Also returns per-stat streaks for the trend stats.
        """
        window = window or self.streak_window
        primary_stat = "ops" if is_hitter else "era"
        trend_stats = HITTER_TREND_STATS if is_hitter else PITCHER_TREND_STATS

        result: dict[str, float] = {}

        # Primary streak
        arr = stat_arrays.get(primary_stat)
        if arr is None or len(arr) < 2:
            result["streak_primary"] = 0.0
        else:
            recent = arr[-window:]
            result["streak_primary"] = float(_compute_streak_count(recent, primary_stat))

        # Per-trend-stat streaks
        for stat_name in trend_stats:
            arr = stat_arrays.get(stat_name)
            if arr is None or len(arr) < 2:
                result[f"streak_{stat_name}"] = 0.0
            else:
                recent = arr[-window:]
                result[f"streak_{stat_name}"] = float(_compute_streak_count(recent, stat_name))

        return result

    def compute_consistency(
        self,
        stat_arrays: dict[str, np.ndarray],
        is_hitter: bool,
    ) -> dict[str, float]:
        """Coefficient of variation (std / mean) over last ``consistency_window`` games.

        Lower values = more consistent. Returns NaN-safe floats (0.0 when
        mean is 0).
        """
        trend_stats = HITTER_TREND_STATS if is_hitter else PITCHER_TREND_STATS
        n = self.consistency_window

        result: dict[str, float] = {}
        for stat_name in trend_stats:
            arr = stat_arrays.get(stat_name)
            if arr is None or len(arr) < 3:
                result[f"consistency_{stat_name}"] = 0.0
                continue
            recent = arr[-n:]
            mean = float(np.mean(recent))
            std = float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0
            if abs(mean) < 1e-9:
                result[f"consistency_{stat_name}"] = 0.0
            else:
                result[f"consistency_{stat_name}"] = std / abs(mean)

        return result

    def compute_volatility_clustering(
        self,
        stat_arrays: dict[str, np.ndarray],
        is_hitter: bool,
    ) -> dict[str, float]:
        """GARCH-like measure: are bad games followed by bad games?

        Computes the autocorrelation of squared deviations from a rolling mean.
        A high positive value means volatility clusters (bad games follow bad
        games, hot streaks follow hot streaks). Near 0 means random walk.

        Returns dict with ``vol_cluster_primary`` and per-stat measures.
        """
        primary_stat = "ops" if is_hitter else "era"
        trend_stats = HITTER_TREND_STATS if is_hitter else PITCHER_TREND_STATS

        result: dict[str, float] = {}

        # Primary
        arr = stat_arrays.get(primary_stat)
        if arr is None or len(arr) < 5:
            result["vol_cluster_primary"] = 0.0
        else:
            result["vol_cluster_primary"] = float(_volatility_clustering(arr))

        for stat_name in trend_stats:
            arr = stat_arrays.get(stat_name)
            if arr is None or len(arr) < 5:
                result[f"vol_cluster_{stat_name}"] = 0.0
            else:
                result[f"vol_cluster_{stat_name}"] = float(_volatility_clustering(arr))

        return result

    # ---- internal helpers ----

    def _get_games(
        self, player_id: int, season: int, is_hitter: bool
    ) -> list[dict]:
        key = (player_id, season, is_hitter)
        if key not in self._games_cache:
            group = "hitting" if is_hitter else "pitching"
            self._games_cache[key] = self.fetcher.fetch_game_log(player_id, season, group)
        return self._games_cache[key]

    def _get_stat_arrays(
        self,
        player_id: int,
        season: int,
        is_hitter: bool,
        games: list[dict],
    ) -> dict[str, np.ndarray]:
        key = (player_id, season, is_hitter)
        if key not in self._stat_cache:
            stat_arrays: dict[str, list[float]] = defaultdict(list)
            derive = derive_hitter_game_stats if is_hitter else derive_pitcher_game_stats
            for game in games:
                stats = derive(game["stat"])
                for k, v in stats.items():
                    stat_arrays[k].append(float(v))
            self._stat_cache[key] = {
                k: np.array(v, dtype=np.float64) for k, v in stat_arrays.items()
            }
        return self._stat_cache[key]

    def _empty_features(self, is_hitter: bool) -> dict[str, float]:
        """Return zero-filled features when insufficient data."""
        trend_stats = HITTER_TREND_STATS if is_hitter else PITCHER_TREND_STATS
        primary = "ops" if is_hitter else "era"
        features: dict[str, float] = {}
        for s in trend_stats:
            features[f"trend_{s}"] = 0.0
            features[f"streak_{s}"] = 0.0
            features[f"consistency_{s}"] = 0.0
            features[f"vol_cluster_{s}"] = 0.0
        features["streak_primary"] = 0.0
        features["vol_cluster_primary"] = 0.0
        return features


# ---------------------------------------------------------------------------
# Pure math helpers (no I/O)
# ---------------------------------------------------------------------------

def _linear_slope(values: np.ndarray) -> float:
    """Simple linear regression slope via least squares.

    values is a 1-D array; x = 0, 1, ..., n-1.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = values.mean()
    num = np.sum((x - x_mean) * (values - y_mean))
    den = np.sum((x - x_mean) ** 2)
    if abs(den) < 1e-12:
        return 0.0
    return float(num / den)


def _compute_streak_count(values: np.ndarray, stat_name: str) -> int:
    """Count consecutive games improving vs declining.

    Returns positive int for improving streak, negative for declining.
    """
    if len(values) < 2:
        return 0
    higher_better = HIGHER_IS_BETTER.get(stat_name, True)
    deltas = np.diff(values)
    if not higher_better:
        deltas = -deltas  # flip so positive = improvement

    streak = 0
    direction = 0
    for d in reversed(deltas):
        if d > 1e-9:
            if direction == -1:
                break
            direction = 1
            streak += 1
        elif d < -1e-9:
            if direction == 1:
                break
            direction = -1
            streak += 1
        # d == 0: neutral, doesn't break streak but doesn't extend it

    return streak * direction


def _volatility_clustering(values: np.ndarray, window: int = 10) -> float:
    """GARCH(1,1)-inspired volatility clustering measure.

    Computes the lag-1 autocorrelation of squared deviations from a rolling
    mean. High positive value → volatility clusters.

    Parameters
    ----------
    values : 1-D array of per-game stat values.
    window : rolling window size for local mean estimation.

    Returns
    -------
    float : autocorrelation of squared deviations in [-1, 1].
    """
    n = len(values)
    if n < window + 2:
        # Fall back to global mean
        residuals = values - np.mean(values)
    else:
        # Rolling mean via convolution
        cumsum = np.cumsum(np.insert(values, 0, 0))
        rolling_mean = (cumsum[window:] - cumsum[:-window]) / window
        residuals = values[window - 1 :] - rolling_mean

    squared = residuals ** 2
    if len(squared) < 3:
        return 0.0

    # Lag-1 autocorrelation
    sq_mean = squared.mean()
    if abs(sq_mean) < 1e-18:
        return 0.0
    autocov = np.mean((squared[1:] - sq_mean) * (squared[:-1] - sq_mean))
    var = np.var(squared)
    if abs(var) < 1e-18:
        return 0.0
    return float(np.clip(autocov / var, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Batch computation & DB storage
# ---------------------------------------------------------------------------

def compute_momentum_for_player(
    player_id: int,
    season: int,
    is_hitter: bool,
    computer: MomentumComputer | None = None,
) -> dict[str, float]:
    """Convenience: compute all momentum features for a single player."""
    computer = computer or MomentumComputer()
    return computer.compute_all(player_id, season, is_hitter)


def store_momentum_in_db(
    momentum_features: dict[str, float],
    player_id: int,
    as_of_date: str,
    is_hitter: bool,
    session_factory=None,
) -> None:
    """Store momentum features in the player_stat_windows table.

    Uses window='momentum' to distinguish from regular stat windows.
    """
    from src.db import PlayerStatWindow, init_db, dumps

    if session_factory is None:
        session_factory = init_db()

    with session_factory() as session:
        existing = (
            session.query(PlayerStatWindow)
            .filter_by(
                mlb_player_id=player_id,
                as_of_date=as_of_date,
                window="momentum",
            )
            .first()
        )
        if existing:
            existing.stats_json = dumps(momentum_features)
        else:
            session.add(
                PlayerStatWindow(
                    mlb_player_id=player_id,
                    as_of_date=as_of_date,
                    window="momentum",
                    is_hitter=int(is_hitter),
                    stats_json=dumps(momentum_features),
                )
            )
        session.commit()
