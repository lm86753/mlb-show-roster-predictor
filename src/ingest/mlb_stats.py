from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import CACHE_DIR, MLB_STATS_API
from src.db import PlayerStatWindow, init_db, dumps
from src.formulas.ratings import LEAGUE_AVG


class MLBStatsClient:
    def __init__(self, delay: float = 0.2):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "mlb-show-roster-predictor/1.0"})

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{MLB_STATS_API}{path}"
        resp = self.session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        time.sleep(self.delay)
        return resp.json()

    def search_player(self, name: str) -> int | None:
        cache_path = CACHE_DIR / "player_ids" / f"{name.lower().replace(' ', '_')}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                return data.get("mlb_id")
            except (json.JSONDecodeError, OSError):
                cache_path.unlink(missing_ok=True)

        try:
            data = self._get("/people/search", params={"names": name})
        except Exception:
            cache_path.write_text(json.dumps({"mlb_id": None}), encoding="utf-8")
            return None
        people = data.get("people", [])
        if not people:
            cache_path.write_text(json.dumps({"mlb_id": None}), encoding="utf-8")
            return None

        mlb_id = people[0]["id"]
        cache_path.write_text(json.dumps({"mlb_id": mlb_id, "name": people[0].get("fullName")}), encoding="utf-8")
        return mlb_id

    def get_season_hitting_stats(self, player_id: int, season: int) -> dict:
        cache_path = CACHE_DIR / "stats" / f"hit_{player_id}_{season}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cache_path.unlink(missing_ok=True)

        try:
            data = self._get(
                f"/people/{player_id}/stats",
                params={"stats": "season", "group": "hitting", "season": season},
            )
        except Exception:
            return {}
        stats = _extract_stat_block(data)
        cache_path.write_text(json.dumps(stats), encoding="utf-8")
        return stats

    def get_season_pitching_stats(self, player_id: int, season: int) -> dict:
        cache_path = CACHE_DIR / "stats" / f"pit_{player_id}_{season}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cache_path.unlink(missing_ok=True)

        try:
            data = self._get(
                f"/people/{player_id}/stats",
                params={"stats": "season", "group": "pitching", "season": season},
            )
        except Exception:
            return {}
        stats = _extract_stat_block(data)
        cache_path.write_text(json.dumps(stats), encoding="utf-8")
        return stats

    def get_game_log_stats(
        self, player_id: int, season: int, group: str, end_date: str
    ) -> dict:
        """Aggregate stats from gameLog up to end_date for rolling windows."""
        cache_key = f"gamelog_{player_id}_{season}_{group}_{end_date}"
        cache_path = CACHE_DIR / "gamelogs" / f"{cache_key}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        data = self._get(
            f"/people/{player_id}/stats",
            params={
                "stats": "gameLog",
                "group": group,
                "season": season,
            },
        )
        games = _extract_game_log(data)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        windows = {}
        for label, days in [("5d", 5), ("21d", 21), ("ytd", 9999)]:
            start_dt = end_dt - timedelta(days=days - 1) if days < 9999 else datetime(season, 1, 1)
            subset = [
                g for g in games
                if start_dt <= datetime.strptime(g["date"], "%Y-%m-%d") <= end_dt
            ]
            windows[label] = _aggregate_games(subset, group)

        cache_path.write_text(json.dumps(windows), encoding="utf-8")
        return windows


def _extract_stat_block(data: dict) -> dict:
    for block in data.get("stats", []):
        splits = block.get("splits", [])
        if splits:
            return splits[0].get("stat", {})
    return {}


def _extract_game_log(data: dict) -> list[dict]:
    games = []
    for block in data.get("stats", []):
        for split in block.get("splits", []):
            stat = split.get("stat", {})
            date_str = split.get("date", "")[:10]
            if date_str:
                games.append({"date": date_str, "stat": stat})
    return games


def _aggregate_games(games: list[dict], group: str) -> dict:
    if not games:
        return {"games": 0}

    if group == "hitting":
        ab = sum(int(g["stat"].get("atBats", 0)) for g in games)
        h = sum(int(g["stat"].get("hits", 0)) for g in games)
        bb = sum(int(g["stat"].get("baseOnBalls", 0)) for g in games)
        so = sum(int(g["stat"].get("strikeOuts", 0)) for g in games)
        hr = sum(int(g["stat"].get("homeRuns", 0)) for g in games)
        pa = ab + bb + sum(int(g["stat"].get("hitByPitch", 0)) for g in games) + sum(
            int(g["stat"].get("sacFlies", 0)) for g in games
        )
        tb = sum(int(g["stat"].get("totalBases", 0)) for g in games)
        return {
            "games": len(games),
            "ab": ab,
            "pa": pa,
            "hits": h,
            "bb": bb,
            "so": so,
            "hr": hr,
            "avg": h / ab if ab else 0.0,
            "k_pct": so / pa if pa else 0.0,
            "bb_pct": bb / pa if pa else 0.0,
            "iso": (tb - h) / ab if ab else 0.0,
        }

    def _parse_ip(ip_str: str) -> float:
        ip_str = str(ip_str or "0")
        if "." in ip_str:
            whole, frac = ip_str.split(".", 1)
            return int(whole) + int(frac) / 3
        return float(ip_str)

    ip = sum(_parse_ip(g["stat"].get("inningsPitched", "0")) for g in games)
    bb = sum(int(g["stat"].get("baseOnBalls", 0)) for g in games)
    so = sum(int(g["stat"].get("strikeOuts", 0)) for g in games)
    hr = sum(int(g["stat"].get("homeRuns", 0)) for g in games)
    bf = sum(int(g["stat"].get("battersFaced", 0)) for g in games)
    return {
        "games": len(games),
        "ip": ip,
        "bb": bb,
        "so": so,
        "hr": hr,
        "bf": bf,
        "k_pct": so / bf if bf else 0.0,
        "bb_pct": bb / bf if bf else 0.0,
        "k9": so * 9 / ip if ip else 0.0,
        "bb9": bb * 9 / ip if ip else 0.0,
        "hr9": hr * 9 / ip if ip else 0.0,
    }


def fetch_pybaseball_season_stats(season: int, is_hitter: bool = True) -> pd.DataFrame:
    """Fetch season-level FanGraphs stats via pybaseball with disk cache."""
    cache_path = CACHE_DIR / "pybaseball" / f"{'bat' if is_hitter else 'pit'}_{season}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        import pybaseball as pyb

        pyb.cache.enable()
        if is_hitter:
            df = pyb.batting_stats(season, qual=1)
        else:
            df = pyb.pitching_stats(season, qual=1)
        df.to_parquet(cache_path)
        return df
    except Exception:
        return pd.DataFrame()


def _season_stats_to_window(stats: dict, is_hitter: bool) -> dict:
    """Convert MLB season stat block to feature window format."""
    if not stats:
        return {k: LEAGUE_AVG.get(k, 0) for k in ("k_pct", "bb_pct", "avg", "iso", "ab", "ip", "k9", "bb9", "hr9", "gs")}
    if is_hitter:
        ab = int(stats.get("atBats", 0))
        pa = int(stats.get("plateAppearances", 0))
        h = int(stats.get("hits", 0))
        bb = int(stats.get("baseOnBalls", 0))
        so = int(stats.get("strikeOuts", 0))
        slg = float(stats.get("sluggingPercentage", 0) or 0)
        avg = float(stats.get("avg", stats.get("battingAverage", 0)) or 0)
        return {
            "ab": ab,
            "pa": pa,
            "avg": h / ab if ab else avg,
            "k_pct": so / pa if pa else LEAGUE_AVG["k_pct"],
            "bb_pct": bb / pa if pa else LEAGUE_AVG["bb_pct"],
            "iso": slg - avg if slg and avg else LEAGUE_AVG["iso"],
            "hr": int(stats.get("homeRuns", 0)),
        }
    ip = float(stats.get("inningsPitched", 0) or 0)
    so = int(stats.get("strikeOuts", 0))
    bb = int(stats.get("baseOnBalls", 0))
    hr = int(stats.get("homeRuns", 0))
    bf = int(stats.get("battersFaced", 0) or 0) or max(so + bb, 1)
    gs = int(stats.get("gamesStarted", 0) or 0)
    return {
        "ip": ip,
        "bf": bf,
        "k_pct": so / bf if bf else LEAGUE_AVG["k_pct"],
        "bb_pct": bb / bf if bf else LEAGUE_AVG["bb_pct"],
        "k9": so * 9 / ip if ip else LEAGUE_AVG["k9"],
        "bb9": bb * 9 / ip if ip else LEAGUE_AVG["bb9"],
        "hr9": hr * 9 / ip if ip else LEAGUE_AVG["hr9"],
        "gs": gs,
    }


def _load_statcast_cache(season: int) -> dict:
    """Load/cache Statcast data for a season. Returns dict keyed by mlbam_id."""
    cache_path = CACHE_DIR / "statcast" / f"statcast_{season}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    result: dict = {"pitchers": {}, "batters": {}, "sprint": {}}
    try:
        import pybaseball as pyb

        pitch_ars = pyb.statcast_pitcher_pitch_arsenal(season, minP=1)
        for _, row in pitch_ars.iterrows():
            pid = int(row["pitcher"])
            result["pitchers"][str(pid)] = {
                "fb_velo": float(row.get("ff_avg_speed") or 0) if pd.notna(row.get("ff_avg_speed")) else None,
            }

        batters_ev = pyb.statcast_batter_exitvelo_barrels(season)
        for _, row in batters_ev.iterrows():
            pid = int(row["player_id"])
            result["batters"][str(pid)] = {
                "exit_velo": float(row.get("avg_hit_speed") or 0) if pd.notna(row.get("avg_hit_speed")) else None,
            }

        sprint = pyb.statcast_sprint_speed(season)
        for _, row in sprint.iterrows():
            pid = int(row["player_id"])
            result["sprint"][str(pid)] = {
                "sprint_speed": float(row.get("sprint_speed") or 0) if pd.notna(row.get("sprint_speed")) else None,
            }
    except Exception:
        pass

    cache_path.write_text(json.dumps(result), encoding="utf-8")
    return result


def _merge_statcast(
    windows: dict[str, dict],
    mlb_id: int,
    is_hitter: bool,
    season: int,
) -> dict[str, dict]:
    """Enrich stat windows with Statcast data (fb_velo, exit_velo, sprint_speed)."""
    sc = _load_statcast_cache(season)
    sid = str(mlb_id)

    for window_key in list(windows.keys()):
        w = windows[window_key]
        # Pitcher: add fb_velo from four-seam avg speed
        if not is_hitter and sid in sc.get("pitchers", {}):
            fb = sc["pitchers"][sid].get("fb_velo")
            if fb is not None and fb > 0:
                w["fb_velo"] = fb
        # Batter: add exit_velo from avg hit speed
        if is_hitter and sid in sc.get("batters", {}):
            ev = sc["batters"][sid].get("exit_velo")
            if ev is not None and ev > 0:
                w["exit_velo"] = ev
        # Both: add sprint_speed
        if sid in sc.get("sprint", {}):
            sp = sc["sprint"][sid].get("sprint_speed")
            if sp is not None and sp > 0:
                w["sprint_speed"] = sp

    return windows


def build_live_stat_windows(
    mlb_player_id: int,
    season: int,
    is_hitter: bool,
    client: MLBStatsClient | None = None,
) -> dict[str, dict]:
    """Actual rolling 5-day and 21-day windows from game log data (cached).

    Computes real 5-day and 21-day rolling windows from the game log
    instead of duplicating season stats across all windows.
    Falls back to season-stats-based windows only when no game log data
    exists for the player.
    """
    from datetime import datetime, timedelta

    cache_path = CACHE_DIR / "live_windows" / f"{mlb_player_id}_{season}_{'hit' if is_hitter else 'pit'}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    client = client or MLBStatsClient()
    group = "hitting" if is_hitter else "pitching"

    # Use yesterday's date as the end of the live window
    end_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Attempt game-log-based rolling windows
    try:
        game_log_windows = get_game_log_stats(mlb_player_id, season, group, end_date)
    except Exception:
        game_log_windows = {"5d": {"games": 0}, "21d": {"games": 0}, "ytd": {"games": 0}}

    # Determine which windows actually have games logged
    has_5d = game_log_windows.get("5d", {}).get("games", 0) > 0
    has_21d = game_log_windows.get("21d", {}).get("games", 0) > 0
    has_ytd = game_log_windows.get("ytd", {}).get("games", 0) > 0

    # Pre-compute season stat windows as fallback
    if is_hitter:
        ytd_stats = client.get_season_hitting_stats(mlb_player_id, season)
        yr_stats = [client.get_season_hitting_stats(mlb_player_id, yr) for yr in range(season - 2, season + 1)]
    else:
        ytd_stats = client.get_season_pitching_stats(mlb_player_id, season)
        yr_stats = [client.get_season_pitching_stats(mlb_player_id, yr) for yr in range(season - 2, season + 1)]

    season_window = _season_stats_to_window(ytd_stats, is_hitter)
    three_yr = _blend_three_year({str(season - 2 + i): s for i, s in enumerate(yr_stats) if s}, is_hitter)

    # Build windows: prefer game-log rolling data, fall back to season stats
    if has_5d:
        window_5d = game_log_windows["5d"]
    else:
        window_5d = season_window

    if has_21d:
        window_21d = game_log_windows["21d"]
    else:
        window_21d = season_window

    if has_ytd:
        window_ytd = game_log_windows["ytd"]
    else:
        window_ytd = season_window

    windows = {
        "5d": window_5d,
        "21d": window_21d,
        "ytd": window_ytd,
        "3yr": three_yr or season_window,
    }
    windows = _merge_statcast(windows, mlb_player_id, is_hitter, season)
    cache_path.write_text(json.dumps(windows), encoding="utf-8")
    return windows


def build_player_stat_windows(
    mlb_player_id: int,
    as_of_date: str,
    is_hitter: bool,
    season: int | None = None,
) -> dict[str, dict]:
    client = MLBStatsClient()
    season = season or int(as_of_date[:4])
    group = "hitting" if is_hitter else "pitching"

    windows = client.get_game_log_stats(mlb_player_id, season, group, as_of_date)

    # 3-year baseline from season stats
    three_yr = {}
    for yr in range(season - 2, season + 1):
        if is_hitter:
            s = client.get_season_hitting_stats(mlb_player_id, yr)
        else:
            s = client.get_season_pitching_stats(mlb_player_id, yr)
        if s:
            three_yr[str(yr)] = s
    windows["3yr"] = _blend_three_year(three_yr, is_hitter)
    windows = _merge_statcast(windows, mlb_player_id, is_hitter, season)
    return windows


def _blend_three_year(yearly: dict[str, dict], is_hitter: bool) -> dict:
    if not yearly:
        return {}
    keys = list(yearly.values())
    if is_hitter:
        ab = sum(int(s.get("atBats", 0)) for s in keys)
        if ab == 0:
            return keys[-1] if keys else {}
        h = sum(int(s.get("hits", 0)) for s in keys)
        bb = sum(int(s.get("baseOnBalls", 0)) for s in keys)
        so = sum(int(s.get("strikeOuts", 0)) for s in keys)
        pa = sum(int(s.get("plateAppearances", 0)) for s in keys)
        hr = sum(int(s.get("homeRuns", 0)) for s in keys)
        return {
            "ab": ab,
            "pa": pa,
            "avg": h / ab if ab else 0,
            "k_pct": so / pa if pa else 0,
            "bb_pct": bb / pa if pa else 0,
            "hr": hr,
            "iso": sum(float(s.get("sluggingPercentage", 0)) - float(s.get("avg", s.get("battingAverage", 0))) for s in keys) / len(keys),
        }
    ip = sum(float(s.get("inningsPitched", 0)) for s in keys)
    so = sum(int(s.get("strikeOuts", 0)) for s in keys)
    bb = sum(int(s.get("baseOnBalls", 0)) for s in keys)
    hr = sum(int(s.get("homeRuns", 0)) for s in keys)
    bf = sum(int(s.get("battersFaced", 0) or 0) for s in keys) or max(so + bb, 1)
    gs = sum(int(s.get("gamesStarted", 0) or 0) for s in keys)
    return {
        "ip": ip,
        "k_pct": so / bf if bf else 0,
        "bb_pct": bb / bf if bf else 0,
        "k9": so * 9 / ip if ip else 0,
        "bb9": bb * 9 / ip if ip else 0,
        "hr9": hr * 9 / ip if ip else 0,
        "gs": gs,
    }


def store_stat_windows_for_updates(game_years: list[int] | None = None) -> int:
    """Join MLB stats to historical attribute changes at each update date."""
    from src.db import AttributeChange

    Session = init_db()
    client = MLBStatsClient()
    count = 0

    with Session() as session:
        q = (
            session.query(
                AttributeChange.game_year,
                AttributeChange.update_id,
                AttributeChange.update_date,
            )
            .filter(AttributeChange.update_date.isnot(None))
            .distinct()
        )
        if game_years:
            q = q.filter(AttributeChange.game_year.in_(game_years))
        updates = q.all()

        for game_year, update_id, update_date in updates:
            if not update_date:
                continue
            players = (
                session.query(AttributeChange)
                .filter_by(game_year=game_year, update_id=update_id)
                .all()
            )
            seen: set[str] = set()
            for row in players:
                if row.player_name in seen:
                    continue
                seen.add(row.player_name)

                mlb_id = row.mlb_player_id or client.search_player(row.player_name)
                if not mlb_id:
                    continue
                if not row.mlb_player_id:
                    session.query(AttributeChange).filter_by(
                        player_name=row.player_name
                    ).update({"mlb_player_id": mlb_id})

                existing_count = (
                    session.query(PlayerStatWindow)
                    .filter_by(mlb_player_id=mlb_id, as_of_date=update_date)
                    .count()
                )
                if existing_count >= 4:
                    continue

                season = int(update_date[:4])
                try:
                    windows = build_player_stat_windows(
                        mlb_id, update_date, bool(row.is_hitter), season
                    )
                except Exception:
                    continue
                for window_name, stats in windows.items():
                    existing = (
                        session.query(PlayerStatWindow)
                        .filter_by(
                            mlb_player_id=mlb_id,
                            as_of_date=update_date,
                            window=window_name,
                        )
                        .first()
                    )
                    if existing:
                        continue
                    session.add(
                        PlayerStatWindow(
                            mlb_player_id=mlb_id,
                            as_of_date=update_date,
                            window=window_name,
                            is_hitter=row.is_hitter,
                            stats_json=dumps(stats),
                        )
                    )
                    count += 1
            session.commit()

    return count
