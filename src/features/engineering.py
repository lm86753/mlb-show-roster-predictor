from __future__ import annotations

import json
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import (
    HITTER_ATTRS,
    MIN_AB_21D,
    MIN_IP_21D,
    PITCHER_ATTRS,
    RARITY_TIERS,
    TIER_ORDER,
)
from sqlalchemy import func as sqlfunc

from src.db import AttributeChange, CardSnapshot, PlayerStatWindow, TrainingExample, init_db, dumps
from src.formulas.ratings import LEAGUE_AVG, ATTR_ALIASES, project_attribute
from src.ingest.mlb_stats import MLBStatsClient, build_player_stat_windows

# ── Feature column lists for models (also imported by train.py, predict.py) ──

REGRESSION_FEATURES = [
    "gap", "gap_abs", "gap_today", "gap_today_abs", "gap_7d", "gap_14d",
    "gap_ytd", "gap_3yr", "gap_7d_abs", "gap_14d_abs", "gap_spread",
    "ovr_gap",
    "days_since_last_update",
    "ovr_distance_to_tier_boundary",
    "is_established_star", "is_prospect", "is_gold_plus",
    "team_market",
    "sample_size_ok", "sample_size_ok_7d", "sample_size_ok_14d",
    "momentum_7d_14d", "momentum_7d_21d", "momentum_14d_21d",
    "streak_count", "consistency_score", "recent_direction",
    "trend_avg", "trend_ops", "trend_k9",
    "streak_primary", "streak_ops", "streak_k9",
    "vol_cluster_primary",
    "consistency_avg", "consistency_k9",
    "stat_k_pct", "stat_bb_pct", "stat_avg", "stat_iso",
    "stat_k9", "stat_bb9", "stat_hr9",
    "stat_k_pct_7d", "stat_avg_7d", "stat_iso_7d",
    "stat_k9_7d", "stat_bb9_7d",
    "stat_k_pct_14d", "stat_avg_14d", "stat_iso_14d",
    "stat_k9_14d", "stat_bb9_14d",
    "tier_distance_up", "tier_distance_down",
]

ANALOG_FEATURES = [
    "gap_today", "gap_7d", "gap_14d", "gap_ytd", "gap_spread",
    "stat_k_pct", "stat_bb_pct", "stat_avg", "stat_iso",
    "stat_k9", "stat_bb9",
    "ovr_before",
    "team_market",
    "days_since_last_update",
    "is_hitter",
]

WINDOW_WEIGHTS = {
    "7d": 0.35,
    "14d": 0.30,
    "21d": 0.20,
    "ytd": 0.15,
}

WINDOW_PRIORITY = ["7d", "14d", "21d", "ytd", "3yr"]


def tier_for_ovr(ovr: int) -> str:
    for tier, (lo, hi) in RARITY_TIERS.items():
        if lo <= ovr <= hi:
            return tier
    return "Common"


def tier_distance(ovr: int, direction: str = "up") -> int:
    tier = tier_for_ovr(ovr)
    idx = TIER_ORDER.index(tier)
    if direction == "up" and idx < len(TIER_ORDER) - 1:
        next_lo = RARITY_TIERS[TIER_ORDER[idx + 1]][0]
        return max(0, next_lo - ovr)
    if direction == "down" and idx > 0:
        prev_hi = RARITY_TIERS[TIER_ORDER[idx - 1]][1]
        return max(0, ovr - prev_hi)
    return 0


def ovr_distance_to_tier_boundary(ovr: int) -> int:
    boundaries = set()
    for lo, hi in RARITY_TIERS.values():
        boundaries.add(lo)
        boundaries.add(hi + 1)
    distances = [abs(ovr - b) for b in boundaries]
    return min(distances) if distances else 99


def sample_size_ok(stats: dict, is_hitter: bool, window: str = "21d") -> bool:
    if window == "21d":
        if is_hitter:
            return stats.get("ab", 0) >= MIN_AB_21D
        return stats.get("ip", 0) >= MIN_IP_21D
    if is_hitter:
        return stats.get("ab", 0) >= 5
    return stats.get("ip", 0) >= 3


def _compute_velocity(stat_windows: dict, key: str, is_hitter: bool) -> float:
    window = stat_windows.get(key, {})
    if not window:
        return 0.0
    if is_hitter:
        return window.get("iso", 0.0) * 10
    return window.get("k9", 0.0)


def _compute_streak(rating_history: list[int] | None) -> int:
    if not rating_history or len(rating_history) < 2:
        return 0
    deltas = [b - a for a, b in zip(rating_history[:-1], rating_history[1:])]
    if not deltas:
        return 0
    streak = 0
    direction = 1 if deltas[-1] > 0 else (-1 if deltas[-1] < 0 else 0)
    if direction == 0:
        return 0
    for d in reversed(deltas):
        if d * direction > 0:
            streak += 1
        elif d == 0:
            continue
        else:
            break
    return streak * direction


def _compute_consistency_score(rating_history: list[int] | None) -> float:
    if not rating_history or len(rating_history) < 2:
        return 0.0
    deltas = [b - a for a, b in zip(rating_history[:-1], rating_history[1:])]
    recent_deltas = deltas[-5:]
    if len(recent_deltas) < 2:
        return 0.0
    return float(np.std(recent_deltas, ddof=0))


# ── Multi-window projection helpers ─────────────────────────────────────────

def compute_window_projections(
    attribute_name: str,
    stat_windows: dict[str, dict],
    is_hitter: bool,
    active_window: str = "21d",
) -> dict:
    """Compute projections from every available stat window.

    Returns dict with keys like proj_7d, proj_14d, proj_21d, proj_ytd, proj_3yr,
    and also gap_7d, gap_14d, etc. relative to the active_window projection.

    Always projects using primary 21d window as the base projection,
    plus all other windows as secondary signals.
    """
    results = {}
    primary = stat_windows.get(active_window, stat_windows.get("ytd", {}))
    results["primary_projection"] = float(project_attribute(attribute_name, primary, is_hitter))

    for w in WINDOW_PRIORITY:
        stats = stat_windows.get(w, {})
        if stats and stats.get("games", 1) > 0:
            results[f"proj_{w}"] = float(project_attribute(attribute_name, stats, is_hitter))
        else:
            results[f"proj_{w}"] = results["primary_projection"]

    # Blended "today" projection: weighted average of available windows
    weighted_sum = 0.0
    total_weight = 0.0
    for w, weight in WINDOW_WEIGHTS.items():
        stats = stat_windows.get(w, {})
        if stats and stats.get("games", 1) > 0:
            proj = project_attribute(attribute_name, stats, is_hitter)
            weighted_sum += proj * weight
            total_weight += weight
    results["proj_today"] = float(weighted_sum / total_weight) if total_weight > 0 else results["primary_projection"]

    return results


def compute_window_gaps(
    projections: dict,
    rating_before: int,
) -> dict:
    """Compute gaps from all projection variants."""
    gaps = {}
    for key, proj in projections.items():
        if key.startswith("proj_"):
            gap_name = key.replace("proj_", "gap_")
            gaps[gap_name] = float(proj) - rating_before
    return gaps


# ── Momentum integration ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_momentum_computer():
    from src.features.momentum import MomentumComputer
    return MomentumComputer()


def compute_momentum_features(
    mlb_id: int,
    season: int,
    is_hitter: bool,
) -> dict[str, float]:
    try:
        computer = _get_momentum_computer()
        return computer.compute_all(mlb_id, season, is_hitter)
    except Exception:
        return {}


# ── Player metadata features ────────────────────────────────────────────────

_TEAM_MARKET = {
    "NYY": 1.0, "LAD": 0.95, "CHC": 0.90, "BOS": 0.88, "NYY": 1.0,
    "HOU": 0.85, "ATL": 0.82, "SFG": 0.80, "NYM": 0.85, "STL": 0.78,
    "PHI": 0.80, "LAA": 0.75, "TEX": 0.72, "SEA": 0.70, "MIL": 0.60,
    "MIN": 0.65, "SDP": 0.68, "CIN": 0.62, "COL": 0.58, "ARI": 0.60,
    "CLE": 0.62, "DET": 0.60, "KCR": 0.50, "MIA": 0.55, "PIT": 0.52,
    "TBR": 0.58, "BAL": 0.62, "TOR": 0.70, "WSN": 0.60, "OAK": 0.45,
    "CHW": 0.65,
}


def team_market_size(team: str) -> float:
    return _TEAM_MARKET.get(team.upper() if team else "", 0.55)


# ── Main feature builder ────────────────────────────────────────────────────

def build_feature_row(
    attribute_name: str,
    rating_before: int,
    ovr_before: int,
    rarity_before: str,
    position: str,
    is_hitter: bool,
    stat_windows: dict[str, dict],
    primary_window: str = "21d",
    rating_history: list[int] | None = None,
    days_since_last_update: int | None = None,
    player_name: str = "",
    team: str = "",
    mlb_id: int | None = None,
    season: int | None = None,
) -> dict:
    primary = stat_windows.get(primary_window, stat_windows.get("ytd", {}))
    ytd = stat_windows.get("ytd", primary)
    three_yr = stat_windows.get("3yr", {})

    # Multi-window projections
    projs = compute_window_projections(attribute_name, stat_windows, is_hitter, primary_window)
    gaps = compute_window_gaps(projs, rating_before)
    gap_today = gaps.get("gap_today", gaps.get("gap_21d", 0.0))

    projected = int(round(projs.get("primary_projection", 50)))
    projected_ytd = int(round(projs.get("proj_ytd", projected)))
    projected_3yr = int(round(projs.get("proj_3yr", projected)))

    gap = gaps.get("gap_21d", 0.0)
    gap_ytd = gaps.get("gap_ytd", 0.0)
    ovr_gap = projs.get("primary_projection", 50) - ovr_before

    # Momentum: 7d velocity vs 14d/21d velocity
    vel_7d = _compute_velocity(stat_windows, "7d", is_hitter)
    vel_14d = _compute_velocity(stat_windows, "14d", is_hitter)
    vel_21d = _compute_velocity(stat_windows, "21d", is_hitter)
    momentum_7d_14d = vel_7d - vel_14d
    momentum_7d_21d = vel_7d - vel_21d
    momentum_14d_21d = vel_14d - vel_21d

    # Game-log derived momentum (trend, streak, consistency, volatility)
    momentum_from_games = {}
    if mlb_id and season:
        momentum_from_games = compute_momentum_features(mlb_id, season, is_hitter)

    # Streak: from recent rating history for this attribute
    streak_count = _compute_streak(rating_history)

    # Consistency: std of last 5 rating changes
    consistency_score = _compute_consistency_score(rating_history)

    # Recent direction: simple sign of last 3 updates' deltas summed
    if rating_history and len(rating_history) >= 2:
        recent_deltas = [b - a for a, b in zip(rating_history[-4:-1], rating_history[-3:])]
        recent_direction = float(sum(recent_deltas))
    else:
        recent_direction = 0.0

    # Player metadata
    market = team_market_size(team)
    is_star = 1 if ovr_before >= 90 else 0
    is_prospect = 1 if ovr_before < 75 else 0
    is_gold_plus = 1 if ovr_before >= 85 else 0

    # Sample size checks per window
    sample_ok_21d = int(sample_size_ok(primary, is_hitter))
    sample_ok_7d = int(sample_size_ok(stat_windows.get("7d", {}), is_hitter, window="7d"))
    sample_ok_14d = int(sample_size_ok(stat_windows.get("14d", {}), is_hitter, window="7d"))

    features = {
        "attribute_name": attribute_name,
        "rating_before": rating_before,
        "ovr_before": ovr_before,
        "position": position,
        "is_hitter": is_hitter,
        "projected_rating": projected,
        "projected_ytd": projected_ytd,
        "projected_3yr": projected_3yr,
        "projected_today": int(round(projs.get("proj_today", projected))),
        # Multi-window gaps
        "gap": gap,
        "gap_ytd": gap_ytd,
        "gap_3yr": gaps.get("gap_3yr", 0.0),
        "gap_7d": gaps.get("gap_7d", 0.0),
        "gap_14d": gaps.get("gap_14d", 0.0),
        "gap_21d": gaps.get("gap_21d", 0.0),
        "gap_today": gap_today,
        "gap_abs": abs(gap),
        "gap_7d_abs": abs(gaps.get("gap_7d", 0.0)),
        "gap_14d_abs": abs(gaps.get("gap_14d", 0.0)),
        "gap_today_abs": abs(gap_today),
        "ovr_gap": ovr_gap,
        # Gap volatility: how much do different windows disagree?
        "gap_spread": float(np.std([abs(gaps.get(f"gap_{w}", 0.0)) for w in ["7d", "14d", "21d", "ytd"]])),
        # Tier
        "tier_distance_up": tier_distance(ovr_before, "up"),
        "tier_distance_down": tier_distance(ovr_before, "down"),
        "is_established_star": is_star,
        "is_prospect": is_prospect,
        "is_gold_plus": is_gold_plus,
        # Sample size
        "sample_size_ok": sample_ok_21d,
        "sample_size_ok_7d": sample_ok_7d,
        "sample_size_ok_14d": sample_ok_14d,
        # Player metadata
        "team_market": market,
        # Stat values from primary window
        "stat_k_pct": primary.get("k_pct", LEAGUE_AVG["k_pct"]) * 100,
        "stat_bb_pct": primary.get("bb_pct", LEAGUE_AVG["bb_pct"]) * 100,
        "stat_avg": primary.get("avg", LEAGUE_AVG["avg"]),
        "stat_iso": primary.get("iso", LEAGUE_AVG["iso"]),
        "stat_ab": primary.get("ab", 0),
        "stat_ip": primary.get("ip", 0),
        "stat_pa": primary.get("pa", 0),
        "stat_hr": primary.get("hr", 0),
        "stat_k9": primary.get("k9", LEAGUE_AVG["k9"]),
        "stat_bb9": primary.get("bb9", LEAGUE_AVG["bb9"]),
        "stat_hr9": primary.get("hr9", LEAGUE_AVG["hr9"]),
        "stat_gs": primary.get("gs", int(LEAGUE_AVG.get("gamesStarted", 1))),
        "stat_sprint_speed": primary.get("sprint_speed", LEAGUE_AVG["sprint_speed"]),
        "stat_fb_velo": primary.get("fb_velo", LEAGUE_AVG["fb_velo"]),
        "stat_exit_velo": primary.get("exit_velo", LEAGUE_AVG["exit_velo"]),
        # Window-specific stats
        "stat_k_pct_7d": stat_windows.get("7d", {}).get("k_pct", LEAGUE_AVG["k_pct"]) * 100,
        "stat_avg_7d": stat_windows.get("7d", {}).get("avg", LEAGUE_AVG["avg"]),
        "stat_iso_7d": stat_windows.get("7d", {}).get("iso", LEAGUE_AVG["iso"]),
        "stat_k9_7d": stat_windows.get("7d", {}).get("k9", LEAGUE_AVG["k9"]),
        "stat_bb9_7d": stat_windows.get("7d", {}).get("bb9", LEAGUE_AVG["bb9"]),
        "stat_k_pct_14d": stat_windows.get("14d", {}).get("k_pct", LEAGUE_AVG["k_pct"]) * 100,
        "stat_avg_14d": stat_windows.get("14d", {}).get("avg", LEAGUE_AVG["avg"]),
        "stat_iso_14d": stat_windows.get("14d", {}).get("iso", LEAGUE_AVG["iso"]),
        "stat_k9_14d": stat_windows.get("14d", {}).get("k9", LEAGUE_AVG["k9"]),
        "stat_bb9_14d": stat_windows.get("14d", {}).get("bb9", LEAGUE_AVG["bb9"]),
        # Momentum features
        "momentum_7d_14d": momentum_7d_14d,
        "momentum_7d_21d": momentum_7d_21d,
        "momentum_14d_21d": momentum_14d_21d,
        "streak_count": streak_count,
        "consistency_score": consistency_score,
        "recent_direction": recent_direction,
        # Game-log momentum features
        "trend_avg": momentum_from_games.get("trend_avg", 0.0),
        "trend_ops": momentum_from_games.get("trend_ops", 0.0),
        "trend_k9": momentum_from_games.get("trend_k9", 0.0),
        "streak_primary": momentum_from_games.get("streak_primary", 0.0),
        "streak_ops": momentum_from_games.get("streak_ops", 0.0),
        "streak_k9": momentum_from_games.get("streak_k9", 0.0),
        "vol_cluster_primary": momentum_from_games.get("vol_cluster_primary", 0.0),
        "consistency_avg": momentum_from_games.get("consistency_avg", 0.0),
        "consistency_k9": momentum_from_games.get("consistency_k9", 0.0),
        # Time-gap feature
        "days_since_last_update": days_since_last_update if days_since_last_update is not None else 0,
        # Tier-boundary proximity
        "ovr_distance_to_tier_boundary": ovr_distance_to_tier_boundary(ovr_before),
    }
    return features


def change_label(delta: int) -> str:
    if delta > 0:
        return "upgrade"
    if delta < 0:
        return "downgrade"
    return "no_change"


def _load_momentum_for_date(
    mlb_id: int,
    as_of_date: str,
    session,
) -> dict | None:
    row = (
        session.query(PlayerStatWindow)
        .filter_by(
            mlb_player_id=mlb_id,
            as_of_date=as_of_date,
            window="momentum",
        )
        .first()
    )
    if row and row.stats_json:
        return json.loads(row.stats_json)
    return None


def build_training_dataset(primary_window: str = "21d") -> pd.DataFrame:
    Session = init_db()
    rows = []

    with Session() as session:
        changes = session.query(AttributeChange).all()
        stat_cache: dict[tuple, dict] = {}
        history_cache: dict[tuple, list[int]] = {}
        last_update_cache: dict[tuple, str] = {}
        momentum_cache: dict[tuple, dict] = {}

        for ch in changes:
            if not ch.update_date:
                continue
            if ch.mlb_player_id is None:
                continue
            if ch.rating_before is not None and not (1 <= ch.rating_before <= 99):
                continue
            if ch.rating_after is not None and not (1 <= ch.rating_after <= 99):
                continue
            if ch.delta is not None and abs(ch.delta) > 30:
                continue
            key = (ch.mlb_player_id, ch.update_date, ch.is_hitter)

            # Days since last update
            player_key = (ch.mlb_player_id, ch.update_date)
            if player_key not in last_update_cache and ch.mlb_player_id:
                prev = (
                    session.query(AttributeChange.update_date)
                    .filter(
                        AttributeChange.mlb_player_id == ch.mlb_player_id,
                        AttributeChange.update_date < ch.update_date,
                    )
                    .order_by(AttributeChange.update_date.desc())
                    .first()
                )
                last_update_cache[player_key] = prev.update_date if prev else None

            days_since = None
            last_date_str = last_update_cache.get(player_key)
            if last_date_str and ch.update_date:
                try:
                    current_dt = datetime.strptime(str(ch.update_date), "%Y-%m-%d")
                    prev_dt = datetime.strptime(str(last_date_str), "%Y-%m-%d")
                    days_since = (current_dt - prev_dt).days
                except (ValueError, TypeError):
                    days_since = None

            # Load stat windows
            if key not in stat_cache:
                if ch.mlb_player_id:
                    cached = (
                        session.query(PlayerStatWindow)
                        .filter_by(mlb_player_id=ch.mlb_player_id, as_of_date=ch.update_date)
                        .all()
                    )
                    stat_cache[key] = {
                        w.window: json.loads(w.stats_json) for w in cached
                    } if cached else {}
                else:
                    stat_cache[key] = {}

            windows = stat_cache[key]
            if not windows:
                _lv = LEAGUE_AVG
                windows = {
                    w: {"k_pct": _lv["k_pct"], "bb_pct": _lv["bb_pct"], "avg": _lv["avg"], "iso": _lv["iso"], "ab": 0, "ip": 0, "k9": _lv["k9"], "bb9": _lv["bb9"], "hr9": _lv["hr9"], "gs": 0, "sprint_speed": _lv["sprint_speed"]}
                    for w in WINDOW_PRIORITY
                }

            # Load momentum features from cache if available
            momentum_key = (ch.mlb_player_id, ch.update_date)
            if momentum_key not in momentum_cache and ch.mlb_player_id:
                momentum_cache[momentum_key] = _load_momentum_for_date(
                    ch.mlb_player_id, ch.update_date, session
                )
            momentum_feats = momentum_cache.get(momentum_key) or {}

            # Build rating history for streak detection
            hist_key = (ch.mlb_player_id, ch.attribute_name, ch.update_date)
            if hist_key not in history_cache and ch.mlb_player_id:
                prev_changes = (
                    session.query(AttributeChange)
                    .filter(
                        AttributeChange.mlb_player_id == ch.mlb_player_id,
                        AttributeChange.attribute_name == ch.attribute_name,
                        AttributeChange.update_date <= ch.update_date,
                    )
                    .order_by(AttributeChange.update_date.asc())
                    .limit(10)
                    .all()
                )
                rating_history = [c.rating_before for c in prev_changes if c.rating_before is not None]
                if ch.rating_before is not None:
                    rating_history.append(ch.rating_before)
                history_cache[hist_key] = rating_history

            season = int(ch.update_date[:4]) if ch.update_date else 26
            feats = build_feature_row(
                ch.attribute_name,
                ch.rating_before or 60,
                ch.ovr_before or 75,
                ch.rarity_before or "Silver",
                ch.position or "",
                bool(ch.is_hitter),
                windows,
                primary_window,
                rating_history=history_cache.get(hist_key),
                days_since_last_update=days_since,
                player_name=ch.player_name or "",
                team=ch.team or "",
                mlb_id=ch.mlb_player_id,
                season=season,
            )
            feats.update(
                {
                    "game_year": ch.game_year,
                    "update_id": ch.update_id,
                    "update_date": ch.update_date,
                    "card_uuid": ch.card_uuid,
                    "player_name": ch.player_name,
                    "mlb_player_id": ch.mlb_player_id,
                    "rating_after": ch.rating_after,
                    "delta": ch.delta,
                    "change_label": change_label(ch.delta or 0),
                    "rarity_before": ch.rarity_before,
                }
            )
            rows.append(feats)

    df = pd.DataFrame(rows)
    if not df.empty:
        out = __import__("src.config", fromlist=["PROCESSED_DIR"]).PROCESSED_DIR
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "training_examples.parquet", index=False)
    return df


def build_live_features(
    game_year: int = 26,
    horizon_days: int = 1,
    primary_window: str = "21d",
    fast: bool = True,
) -> pd.DataFrame:
    from src.ingest.mlb_stats import build_live_stat_windows

    Session = init_db()
    client = MLBStatsClient()
    season = datetime.utcnow().year
    rows = []
    stat_cache: dict[int, dict] = {}
    momentum_cache: dict[int, dict] = {}

    # Latest rating lookup
    latest_rating_lookup: dict[tuple[int, str], int] = {}
    with Session() as session:
        subq = (
            session.query(
                AttributeChange.mlb_player_id,
                AttributeChange.attribute_name,
                sqlfunc.max(AttributeChange.update_date).label("max_date"),
            )
            .filter(
                AttributeChange.mlb_player_id.isnot(None),
                AttributeChange.attribute_name.isnot(None),
            )
            .group_by(AttributeChange.mlb_player_id, AttributeChange.attribute_name)
            .subquery()
        )
        latest_rows = (
            session.query(AttributeChange)
            .join(
                subq,
                (AttributeChange.mlb_player_id == subq.c.mlb_player_id)
                & (AttributeChange.attribute_name == subq.c.attribute_name)
                & (AttributeChange.update_date == subq.c.max_date),
            )
            .all()
        )
        for r in latest_rows:
            if r.mlb_player_id and r.attribute_name and r.rating_after is not None:
                lookup_key = (int(r.mlb_player_id), r.attribute_name)
                latest_rating_lookup[lookup_key] = int(r.rating_after)

    with Session() as session:
        cards = (
            session.query(CardSnapshot)
            .filter_by(game_year=game_year, series="Live")
            .filter(CardSnapshot.mlb_player_id.isnot(None))
            .all()
        )
        if not cards:
            cards = session.query(CardSnapshot).filter_by(game_year=game_year, series="Live").all()

        for card in cards:
            mlb_id = card.mlb_player_id or client.search_player(card.player_name)
            has_mlb_id = bool(mlb_id)
            if not mlb_id:
                mlb_id = -(hash(card.card_uuid) % 10**9)
            attrs = json.loads(card.attributes_json or "{}")
            is_hitter = bool(card.is_hitter)

            # Days since last update
            days_since_live = None
            if has_mlb_id:
                last_live = (
                    session.query(AttributeChange.update_date)
                    .filter(AttributeChange.mlb_player_id == mlb_id)
                    .order_by(AttributeChange.update_date.desc())
                    .first()
                )
                if last_live and last_live.update_date:
                    try:
                        last_dt = datetime.strptime(str(last_live.update_date), "%Y-%m-%d")
                        days_since_live = (datetime.utcnow() - last_dt).days
                    except (ValueError, TypeError):
                        days_since_live = None

            # Stat windows
            if mlb_id not in stat_cache:
                if not has_mlb_id:
                    stat_cache[mlb_id] = {w: {} for w in WINDOW_PRIORITY}
                elif fast:
                    stat_cache[mlb_id] = build_live_stat_windows(mlb_id, season, is_hitter, client)
                else:
                    as_of = (datetime.utcnow() - timedelta(days=horizon_days)).strftime("%Y-%m-%d")
                    try:
                        stat_cache[mlb_id] = build_player_stat_windows(mlb_id, as_of, is_hitter, season)
                    except Exception:
                        stat_cache[mlb_id] = {w: {} for w in WINDOW_PRIORITY}
            windows = stat_cache[mlb_id]

            # Pre-compute momentum features once per player
            if mlb_id not in momentum_cache and has_mlb_id:
                try:
                    momentum_cache[mlb_id] = compute_momentum_features(mlb_id, season, is_hitter)
                except Exception:
                    momentum_cache[mlb_id] = {}
            momentum_feats = momentum_cache.get(mlb_id, {})
            windows_with_momentum = dict(windows)
            windows_with_momentum["momentum"] = momentum_feats

            attr_list = HITTER_ATTRS if is_hitter else PITCHER_ATTRS
            stat_windows_json = dumps(windows) if windows else None
            for attr in attr_list:
                current = attrs.get(attr)
                if current is None:
                    alias_key = next((k for k, v in ATTR_ALIASES.items() if v == attr), attr)
                    lookup = latest_rating_lookup.get((mlb_id, attr)) or latest_rating_lookup.get((mlb_id, alias_key))
                    if lookup is not None:
                        current = lookup
                    else:
                        continue
                feats = build_feature_row(
                    attr,
                    current,
                    card.ovr or 75,
                    card.rarity or "Silver",
                    card.position or "",
                    is_hitter,
                    windows,
                    primary_window,
                    days_since_last_update=days_since_live,
                    player_name=card.player_name or "",
                    team=card.team or "",
                    mlb_id=mlb_id if has_mlb_id else None,
                    season=season,
                )
                feats.update(
                    {
                        "card_uuid": card.card_uuid,
                        "player_name": card.player_name,
                        "mlb_player_id": mlb_id,
                        "current_ovr": card.ovr,
                        "current_rarity": card.rarity,
                        "horizon_days": horizon_days,
                        "stat_windows_json": stat_windows_json,
                    }
                )
                rows.append(feats)

    return pd.DataFrame(rows)


def persist_training_examples(df: pd.DataFrame) -> int:
    Session = init_db()
    count = 0
    with Session() as session:
        session.query(TrainingExample).delete()
        for _, row in df.iterrows():
            feat_cols = {k: v for k, v in row.items() if k not in {
                "rating_after", "delta", "change_label", "game_year", "update_id",
                "update_date", "card_uuid", "player_name", "mlb_player_id", "attribute_name",
                "rating_before", "rarity_before",
            }}
            session.add(
                TrainingExample(
                    game_year=int(row.get("game_year", 26)),
                    update_id=int(row.get("update_id", 0)),
                    update_date=str(row.get("update_date", "")),
                    card_uuid=str(row.get("card_uuid", "")),
                    player_name=str(row.get("player_name", "")),
                    mlb_player_id=int(row["mlb_player_id"]) if pd.notna(row.get("mlb_player_id")) else None,
                    attribute_name=str(row.get("attribute_name", "")),
                    rating_before=int(row.get("rating_before", 60)),
                    rating_after=int(row.get("rating_after", 60)),
                    delta=int(row.get("delta", 0)),
                    change_label=str(row.get("change_label", "no_change")),
                    ovr_before=int(row.get("ovr_before", 75)),
                    rarity_before=str(row.get("rarity_before", "")),
                    position=str(row.get("position", "")),
                    is_hitter=int(row.get("is_hitter", 1)),
                    features_json=dumps(feat_cols),
                )
            )
            count += 1
        session.commit()
    return count
