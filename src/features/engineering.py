from __future__ import annotations

import json
from datetime import datetime, timedelta

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
    """Distance (in OVR points) to the *nearest* tier boundary.

    Players near a boundary are more volatile — a small change can
    shift their rarity tier (e.g. 84→85 = Silver→Gold).  Returns 0
    if the OVR sits exactly on a boundary.
    """
    boundaries = set()
    for lo, hi in RARITY_TIERS.values():
        boundaries.add(lo)
        boundaries.add(hi + 1)  # +1 because crossing hi+1 enters next tier
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
    """Compute a single normalised velocity value for a given stat window.

    Hitters: ISO scaled by 10 (so ~1.5 for a slugger).
    Pitchers: K/9 directly.
    Returns 0 when the window is empty.
    """
    window = stat_windows.get(key, {})
    if not window:
        return 0.0
    if is_hitter:
        return window.get("iso", 0.0) * 10
    return window.get("k9", 0.0)


def _compute_streak(rating_history: list[int] | None) -> int:
    """Count consecutive rating changes in same direction.

    Positive  = consecutive upgrades.
    Negative  = consecutive downgrades.
    0         = no streak (or insufficient history).
    """
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
            continue  # neutral change doesn't break streak
        else:
            break
    return streak * direction


def _compute_consistency_score(rating_history: list[int] | None) -> float:
    """Standard deviation of the last 5 rating changes (deltas).

    A low value means the player's ratings have been stable;
    a high value means volatile swings.  Returns 0.0 when there
    are fewer than 2 history entries (no deltas to compute).
    """
    if not rating_history or len(rating_history) < 2:
        return 0.0
    deltas = [b - a for a, b in zip(rating_history[:-1], rating_history[1:])]
    # Use last 5 deltas (or fewer if history is short)
    recent_deltas = deltas[-5:]
    if len(recent_deltas) < 2:
        return 0.0
    return float(np.std(recent_deltas, ddof=0))


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
) -> dict:
    primary = stat_windows.get(primary_window, stat_windows.get("ytd", {}))
    ytd = stat_windows.get("ytd", primary)
    three_yr = stat_windows.get("3yr", {})
    five_d = stat_windows.get("5d", stat_windows.get("21d", {}))

    projected = project_attribute(attribute_name, primary, is_hitter)
    projected_ytd = project_attribute(attribute_name, ytd, is_hitter)
    projected_3yr = project_attribute(attribute_name, three_yr, is_hitter)

    gap = projected - rating_before
    gap_ytd = projected_ytd - rating_before
    ovr_gap = projected - ovr_before

    # Momentum: 5-day velocity vs 21-day velocity
    vel_5d = _compute_velocity(stat_windows, "5d", is_hitter)
    vel_21d = _compute_velocity(stat_windows, "21d", is_hitter)
    momentum_5d_21d = vel_5d - vel_21d

    # Streak: from recent rating history for this attribute (if supplied)
    streak_count = _compute_streak(rating_history)

    # Consistency: std of last 5 rating changes (lower = more stable)
    consistency_score = _compute_consistency_score(rating_history)

    # Recent direction: simple sign of last 3 updates' deltas summed
    if rating_history and len(rating_history) >= 2:
        recent_deltas = [b - a for a, b in zip(rating_history[-4:-1], rating_history[-3:])]
        recent_direction = float(sum(recent_deltas))
    else:
        recent_direction = 0.0

    features = {
        "attribute_name": attribute_name,
        "rating_before": rating_before,
        "ovr_before": ovr_before,
        "position": position,
        "is_hitter": is_hitter,
        "projected_rating": projected,
        "projected_ytd": projected_ytd,
        "projected_3yr": projected_3yr,
        "gap": gap,
        "gap_ytd": gap_ytd,
        "gap_abs": abs(gap),
        "ovr_gap": ovr_gap,
        "tier_distance_up": tier_distance(ovr_before, "up"),
        "tier_distance_down": tier_distance(ovr_before, "down"),
        "is_established_star": 1 if ovr_before >= 90 else 0,
        "sample_size_ok": int(sample_size_ok(primary, is_hitter)),
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
        # New momentum features
        "momentum_5d_21d": momentum_5d_21d,
        "streak_count": streak_count,
        "consistency_score": consistency_score,
        "recent_direction": recent_direction,
        # Time-gap feature: longer gaps between updates → more volatile swings
        "days_since_last_update": days_since_last_update if days_since_last_update is not None else 0,
        # Tier-boundary proximity: players near a boundary are more volatile
        "ovr_distance_to_tier_boundary": ovr_distance_to_tier_boundary(ovr_before),
    }
    return features


def change_label(delta: int) -> str:
    if delta > 0:
        return "upgrade"
    if delta < 0:
        return "downgrade"
    return "no_change"


def build_training_dataset(primary_window: str = "21d") -> pd.DataFrame:
    Session = init_db()
    rows = []

    with Session() as session:
        changes = session.query(AttributeChange).all()
        stat_cache: dict[tuple, dict] = {}
        # For streak we want rating history per (player, attr, date)
        history_cache: dict[tuple, list[int]] = {}
        # Cache last update date per player for days_since_last_update
        last_update_cache: dict[tuple, str] = {}

        for ch in changes:
            if not ch.update_date:
                continue
            # ── Data-quality gate ───────────────────────────────────────
            # Drop program/upgrade cards with no real MLB player, impossible
            # ratings (<1 or >99), and implausible single-update deltas.
            # These corrupt the refit coefficients and inflate error.
            if ch.mlb_player_id is None:
                continue
            if ch.rating_before is not None and not (1 <= ch.rating_before <= 99):
                continue
            if ch.rating_after is not None and not (1 <= ch.rating_after <= 99):
                continue
            if ch.delta is not None and abs(ch.delta) > 30:
                continue
            key = (ch.mlb_player_id, ch.update_date, ch.is_hitter)

            # Compute days_since_last_update for this (player, update_date)
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

            # Calculate the day gap
            days_since = None
            last_date_str = last_update_cache.get(player_key)
            if last_date_str and ch.update_date:
                try:
                    current_dt = datetime.strptime(str(ch.update_date), "%Y-%m-%d")
                    prev_dt = datetime.strptime(str(last_date_str), "%Y-%m-%d")
                    days_since = (current_dt - prev_dt).days
                except (ValueError, TypeError):
                    days_since = None
            if key not in stat_cache:
                if ch.mlb_player_id:
                    cached = (
                        session.query(PlayerStatWindow)
                        .filter_by(mlb_player_id=ch.mlb_player_id, as_of_date=ch.update_date)
                        .all()
                    )
                    if cached:
                        stat_cache[key] = {
                            w.window: json.loads(w.stats_json) for w in cached
                        }
                    else:
                        stat_cache[key] = {}
                else:
                    stat_cache[key] = {}

            windows = stat_cache[key]
            if not windows:
                _lv = LEAGUE_AVG
                windows = {
                    "5d": {"k_pct": _lv["k_pct"], "bb_pct": _lv["bb_pct"], "avg": _lv["avg"], "iso": _lv["iso"], "ab": 0, "ip": 0, "k9": _lv["k9"], "bb9": _lv["bb9"], "hr9": _lv["hr9"], "gs": 0, "sprint_speed": _lv["sprint_speed"]},
                    "21d": {"k_pct": _lv["k_pct"], "bb_pct": _lv["bb_pct"], "avg": _lv["avg"], "iso": _lv["iso"], "ab": 0, "ip": 0, "k9": _lv["k9"], "bb9": _lv["bb9"], "hr9": _lv["hr9"], "gs": 0, "sprint_speed": _lv["sprint_speed"]},
                    "ytd": {"k_pct": _lv["k_pct"], "bb_pct": _lv["bb_pct"], "avg": _lv["avg"], "iso": _lv["iso"], "k9": _lv["k9"], "bb9": _lv["bb9"], "hr9": _lv["hr9"], "gs": 0, "sprint_speed": _lv["sprint_speed"]},
                    "3yr": {"k_pct": _lv["k_pct"], "bb_pct": _lv["bb_pct"], "avg": _lv["avg"], "iso": _lv["iso"], "k9": _lv["k9"], "bb9": _lv["bb9"], "hr9": _lv["hr9"], "gs": 0, "sprint_speed": _lv["sprint_speed"]},
                }

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

    # Build latest rating lookup for ALL attrs from attribute_changes
    # (some attrs like k/9_r, stamina aren't in attributes_json)
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

            # Compute days_since_last_update for live predictions
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

            if mlb_id not in stat_cache:
                if not has_mlb_id:
                    stat_cache[mlb_id] = {"5d": {}, "21d": {}, "ytd": {}, "3yr": {}}
                elif fast:
                    stat_cache[mlb_id] = build_live_stat_windows(mlb_id, season, is_hitter, client)
                else:
                    as_of = (datetime.utcnow() - timedelta(days=horizon_days)).strftime("%Y-%m-%d")
                    try:
                        stat_cache[mlb_id] = build_player_stat_windows(mlb_id, as_of, is_hitter, season)
                    except Exception:
                        stat_cache[mlb_id] = {"5d": {}, "21d": {}, "ytd": {}, "3yr": {}}
            windows = stat_cache[mlb_id]

            attr_list = HITTER_ATTRS if is_hitter else PITCHER_ATTRS
            stat_windows_json = dumps(windows) if windows else None
            for attr in attr_list:
                # Try attributes_json first, then latest-rating lookup for
                # split attrs (k/9_r, stamina, etc.) not stored in card data
                current = attrs.get(attr)
                if current is None:
                    alias_key = next((k for k, v in ATTR_ALIASES.items() if v == attr), attr)
                    lookup = latest_rating_lookup.get((mlb_id, attr)) or latest_rating_lookup.get((mlb_id, alias_key))
                    if lookup is not None:
                        current = lookup
                    else:
                        continue  # no rating available for this attr
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
