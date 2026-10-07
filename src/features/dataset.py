"""Training and live datasets for the attribute-update model.

One row = (card, core attribute, as-of date). For each monthly attribute
update we reconstruct every Live card's ratings *as they stood before the
update* and attach point-in-time stat windows. Attributes that weren't
listed in the update are real "no change" examples, which the model needs
to learn that most attributes don't move.

Live rows go through exactly the same code with ``as_of = today``.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ATTR_GROUP, CARD_FIELDS, CORE_ATTRS, DB_PATH, MAJOR_UPDATE_MIN_CARDS,
    PROCESSED_DIR, RAW_DIR, STAT_CUTOFF_DAYS,
)
from src.features.windows import HIT_RATES, PIT_RATES, load_game_log, prior_season_window, windows_as_of
from src.models.registry import normalize_attr_name

logger = logging.getLogger(__name__)

PITCHER_POSITIONS = {"SP", "RP", "CP"}
WINDOW_NAMES = ["7d", "14d", "30d", "ytd", "prev", "stab"]
ATTR_CODES = {a: i for i, a in enumerate(CORE_ATTRS["hitting"] + CORE_ATTRS["pitching"])}


def _stat_feature_names(group: str) -> list[str]:
    rates = HIT_RATES if group == "hitting" else PIT_RATES
    n_key = "pa" if group == "hitting" else "bf"
    names = [f"{w}_{n_key}" for w in WINDOW_NAMES if w != "stab"]
    names += [f"{w}_{r}" for w in WINDOW_NAMES for r in rates]
    if group == "pitching":
        names += ["ytd_gs_share", "prev_gs_share"]
    return names


def feature_columns(group: str) -> list[str]:
    """Model inputs for one stat group (hitting / pitching)."""
    attrs = CORE_ATTRS[group]
    return (
        ["attr_code", "rating_before", "ovr_before", "rating_minus_ovr"]
        + [f"r_{a}" for a in attrs]
        + ["last_delta", "season_delta", "n_changes", "prev_major_delta"]
        + _stat_feature_names(group)
    )


# ── Loading ────────────────────────────────────────────────────────────────

def _connect():
    import sqlite3
    return sqlite3.connect(str(DB_PATH))


def load_changes(game_year: int) -> pd.DataFrame:
    with _connect() as conn:
        df = pd.read_sql_query(
            "SELECT card_uuid, player_name, mlb_player_id, update_id, update_date, position, "
            "attribute_name, rating_before, rating_after, delta, ovr_before, ovr_after "
            "FROM attribute_changes WHERE game_year = ? AND update_date IS NOT NULL",
            conn, params=(game_year,),
        )
    df["attr"] = df["attribute_name"].map(lambda a: normalize_attr_name(str(a), game_year))
    return df


def load_cards(game_year: int) -> pd.DataFrame:
    with _connect() as conn:
        cards = pd.read_sql_query(
            "SELECT card_uuid, player_name, team, position, ovr, rarity, is_hitter, "
            "mlb_player_id, attributes_json FROM card_snapshots "
            "WHERE game_year = ? AND series = 'Live'",
            conn, params=(game_year,),
        )
    cards["attributes"] = cards["attributes_json"].map(lambda s: json.loads(s or "{}"))
    return cards.drop(columns="attributes_json")


def major_update_dates(changes: pd.DataFrame) -> list[str]:
    per_date = changes.groupby("update_date")["card_uuid"].nunique()
    return sorted(per_date[per_date >= MAJOR_UPDATE_MIN_CARDS].index)


def card_added_dates(game_year: int) -> dict[str, str]:
    """card_uuid / player name → date the card was added mid-season."""
    from src.ingest.sds_client import parse_update_date

    added: dict[str, str] = {}
    folder = RAW_DIR / f"mlb{game_year}" / "roster_updates"
    with _connect() as conn:
        names = dict(conn.execute(
            "SELECT update_id, update_name FROM roster_updates WHERE game_year = ?", (game_year,)
        ).fetchall())
    for path in folder.glob("update_*.json"):
        update_id = int(path.stem.split("_")[1])
        when = parse_update_date(names.get(update_id, "") or "")
        if not when:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data.get("newly_added", []):
            for key in (entry.get("obfuscated_id"), entry.get("name")):
                if isinstance(key, str) and key and key != "-1":
                    added[key] = min(added.get(key, when), when)
    return added


# ── Point-in-time reconstruction ─────────────────────────────────────────────

def _rating_lookup(changes: pd.DataFrame, as_of: str, attrs: list[str]):
    """For each (card, attr): rating *before* the update on ``as_of``.

    Uses the latest earlier change's rating_after, else the next later
    change's rating_before; cards with no changes fall back to the snapshot.
    Also returns per-(card, attr) history features.
    """
    ch = changes[changes["attr"].isin(attrs)].sort_values(["update_date", "update_id"])
    before = ch[ch["update_date"] < as_of]
    at = ch[ch["update_date"] == as_of]
    after = ch[ch["update_date"] > as_of]

    key = ["card_uuid", "attr"]
    rating = before.groupby(key)["rating_after"].last()
    rating = rating.combine_first(at.groupby(key)["rating_before"].first())
    rating = rating.combine_first(after.groupby(key)["rating_before"].first())

    hist = pd.DataFrame({
        "last_delta": before.groupby(key)["delta"].last(),
        "season_delta": before.groupby(key)["delta"].sum(),
        "n_changes": before.groupby(key)["delta"].size(),
    })
    target = at.groupby(key)["delta"].sum()
    return rating, hist, target


def _ovr_lookup(changes: pd.DataFrame, as_of: str):
    ch = changes.sort_values(["update_date", "update_id"])
    per_card_update = ch.groupby(["card_uuid", "update_date", "update_id"])[["ovr_before", "ovr_after"]].first().reset_index()
    before = per_card_update[per_card_update["update_date"] < as_of]
    at = per_card_update[per_card_update["update_date"] == as_of]
    after = per_card_update[per_card_update["update_date"] > as_of]
    ovr = at.groupby("card_uuid")["ovr_before"].first()
    ovr = ovr.combine_first(before.groupby("card_uuid")["ovr_after"].last())
    ovr = ovr.combine_first(after.groupby("card_uuid")["ovr_before"].first())
    ovr_after = at.groupby("card_uuid")["ovr_after"].first()
    return ovr, ovr_after


class _StatCache:
    """Game logs + prior-season stats, memoized per player."""

    def __init__(self, season: int, max_age_hours: float | None, lag_days: int = 0):
        self.season = season
        self.max_age_hours = max_age_hours
        # SDS rates players on stats frozen some days before an update ships.
        self.lag_days = lag_days
        self._logs: dict = {}
        self._prev: dict = {}

    def windows(self, mlb_id, group: str, as_of: str) -> dict:
        if self.lag_days:
            as_of = (datetime.strptime(as_of[:10], "%Y-%m-%d").date() - timedelta(days=self.lag_days)).isoformat()
        if mlb_id is None or pd.isna(mlb_id):
            return windows_as_of([], as_of, group)
        k = (int(mlb_id), group)
        if k not in self._logs:
            try:
                self._logs[k] = load_game_log(k[0], self.season, group, self.max_age_hours)
            except Exception:
                self._logs[k] = []
            self._prev[k] = prior_season_window(k[0], self.season, group)
        return windows_as_of(self._logs[k], as_of, group, self._prev[k])


def _flatten_windows(win: dict, group: str) -> dict:
    out = {}
    for name in _stat_feature_names(group):
        w, stat = name.split("_", 1)
        out[name] = win.get(w, {}).get(stat)
    return out


def build_rows(
    cards: pd.DataFrame,
    changes: pd.DataFrame,
    as_of: str,
    stats: _StatCache,
    prev_major: str | None = None,
    with_targets: bool = False,
) -> pd.DataFrame:
    rows = []
    ovr_now, ovr_after = _ovr_lookup(changes, as_of)
    prev_major_delta = None
    if prev_major:
        pm = changes[changes["update_date"] == prev_major]
        prev_major_delta = pm.groupby(["card_uuid", "attr"])["delta"].sum()

    # Pitchers carry the pitching attrs; hitters (and two-way cards that have
    # had hitting attrs move, e.g. Ohtani) carry hitting attrs.
    hitting_two_way = set(changes.loc[changes["attr"].isin(CORE_ATTRS["hitting"]), "card_uuid"])
    for group, attrs in CORE_ATTRS.items():
        rating, hist, target = _rating_lookup(changes, as_of, attrs)
        for card in cards.itertuples(index=False):
            is_pitcher = card.position in PITCHER_POSITIONS
            if group == "pitching" and not is_pitcher:
                continue
            if group == "hitting" and is_pitcher and card.card_uuid not in hitting_two_way:
                continue

            ratings = {}
            for a in attrs:
                r = rating.get((card.card_uuid, a))
                if r is None or pd.isna(r):
                    r = card.attributes.get(CARD_FIELDS[a])
                ratings[a] = r
            if any(v is None for v in ratings.values()):
                continue

            ovr = ovr_now.get(card.card_uuid, card.ovr)
            win = _flatten_windows(stats.windows(card.mlb_player_id, group, as_of), group)
            base = {
                "card_uuid": card.card_uuid, "player_name": card.player_name,
                "mlb_player_id": card.mlb_player_id, "team": card.team,
                "position": card.position, "rarity": card.rarity,
                "is_pitcher_card": int(is_pitcher), "group": group,
                "as_of": as_of, "ovr_before": ovr,
                **{f"r_{a}": v for a, v in ratings.items()}, **win,
            }
            if with_targets:
                base["ovr_after"] = ovr_after.get(card.card_uuid, ovr)
            for a in attrs:
                h = hist.loc[(card.card_uuid, a)] if (card.card_uuid, a) in hist.index else None
                row = dict(base)
                row.update({
                    "attr": a, "attr_code": ATTR_CODES[a],
                    "rating_before": ratings[a],
                    "rating_minus_ovr": ratings[a] - ovr,
                    "last_delta": h["last_delta"] if h is not None else 0,
                    "season_delta": h["season_delta"] if h is not None else 0,
                    "n_changes": h["n_changes"] if h is not None else 0,
                    "prev_major_delta": (
                        prev_major_delta.get((card.card_uuid, a), 0) if prev_major_delta is not None else 0
                    ),
                })
                if with_targets:
                    row["delta"] = int(target.get((card.card_uuid, a), 0))
                rows.append(row)

    df = pd.DataFrame(rows)
    numeric = {c for g in CORE_ATTRS for c in feature_columns(g)} & set(df.columns)
    df[list(numeric)] = df[list(numeric)].astype(float)
    return df


def build_training_dataset(
    game_year: int = 26, save: bool = True, refresh_hours: float | None = 20, lag_days: int = STAT_CUTOFF_DAYS,
) -> pd.DataFrame:
    changes = load_changes(game_year)
    cards = load_cards(game_year)
    if cards.empty:
        raise ValueError("No Live cards in the database. Run scripts/daily_predict.py (or fetch cards) first.")
    added = card_added_dates(game_year)
    majors = major_update_dates(changes)
    season = 2000 + game_year
    # Logs cached mid-season would be missing later games, so refresh stale ones.
    refresh_game_logs(cards, changes, season, refresh_hours)
    stats = _StatCache(season, max_age_hours=None, lag_days=lag_days)
    logger.info("Major attribute updates: %s (stats cut off %d days before each)", ", ".join(majors), lag_days)

    # A card added mid-season isn't a "no change" example for earlier updates.
    first_seen = changes.groupby("card_uuid")["update_date"].min()
    added_on = [
        min(first_seen.get(c.card_uuid, "9999"), added.get(c.card_uuid) or added.get(c.player_name) or "0000")
        for c in cards.itertuples()
    ]
    cards = cards.assign(added_on=added_on)

    frames = []
    for i, d in enumerate(majors):
        existed = cards[cards["added_on"] <= d]
        df = build_rows(existed, changes, d, stats, prev_major=majors[i - 1] if i else None, with_targets=True)
        logger.info("  %s: %d rows, %d changed", d, len(df), int((df["delta"] != 0).sum()))
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data.attrs["lag_days"] = lag_days
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        data.to_parquet(PROCESSED_DIR / "training_examples.parquet", index=False)
    return data


def refresh_game_logs(cards: pd.DataFrame, changes: pd.DataFrame, season: int, max_age_hours: float | None) -> None:
    """Fetch every needed game log concurrently (missing or older than max_age)."""
    from src.features.windows import prefetch_game_logs

    two_way = set(changes.loc[changes["attr"].isin(CORE_ATTRS["hitting"]), "card_uuid"])
    keys = []
    for c in cards.dropna(subset=["mlb_player_id"]).itertuples():
        pitcher = c.position in PITCHER_POSITIONS
        keys.append((int(c.mlb_player_id), season, "pitching" if pitcher else "hitting"))
        if pitcher and c.card_uuid in two_way:
            keys.append((int(c.mlb_player_id), season, "hitting"))
    logger.info("Refreshing %d game logs (max age %sh)...", len(keys), max_age_hours)
    prefetch_game_logs(keys, max_age_hours=max_age_hours)


def build_live_dataset(game_year: int = 26, as_of: str | None = None, refresh_hours: float | None = 20) -> pd.DataFrame:
    """Features for every Live card as of today (game logs refreshed daily)."""
    as_of = as_of or date.today().isoformat()
    changes = load_changes(game_year)
    cards = load_cards(game_year)
    majors = [d for d in major_update_dates(changes) if d < as_of]
    season = 2000 + game_year
    refresh_game_logs(cards, changes, season, refresh_hours)
    stats = _StatCache(season, max_age_hours=None)
    return build_rows(cards, changes, as_of, stats, prev_major=majors[-1] if majors else None)
