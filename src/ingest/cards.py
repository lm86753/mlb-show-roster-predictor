from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache

from src.config import GAME_YEARS
from src.db import CardSnapshot, init_db, dumps
from src.ingest.sds_client import SDSClient, extract_card_attributes


def fetch_live_series_cards(game_year: int = 26) -> dict:
    """Pull all Live Series cards from SDS items API."""
    client = SDSClient(game_year=game_year)
    Session = init_db()
    stats = {"pages": 0, "cards": 0}

    first = client.list_items(page=1, series="Live")
    total_pages = first.get("total_pages", 1)

    with Session() as session:
        # Keep MLB ID links across refreshes so we don't re-search ~2k names.
        known_ids = {
            uuid: mlb_id
            for uuid, mlb_id in session.query(CardSnapshot.card_uuid, CardSnapshot.mlb_player_id)
            .filter(CardSnapshot.mlb_player_id.isnot(None))
        }
        session.query(CardSnapshot).filter_by(game_year=game_year).delete()

        for page in range(1, total_pages + 1):
            data = client.list_items(page=page, series="Live") if page > 1 else first
            stats["pages"] += 1
            for item in data.get("items", []):
                if item.get("series") != "Live":
                    continue
                attrs = extract_card_attributes(item)
                session.add(
                    CardSnapshot(
                        game_year=game_year,
                        card_uuid=item.get("uuid", ""),
                        player_name=item.get("name", ""),
                        team=item.get("team", ""),
                        position=item.get("display_position", ""),
                        ovr=item.get("ovr"),
                        rarity=item.get("rarity", ""),
                        series=item.get("series", "Live"),
                        is_hitter=0 if item.get("display_position") in {"SP", "RP", "CP"} else 1,
                        mlb_player_id=known_ids.get(item.get("uuid", "")),
                        attributes_json=dumps(attrs),
                    )
                )
                stats["cards"] += 1
            session.commit()

    return stats


PITCHER_POSITIONS = {"SP", "RP", "CP"}
# SDS team names that differ from the MLB Stats API teamName.
_TEAM_ALIASES = {"Diamondbacks": "D-backs"}


def _norm_name(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    plain = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", plain)
    return re.sub(r"[^a-z]", "", plain)


@lru_cache(maxsize=1)
def _team_org_names() -> dict[int, str]:
    """Team id (MLB or affiliate) -> parent MLB club's short name, e.g. 'Dodgers'."""
    import requests

    from src.config import MLB_STATS_API

    teams = requests.get(
        f"{MLB_STATS_API}/teams", params={"sportIds": "1,11,12,13,14,16"}, timeout=60
    ).json()["teams"]
    by_full = {t["name"]: t["teamName"] for t in teams if t.get("sport", {}).get("id") == 1}
    return {t["id"]: by_full.get(t.get("parentOrgName") or t["name"], t.get("teamName", "")) for t in teams}


def score_candidate(person: dict, name: str, team: str | None, position: str | None) -> float:
    """How well an MLB Stats API person matches a card. Higher is better."""
    pos = (person.get("primaryPosition") or {}).get("abbreviation", "")
    is_pitcher_card = position in PITCHER_POSITIONS
    role_ok = pos == "TWP" or (pos == "P") == is_pitcher_card
    org = _team_org_names().get((person.get("currentTeam") or {}).get("id"), "")
    team_ok = bool(team) and team != "Free Agents" and org == _TEAM_ALIASES.get(team, team)
    return (
        3.0 * role_ok
        + 2.0 * team_ok
        + 1.0 * (_norm_name(person.get("fullName", "")) == _norm_name(name))
        + 0.5 * bool(person.get("active"))
        + 0.5 * bool(person.get("mlbDebutDate"))
    )


def resolve_player(client, name: str, team: str | None, position: str | None) -> int | None:
    """Pick the MLB player a card represents among every same-name candidate.

    Name search alone returns the wrong player for shared names (two Max Muncys,
    a pitcher and an infielder both called Luis Garcia), so candidates are
    scored on role, organisation, exact name and activity.
    """
    try:
        data = client._get("/people/search", params={"names": name, "hydrate": "currentTeam"})
    except Exception:
        return None
    people = data.get("people", [])
    if not people:
        return None
    best = max(people, key=lambda p: score_candidate(p, name, team, position))
    # Require at least the right role; a wrong-role match feeds the wrong stats.
    return best["id"] if score_candidate(best, name, team, position) >= 3.0 else None


def link_cards_to_mlb_ids(game_year: int = 26, limit: int | None = None, verify_all: bool = False) -> int:
    """Link unlinked cards, and re-resolve existing links that look wrong.

    An existing link is re-checked when its player has the wrong role for the
    card, plays for a different organisation, or is shared with another card.
    """
    import requests

    from src.config import MLB_STATS_API
    from src.ingest.mlb_stats import MLBStatsClient

    Session = init_db()
    client = MLBStatsClient()
    changed = 0

    with Session() as session:
        cards = session.query(CardSnapshot).filter_by(game_year=game_year).all()
        ids = sorted({c.mlb_player_id for c in cards if c.mlb_player_id})
        people: dict[int, dict] = {}
        for i in range(0, len(ids), 150):
            chunk = ",".join(map(str, ids[i:i + 150]))
            resp = requests.get(f"{MLB_STATS_API}/people", params={"personIds": chunk, "hydrate": "currentTeam"}, timeout=60)
            people.update({p["id"]: p for p in resp.json().get("people", [])})
        shared = Counter(c.mlb_player_id for c in cards if c.mlb_player_id)

        def suspicious(card) -> bool:
            p = people.get(card.mlb_player_id)
            if p is None or shared[card.mlb_player_id] > 1 or verify_all:
                return True
            return score_candidate(p, card.player_name, card.team, card.position) < 5.0 and card.team != "Free Agents"

        todo = [c for c in cards if not c.mlb_player_id or suspicious(c)]
        if limit:
            todo = todo[:limit]
        for i, card in enumerate(todo, 1):
            mlb_id = resolve_player(client, card.player_name, card.team, card.position)
            if mlb_id and mlb_id != card.mlb_player_id:
                card.mlb_player_id = mlb_id
                changed += 1
            elif (
                not mlb_id
                and card.mlb_player_id in people
                and score_candidate(people[card.mlb_player_id], card.player_name, card.team, card.position) < 3.0
            ):
                card.mlb_player_id = None  # wrong-role link is worse than none
                changed += 1
            if i % 50 == 0:
                session.commit()
                print(f"  Checked {i}/{len(todo)} card links ({changed} changed)...", flush=True)
        session.commit()

    return changed


def get_card_attributes_dict(card: CardSnapshot) -> dict:
    if not card.attributes_json:
        return {}
    return json.loads(card.attributes_json)
