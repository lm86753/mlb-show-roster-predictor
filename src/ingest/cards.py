from __future__ import annotations

import json

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
                        attributes_json=dumps(attrs),
                    )
                )
                stats["cards"] += 1
            session.commit()

    return stats


def link_cards_to_mlb_ids(game_year: int = 26, limit: int | None = None) -> int:
    from src.ingest.mlb_stats import MLBStatsClient

    Session = init_db()
    client = MLBStatsClient()
    linked = 0

    with Session() as session:
        q = session.query(CardSnapshot).filter_by(game_year=game_year).filter(
            CardSnapshot.mlb_player_id.is_(None)
        )
        if limit:
            q = q.limit(limit)
        cards = q.all()
        for i, card in enumerate(cards, 1):
            mlb_id = client.search_player(card.player_name)
            if mlb_id:
                card.mlb_player_id = mlb_id
                linked += 1
            if i % 50 == 0:
                session.commit()
                print(f"  Linked {linked}/{i} cards...", flush=True)
        session.commit()

    return linked


def get_card_attributes_dict(card: CardSnapshot) -> dict:
    if not card.attributes_json:
        return {}
    return json.loads(card.attributes_json)
