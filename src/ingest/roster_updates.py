from __future__ import annotations

from src.config import GAME_YEARS
from src.db import AttributeChange, RosterUpdate, init_db
from src.ingest.sds_client import (
    SDSClient,
    parse_attribute_changes,
    save_raw_update,
)


def backfill_roster_updates(game_years: list[int] | None = None, force: bool = False) -> dict:
    """Fetch and store all SDS roster updates for given game years."""
    game_years = game_years or GAME_YEARS
    Session = init_db()
    stats = {"updates": 0, "changes": 0, "errors": []}

    with Session() as session:
        for year in game_years:
            client = SDSClient(game_year=year)
            try:
                updates = client.list_roster_updates()
            except Exception as exc:
                stats["errors"].append(f"mlb{year} list: {exc}")
                continue

            for upd in updates:
                update_id = upd["id"]
                update_name = upd["name"]

                existing = (
                    session.query(RosterUpdate)
                    .filter_by(game_year=year, update_id=update_id)
                    .first()
                )
                if existing and not force:
                    continue

                try:
                    data = client.get_roster_update(update_id)
                    raw_path = save_raw_update(year, update_id, data)
                except Exception as exc:
                    stats["errors"].append(f"mlb{year} update {update_id}: {exc}")
                    continue

                if existing:
                    session.delete(existing)
                    session.query(AttributeChange).filter_by(
                        game_year=year, update_id=update_id
                    ).delete()

                session.add(
                    RosterUpdate(
                        game_year=year,
                        update_id=update_id,
                        update_name=update_name,
                        update_date=parse_update_date_safe(update_name),
                        raw_json_path=str(raw_path),
                    )
                )

                rows = parse_attribute_changes(data, year, update_id, update_name)
                for row in rows:
                    session.add(AttributeChange(**row))
                stats["updates"] += 1
                stats["changes"] += len(rows)

            session.commit()

    return stats


def parse_update_date_safe(name: str) -> str | None:
    from src.ingest.sds_client import parse_update_date

    return parse_update_date(name)
