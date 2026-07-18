from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from src.config import GAME_BASE_URLS, RAW_DIR
from src.models.registry import normalize_attr_name


class SDSClient:
    def __init__(self, game_year: int = 26, timeout: int = 60, delay: float = 0.3):
        self.game_year = game_year
        self.base_url = GAME_BASE_URLS[game_year]
        self.timeout = timeout
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "mlb-show-roster-predictor/1.0"})

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        time.sleep(self.delay)
        return resp.json()

    def list_roster_updates(self) -> list[dict]:
        data = self._get("/apis/roster_updates.json")
        return data.get("roster_updates", [])

    def get_roster_update(self, update_id: int) -> dict:
        return self._get("/apis/roster_update.json", params={"id": update_id})

    def list_items(self, page: int = 1, series: str = "Live") -> dict:
        return self._get(
            "/apis/items.json",
            params={"type": "mlb_card", "page": page, "series": series},
        )

    def get_item(self, uuid: str) -> dict:
        data = self._get("/apis/item.json", params={"uuid": uuid})
        return data.get("item", data)


def parse_update_date(name: str) -> str | None:
    """Parse 'June 18, 2026' -> ISO date string."""
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(name.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def parse_delta(delta_str: str) -> int:
    if not delta_str:
        return 0
    cleaned = delta_str.replace("+", "").strip()
    try:
        return int(cleaned)
    except ValueError:
        return 0





def extract_card_attributes(item: dict) -> dict[str, int]:
    attrs = {}
    for key, val in item.items():
        if isinstance(val, int) and key not in {
            "ovr", "age", "stamina", "series_year", "jersey_number", "ui_anim_index",
        }:
            if any(
                key.startswith(p)
                for p in (
                    "contact_", "power_", "plate_", "batting_", "bunting_", "drag_",
                    "fielding_", "arm_", "blocking", "speed", "baserunning_",
                    "pitch_", "bb_per_", "hr_per_", "hitting_", "pitching_",
                )
            ):
                attrs[key] = val
    return attrs


def parse_attribute_changes(
    update_data: dict,
    game_year: int,
    update_id: int,
    update_name: str,
) -> list[dict]:
    rows = []
    update_date = parse_update_date(update_name)
    for entry in update_data.get("attribute_changes", []):
        item = entry.get("item", {})
        card_uuid = entry.get("obfuscated_id") or item.get("uuid", "")
        player_name = entry.get("name") or item.get("name", "")
        team = entry.get("team") or item.get("team", "")
        position = item.get("display_position", "")
        is_hitter = 0 if position in {"SP", "RP", "CP"} else 1
        ovr_before = entry.get("old_rank") or item.get("ovr")
        ovr_after = entry.get("current_rank") or item.get("new_rank") or item.get("ovr")
        rarity_before = entry.get("old_rarity", "")
        rarity_after = entry.get("current_rarity") or item.get("rarity", "")

        if not entry.get("changes"):
            continue

        for change in entry["changes"]:
            attr = normalize_attr_name(change.get("name", ""), game_year)
            delta = parse_delta(change.get("delta", "0"))
            try:
                rating_after = int(change.get("current_value", 0))
            except (TypeError, ValueError):
                rating_after = 0
            rating_before = rating_after - delta
            rows.append(
                {
                    "game_year": game_year,
                    "update_id": update_id,
                    "update_name": update_name,
                    "update_date": update_date,
                    "card_uuid": card_uuid,
                    "player_name": player_name,
                    "team": team,
                    "position": position,
                    "is_hitter": is_hitter,
                    "attribute_name": attr,
                    "rating_before": rating_before,
                    "rating_after": rating_after,
                    "delta": delta,
                    "ovr_before": ovr_before,
                    "ovr_after": ovr_after,
                    "rarity_before": rarity_before,
                    "rarity_after": rarity_after,
                }
            )
    return rows


def save_raw_update(game_year: int, update_id: int, data: dict) -> Path:
    out_dir = RAW_DIR / f"mlb{game_year}" / "roster_updates"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"update_{update_id}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
