#!/usr/bin/env python3
"""Daily pipeline: pull new updates + cards, retrain if a new attribute update
landed, then score every Live card for the next one."""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.dataset import build_live_dataset, build_training_dataset, load_changes, major_update_dates
from src.ingest.cards import fetch_live_series_cards, link_cards_to_mlb_ids
from src.ingest.roster_updates import backfill_roster_updates
from src.models.predict import run_predictions
from src.models.train import MODEL_PATH, load_summary, train_all


def needs_retrain(game_year: int) -> str | None:
    """Reason to retrain, or None if the saved model is current."""
    if not MODEL_PATH.exists():
        return "no trained model"
    majors = major_update_dates(load_changes(game_year))
    trained = load_summary().get("trained_on") or []
    if majors and (not trained or majors[-1] > trained[-1]):
        return f"new attribute update on {majors[-1]}"
    return None


def main():
    parser = argparse.ArgumentParser(description="Score every Live Series card for the next attribute update")
    parser.add_argument("--game-year", type=int, default=26)
    parser.add_argument("--skip-backfill", action="store_true", help="Don't pull new SDS roster updates")
    parser.add_argument("--skip-cards", action="store_true", help="Don't refetch the Live card list")
    parser.add_argument("--skip-link", action="store_true", help="Don't look up MLB IDs for new cards")
    parser.add_argument("--retrain", action="store_true", help="Retrain even if no new update landed")
    parser.add_argument("--refresh-hours", type=float, default=20, help="Refetch game logs older than this")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    warnings.filterwarnings("ignore", category=UserWarning)

    if not args.skip_backfill:
        print("Backfilling roster updates...")
        print(json.dumps(backfill_roster_updates([args.game_year])))
    if not args.skip_cards:
        print("Fetching Live Series cards...")
        print(json.dumps(fetch_live_series_cards(game_year=args.game_year)))
    if not args.skip_link:
        print(f"Linked {link_cards_to_mlb_ids(game_year=args.game_year)} new cards to MLB IDs")

    reason = "requested" if args.retrain else needs_retrain(args.game_year)
    if reason:
        print(f"Retraining ({reason})...")
        summary = train_all(build_training_dataset(args.game_year, refresh_hours=args.refresh_hours))
        print(json.dumps(summary["summary"], indent=2))

    live = build_live_dataset(args.game_year, refresh_hours=args.refresh_hours)
    players = run_predictions(live)
    cols = ["player_name", "ovr_before", "mu", "upgrade_probability", "expected_qs_change"]
    print(f"Scored {len(players)} cards. Top expected quicksell gains:")
    print(players.head(15)[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
