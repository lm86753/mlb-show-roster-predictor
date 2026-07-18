#!/usr/bin/env python3
"""Daily prediction pipeline with T-7/T-3/T-1 horizons."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.engineering import build_live_features
from src.ingest.cards import fetch_live_series_cards, link_cards_to_mlb_ids
from src.models.predict import run_predictions


def main():
    parser = argparse.ArgumentParser(description="Run daily roster update predictions")
    parser.add_argument("--game-year", type=int, default=26)
    parser.add_argument("--horizons", type=int, nargs="+", default=[7, 3, 1])
    parser.add_argument("--skip-cards", action="store_true")
    parser.add_argument("--skip-link", action="store_true")
    parser.add_argument("--link-limit", type=int, default=None)
    args = parser.parse_args()

    if not args.skip_cards:
        print("Fetching Live Series cards...")
        card_stats = fetch_live_series_cards(game_year=args.game_year)
        print(json.dumps(card_stats, indent=2))

    if not args.skip_link:
        print("Linking cards to MLB IDs...")
        linked = link_cards_to_mlb_ids(game_year=args.game_year, limit=args.link_limit)
        print(f"Linked {linked} cards to MLB IDs")

    results = {}
    for horizon in args.horizons:
        print(f"Scoring T-{horizon} horizon...")
        live_df = build_live_features(game_year=args.game_year, horizon_days=horizon)
        preds = run_predictions(live_df, horizon_days=horizon, persist=True)
        results[f"T-{horizon}"] = {
            "players_scored": len(preds),
            "top_upgrades": preds.head(10)[
                ["player_name", "current_ovr", "predicted_ovr_delta", "upgrade_probability"]
            ].to_dict(orient="records") if not preds.empty else [],
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
