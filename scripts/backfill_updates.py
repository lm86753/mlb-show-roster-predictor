#!/usr/bin/env python3
"""Backfill SDS roster updates for MLB The Show 22-26."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GAME_YEARS
from src.ingest.roster_updates import backfill_roster_updates


def main():
    parser = argparse.ArgumentParser(description="Backfill SDS roster updates")
    parser.add_argument("--years", type=int, nargs="+", default=GAME_YEARS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stats = backfill_roster_updates(game_years=args.years, force=args.force)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
