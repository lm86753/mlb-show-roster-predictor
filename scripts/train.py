#!/usr/bin/env python3
"""Build training dataset, join MLB stats, train models, and run backtest."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.engineering import build_training_dataset, persist_training_examples
from src.ingest.mlb_stats import store_stat_windows_for_updates
from src.ingest.roster_updates import backfill_roster_updates
from src.models.evaluate import run_backtest
from src.models.train import train_all


def main():
    parser = argparse.ArgumentParser(description="Train roster update predictor models")
    parser.add_argument("--skip-backfill", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--stats-years", type=int, nargs="+", default=[26])
    args = parser.parse_args()

    if not args.skip_backfill:
        print("Backfilling roster updates...")
        stats = backfill_roster_updates()
        print(json.dumps(stats, indent=2))

    if not args.skip_stats:
        print("Linking MLB stats to update dates (this may take a while)...")
        count = store_stat_windows_for_updates(game_years=args.stats_years)
        print(f"Stored {count} stat windows")

    print("Building training dataset...")
    df = build_training_dataset()
    print(f"Training examples: {len(df)}")
    persist_training_examples(df)

    print("Training models...")
    result = train_all()
    print(json.dumps(result, indent=2))

    print("Running backtest...")
    backtest = run_backtest(df)
    print(json.dumps(backtest, indent=2))


if __name__ == "__main__":
    main()
