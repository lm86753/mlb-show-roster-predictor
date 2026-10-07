#!/usr/bin/env python3
"""Build the training dataset, run the walk-forward backtest, and fit the model."""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.dataset import build_training_dataset
from src.ingest.roster_updates import backfill_roster_updates
from src.models.train import train_all


def main():
    parser = argparse.ArgumentParser(description="Train the roster update predictor")
    parser.add_argument("--game-year", type=int, default=26)
    parser.add_argument("--skip-backfill", action="store_true", help="Don't pull new SDS roster updates first")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    warnings.filterwarnings("ignore", category=UserWarning)

    if not args.skip_backfill:
        print("Backfilling roster updates...")
        print(json.dumps(backfill_roster_updates([args.game_year]), indent=2))

    print("Building training dataset...")
    df = build_training_dataset(args.game_year)
    print(f"Training rows: {len(df):,} across {df['as_of'].nunique()} attribute updates")

    summary = train_all(df)
    print(json.dumps(summary["summary"], indent=2))


if __name__ == "__main__":
    main()
