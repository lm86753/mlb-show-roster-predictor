#!/usr/bin/env python3
"""Backtest checks for the attribute-update model.

1. Stat cutoff: SDS rates players on stats frozen some days before an update
   ships. Run the walk-forward backtest with training stats cut off N days
   early and report which alignment predicts best.
2. Shuffle test (--shuffle): shuffle the outcomes within each update. A
   pipeline with no leakage must fall back to base-rate skill.
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.dataset import build_training_dataset
from src.models.train import summarize, walk_forward

KEYS = [
    "top25_up_hit_rate", "top25_down_hit_rate", "ovr_spearman", "upgrade_brier",
    "upgrade_brier_baseline", "attr_skill", "attr_spearman", "top25_ev_avg_qs_gain",
]


def backtest(data) -> dict:
    _, folds, _ = walk_forward(data)
    s = summarize(folds)
    return {k: round(s[k], 4) for k in KEYS if s.get(k) is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lags", type=int, nargs="+", default=[0, 3, 7, 10, 14])
    parser.add_argument("--shuffle", action="store_true", help="Also run the shuffled-outcome leakage check")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")

    results = {}
    for lag in args.lags:
        data = build_training_dataset(save=False, refresh_hours=None, lag_days=lag)
        results[lag] = backtest(data)
        print(f"cutoff {lag:>2}d before update: {json.dumps(results[lag])}", flush=True)

    if args.shuffle:
        data = build_training_dataset(save=False, refresh_hours=None, lag_days=args.lags[0])
        rng = np.random.default_rng(0)
        for _, idx in data.groupby("as_of").groups.items():
            data.loc[idx, "delta"] = rng.permutation(data.loc[idx, "delta"].to_numpy())
            players = data.loc[idx].drop_duplicates("card_uuid")["card_uuid"]
            ovr_after = data.loc[idx].groupby("card_uuid")["ovr_after"].first()
            ovr_before = data.loc[idx].groupby("card_uuid")["ovr_before"].first()
            shuffled = dict(zip(players, rng.permutation((ovr_after - ovr_before).loc[players].to_numpy())))
            data.loc[idx, "ovr_after"] = data.loc[idx, "ovr_before"] + data.loc[idx, "card_uuid"].map(shuffled)
        print(f"SHUFFLED outcomes:          {json.dumps(backtest(data))}", flush=True)


if __name__ == "__main__":
    main()
