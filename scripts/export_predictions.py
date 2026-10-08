#!/usr/bin/env python3
"""Export all predictions to a CSV file.

Reads every prediction row from the database and writes a CSV with columns:
player_name, team, position, current_ovr, current_rarity, predicted_ovr_delta,
projected_ovr, upgrade_probability, downgrade_probability, expected_qs_gain, signal

The ``signal`` column is derived from the prediction:
  - "UP"    if upgrade_probability >= 0.50
  - "DOWN"  if downgrade_probability >= 0.50
  - "HOLD"  otherwise

Team and position come from the current Live card snapshot.

Usage:
    python scripts/export_predictions.py               # writes data/predictions_export.csv
    python scripts/export_predictions.py --out custom.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import CardSnapshot, Prediction, init_db


COLUMNS = [
    "player_name",
    "team",
    "position",
    "current_ovr",
    "current_rarity",
    "predicted_ovr_delta",
    "projected_ovr",
    "upgrade_probability",
    "downgrade_probability",
    "expected_qs_gain",
    "signal",
]


def _derive_signal(up_prob: float, dn_prob: float) -> str:
    # Same thresholds as the dashboard's BUY/SELL signals.
    if up_prob >= 0.50 and up_prob > dn_prob:
        return "UP"
    if dn_prob >= 0.50 and dn_prob > up_prob:
        return "DOWN"
    return "HOLD"


def export_predictions(out_path: Path) -> int:
    Session = init_db()
    with Session() as session:
        predictions = (
            session.query(Prediction)
            .order_by(Prediction.upgrade_probability.desc())
            .all()
        )
        cards = {c.card_uuid: c for c in session.query(CardSnapshot)}

        out_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()

            for p in predictions:
                card = cards.get(p.card_uuid)
                projected = (
                    int(round(p.current_ovr + p.predicted_ovr_delta))
                    if p.current_ovr is not None and p.predicted_ovr_delta is not None
                    else ""
                )
                writer.writerow({
                    "player_name": p.player_name,
                    "team": card.team if card else "",
                    "position": card.position if card else "",
                    "current_ovr": p.current_ovr,
                    "current_rarity": p.current_rarity,
                    "predicted_ovr_delta": round(p.predicted_ovr_delta, 2) if p.predicted_ovr_delta is not None else "",
                    "projected_ovr": projected,
                    "upgrade_probability": round(p.upgrade_probability, 4) if p.upgrade_probability is not None else "",
                    "downgrade_probability": round(p.downgrade_probability, 4) if p.downgrade_probability is not None else "",
                    "expected_qs_gain": round(p.expected_value_per_card or 0.0, 1),
                    "signal": _derive_signal(p.upgrade_probability or 0.0, p.downgrade_probability or 0.0),
                })
                written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description="Export predictions to CSV")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/predictions_export.csv"),
        help="Output CSV path (default: data/predictions_export.csv)",
    )
    args = parser.parse_args()

    out_path = args.out
    print(f"[export_predictions] Exporting predictions -> {out_path}")
    count = export_predictions(out_path)
    print(f"[export_predictions] Wrote {count} rows to {out_path}")
    print("[export_predictions] Done.")


if __name__ == "__main__":
    main()
