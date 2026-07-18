#!/usr/bin/env python3
"""Export all predictions to a CSV file.

Reads every prediction row from the database and writes a CSV with columns:
player_name, team, position, current_ovr, current_rarity, predicted_ovr_delta,
projected_ovr, upgrade_probability, downgrade_probability, signal

The ``signal`` column is derived from the prediction:
  - "UP"    if upgrade_probability >= 0.60
  - "DOWN"  if downgrade_probability >= 0.60
  - "HOLD"  otherwise

Team, position, and projected_ovr come from the most recent card snapshot
in the ``attribute_changes`` table (the latest row for that mlb_player_id).
If no historical record exists those fields are empty/zero.

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

from src.db import AttributeChange, Prediction, init_db


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
    "signal",
]


def _derive_signal(up_prob: float, dn_prob: float) -> str:
    if up_prob >= 0.60:
        return "UP"
    if dn_prob >= 0.60:
        return "DOWN"
    return "HOLD"


def _latest_card_info(session, mlb_player_id: int | None) -> dict:
    """Return {team, position} from the most recent attribute_changes row."""
    if mlb_player_id is None:
        return {"team": "", "position": ""}
    row = (
        session.query(AttributeChange.team, AttributeChange.position)
        .filter(AttributeChange.mlb_player_id == mlb_player_id)
        .order_by(AttributeChange.update_date.desc(), AttributeChange.id.desc())
        .first()
    )
    if row:
        return {"team": row.team or "", "position": row.position or ""}
    return {"team": "", "position": ""}


def export_predictions(out_path: Path) -> int:
    Session = init_db()
    with Session() as session:
        predictions = (
            session.query(Prediction)
            .order_by(Prediction.upgrade_probability.desc())
            .all()
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()

            for p in predictions:
                info = _latest_card_info(session, p.mlb_player_id)
                projected = (
                    int(round(p.current_ovr + p.predicted_ovr_delta))
                    if p.current_ovr is not None and p.predicted_ovr_delta is not None
                    else ""
                )
                writer.writerow({
                    "player_name": p.player_name,
                    "team": info["team"],
                    "position": info["position"],
                    "current_ovr": p.current_ovr,
                    "current_rarity": p.current_rarity,
                    "predicted_ovr_delta": round(p.predicted_ovr_delta, 2) if p.predicted_ovr_delta is not None else "",
                    "projected_ovr": projected,
                    "upgrade_probability": round(p.upgrade_probability, 4) if p.upgrade_probability is not None else "",
                    "downgrade_probability": round(p.downgrade_probability, 4) if p.downgrade_probability is not None else "",
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
