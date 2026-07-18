#!/usr/bin/env python3
"""Fetch high-resolution MLB The Show card images (-baked-lg.webp) for all
cards that have predictions, saving them as PNG in data/card_images_real/.

Previous fetches used the "-baked-sm" (210x296) variant. The "-baked-lg"
variant is 363x512 and uses the same aspect ratio, so it displays identically
but much sharper in the dashboard.

Usage:
    python scripts/fetch_card_images_lg.py
    python scripts/fetch_card_images_lg.py --dry-run
    python scripts/fetch_card_images_lg.py --force   # re-download even if present
"""
from __future__ import annotations

import argparse
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import init_db, Prediction  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "data" / "card_images_real"
CDN = "https://cards.theshow.com/mlb26/{uuid}-baked-lg.webp"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "image/webp,image/png,image/*,*/*;q=0.8",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch high-res card images.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    Session = init_db()
    with Session() as session:
        uuids = {r[0] for r in session.query(Prediction.card_uuid).all() if r[0]}
    print(f"[lg] {len(uuids)} prediction card UUIDs")

    uuids = sorted(uuids)
    if args.limit:
        uuids = uuids[: args.limit]

    if args.dry_run:
        print("[lg] DRY RUN — would fetch", len(uuids), "images")
        return

    downloaded = skipped = failed = 0
    for uuid in uuids:
        out = OUTPUT_DIR / f"{uuid}.png"
        if out.exists() and not args.force:
            skipped += 1
            continue
        try:
            r = requests.get(CDN.format(uuid=uuid), headers=HEADERS, timeout=30, allow_redirects=True)
            if r.status_code != 200:
                failed += 1
                continue
            img = Image.open(BytesIO(r.content))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")
            # Only replace if we got a strictly larger image than what we have
            if out.exists() and not args.force:
                try:
                    old = Image.open(out)
                    if old.size[0] >= img.size[0]:
                        skipped += 1
                        continue
                except Exception:
                    pass
            img.save(out, "PNG", optimize=True)
            downloaded += 1
            if downloaded % 50 == 0:
                print(f"  ... {downloaded} downloaded")
            time.sleep(0.1)
        except Exception:
            failed += 1

    print(f"\n[lg] Done. downloaded={downloaded} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
