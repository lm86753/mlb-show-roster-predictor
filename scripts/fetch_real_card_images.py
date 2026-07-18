#!/usr/bin/env python3
"""Fetch real MLB The Show card images from the game's CDN.

This script:
1. Extracts card UUID -> baked_img URL from update JSON files
2. Downloads only cards that are in the predictions table
3. Caches them as PNG files in data/card_images/

Usage:
    python scripts/fetch_real_card_images.py
    python scripts/fetch_real_card_images.py --dry-run  # Show stats without downloading
    python scripts/fetch_real_card_images.py --limit 50   # Download first 50 only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from io import BytesIO
from PIL import Image

# ─── Project setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import init_db, Prediction  # noqa: E402

# ─── Constants ────────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "data" / "card_images_real"
UPDATE_DIRS = [
    PROJECT_ROOT / "data" / "raw" / "mlb26" / "roster_updates",
    PROJECT_ROOT / "data" / "raw" / "mlb22" / "roster_updates",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def extract_image_urls_from_updates() -> dict[str, str]:
    """Parse all update JSONs and build a mapping of card_uuid -> baked_img URL."""
    uuid_to_url: dict[str, str] = {}
    
    for update_dir in UPDATE_DIRS:
        if not update_dir.exists():
            print(f"[!] Update dir not found: {update_dir}")
            continue
        
        for json_file in sorted(update_dir.glob("*.json")):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                continue
            
            # Check all top-level keys for player-like objects
            for key in data:
                entries = data.get(key, [])
                if not isinstance(entries, list):
                    continue
                
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    
                    # Extract UUID from the item or top level
                    card_uuid = None
                    if "item" in entry and isinstance(entry["item"], dict):
                        card_uuid = entry["item"].get("uuid") or entry.get("obfuscated_id")
                        img_url = entry["item"].get("baked_img") or entry["item"].get("img")
                    else:
                        card_uuid = entry.get("uuid") or entry.get("obfuscated_id")
                        img_url = entry.get("baked_img") or entry.get("img")
                    
                    if card_uuid and img_url and isinstance(img_url, str):
                        # Prefer baked_img if available (higher quality)
                        if "baked" in img_url or card_uuid not in uuid_to_url:
                            uuid_to_url[str(card_uuid)] = img_url
    
    return uuid_to_url


def main():
    parser = argparse.ArgumentParser(description="Fetch real MLB The Show card images.")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without downloading.")
    parser.add_argument("--limit", type=int, default=None, help="Only download N cards.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--threads", type=int, default=8, help="Number of download threads.")
    args = parser.parse_args()
    
    # ── Prepare output dir ──────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Get prediction card UUIDs ──────────────────────────────────
    Session = init_db()
    with Session() as session:
        prediction_uuids = set()
        results = session.query(Prediction.card_uuid).all()
        prediction_uuids = {r[0] for r in results if r[0]}
    
    print(f"[fetch] {len(prediction_uuids)} card UUIDs in predictions table")
    
    # ── Extract image URLs from updates ────────────────────────────
    print("[fetch] Extracting image URLs from update JSONs...")
    uuid_to_url = extract_image_urls_from_updates()
    print(f"[fetch] Found {len(uuid_to_url)} total image URLs in updates")
    
    # Find which prediction UUIDs have image URLs
    download_tasks: list[tuple[str, str]] = []
    for uuid in prediction_uuids:
        if uuid in uuid_to_url:
            download_tasks.append((uuid, uuid_to_url[uuid]))
    
    # Also check for w/o hex variant
    remaining = prediction_uuids - {t[0] for t in download_tasks}
    for uuid in remaining:
        # Strip any potential formatting differences
        clean_uuid = uuid.replace('-', '').lower()
        if clean_uuid in uuid_to_url:
            download_tasks.append((uuid, uuid_to_url[clean_uuid]))
    
    print(f"[fetch] {len(download_tasks)} prediction cards have image URLs")
    
    if args.dry_run:
        print("[fetch] DRY RUN — no downloads")
        return
    
    if args.limit:
        download_tasks = download_tasks[:args.limit]
        print(f"[fetch] Limited to {args.limit} cards")
    
    # ── Download images ────────────────────────────────────────────
    print(f"[fetch] Downloading {len(download_tasks)} card images...")
    downloaded = 0
    failed = 0
    already_cached = 0
    
    for uuid, url in download_tasks:
        out_path = args.out_dir / f"{uuid}.png"
        if out_path.exists():
            already_cached += 1
            continue
        
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
            resp.raise_for_status()
            
            img = Image.open(BytesIO(resp.content))
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGBA')
            else:
                img = img.convert('RGB')
            
            img.save(out_path, "PNG", optimize=True)
            downloaded += 1
            
            if downloaded % 50 == 0:
                print(f"  ... {downloaded} downloaded")
            
            time.sleep(0.15)  # Be nice to the CDN
            
        except Exception as e:
            # print(f"  ERROR downloading {uuid}: {e}")
            failed += 1
    
    print(f"\n[fetch] Complete!")
    print(f"  Downloaded:     {downloaded}")
    print(f"  Already cached: {already_cached}")
    print(f"  Failed:         {failed}")
    print(f"  Total:          {len(download_tasks)}")
    print(f"  Output:         {args.out_dir}")


if __name__ == "__main__":
    main()
