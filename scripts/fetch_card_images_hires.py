#!/usr/bin/env python3
"""Fetch high-resolution card images (replace -sm with full size)."""
from __future__ import annotations
import sys, json, time, os
from pathlib import Path
import requests
from io import BytesIO
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "data" / "card_images_real"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "image/webp,image/png,image/*,*/*;q=0.8",
}

# Find existing files and purge small ones
existing = list(OUTPUT_DIR.glob("*.png"))
print(f"Existing images: {len(existing)}")
# Check size of first few
small_count = 0
for f in existing[:300]:
    img = Image.open(f)
    if img.size == (210, 296):
        small_count += 1
print(f"Small size images: {small_count}")

# Re-download larger versions
# First, get all UUIDs from filenames
uuids = [f.stem for f in existing]
print(f"UUIDs to process: {len(uuids)}")

# For each UUID, build the large image URL and download
downloaded = 0
failed = 0
skipped_big = 0

for uuid in uuids:
    out_path = OUTPUT_DIR / f"{uuid}.png"
    
    # Check if already large
    try:
        img = Image.open(out_path)
        if img.size[0] > 210:
            skipped_big += 1
            continue
    except:
        pass
    
    # Build large URL - try baked without -sm first
    large_url = f"https://cards.theshow.com/mlb26/{uuid}-baked.webp"
    
    try:
        resp = requests.get(large_url, headers=HEADERS, timeout=15, allow_redirects=True)
        if resp.status_code == 200:
            img = Image.open(BytesIO(resp.content))
            img.save(out_path, "PNG", optimize=True)
            downloaded += 1
        else:
            # Try alternative: baked-sm (already have)
            failed += 1
    except:
        failed += 1

    if downloaded % 100 == 0 and downloaded > 0:
        print(f"  ... {downloaded} downloaded")

print(f"Downloaded larger versions: {downloaded}")
print(f"Failed: {failed}")
print(f"Skipped (already large): {skipped_big}")
