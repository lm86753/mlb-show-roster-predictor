#!/usr/bin/env python3
"""Pre-generate all player card PNG images for the roster predictor dashboard.

Reads predictions (horizon_days=1) from the database, looks up team/position
from the most recent AttributeChange row, and renders a 400x560 PNG for each
of the 1826 cards.

Output: data/card_images/<card_uuid>.png

Card design:
  - Team color gradient background (top-left bright → bottom-right dark)
  - OVR number (large, top-left)
  - Delta badge (top-right, colored by direction:
      green for up (+), red for down (-), grey for flat (≈0))
  - Player initials in a circle (center)
  - Player full name + position (below initials)
  - Rarity badge (bottom-left, colored by tier:
      Common=grey, Bronze=brown, Silver=silver, Gold=gold, Diamond=cyan)
  - Signal/badge (bottom-right: BUY/HOLD/SELL based on delta)

Usage:
    python scripts/generate_card_images.py            # generate all
    python scripts/generate_card_images.py --limit 10 # debug: only first 10
    python scripts/generate_card_images.py --dry-run  # print stats, no files
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import or_

# ─── Project setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import AttributeChange, Prediction, init_db  # noqa: E402

# ─── Constants ────────────────────────────────────────────────────────────────

CARD_W, CARD_H = 400, 560  # output size in pixels

# Rarity badge colors (matches dashboard HTML version)
RARITY_COLORS = {
    "Common":   (107, 114, 128),   # grey
    "Bronze":   (205, 127, 50),    # brown
    "Silver":   (192, 192, 192),   # silver
    "Gold":     (255, 215, 0),     # gold
    "Diamond":  (79, 209, 255),    # cyan
}
DEFAULT_RARITY_COLOR = (107, 114, 128)  # fallback grey

# Delta badge colors
DELTA_UP   = (74, 222, 128)     # green
DELTA_DOWN = (248, 113, 113)    # red
DELTA_FLAT = (107, 114, 128)    # grey

# Signal thresholds
DELTA_BUY  = 0.5
DELTA_SELL = -0.5

# Team colors (same as dashboard.py TEAM_COLORS)
TEAM_COLORS = {
    "Dodgers":      (0,   90,  156),
    "Yankees":      (0,   48,  135),
    "Astros":       (0,   45,  98),
    "Braves":       (206, 17,  65),
    "Rangers":      (0,   50,  120),
    "Orioles":      (223, 70,  1),
    "Rays":         (9,   44,  92),
    "Phillies":     (232, 24,  40),
    "Twins":        (0,   43,  92),
    "Blue Jays":    (19,  74,  142),
    "Padres":       (47,  36,  29),
    "Giants":       (253, 90,  30),
    "Cardinals":    (196, 30,  58),
    "Cubs":         (14,  51,  134),
    "Brewers":      (255, 197, 47),
    "Mets":         (0,   45,  114),
    "Diamondbacks": (167, 25,  48),
    "Reds":         (198, 1,   31),
    "Pirates":      (253, 184, 39),
    "Rockies":      (51,  51,  102),
    "Red Sox":      (189, 48,  57),
    "White Sox":    (39,  37,  31),
    "Guardians":    (12,  35,  64),
    "Tigers":       (12,  35,  64),
    "Royals":       (0,   70,  135),
    "Angels":       (186, 0,   33),
    "Mariners":     (12,  44,  86),
    "Athletics":    (0,   56,  49),
    "Marlins":      (0,   163, 224),
    "Nationals":    (171, 0,   3),
}
DEFAULT_TEAM_COLOR = (30, 41, 59)  # dark slate

OUTPUT_DIR = PROJECT_ROOT / "data" / "card_images"


def lookup_team_color(team_name: str | None) -> tuple[int, int, int]:
    """Return RGB color for a team name fuzzy match."""
    if not team_name:
        return DEFAULT_TEAM_COLOR
    lower = team_name.lower()
    for name, rgb in TEAM_COLORS.items():
        if name.lower() in lower:
            return rgb
    return DEFAULT_TEAM_COLOR


def get_initials(name: str) -> str:
    """Get player initials from full name."""
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper()


def get_signal(pred_delta: float) -> tuple[str, tuple[int, int, int]]:
    """Return (signal_label, color) based on predicted OVR delta."""
    if pred_delta >= DELTA_BUY:
        return "BUY", DELTA_UP
    elif pred_delta <= DELTA_SELL:
        return "SELL", DELTA_DOWN
    else:
        return "HOLD", DELTA_FLAT


def draw_gradient_bg(
    draw: ImageDraw.Draw,
    w: int,
    h: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a team-color gradient: vibrant top-left fading to dark bottom-right."""
    for y in range(h):
        frac = y / h
        # Interpolate from bright → dark
        r = int(color[0] * (1 - frac * 0.7))
        g = int(color[1] * (1 - frac * 0.7))
        b = int(color[2] * (1 - frac * 0.7))
        draw.line([(0, y), (w, y)], fill=(r, g, b))


def draw_delta_badge(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    pred_delta: float,
    font: ImageFont.FreeTypeFont,
) -> None:
    """Draw the delta badge (top-right) with arrow + sign."""
    if pred_delta > 0.1:
        badge_color = DELTA_UP
        text = f"+{pred_delta:.1f}"
        arrow = "▲"
    elif pred_delta < -0.1:
        badge_color = DELTA_DOWN
        text = f"{pred_delta:.1f}"
        arrow = "▼"
    else:
        badge_color = DELTA_FLAT
        text = f"{pred_delta:+.1f}"
        arrow = "─"

    badge_text = f"{arrow} {text}"
    # Measure text width
    bbox = draw.textbbox((0, 0), badge_text, font=font)
    tw = bbox[2] - bbox[0]
    padding = 14
    box_w = tw + padding * 2
    box_h = 38

    # Rounded rectangle background
    draw.rounded_rectangle(
        [x - box_w, y, x, y + box_h],
        radius=8,
        fill=(*badge_color, 40),
        outline=badge_color,
        width=2,
    )
    # Text inside
    draw.text(
        (x - box_w + padding, y + 7),
        badge_text,
        fill=badge_color,
        font=font,
    )


def draw_rarity_badge(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    rarity: str,
    font: ImageFont.FreeTypeFont,
) -> None:
    """Draw the rarity badge (bottom-left) colored by tier."""
    tier = rarity.split()[0] if rarity else "Common"
    color = RARITY_COLORS.get(tier, DEFAULT_RARITY_COLOR)
    label = rarity.upper()

    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    padding = 12
    box_w = tw + padding * 2
    box_h = 34

    # Background (semi-transparent tint of the rarity color)
    draw.rounded_rectangle(
        [x, y, x + box_w, y + box_h],
        radius=6,
        fill=(*color, 50),
        outline=color,
        width=2,
    )
    draw.text(
        (x + padding, y + 6),
        label,
        fill=color,
        font=font,
    )


def draw_signal_badge(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    pred_delta: float,
    font: ImageFont.FreeTypeFont,
) -> None:
    """Draw the BUY/HOLD/SELL signal badge (bottom-right)."""
    signal, color = get_signal(pred_delta)
    bbox = draw.textbbox((0, 0), signal, font=font)
    tw = bbox[2] - bbox[0]
    padding = 14
    box_w = tw + padding * 2
    box_h = 34

    # Right-aligned box
    draw.rounded_rectangle(
        [x - box_w, y, x, y + box_h],
        radius=6,
        fill=color,
    )
    # White text on colored badge
    draw.text(
        (x - box_w + padding, y + 6),
        signal,
        fill=(255, 255, 255),
        font=font,
    )


def make_fonts():
    """Load TrueType fonts with fallbacks."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    font_large = font_med = font_small = None
    for path in candidates:
        try:
            font_large = ImageFont.truetype(path, 64)
            font_med = ImageFont.truetype(font=ImageFont.truetype(path, 28))
            font_small = ImageFont.truetype(path, 22)
            break
        except OSError:
            continue

    if font_large is None:
        font_large = ImageFont.load_default()
        font_med = ImageFont.load_default()
        font_small = ImageFont.load_default()

    return font_large, font_med, font_small


def render_card(
    player_name: str,
    team: str | None,
    position: str,
    ovr: int,
    rarity: str,
    pred_delta: float,
    font_ovr: ImageFont.FreeTypeFont,
    font_delta: ImageFont.FreeTypeFont,
    font_initials: ImageFont.FreeTypeFont,
    font_name: ImageFont.FreeTypeFont,
    font_pos: ImageFont.FreeTypeFont,
    font_badge: ImageFont.FreeTypeFont,
    font_signal: ImageFont.FreeTypeFont,
) -> Image.Image:
    """Render a single player card as a PIL Image."""
    color = lookup_team_color(team)

    # ── Background gradient ───────────────────────────────────────────────
    img = Image.new("RGB", (CARD_W, CARD_H), color)
    draw = ImageDraw.Draw(img)
    draw_gradient_bg(draw, CARD_W, CARD_H, color)

    # ── Card border ───────────────────────────────────────────────────────
    border_color = (*color, 255)
    draw.rounded_rectangle(
        [6, 6, CARD_W - 7, CARD_H - 7],
        radius=20,
        outline=(255, 255, 255, 60),
        width=3,
    )

    # ── OVR (top-left) ───────────────────────────────────────────────────
    draw.text((20, 20), str(ovr), fill=(255, 255, 255), font=font_ovr)

    # ── Delta badge (top-right) ──────────────────────────────────────────
    draw_delta_badge(draw, CARD_W - 20, 24, pred_delta, font_delta)

    # ── Initials circle (center-top) ──────────────────────────────────────
    cx, cy = CARD_W // 2, 195
    radius = 64
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=(20, 20, 25),
        outline=(255, 255, 255),
        width=3,
    )
    initials = get_initials(player_name)
    bbox = draw.textbbox((0, 0), initials, font=font_initials)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        (cx - tw / 2, cy - th / 2 - 4),
        initials,
        fill=(255, 255, 255),
        font=font_initials,
    )

    # ── Player name ───────────────────────────────────────────────────────
    name_y = 290
    display_name = player_name if len(player_name) <= 20 else player_name[:18] + "…"
    bbox = draw.textbbox((0, 0), display_name, font=font_name)
    tw = bbox[2] - bbox[0]
    draw.text(
        ((CARD_W - tw) // 2, name_y),
        display_name,
        fill=(240, 246, 252),
        font=font_name,
    )

    # ── Position ─────────────────────────────────────────────────────────
    pos_y = name_y + 36
    display_pos = position if position else "N/A"
    bbox = draw.textbbox((0, 0), display_pos, font=font_pos)
    tw = bbox[2] - bbox[0]
    draw.text(
        ((CARD_W - tw) // 2, pos_y),
        display_pos,
        fill=(180, 190, 200),
        font=font_pos,
    )

    # ── Team name (subtitle) ─────────────────────────────────────────────
    team_y = pos_y + 30
    display_team = team if team else ""
    if display_team:
        bbox = draw.textbbox((0, 0), display_team, font=font_pos)
        tw = bbox[2] - bbox[0]
        draw.text(
            ((CARD_W - tw) // 2, team_y),
            display_team,
            fill=tuple(min(c + 60, 255) for c in color),
            font=font_pos,
        )

    # ── Bottom badges ────────────────────────────────────────────────────
    badge_y = CARD_H - 60

    # Rarity badge (left)
    draw_rarity_badge(draw, 20, badge_y, rarity, font_badge)

    # Signal badge (right)
    draw_signal_badge(draw, CARD_W - 20, badge_y, pred_delta, font_signal)

    return img


def main():
    parser = argparse.ArgumentParser(description="Pre-generate player card PNG images.")
    parser.add_argument("--limit", type=int, default=None, help="Only generate N cards (for testing).")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing files.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args()

    Session = init_db()
    with Session() as session:
        predictions = (
            session.query(Prediction)
            .filter_by(horizon_days=1)
            .order_by(Prediction.card_uuid)
            .all()
        )

        if args.limit:
            predictions = predictions[: args.limit]

        print(f"[generate_card_images] Found {len(predictions)} cards to render.")

        if args.dry_run:
            print(f"[generate_card_images] DRY RUN — no files written.")
            print(f"[generate_card_images] Would write to: {args.out_dir}")
            return

        # ── Prepare output dir ────────────────────────────────────────────
        args.out_dir.mkdir(parents=True, exist_ok=True)

        # ── Load fonts ────────────────────────────────────────────────────
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]
        bold_path = regular_path = None
        for p in candidates:
            try:
                ImageFont.truetype(p, 28)
                if "bold" in p.lower() or "bd" in p.lower():
                    bold_path = p
                else:
                    regular_path = p
            except OSError:
                continue
        if bold_path is None:
            bold_path = regular_path
        if regular_path is None:
            regular_path = bold_path or "C:/Windows/Fonts/arial.ttf"

        font_ovr = ImageFont.truetype(bold_path, 64)
        font_initials = ImageFont.truetype(bold_path, 52)
        font_delta = ImageFont.truetype(regular_path, 24)
        font_name = ImageFont.truetype(bold_path, 28) if bold_path else ImageFont.load_default()
        font_pos = ImageFont.truetype(regular_path, 22)
        font_badge = ImageFont.truetype(regular_path, 20)
        font_signal = ImageFont.truetype(bold_path if bold_path else regular_path, 22)

        # ── Render loop ──────────────────────────────────────────────────
        written = 0
        errors = 0
        skipped = 0

        # Pre-fetch all latest attribute_changes for efficiency:
        # Build a dict: mlb_player_id -> (team, position)
        print("[generate_card_images] Pre-fetching team/position data...")
        latest_info: dict[int, tuple[str | None, str | None]] = {}
        all_changes = (
            session.query(
                AttributeChange.mlb_player_id,
                AttributeChange.team,
                AttributeChange.position,
                AttributeChange.update_date,
            )
            .filter(AttributeChange.mlb_player_id.isnot(None))
            .order_by(AttributeChange.mlb_player_id, AttributeChange.update_date.asc(), AttributeChange.id.asc())
            .all()
        )
        for row in all_changes:
            mlb_id, team_val, pos_val, _ = row
            # Keep last (most recent update_date wins)
            latest_info[mlb_id] = (team_val, pos_val)

        print(f"[generate_card_images] Team info for {len(latest_info)} MLB players.")
        print("[generate_card_images] Rendering cards...")

        for i, p in enumerate(predictions):
            try:
                # Look up team/position from pre-fetched dict
                info = latest_info.get(p.mlb_player_id, (None, None))
                team = info[0] if info and info[0] else None
                position = info[1] if info and info[1] else "N/A"

                # Filter for non-empty team
                if team == "":
                    team = None

                img = render_card(
                    player_name=p.player_name or "",
                    team=team,
                    position=position or "N/A",
                    ovr=p.current_ovr or 0,
                    rarity=p.current_rarity or "Common",
                    pred_delta=p.predicted_ovr_delta or 0.0,
                    font_ovr=font_ovr,
                    font_delta=font_delta,
                    font_initials=font_initials,
                    font_name=font_name,
                    font_pos=font_pos,
                    font_badge=font_badge,
                    font_signal=font_signal,
                )

                out_path = args.out_dir / f"{p.card_uuid}.png"
                img.save(str(out_path), "PNG")
                written += 1

                if (i + 1) % 200 == 0:
                    print(f"  ... {i + 1}/{len(predictions)} done")

            except Exception as exc:
                print(f"  ERROR card_uuid={p.card_uuid} ({p.player_name}): {exc}")
                errors += 1

        # ── Summary ───────────────────────────────────────────────────────
        print(f"[generate_card_images] Complete!")
        print(f"  Written:  {written}")
        print(f"  Errors:   {errors}")
        print(f"  Skipped:  {skipped}")
        print(f"  Output:   {args.out_dir}")


if __name__ == "__main__":
    main()
