#!/usr/bin/env python3
"""Pre-compute momentum features and store in player_stat_windows table.

Iterates over all (player, date) pairs already present in the
player_stat_windows table (or attribute_changes table), fetches individual
game logs from the MLB Stats API, computes momentum features, and stores
them back as new rows with window='momentum'.

Usage:
    python scripts/compute_momentum.py                  # all players, current season
    python scripts/compute_momentum.py --season 2026     # specific season
    python scripts/compute_momentum.py --player 456781   # single player
    python scripts/compute_momentum.py --dry-run         # print, don't write
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR
from src.db import AttributeChange, PlayerStatWindow, init_db, dumps
from src.features.momentum import MomentumComputer, GameLogFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("compute_momentum")


def get_target_players(session_factory, season: int | None, player_id: int | None):
    """Get unique (player_id, is_hitter, as_of_date) tuples to process."""
    Session = session_factory

    with Session() as session:
        # Get unique (player_id, is_hitter, date) from attribute_changes
        query = (
            session.query(
                AttributeChange.mlb_player_id,
                AttributeChange.is_hitter,
                AttributeChange.update_date,
            )
            .filter(AttributeChange.mlb_player_id.isnot(None))
            .filter(AttributeChange.update_date.isnot(None))
            .distinct()
        )

        if player_id is not None:
            query = query.filter(AttributeChange.mlb_player_id == player_id)

        rows = query.all()

    # Filter by season
    results = []
    for mlb_id, is_hitter, update_date in rows:
        if update_date is None:
            continue
        yr = int(update_date[:4])
        if season is not None and yr != season:
            continue
        results.append((mlb_id, bool(is_hitter), update_date))

    return results


def process_player(
    player_id: int,
    is_hitter: bool,
    as_of_date: str,
    season: int,
    computer: MomentumComputer,
    dry_run: bool = False,
) -> dict[str, float] | None:
    """Compute and store momentum features for one (player, date)."""
    # Only use game logs UP TO as_of_date
    games = [
        g for g in computer._get_games(player_id, season, is_hitter)
        if g["date"] <= as_of_date
    ]

    if len(games) < 3:
        logger.debug(
            "Skipping %s/%s: only %d games available (need >= 3)",
            player_id, as_of_date, len(games),
        )
        return None

    # Temporarily override the cache to use filtered games
    cache_key = (player_id, season, is_hitter)
    computer._games_cache[cache_key] = games
    # Invalidate stat array cache for this key
    computer._stat_cache.pop(cache_key, None)

    try:
        stat_arrays = computer._get_stat_arrays(
            player_id, season, is_hitter, games
        )
        features: dict[str, float] = {}
        features.update(computer.compute_trend(games, stat_arrays, is_hitter))
        features.update(computer.compute_streak(stat_arrays, is_hitter))
        features.update(computer.compute_consistency(stat_arrays, is_hitter))
        features.update(computer.compute_volatility_clustering(stat_arrays, is_hitter))
    except Exception as exc:
        logger.error("Error computing momentum for %s/%s: %s", player_id, as_of_date, exc)
        return None

    if dry_run:
        logger.info(
            "[DRY RUN] %s/%s: %s", player_id, as_of_date,
            {k: round(v, 4) for k, v in features.items()},
        )
        return features

    return features


def main():
    parser = argparse.ArgumentParser(description="Compute momentum features for players")
    parser.add_argument("--season", type=int, default=None, help="Season year (default: current)")
    parser.add_argument("--player", type=int, default=None, help="Single MLB player ID")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing to DB")
    parser.add_argument("--delay", type=float, default=0.15, help="API delay between requests (seconds)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    season = args.season or datetime.utcnow().year
    session_factory = init_db()

    targets = get_target_players(session_factory, season, args.player)
    if not targets:
        logger.warning("No players found for season=%s player=%s", season, args.player)
        return

    logger.info(
        "Computing momentum for %d (player, date) pairs, season=%s",
        len(targets), season,
    )

    fetcher = GameLogFetcher(delay=args.delay)
    computer = MomentumComputer(fetcher=fetcher)
    Session = session_factory

    computed = 0
    skipped = 0
    errors = 0

    for i, (mlb_id, is_hitter, as_of_date) in enumerate(targets):
        if (i + 1) % 50 == 0:
            logger.info("Progress: %d/%d", i + 1, len(targets))

        features = process_player(
            mlb_id, is_hitter, as_of_date, season, computer, dry_run=args.dry_run,
        )

        if features is None:
            skipped += 1
            continue

        if not args.dry_run:
            try:
                with Session() as session:
                    existing = (
                        session.query(PlayerStatWindow)
                        .filter_by(
                            mlb_player_id=mlb_id,
                            as_of_date=as_of_date,
                            window="momentum",
                        )
                        .first()
                    )
                    if existing:
                        existing.stats_json = dumps(features)
                    else:
                        session.add(
                            PlayerStatWindow(
                                mlb_player_id=mlb_id,
                                as_of_date=as_of_date,
                                window="momentum",
                                is_hitter=int(is_hitter),
                                stats_json=dumps(features),
                            )
                        )
                    session.commit()
                computed += 1
            except Exception as exc:
                logger.error("DB write failed for %s/%s: %s", mlb_id, as_of_date, exc)
                errors += 1
        else:
            computed += 1

    logger.info(
        "Done. Computed: %d, Skipped: %d, Errors: %d",
        computed, skipped, errors,
    )


if __name__ == "__main__":
    main()
