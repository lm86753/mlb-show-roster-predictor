# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

A small group of friends who play MLB The Show 26 Diamond Dynasty and invest stubs in Live Series
cards ahead of roster updates, then sell or quicksell after the update. They are knowledgeable
about the game and its market; they are not data scientists.

## Product Purpose

Roster Forecast predicts how every Live Series card's attributes and overall rating will move at
the next monthly attribute update, with calibrated odds, and turns that into expected quicksell
value. Success: the group finds cards worth buying before an update, and can look up any specific
card to see its forecast and why.

## Positioning

Forecasts come from a model trained on this season's real attribute updates against point-in-time
MLB game stats, and its accuracy is backtested and shown in the product rather than asserted.

## Operating Context

- Primary device: desktop or laptop, planning sessions between updates.

- Two equal jobs: browsing a ranked shortlist (likely upgrades, likely downgrades, best quicksell
  value) and looking up a specific player.
- Used between monthly attribute updates (roughly every four to five weeks); a daily pipeline
  refreshes cards, stats and predictions.
- Data: SDS roster-update and card APIs, MLB Stats API game logs. Card art comes from The Show's CDN.

## Capabilities and Constraints

- Tool set (TBR parity where real data exists): player projections, model-built buy lists,
  investment rankings, MLB stats tool, track record, flip/profit calculator. Not built: the card
  exchange calculator and game-mode guides, whose rules we don't have as data.

- React + Vite frontend, FastAPI backend; can also ship as a static build with a frozen JSON payload.
- Quicksell values follow the real Live Series table; market prices are not available.
- Only MLB 26 has point-in-time stats, so the model learns from that season's attribute updates.
- Fan project, not affiliated with San Diego Studio or MLB; no official logos.

## Brand Commitments

- The product name is "Roster Forecast".
- Look like an established business, not a gamer site: monotone and restrained, no neon or glow.
  (User choice, Oct 2026, taking the category-standard route over the dealt directions.)
- Feature reference: tbrshowtime.com (TBR). Match its tool set and navigation model with our own
  name, copy and data; never reuse its branding, logo, copy, or member reviews.

## Evidence on Hand

- Walk-forward backtest results in `data/models/backtest_summary.json` (top-25 hit rates, Brier
  score, realized quicksell gains). No testimonials, users counts, or press exist; do not invent them.

## Product Principles

1. Honest odds over hype: show probabilities and ranges, and the track record behind them.
2. Shortlist and lookup are equally first-class.
3. Speak the game's language (OVR, rarity tiers, quicksell, stubs) without explaining it to death.
4. Dense but calm: lots of cards and numbers, readable at a glance.
