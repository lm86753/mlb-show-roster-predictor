# MLB The Show 26 Roster Update Predictor

Predicts which Live Series cards move at the next monthly **attribute update**, how likely
each OVR move is, and what that's worth in quicksell stubs.

## How it works

**Data.** Every SDS roster update for MLB 26 (`/apis/roster_update.json`), the Live card list,
and per-game MLB logs from the MLB Stats API. The monthly attribute updates (the ones that move
~700 cards) are the training targets; the small weekly updates are injury/role changes.

Card art is loaded straight from The Show's CDN (`cards.theshow.com/mlb26/{uuid}-baked-{sm,lg}.webp`),
which re-renders cards after every update, so art and OVR badges are always current.

**Training rows.** For each attribute update, every Live card that existed at the time is
reconstructed *as it stood before the update* (ratings and OVR rebuilt from the change history),
with stat windows (7/14/30-day, season-to-date, last season, and a sample-size-regressed blend)
built only from games played before that date. Core attributes that weren't touched are real
"no change" examples. Live predictions go through the same code with today's date.

**Model** (`src/models/train.py`), per stat group (hitting / pitching):
- a per-attribute Ridge "what rating do these stats deserve" projector → gap feature
- LightGBM delta regressor + an analog (k-NN) regressor, stacked with non-negative weights
- LightGBM classifier for P(attribute up / down)
- a learned map from attribute deltas to OVR delta, then a multinomial model of the
  integer OVR move (−4…+4) that gives calibrated upgrade/downgrade/tier-jump probabilities and
  expected quicksell change

Stacking weights, prediction intervals and the OVR distribution are all fit on out-of-fold
predictions from a walk-forward backtest, so the reported numbers are what you'd have seen live.

## Backtest (walk-forward over the 2026 attribute updates)

Averaged over the Jul 16 – Oct 2 updates, each predicted using only earlier data:

| | Model | Baseline |
|---|---|---|
| Top-25 predicted upgrades that went up | 93% (avg +3.9 OVR) | 15% base rate |
| Top-25 predicted downgrades that went down | 81% | 11% base rate |
| Realized QS gain/card, top-25 expected-value picks | +796 stubs | +3 (average card) |
| Upgrade-probability Brier score | 0.076 | 0.126 |
| Attribute direction (when it moved, model called ≥0.5) | 78% | — |
| 80% attribute intervals containing the outcome | 84% | — |

Training and backtest stats are cut off 3 days before each update ships, which is when SDS
appears to freeze the numbers it rates on (`scripts/tune_stat_cutoff.py` sweeps the cutoff and
runs a shuffled-outcome leakage check). Full per-update results are in
`data/models/backtest_summary.json` and the dashboard's Track Record page.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Pull updates + cards, retrain if a new attribute update landed, score every card
python scripts/daily_predict.py

# Force a retrain (also prints the backtest)
python scripts/train.py

# API + dashboard (or just run_dashboard.bat on Windows)
uvicorn src.api.main:app --reload
cd web && npm install && npm run dev
```

`scripts/export_static_predictions.py` freezes the dashboard payload to
`data/static_predictions.json` for a DB-less deploy.

## Project structure

```
src/ingest/     SDS roster updates + cards, MLB Stats API client
src/features/   windows.py (point-in-time stat windows), dataset.py (train/live rows)
src/models/     train.py (models + walk-forward backtest), predict.py (live scoring)
src/api/        FastAPI: /dashboard, /player/{uuid}, /player-history/{uuid}, /accuracy
web/            React + Vite dashboard (Projections, Buy Lists, Player Stats, Track Record, player drawer)
scripts/        daily_predict.py, train.py, tune_stat_cutoff.py, exports
```

## Limitations

- Only one season (MLB 26) has point-in-time stats, so the model has six attribute updates to
  learn from; early-season predictions will be rougher than late-season ones.
- L/R split stats aren't available as of past dates from the MLB API, so contact/power L vs R
  lean on the card's existing split ratings plus overall stats.
- Quicksell math assumes you buy at the quicksell floor; market prices are higher.

## Disclaimer

Predictions are estimates. San Diego Studio decides all rating changes.
THIS IS JUST FOR FUN AND 100% VIBE CODED JUST TO TEST RANDOM AI STUFF
