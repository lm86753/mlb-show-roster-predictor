# MLB The Show 26 Roster Update Predictor

Predict Live Series rating changes using MLB stats and historical SDS roster update data.

## Features

- **SDS backfill** — scrapes roster updates from MLB The Show 22–26 via the official API
- **MLB stats ingestion** — rolling 5d/21d/YTD/3yr windows from MLB Stats API + pybaseball
- **Hybrid ML pipeline** — formula-based attribute projection + LightGBM classifiers trained on past updates
- **FastAPI backend** — `/predictions`, `/player/{uuid}`, `/accuracy` endpoints
- **Streamlit dashboard** — Hot/Cold board, attribute explorer, accuracy tracker

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 1. Backfill historical roster updates
python scripts/backfill_updates.py

# 2. Train models (use --skip-stats for faster first run)
python scripts/train.py --skip-stats

# 3. Fetch cards and run predictions
python scripts/daily_predict.py

# 4. Launch dashboard
streamlit run web/dashboard.py

# 5. Launch API
uvicorn src.api.main:app --reload
```

## Project Structure

```
src/ingest/     SDS API, MLB stats, card snapshots
src/formulas/   Stat-to-rating projection formulas
src/features/   Feature engineering pipeline
src/models/     Train, predict, evaluate
src/api/        FastAPI endpoints
web/            Streamlit dashboard
scripts/        CLI entry points
```

## Prediction Horizons

The daily pipeline scores three snapshots before each roster update:

- **T-7** — early cycle signal
- **T-3** — mid-cycle refinement
- **T-1** — sharpest pre-update signal

## Disclaimer

Predictions are experimental estimates, not guarantees. San Diego Studio controls all final rating changes.
