# Plan: Fix Dashboard Predictor Output Numbers

## Problem
The predictor numbers displayed on the dashboard are inconsistent, use arbitrary scaling, and don't align between backend and frontend. The user reports they "seem bad."

## Root Causes Identified

### 1. ROI % Formula Mismatch
- **Backend** (`predict.py:764`): `roi_pct = round(delta * 2.5, 1)` — arbitrary multiplier
- **Dashboard** (`dashboard.py:234`): `roi_pct = (predicted_delta * 100 / current_ovr)` — actual percentage return
- These produce completely different numbers for the same prediction

### 2. QS Tier Tables Diverge
- **Backend** (`predict.py:619-623`): 10 tiers including `(80, 600)`, `(92, 10000)`, `(94, 25000)`, `(97, 100000)`
- **Dashboard** (`dashboard.py:333`): only 6 tiers, missing 80/92/94/97
- Dashboard detail view computes wrong QS profits because it uses a different tier table than the backend prediction

### 3. Arbitrary Scaling in EV/ROI
- `roi_pct = delta * 2.5` — no economic justification
- `stub_ev = (p_up - p_down) * cur_qs * 0.05` — 0.05 is a guess
- `tier_bonus * 0.5` in investment score — arbitrary halving

### 4. Backtest vs Production Mismatch
- `evaluate.py:174`: `pred_ovr_delta = (pred_delta_sum / n_attrs * 1.5).clip(-8.0, 8.0)` — simple avg * 1.5
- `predict.py:681`: `predicted_ovr_delta = weighted_sum.clip(-5.0, 5.0)` — weighted sum
- Backtest metrics don't reflect actual production numbers

### 5. Overly Aggressive Clipping
- Per-attribute hard clip at ±5 suppresses realistic deltas
- OVR clip at ±5 further restricts output range
- Combined with conservative scale (0.15) and signal1 weight (0.5), most predictions cluster near 0

## Proposed Changes

### A. Unify ROI/EV Math (`src/models/predict.py`)
1. Replace `roi_pct = round(delta * 2.5, 1)` with a proper formula: `roi_pct = round((delta / current_ovr) * 100, 1)` if current_ovr > 0 else 0
2. Replace arbitrary `stub_ev = (p_up - p_down) * cur_qs * 0.05` with: `stub_ev = (p_up - p_down) * cur_qs` (net expected value in stubs)
3. Remove `tier_bonus * 0.5` arbitrary scaling; use `tier_bonus` directly

### B. Sync QS Tiers (`web/dashboard.py`)
1. Replace dashboard's 6-tier QS table with the same 10-tier table from `predict.py:619-623`
2. Use the exact same `_qs_value` logic as the backend

### C. Fix Backtest OVR Calculation (`src/models/evaluate.py`)
1. Change `pred_ovr_delta = (pred_delta_sum / n_attrs * 1.5).clip(-8.0, 8.0)` to match production: `pred_ovr_delta = weighted_delta_sum.clip(-5.0, 5.0)`
2. This ensures backtest MAE reflects real production behavior

### D. Relax Clipping / Calibration (`src/models/predict.py`)
1. Raise per-attribute hard clip from ±5 to ±8 to match signal2/signal3 bounds
2. Keep OVR clip at ±5 but raise calibration `max` defaults from 5.0 to 8.0 for key attributes
3. Consider raising signal1 scale default from 0.15 to 0.20 for attributes with strong historical data

### E. Centralize QS Tier Table
1. Move `_QS_TIERS` to `src/config.py` as a shared constant
2. Import it in both `predict.py` and `dashboard.py` to prevent future drift

## Validation Plan
1. Run `pytest tests/` to ensure no regressions
2. Run `python scripts/daily_predict.py --skip-cards --skip-link` and verify top predictions have sensible deltas (±1 to ±4 range instead of clustering near 0)
3. Run backtest (`python -c "from src.models.evaluate import run_backtest; print(run_backtest())"`) and verify `ovr_delta_mae` is reasonable
4. Launch dashboard and confirm:
   - ROI % in card view matches detail view
   - EV calculations use correct QS tiers
   - Deltas show realistic spread

## Open Questions
- Should the `roi_pct` in the DB be recalculated for existing predictions, or only apply to new predictions? **Recommendation: recalculate on next `run_predictions` call since it's derived, not stored independently.**
- Should we bump signal1 scale to 0.20 now, or calibrate first from historical ratios? **Recommendation: bump to 0.20 as a first pass; if `compute_calibration` produces reliable per-attr scales, use those instead.**
