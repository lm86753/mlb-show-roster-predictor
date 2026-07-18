"""Validate refitted coefficients are actually changing projections."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.formulas.ratings import (
    DEFAULT_COEFFICIENTS,
    project_attribute,
    refit_coefficients,
    _load_active_coefficients,
    load_coefficients,
)

# Load refitted coefficients
refitted = load_coefficients()
active = _load_active_coefficients()

# The refitted k/9_r: k9=1.655, intercept=46.89
# With k9=9.0: 46.89 + 1.655*9 = 46.89 + 14.9 = 61.8 ≈ 62
# The DEFAULT k/9_r: k9=3.50, intercept=31.0
# With k9=9.0: 31.0 + 3.50*9 = 31.0 + 31.5 = 62.5 ≈ 62

# The issue is that the refitted coefficients for many pitcher attrs are
# coming from the "fallback all stat_ columns" path, not ATTR_FEATURE_MAP.
# Let me verify the refitted coefficients for key pitcher attrs

print("=== Refitted vs Default coefficient comparison ===")
for attr in ["k/9_r", "k/9_l", "h/9_r", "pitch_control", "contact_right", "power_right", "pitching_clutch"]:
    default_c = DEFAULT_COEFFICIENTS.get(attr)
    refitted_c = refitted.get(attr) if refitted else None
    active_c = active.get(attr)
    print(f"\n  {attr}:")
    print(f"    Default:  {default_c}")
    print(f"    Refitted: {refitted_c}")
    print(f"    Active:   {active_c}")

# Now test projection with actual good stats
print("\n=== Projections with good pitcher stats (K/9=11, BB/9=2) ===")
good_stats = {"k_pct": 0.30, "bb_pct": 0.06, "k9": 11.0, "bb9": 2.0, "hr9": 0.8, "fb_velo": 96.0}
for attr in ["k/9_r", "k/9_l", "h/9_r", "pitch_control", "pitch_velocity", "pitching_clutch", "bb/9"]:
    val_default = project_attribute(attr, good_stats, False, coeffs=DEFAULT_COEFFICIENTS)
    val_active = project_attribute(attr, good_stats, False, coeffs=active)
    print(f"  {attr}: default_coeff={val_default}, active_coeff={val_active}")

# Check: are pitcher stats with missing k9/bb9 now using league avg instead of 0?
print("\n=== Projections with missing pitcher stats ===")
missing_stats = {"k_pct": 0.22, "bb_pct": 0.08}  # no k9, bb9
for attr in ["k/9_r", "pitch_control", "pitching_clutch"]:
    val = project_attribute(attr, missing_stats, False, coeffs=active)
    print(f"  {attr} (no k9/bb9): {val}")

# Compare against ground truth ratings from training data
parquet_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "training_examples.parquet"
if parquet_path.exists():
    df = pd.read_parquet(parquet_path)
    
    print("\n=== MAE comparison: refitted vs default coefficients ===")
    from src.formulas.ratings import STAT_COLUMN_TO_VAR
    import numpy as np
    
    for attr in ["k/9_r", "contact_right", "power_right"]:
        grp = df[df["attribute_name"] == attr]
        if len(grp) < 20:
            continue
        
        # Build stats dict from stat_ columns
        stat_cols = [c for c in grp.columns if c in STAT_COLUMN_TO_VAR]
        is_hitter = int(grp["is_hitter"].mode().iloc[0])
        
        errors_default = []
        errors_active = []
        for _, row in grp.iterrows():
            stats = {}
            for sc in stat_cols:
                var = STAT_COLUMN_TO_VAR[sc]
                val = row[sc]
                if var.endswith("_pct"):
                    val = val / 100.0  # stat_k_pct is already *100
                stats[var] = val
            
            pred_default = project_attribute(attr, stats, bool(is_hitter), coeffs=DEFAULT_COEFFICIENTS)
            pred_active = project_attribute(attr, stats, bool(is_hitter), coeffs=active)
            actual = row["rating_after"]
            
            errors_default.append(abs(pred_default - actual))
            errors_active.append(abs(pred_active - actual))
        
        mae_default = np.mean(errors_default)
        mae_active = np.mean(errors_active)
        print(f"  {attr}: MAE_default={mae_default:.1f}, MAE_refitted={mae_active:.1f}, improvement={mae_default - mae_active:+.1f}")
