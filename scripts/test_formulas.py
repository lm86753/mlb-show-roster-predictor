"""Test the improved formula projection system."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.formulas.ratings import (
    DEFAULT_COEFFICIENTS,
    LEAGUE_AVG,
    ATTR_ALIASES,
    STAT_COLUMN_TO_VAR,
    _resolve_attr_name,
    _get_stat,
    clip_rating,
    project_hitter_attribute,
    project_pitcher_attribute,
    project_attribute,
    refit_coefficients,
    refit_and_save,
    save_coefficients,
    load_coefficients,
    _load_active_coefficients,
)

print("=== Test 1: League average defaults ===")
for key in ["k_pct", "bb_pct", "avg", "iso", "fb_velo", "k9", "bb9"]:
    print(f"  {key}: {LEAGUE_AVG[key]}")

print("\n=== Test 2: _get_stat with missing values ===")
stats_empty = {}
stats_partial = {"k_pct": 0.30, "avg": 0.280}
for key in ["k_pct", "avg", "iso", "fb_velo", "k9"]:
    v_empty = _get_stat(stats_empty, key)
    v_partial = _get_stat(stats_partial, key)
    print(f"  {key}: empty={v_empty}, partial={v_partial}")

print("\n=== Test 3: _resolve_attr_name ===")
for abbrev in ["clt", "vis", "pclt", "sta", "k/9_r", "contact_right"]:
    print(f"  {abbrev} -> {_resolve_attr_name(abbrev)}")

print("\n=== Test 4: Hitter projections with league-average stats ===")
league_stats = {"k_pct": 0.22, "bb_pct": 0.08, "avg": 0.250, "iso": 0.150}
for attr in ["plate_vision", "plate_discipline", "contact_right", "contact_left", "power_right", "power_left", "speed", "batting_clutch"]:
    val = project_hitter_attribute(attr, league_stats)
    print(f"  {attr}: {val}")

print("\n=== Test 5: Hitter projections with empty stats (graceful defaults) ===")
empty_stats = {}
for attr in ["plate_vision", "plate_discipline", "contact_right", "contact_left", "power_right", "power_left"]:
    val = project_hitter_attribute(attr, empty_stats)
    print(f"  {attr}: {val}")

print("\n=== Test 6: Pitcher projections with league-average stats ===")
pitcher_stats = {"k_pct": 0.25, "bb_pct": 0.08, "fb_velo": 94.0, "k9": 9.0, "bb9": 3.0, "hr9": 1.0, "ip": 180, "gamesStarted": 30}
for attr in ["pitch_velocity", "pitch_control", "pitch_movement", "bb_per_bf", "hr_per_bf", "pitching_clutch", "stamina", "k/9_r", "k/9_l", "h/9_r", "h/9", "bb/9"]:
    val = project_pitcher_attribute(attr, pitcher_stats)
    print(f"  {attr}: {val}")

print("\n=== Test 7: Pitcher projections with empty stats ===")
empty_stats = {}
for attr in ["pitch_velocity", "pitch_control", "pitch_movement", "bb_per_bf", "hr_per_bf", "pitching_clutch", "stamina", "k/9_r", "h/9_r", "bb/9"]:
    val = project_pitcher_attribute(attr, empty_stats)
    print(f"  {attr}: {val}")

print("\n=== Test 8: Abbreviated pitcher attrs ===")
for attr in ["pclt", "sta", "k/9_r", "k/9_l", "h/9_r"]:
    val = project_pitcher_attribute(attr, pitcher_stats)
    print(f"  {attr}: {val}")

print("\n=== Test 9: Refit coefficients from training data ===")
parquet_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "training_examples.parquet"
if parquet_path.exists():
    df = pd.read_parquet(parquet_path)
    print(f"  Training data: {df.shape}")
    
    coeffs = refit_coefficients(df)
    print(f"  Refitted {len(coeffs)} attributes")
    for attr in sorted(coeffs.keys()):
        c = coeffs[attr]
        print(f"    {attr}: {c}")
    
    # Save and reload
    path = save_coefficients(coeffs)
    print(f"\n  Saved to: {path}")
    
    loaded = load_coefficients()
    print(f"  Loaded {len(loaded)} attributes from disk")
    
    # Test active coefficients (merged defaults + refitted)
    active = _load_active_coefficients()
    print(f"  Active coefficients: {len(active)} attributes")
    
    # Test projection with refitted coefficients
    print("\n  === Projections with refitted coefficients ===")
    for attr in ["contact_right", "plate_vision", "k/9_r", "pitch_control"]:
        hitter = attr in ["contact_right", "plate_vision"]
        val_default = project_attribute(attr, {"k_pct": 0.25, "bb_pct": 0.08, "avg": 0.280, "iso": 0.180, "k9": 9.0, "bb9": 3.0}, hitter)
        val_refitted = project_attribute(attr, {"k_pct": 0.25, "bb_pct": 0.08, "avg": 0.280, "iso": 0.180, "k9": 9.0, "bb9": 3.0}, hitter, coeffs=active)
        print(f"    {attr}: default={val_default}, refitted={val_refitted}")
    
    # Quick refit_and_save test
    print("\n  === refit_and_save convenience function ===")
    coeffs2 = refit_and_save(df)
    print(f"  Refitted and saved {len(coeffs2)} attributes")
else:
    print("  No training data found, skipping refit test")

print("\n=== All tests passed! ===")
