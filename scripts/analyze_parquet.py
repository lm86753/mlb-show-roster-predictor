"""Analyze parquet training data."""
import pandas as pd
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "data" / "processed" / "training_examples.parquet"
if not p.exists():
    print("No parquet found")
    exit()

df = pd.read_parquet(p)
print(f"Shape: {df.shape}")
print(f"Columns: {sorted(df.columns.tolist())}")

if "attribute_name" in df.columns:
    print(f"\nAttributes: {sorted(df['attribute_name'].unique().tolist())}")

stat_cols = [c for c in df.columns if c.startswith("stat_")]
print(f"\nStat columns: {stat_cols}")

# Check non-zero stats for pitcher attrs
pitcher_attrs = ["pitch_velocity", "pitch_control", "pitch_movement", "stamina", "bb_per_bf", "hr_per_bf", "k/9_r", "k/9_l", "h/9_r", "h/9", "bb/9", "pclt"]
for attr in pitcher_attrs:
    if attr not in df["attribute_name"].values:
        print(f"  {attr}: NOT in data")
        continue
    grp = df[df["attribute_name"] == attr]
    print(f"  {attr}: n={len(grp)}, rating_after mean={grp['rating_after'].mean():.1f}")
    for sc in stat_cols:
        nonzero = (grp[sc] != 0).sum()
        if nonzero > 0:
            print(f"    {sc}: nonzero={nonzero}/{len(grp)}, mean={grp[sc].mean():.4f}")
