"""Analyze attribute name mapping and which attrs need formula support."""
import pandas as pd
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "data" / "processed" / "training_examples.parquet"
df = pd.read_parquet(p)

# SDS_ATTR_MAP short -> canonical
SDS_ATTR_MAP = {
    "CON L": "contact_left",
    "CON R": "contact_right",
    "POW L": "power_left",
    "POW R": "power_right",
    "VISION": "plate_vision",
    "DISC": "plate_discipline",
    "CLTCH": "batting_clutch",
    "BUNT": "bunting_ability",
    "DRAG": "drag_bunting_ability",
    "FLD": "fielding_ability",
    "ARM STR": "arm_strength",
    "ARM ACC": "arm_accuracy",
    "REACT": "fielding_ability",
    "BLK": "blocking",
    "SPD": "speed",
    "BR AGG": "baserunning_aggression",
    "BR ABIL": "baserunning_ability",
    "H DUR": "hitting_durability",
    "F DUR": "fielding_durability",
    "VEL": "pitch_velocity",
    "CTRL": "pitch_control",
    "BRK": "pitch_movement",
    "STAM": "stamina",
    "P CLTCH": "pitching_clutch",
    "K/9": "bb_per_bf",
    "HR/9": "hr_per_bf",
}

# Short name mapping (from what we see in the DB)
# In DB we see: clt, vis, pclt, sta, k/9_r, k/9_l, h/9_r, h/9, bb/9, acc, arm, reac, steal, blocking, pop, bnt, drg_bnt
# These map to canonical names in config

HITTER_ATTRS = [
    "contact_left", "contact_right", "power_left", "power_right",
    "plate_vision", "plate_discipline", "batting_clutch", "speed",
    "fielding_ability", "arm_strength", "arm_accuracy",
]

PITCHER_ATTRS = [
    "pitch_velocity", "pitch_control", "pitch_movement", "stamina",
    "bb_per_bf", "hr_per_bf", "pitching_clutch",
]

# Show what attrs are in the data and which have formula support
attrs_in_data = sorted(df['attribute_name'].unique().tolist())
print("=== All attrs in training data ===")
for attr in attrs_in_data:
    is_hitter_col = df[df['attribute_name'] == attr]['is_hitter']
    is_hit = is_hitter_col.mode().iloc[0] if len(is_hitter_col) > 0 else "?"
    n = len(df[df['attribute_name'] == attr])
    print(f"  {attr:30s}  n={n:5d}  is_hitter={int(is_hit)}")

# Which pitcher attrs in data lack formula support?
print("\n=== Pitcher attrs needing formula support ===")
pitcher_in_data = [a for a in attrs_in_data if df[df['attribute_name'] == a]['is_hitter'].mode().iloc[0] == 0]
print(f"In data: {pitcher_in_data}")
print(f"In PITCHER_ATTRS: {PITCHER_ATTRS}")
print(f"Missing from PITCHER_ATTRS: {set(pitcher_in_data) - set(PITCHER_ATTRS)}")

# Key pitcher stat relationships
print("\n=== Pitcher attribute vs k9/bb9 correlation ===")
for attr in ['k/9_r', 'k/9_l', 'h/9_r', 'h/9', 'bb/9', 'pitch_control', 'pclt', 'bb_per_bf']:
    grp = df[df['attribute_name'] == attr]
    if len(grp) < 20:
        continue
    corr_k9 = grp['stat_k9'].corr(grp['rating_after'])
    corr_bb9 = grp['stat_bb9'].corr(grp['rating_after'])
    corr_k_pct = grp['stat_k_pct'].corr(grp['rating_after'])
    corr_bb_pct = grp['stat_bb_pct'].corr(grp['rating_after'])
    print(f"  {attr:20s}  corr(k9,rating)={corr_k9:.3f}  corr(bb9,rating)={corr_bb9:.3f}  corr(k_pct,rating)={corr_k_pct:.3f}  corr(bb_pct,rating)={corr_bb_pct:.3f}")
