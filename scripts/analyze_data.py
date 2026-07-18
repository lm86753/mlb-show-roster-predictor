"""Analyze training data to understand available stat columns."""
import sqlite3
import json
from pathlib import Path

db_path = Path(__file__).resolve().parent.parent / "data" / "predictor.db"
print(f"DB exists: {db_path.exists()}")

if not db_path.exists():
    print("No database found, exiting")
    exit()

conn = sqlite3.connect(str(db_path))

# Check attribute names
print("\n=== Attribute Changes by type ===")
cur = conn.execute(
    "SELECT attribute_name, COUNT(*), is_hitter "
    "FROM attribute_changes "
    "GROUP BY attribute_name, is_hitter "
    "ORDER BY is_hitter, COUNT(*) DESC"
)
for row in cur.fetchall():
    print(f"  {row[0]:30s} count={row[1]:5d}  is_hitter={row[2]}")

# Check stat windows
cur = conn.execute("SELECT COUNT(*) FROM player_stat_windows")
print(f"\nTotal stat windows: {cur.fetchone()[0]}")

# Sample stat windows  
print("\n=== Sample stat windows ===")
cur = conn.execute("SELECT stats_json, is_hitter FROM player_stat_windows LIMIT 5")
for row in cur.fetchall():
    stats = json.loads(row[0])
    print(f"  is_hitter={row[1]}  keys={sorted(stats.keys())}")
    print(f"  {json.dumps(stats, indent=2)[:300]}")
    print("  ---")

# All unique keys in stat windows
print("\n=== All unique stat keys ===")
cur = conn.execute("SELECT stats_json, is_hitter FROM player_stat_windows")
hitter_keys = set()
pitcher_keys = set()
for row in cur.fetchall():
    stats = json.loads(row[0])
    if row[1] == 1:
        hitter_keys.update(stats.keys())
    else:
        pitcher_keys.update(stats.keys())
print(f"  Hitter stat keys: {sorted(hitter_keys)}")
print(f"  Pitcher stat keys: {sorted(pitcher_keys)}")

# Check if there are parquet files
processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
if processed_dir.exists():
    for f in processed_dir.glob("*.parquet"):
        print(f"\nFound parquet: {f}")
        try:
            import pandas as pd
            df = pd.read_parquet(f)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {sorted(df.columns.tolist())}")
            if "attribute_name" in df.columns:
                print(f"  Attributes: {sorted(df['attribute_name'].unique().tolist())}")
        except Exception as e:
            print(f"  Error: {e}")

conn.close()
