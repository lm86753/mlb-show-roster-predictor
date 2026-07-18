import sqlite3, json

conn = sqlite3.connect('data/predictor.db')

# Check how many live cards we have
print('=== Live cards ===')
row = conn.execute("SELECT COUNT(*) FROM card_snapshots WHERE series='Live'").fetchone()
print(f"  Total Live cards: {row[0]}")

row = conn.execute("SELECT COUNT(DISTINCT mlb_player_id) FROM card_snapshots WHERE series='Live' AND mlb_player_id IS NOT NULL").fetchone()
print(f"  With MLB ID: {row[0]}")

# Check OVR distribution
print()
print('=== OVR distribution ===')
for row in conn.execute("SELECT ovr, COUNT(*) FROM card_snapshots WHERE series='Live' GROUP BY ovr ORDER BY ovr").fetchall():
    print(f"  OVR {row[0]:2d}: {row[1]:4d} cards")

conn.close()
