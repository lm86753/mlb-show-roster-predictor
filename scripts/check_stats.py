import sqlite3, json

conn = sqlite3.connect('data/predictor.db')

# See what windows exist
print('=== Window types ===')
for row in conn.execute('SELECT window, COUNT(*) FROM player_stat_windows GROUP BY window').fetchall():
    print(f'  {row}')

print()

# Sample hitter stat windows
print('=== Hitter stat sample ===')
for row in conn.execute('SELECT stats_json FROM player_stat_windows WHERE is_hitter=1 LIMIT 2').fetchall():
    print(json.dumps(json.loads(row[0]), indent=2))

print()
print('=== Pitcher stat sample ===')
for row in conn.execute('SELECT stats_json FROM player_stat_windows WHERE is_hitter=0 LIMIT 2').fetchall():
    print(json.dumps(json.loads(row[0]), indent=2))

conn.close()
