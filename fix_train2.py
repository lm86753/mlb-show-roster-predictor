with open('src/models/train.py', 'r') as f:
    lines = f.readlines()

# Remove lines 321-377 (0-indexed: 320-376)
# Lines to keep: 0-319, then 378 onwards (which is Orchestrator header)
new_lines = lines[:320] + lines[378:]

# Now fix train_all function - remove market_cal call and return value
# Find the train_all function and modify it
# The function starts around line 383 in original (now shifted)
# We need to remove:
#     # 8. Market simulation calibration
#     market_cal = calibrate_market_simulation(df)
#
# and from return dict:
#         "market_cal_samples": sum(b.get("n", 0) for b in market_cal.get("prob_buckets", [])),

# Find the return statement and remove the market_cal line
for i, line in enumerate(new_lines):
    if '# 8. Market simulation calibration' in line:
        # Remove this line and the next line (market_cal = ...)
        new_lines[i] = ''
        new_lines[i+1] = ''
    if 'market_cal_samples' in line:
        # Remove this line
        new_lines[i] = ''

with open('src/models/train.py', 'w') as f:
    f.writelines(new_lines)

print('Done')