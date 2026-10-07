with open('src/models/train.py', 'rb') as f:
    content = f.read().decode('utf-8', errors='replace')

# Split into lines
lines = content.splitlines(keepends=True)

# Remove lines 321-377 (0-indexed: 320-376) - market simulation section
# Lines to keep: 0-319, then 378 onwards
new_lines = lines[:320] + lines[378:]

# Now fix train_all function
for i, line in enumerate(new_lines):
    if '# 8. Market simulation calibration' in line:
        new_lines[i] = ''
        if i+1 < len(new_lines):
            new_lines[i+1] = ''
    if 'market_cal_samples' in line:
        new_lines[i] = ''

with open('src/models/train.py', 'wb') as f:
    f.write(''.join(new_lines).encode('utf-8'))

print('Done')