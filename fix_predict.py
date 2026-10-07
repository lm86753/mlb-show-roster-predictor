with open('src/models/predict.py', 'rb') as f:
    content = f.read().decode('utf-8', errors='replace')

lines = content.splitlines(keepends=True)

# Track changes
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # 1. Remove self._market_cal from __init__
    if 'self._market_cal: dict | None = None' in line:
        i += 1
        continue
    
    # 2. Remove market_cal property (lines 196-202 in original)
    if 'def market_cal' in line and '@property' in lines[i-1]:
        # Skip until next property or method
        while i < len(lines) and not (lines[i].strip().startswith('@property') or lines[i].strip().startswith('def ') and i > 0):
            i += 1
        # We've gone one past - step back
        i -= 1
        continue
    
    # 3. Remove calibrated_prob method (lines 272-290)
    if 'def calibrated_prob' in line:
        # Skip until next method (def with indent 4)
        while i < len(lines):
            if i+1 < len(lines) and lines[i+1].startswith('    def ') and not lines[i+1].startswith('        '):
                break
            i += 1
        continue
    
    # 4. In predict_attributes: remove up_prob/dn_prob computation (lines 621-632)
    # We'll handle this by removing specific lines
    # The computation block starts with "delta_strength = min(1.0, abs(delta) / 2.0)"
    if 'delta_strength = min(1.0, abs(delta) / 2.0)' in line:
        # Skip the next 11 lines (up to "dn_prob = max(split, 0.001)")
        for _ in range(12):
            if i < len(lines):
                i += 1
        continue
    
    # 5. Remove upgrade_prob_attr and downgrade_prob_attr from results
    if '"upgrade_prob_attr": round(up_prob, 3),' in line:
        i += 1
        continue
    if '"downgrade_prob_attr": round(dn_prob, 3),' in line:
        i += 1
        continue
    
    # 6. In aggregate_player_predictions: remove up_prob_mean/dn_prob_mean from agg
    if 'up_prob_mean=("upgrade_prob_attr", "mean"),' in line:
        i += 1
        continue
    if 'dn_prob_mean=("downgrade_prob_attr", "mean"),' in line:
        i += 1
        continue
    
    # 7. Remove _ovr_probs function and its usage (lines 698-716)
    if 'def _ovr_probs(row):' in line:
        # Skip until "probs = grouped.apply(_ovr_probs, axis=1)" and the if block
        while i < len(lines):
            if 'probs = grouped.apply(_ovr_probs, axis=1)' in lines[i]:
                # Skip this line and the next 6 lines (if/else block)
                for _ in range(7):
                    if i < len(lines):
                        i += 1
                break
            i += 1
        continue
    
    # 8. Update _expected_value function
    # Replace the function body to use direction_consensus instead of p_up/p_down
    if 'def _expected_value(row):' in line:
        # Copy this line
        new_lines.append(line)
        i += 1
        # Skip until the return statement, then replace the body
        indent = None
        while i < len(lines):
            if 'direction = np.sign(delta) * min(1.0, abs(delta) / 3.0)' in lines[i]:
                # Found the start of the body we want to replace
                # Replace from here until "return pd.Series({"
                break
            new_lines.append(lines[i])
            i += 1
        
        # Now insert the new body
        new_body = '''        delta = row["predicted_ovr_delta"]
        consensus = row["direction_consensus"]
        tier_jump = row["tier_jump_probability"]

        direction = np.sign(delta) * min(1.0, abs(delta) / 3.0)
        confidence = consensus
        tier_bonus = np.sign(delta) * tier_jump * 20.0 if abs(delta) > 0.5 else 0.0

        score = (direction * 50 + confidence * 30 + tier_bonus)
        score = max(-100, min(100, score))

        ovr = row["current_ovr"]
        cur_qs = _qs_value(ovr)
        stub_ev = consensus * cur_qs
        stub_ev = max(-cur_qs, min(cur_qs, stub_ev))

        return pd.Series({'''
        new_lines.append(new_body)
        # Continue until we hit the closing brace of the return
        while i < len(lines):
            if '        })' in lines[i] and lines[i].strip() == '})':
                new_lines.append(lines[i])  # Keep the closing brace
                i += 1
                break
            i += 1
        continue
    
    # 9. In run_predictions: remove upgrade_probability/downgrade_probability from Prediction
    if 'upgrade_probability=float(row["upgrade_probability"]),' in line:
        i += 1
        continue
    if 'downgrade_probability=float(row["downgrade_probability"]),' in line:
        i += 1
        continue
    
    # Keep the line
    new_lines.append(line)
    i += 1

with open('src/models/predict.py', 'wb') as f:
    f.write(''.join(new_lines).encode('utf-8'))

print('Done')