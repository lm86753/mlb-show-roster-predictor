import re

with open('src/models/train.py', 'r') as f:
    content = f.read()

# Remove market simulation calibration section
old = '''# ═══════════════════════════════════════════════════════════════════════════
#  Market simulation calibration
# ══════════════════════════════════════════════════════════════════════════

_QS_TIERS = [
    (0, 25), (65, 100), (75, 300), (80, 600),
    (85, 1000), (90, 5000), (92, 10000),
    (94, 25000), (95, 50000), (97, 100000),
]


def qs_value(ovr: int) -> int:
    return max((v for k, v in _QS_TIERS if ovr >= k), default=0)


def calibrate_market_simulation(df: pd.DataFrame) -> dict:
    """Calibrate market simulation from historical changes.

    Computes:
      - avg_delta_per_attribute_group: how many OVR points change per update
      - upgrade_prob_vs_gap: logistic mapping from gap_today \u2192 P(upgrade)
      - downgrade_prob_vs_gap: logistic mapping from gap_today \u2192 P(downgrade)
    """
    cal: dict = {}

    df = df.copy()
    df["gap_today_abs"] = df["gap_today"].abs()
    df["positive_delta"] = (df["delta"] > 0).astype(float)
    df["negative_delta"] = (df["delta"] < 0).astype(float)

    # Logistic calibration: bin by abs gap, compute observed probability
    boundaries = [0, 0.5, 1.5, 2.5, 3.5, 5, 7, 10, 99]
    prob_buckets = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        subset = df[(df["gap_today_abs"] >= lo) & (df["gap_today_abs"] < hi)]
        if len(subset) < 10:
            continue
        n_up = subset["positive_delta"].sum()
        n_down = subset["negative_delta"].sum()
        mid = (lo + hi) / 2
        prob_buckets.append({
            "gap_mid": mid,
            "p_up": float(n_up / len(subset)),
            "p_down": float(n_down / len(subset)),
            "p_change": float((n_up + n_down) / len(subset)),
            "n": int(len(subset)),
        })

    cal["prob_buckets"] = prob_buckets
    cal["qs_tiers"] = [{"min_ovr": k, "value": v} for k, v in _QS_TIERS]

    path = MODELS_DIR / "market_calibration.json"
    path.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    logger.info("Market simulation calibrated from %d total samples", len(df))
    return cal


# ═══════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════════'''

new = '''# ═══════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════════'''

if old in content:
    content = content.replace(old, new)
    with open('src/models/train.py', 'w') as f:
        f.write(content)
    print('Done')
else:
    print('Not found')
    # Find where it differs
    idx = content.find('Market simulation calibration')
    if idx >= 0:
        print('Found at', idx)
        print(repr(content[idx:idx+100]))