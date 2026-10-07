import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_is_vercel = os.environ.get("VERCEL") == "1"

if _is_vercel:
    DATA_DIR = Path("/tmp") / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Model files live in the project (read-only, committed to git)
    MODELS_DIR = PROJECT_ROOT / "data" / "models"
else:
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = DATA_DIR / "models"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "predictor.db"

GAME_YEARS = [22, 23, 24, 25, 26]
GAME_BASE_URLS = {y: f"https://mlb{y}.theshow.com" for y in GAME_YEARS}

MLB_STATS_API = "https://statsapi.mlb.com/api/v1"

RARITY_TIERS = {
    "Common": (0, 64),
    "Bronze": (65, 74),
    "Silver": (75, 84),
    "Gold": (85, 89),
    "Diamond": (90, 94),
    "Red Diamond": (95, 99),
}

TIER_ORDER = ["Common", "Bronze", "Silver", "Gold", "Diamond", "Red Diamond"]

# ── SDS API label → canonical name ──────────────────────────────────────────
# FIXED: K/9 → k_per_9 (was incorrectly bb_per_bf)
# ADDED: BB/9 for MLB 26, and split attrs use _per_ naming for consistency
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
    "REACT": "reaction_time",
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
    "K/9": "k_per_9",         # FIXED: was "bb_per_bf" (K/9 ≠ walks/bf)
    "HR/9": "hr_per_9",
    "H/9 R": "h_per_9_r",
    "H/9": "h_per_9",
    "K/9 R": "k_per_9_r",
    "K/9 L": "k_per_9_l",
    "BB/9": "bb_per_9",       # NEW: MLB 26
}

# ── Alias map: old/informal names → canonical names ──────────────────────────
# Handles DB data stored with old conventions, abbreviated names,
# and common misspellings.
ALIAS_MAP = {
    # Fixed naming (old /9 style → _per_9 style)
    "k/9_r": "k_per_9_r",
    "k/9_l": "k_per_9_l",
    "h/9_r": "h_per_9_r",
    "h/9": "h_per_9",
    "bb/9": "bb_per_9",
    "bb_per_bf": "k_per_9",   # Old buggy name for K/9
    "hr_per_bf": "hr_per_9",  # Old buggy name for HR/9
    # Abbreviated names (from DB feature stores)
    "clt": "batting_clutch",
    "vis": "plate_vision",
    "pclt": "pitching_clutch",
    "sta": "stamina",
    "acc": "arm_accuracy",
    "arm": "arm_strength",
    "reac": "fielding_ability",
    "reac_b": "fielding_ability",
    "reac_f": "fielding_ability",
    "reac_r": "fielding_ability",
    "steal": "speed",
    "bnt": "bunting_ability",
    "drg_bnt": "drag_bunting_ability",
    "pop": "blocking",
}

# ── Attributes to predict per position type ──────────────────────────────────
HITTER_ATTRS = [
    "contact_left", "contact_right", "power_left", "power_right",
    "plate_vision", "plate_discipline", "batting_clutch", "speed",
    "fielding_ability", "arm_strength", "arm_accuracy", "reaction_time",
]

PITCHER_ATTRS = [
    "pitch_velocity", "pitch_control", "pitch_movement",
    "pitching_clutch", "stamina",
    "k_per_9_r", "k_per_9_l", "h_per_9_r", "h_per_9", "k_per_9", "hr_per_9", "bb_per_9",
]

MIN_AB_21D = 20
MIN_IP_21D = 10

# Attribute ratings run 0-125 in The Show (OVR is capped at 99).
RATING_MAX = 125

# The attributes SDS actually moves in the monthly attribute updates, keyed by
# which MLB stat group drives them. Everything else changes so rarely
# (defense, speed, velocity) that we predict no change for it.
CORE_ATTRS = {
    "hitting": [
        "contact_left", "contact_right", "power_left", "power_right",
        "plate_vision", "batting_clutch",
    ],
    "pitching": [
        "h_per_9_r", "k_per_9_r", "k_per_9_l", "pitching_clutch", "stamina",
    ],
}
ATTR_GROUP = {a: g for g, attrs in CORE_ATTRS.items() for a in attrs}

# Canonical attribute → field name on the SDS card item.
CARD_FIELDS = {
    "contact_left": "contact_left", "contact_right": "contact_right",
    "power_left": "power_left", "power_right": "power_right",
    "plate_vision": "plate_vision", "batting_clutch": "batting_clutch",
    "h_per_9_r": "hits_per_bf_right", "k_per_9_r": "k_per_bf_right",
    "k_per_9_l": "k_per_bf_left", "pitching_clutch": "pitching_clutch",
    "stamina": "stamina",
}

# An update that touches at least this many cards is a monthly attribute
# update; the small weekly ones are injuries/role changes.
MAJOR_UPDATE_MIN_CARDS = 100

# Live Series quicksell values by OVR (MLB The Show 26).
_QS_BY_OVR = {
    75: 50, 76: 75, 77: 100, 78: 125, 79: 150,
    80: 400, 81: 600, 82: 900, 83: 1200, 84: 1500,
    85: 3000, 86: 3750, 87: 4500, 88: 5500, 89: 7000,
    90: 8000, 91: 9000,
}


def quicksell_value(ovr: int) -> int:
    if ovr < 65:
        return 5
    if ovr < 75:
        return 25
    if ovr >= 92:
        return 10000
    return _QS_BY_OVR[ovr]
