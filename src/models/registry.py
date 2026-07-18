"""
Year-Versioned Attribute Registry.

Tracks which attributes existed in each game year (MLB 22–26+), how SDS
labels them in the API, and the canonical names used internally.

This replaces the static SDS_ATTR_MAP with a year-aware system so we
can:
  1. Correctly map historical data (old names → current canonical names)
  2. Know which attributes existed in each game year
  3. Detect when new attributes appear or old ones are removed
"""

# ── SDS API label → canonical name (current / MLB 26) ────────────────────────
# These are the names as they appear in the SDS roster update JSON.
_CANONICAL: dict[str, str] = {
    "CON L":    "contact_left",
    "CON R":    "contact_right",
    "POW L":    "power_left",
    "POW R":    "power_right",
    "VIS":      "plate_vision",
    "DISC":     "plate_discipline",
    "CLTCH":    "batting_clutch",
    "BUNT":     "bunting_ability",
    "DRAG":     "drag_bunting_ability",
    "FLD":      "fielding_ability",
    "ARM STR":  "arm_strength",
    "ARM ACC":  "arm_accuracy",
    "REACT":    "reaction_time",
    "BLK":      "blocking",
    "SPD":      "speed",
    "BR AGG":   "baserunning_aggression",
    "BR ABIL":  "baserunning_ability",
    "H DUR":    "hitting_durability",
    "F DUR":    "fielding_durability",
    "VEL":      "pitch_velocity",
    "CTRL":     "pitch_control",
    "BRK":      "pitch_movement",
    "STAM":     "stamina",
    "P CLTCH":  "pitching_clutch",
    "K/9":      "k_per_9",       # FIXED: was incorrectly mapped to "bb_per_bf"
    "HR/9":     "hr_per_9",
    "H/9 R":    "h_per_9_r",
    "H/9":      "h_per_9",
    "K/9 R":    "k_per_9_r",
    "K/9 L":    "k_per_9_l",
    "BB/9":     "bb_per_9",      # NEW: MLB 26
}

# ── Canonical name → human-readable display label ────────────────────────────
DISPLAY_LABELS: dict[str, str] = {
    "contact_left":          "Contact L",
    "contact_right":         "Contact R",
    "power_left":            "Power L",
    "power_right":           "Power R",
    "plate_vision":          "Vision",
    "plate_discipline":      "Discipline",
    "batting_clutch":        "Clutch",
    "bunting_ability":       "Bunting",
    "drag_bunting_ability":  "Drag Bunting",
    "fielding_ability":      "Fielding",
    "arm_strength":          "Arm Str",
    "arm_accuracy":          "Arm Acc",
    "reaction_time":         "Reaction",
    "blocking":              "Blocking",
    "speed":                 "Speed",
    "baserunning_aggression":"BR Agg",
    "baserunning_ability":   "BR Abil",
    "hitting_durability":    "H Dur",
    "fielding_durability":   "F Dur",
    "pitch_velocity":        "Velocity",
    "pitch_control":         "Control",
    "pitch_movement":        "Movement",
    "stamina":               "Stamina",
    "pitching_clutch":       "P Clutch",
    "k_per_9":               "K/9",
    "hr_per_9":              "HR/9",
    "h_per_9_r":             "H/9 R",
    "h_per_9":               "H/9",
    "k_per_9_r":             "K/9 R",
    "k_per_9_l":             "K/9 L",
    "bb_per_9":              "BB/9",
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

# ── Year-to-year attribute evolution ──────────────────────────────────────────
# Key: SDS API label as seen in that year's roster update JSON
# Value: canonical name in the CURRENT (MLB 26) schema

_YEAR_MAPS: dict[int, dict[str, str]] = {
    22: {
        "CON L":    "contact_left", "CON R":    "contact_right",
        "POW L":    "power_left",   "POW R":    "power_right",
        "VIS":      "plate_vision", "DISC":     "plate_discipline",
        "CLTCH":    "batting_clutch",
        "FLD":      "fielding_ability",
        "ARM STR":  "arm_strength", "ARM ACC":  "arm_accuracy",
        "REACT":    "reaction_time", "BLK":     "blocking",
        "SPD":      "speed",
        "BR AGG":   "baserunning_aggression",
        "BR ABIL":  "baserunning_ability",
        "H DUR":    "hitting_durability",
        "F DUR":    "fielding_durability",
        "VEL":      "pitch_velocity",
        "CTRL":     "pitch_control", "BRK":     "pitch_movement",
        "STAM":     "stamina",       "P CLTCH": "pitching_clutch",
        "K/9":      "k_per_9",       "HR/9":    "hr_per_9",
        "H/9":      "h_per_9",
        # MLB 22-23: no splits
    },
    23: "inherit_22",
    24: {
        # MLB 24: added H/9 R, K/9 R, K/9 L splits
        "CON L":    "contact_left", "CON R":    "contact_right",
        "POW L":    "power_left",   "POW R":    "power_right",
        "VIS":      "plate_vision", "DISC":     "plate_discipline",
        "CLTCH":    "batting_clutch",
        "FLD":      "fielding_ability",
        "ARM STR":  "arm_strength", "ARM ACC":  "arm_accuracy",
        "REACT":    "reaction_time", "BLK":     "blocking",
        "SPD":      "speed",
        "BR AGG":   "baserunning_aggression",
        "BR ABIL":  "baserunning_ability",
        "H DUR":    "hitting_durability",
        "F DUR":    "fielding_durability",
        "VEL":      "pitch_velocity",
        "CTRL":     "pitch_control", "BRK":     "pitch_movement",
        "STAM":     "stamina",       "P CLTCH": "pitching_clutch",
        "K/9":      "k_per_9",       "HR/9":    "hr_per_9",
        "H/9 R":    "h_per_9_r",     "H/9":     "h_per_9",
        "K/9 R":    "k_per_9_r",     "K/9 L":   "k_per_9_l",
    },
    25: "inherit_24",
    26: {
        # MLB 26: added BB/9
        "CON L":    "contact_left", "CON R":    "contact_right",
        "POW L":    "power_left",   "POW R":    "power_right",
        "VIS":      "plate_vision", "DISC":     "plate_discipline",
        "CLTCH":    "batting_clutch",
        "FLD":      "fielding_ability",
        "ARM STR":  "arm_strength", "ARM ACC":  "arm_accuracy",
        "REACT":    "reaction_time", "BLK":     "blocking",
        "SPD":      "speed",
        "BR AGG":   "baserunning_aggression",
        "BR ABIL":  "baserunning_ability",
        "H DUR":    "hitting_durability",
        "F DUR":    "fielding_durability",
        "VEL":      "pitch_velocity",
        "CTRL":     "pitch_control", "BRK":     "pitch_movement",
        "STAM":     "stamina",       "P CLTCH": "pitching_clutch",
        "K/9":      "k_per_9",       "HR/9":    "hr_per_9",
        "H/9 R":    "h_per_9_r",     "H/9":     "h_per_9",
        "K/9 R":    "k_per_9_r",     "K/9 L":   "k_per_9_l",
        "BB/9":     "bb_per_9",
    },
}


def _resolve_year_map(game_year: int) -> dict[str, str]:
    """Resolve inherit_XX references in _YEAR_MAPS."""
    resolved: dict[str, str] = {}
    for y in range(min(_YEAR_MAPS), game_year + 1):
        entry = _YEAR_MAPS.get(y)
        if entry is None:
            continue
        if isinstance(entry, str) and entry.startswith("inherit_"):
            continue  # already merged
        if isinstance(entry, dict):
            resolved.update(entry)
    return resolved

# ── Attributes grouped by the MLB stat they primarily map to ─────────────────
# Used for calibration grouping: if we don't have enough data for a specific
# attribute, we fall back to its group's calibration.
ATTR_STAT_GROUP: dict[str, str] = {
    "contact_left":          "contact",
    "contact_right":         "contact",
    "power_left":            "power",
    "power_right":           "power",
    "plate_vision":          "vision",
    "plate_discipline":      "discipline",
    "batting_clutch":        "clutch",
    "speed":                 "speed",
    "fielding_ability":      "fielding",
    "arm_strength":          "fielding",
    "arm_accuracy":          "fielding",
    "reaction_time":         "fielding",
    "pitch_velocity":        "velocity",
    "pitch_control":         "control",
    "pitch_movement":        "movement",
    "pitching_clutch":       "clutch",
    "stamina":               "stamina",
    "k_per_9":               "k_per_9",
    "k_per_9_r":             "k_per_9",
    "k_per_9_l":             "k_per_9",
    "h_per_9":               "h_per_9",
    "h_per_9_r":             "h_per_9",
    "hr_per_9":              "hr_per_9",
    "bb_per_9":              "bb_per_9",
}


# ── Public API ───────────────────────────────────────────────────────────────

def _load_year_map(game_year: int) -> dict[str, str]:
    """Load the complete attribute map for a given game year, resolving inheritance."""
    entry = _YEAR_MAPS.get(game_year)
    if entry is None:
        return _CANONICAL
    if isinstance(entry, str) and entry.startswith("inherit_"):
        base_year = int(entry.split("_")[1])
        return _load_year_map(base_year)
    if isinstance(entry, dict):
        return entry
    return _CANONICAL


def canonical_name(sds_label: str, game_year: int | None = None) -> str:
    """Map an SDS API attribute label to the current canonical name.

    If game_year is provided, uses the year-specific mapping so that
    historical attribute names resolve correctly even if they differ
    from the current schema.
    """
    label = sds_label.strip()
    if game_year:
        year_map = _load_year_map(game_year)
        result = year_map.get(label)
        if result:
            return result
    return _CANONICAL.get(label, label)


def sds_label(canonical: str) -> str:
    """Reverse-lookup: canonical name → SDS API label."""
    for label, name in _CANONICAL.items():
        if name == canonical:
            return label
    return canonical


def display_label(canonical: str) -> str:
    """Canonical name → human-friendly display string."""
    return DISPLAY_LABELS.get(canonical, canonical)


def attrs_for_year(game_year: int) -> set[str]:
    """Return the set of canonical attribute names that existed in a given year."""
    year_map = _YEAR_MAPS.get(game_year, _YEAR_MAPS[max(_YEAR_MAPS.keys())])
    return {year_map[k] for k in year_map}


def attrs_for_position(is_hitter: bool, game_year: int = 26) -> list[str]:
    """Return the list of attributes to predict for a hitter or pitcher."""
    year_set = attrs_for_year(game_year)
    source = HITTER_ATTRS if is_hitter else PITCHER_ATTRS
    return [a for a in source if a in year_set]


def stat_group(attr: str) -> str:
    """Return the stat group for a given attribute (for fallback calibration)."""
    return ATTR_STAT_GROUP.get(attr, "other")


# ── Internal alias map (avoids circular import with config) ─────────────────
_ALIASES: dict[str, str] = {
    "k/9_r": "k_per_9_r", "k/9_l": "k_per_9_l",
    "h/9_r": "h_per_9_r", "h/9": "h_per_9", "bb/9": "bb_per_9",
    "bb_per_bf": "k_per_9", "hr_per_bf": "hr_per_9",
    "clt": "batting_clutch", "vis": "plate_vision",
    "pclt": "pitching_clutch", "sta": "stamina",
    "acc": "arm_accuracy", "arm": "arm_strength",
    "reac": "fielding_ability",
    "reac_b": "fielding_ability", "reac_f": "fielding_ability",
    "reac_r": "fielding_ability",
    "steal": "speed", "bnt": "bunting_ability",
    "drg_bnt": "drag_bunting_ability", "pop": "blocking",
}


def normalize_attr_name(name: str, game_year: int | None = None) -> str:
    """Normalize any attribute name to canonical form.

    Handles SDS labels, abbreviated aliases, old naming conventions,
    and already-canonical names.
    """
    name = name.strip()

    result = canonical_name(name, game_year)
    if result != name:
        return result

    if name in _ALIASES:
        return _ALIASES[name]

    for label, canon in _CANONICAL.items():
        if name.lower() == canon.lower():
            return canon

    return name


def is_split_attr(attr: str) -> bool:
    """Return True if this is a left/right split attribute (e.g., k_per_9_r)."""
    return attr.endswith("_r") or attr.endswith("_l")


def split_base(attr: str) -> str:
    """For split attrs, return the base name (e.g., k_per_9_r → k_per_9)."""
    if attr.endswith("_r"):
        return attr[:-2]
    if attr.endswith("_l"):
        return attr[:-2]
    return attr
