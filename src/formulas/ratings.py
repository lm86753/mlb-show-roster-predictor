from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import ALIAS_MAP, MODELS_DIR

# ─── League-average defaults (used when a stat is missing) ──────────────────
# These are typical MLB averages so formulas produce a reasonable ~50-60 rating
# instead of clipping to 0 or 99 when data is absent.
LEAGUE_AVG = {
    # Hitter stats (MLB 26 YTD averages)
    "k_pct": 0.225,       # strikeout rate (up from 0.22 in 25)
    "bb_pct": 0.085,      # walk rate (up slightly)
    "avg": 0.248,         # batting average (down from 0.250)
    "iso": 0.155,         # isolated power (up slightly)
    "exit_velo": 88.5,    # avg exit velocity (mph)
    "sprint_speed": 27.0, # ft/sec
    "ab": 0,              # at-bats (count, not rate)
    "pa": 0,              # plate appearances
    "hr": 0,              # home runs
    # Pitcher stats (MLB 26 YTD)
    "fb_velo": 93.5,      # fastball velocity (mph, up from 93.0)
    "k9": 8.7,            # strikeouts per 9 (up from 8.5)
    "bb9": 3.1,           # walks per 9 (up slightly)
    "hr9": 1.15,          # homers per 9 (down slightly)
    "ip": 0,              # innings pitched
    "bf": 0,              # batters faced
    "gs": 1,              # games started
    "gamesStarted": 1,
    "whip": 1.28,         # walks + hits per IP
    "oba": 0.305,         # on-base against
    "slga": 0.395,        # slugging against
}

# ─── Default coefficients from community reverse-engineering ────────────────
# These are the initial guesses; refit_coefficients() will produce better ones.
DEFAULT_COEFFICIENTS = {
    # Hitter attributes
    # Intercepts calibrated so league-average stats (LEAGUE_AVG) produce ratings ~70-80
    # matching the actual Live Series card distribution in MLB The Show.
    "plate_vision": {"k_pct": -2.55, "intercept": 132.0},
    "plate_discipline": {"bb_pct": 4.86, "intercept": 35.0},
    "contact_right": {"avg": 374.6, "exit_velo": 2.30, "intercept": -218.5},
    "contact_left": {"avg": 347.6, "intercept": -10.0},
    "power_right": {"iso": 336.0, "intercept": 20.0},
    "power_left": {"iso": 244.0, "intercept": 32.0},
    "speed": {"sprint_speed": 15.01, "intercept": -335.0},
    "batting_clutch": {"avg": 200.0, "iso": 100.0, "intercept": 10.0},
    # Pitcher attributes
    "pitch_velocity": {"fb_velo": 3.14, "intercept": -215.0},
    "pitch_control": {"bb_pct": -4.44, "k_pct": 2.80, "intercept": 48.0},
    "pitch_movement": {"k9": 3.50, "bb9": -2.00, "intercept": 45.0},
    "k_per_9": {"k_pct": 2.2, "intercept": 28.0},
    "hr_per_9": {"hr9": -3.5, "k_pct": 1.0, "intercept": 62.0},
    "pitching_clutch": {"k_pct": 1.5, "bb_pct": -2.0, "intercept": 55.0},
    "k_per_9_r": {"k9": 3.50, "intercept": 44.0},
    "k_per_9_l": {"k9": 3.50, "intercept": 44.0},
    "h_per_9_r": {"hr9": -5.0, "bb9": -1.5, "k9": 1.0, "intercept": 77.0},
    "h_per_9": {"hr9": -5.0, "bb9": -1.5, "intercept": 77.0},
    "bb_per_9": {"bb9": 5.0, "intercept": 62.0},
    "stamina": None,  # Special formula (not linear)
}

# ─── Mapping from training dataframe stat_ columns to formula variable names ─
# The training df has columns like stat_k_pct, stat_bb_pct, stat_avg, stat_iso,
# stat_k9, stat_bb9, stat_ip, stat_ab, stat_pa, stat_hr.
# These need to map to the variable names used in formula functions.
STAT_COLUMN_TO_VAR = {
    "stat_k_pct": "k_pct",
    "stat_bb_pct": "bb_pct",
    "stat_avg": "avg",
    "stat_iso": "iso",
    "stat_ab": "ab",
    "stat_ip": "ip",
    "stat_pa": "pa",
    "stat_hr": "hr",
    "stat_k9": "k9",
    "stat_bb9": "bb9",
    "stat_hr9": "hr9",
    "stat_gs": "gamesStarted",
    "stat_sprint_speed": "sprint_speed",
}

# Reverse: formula var name -> stat_ column in training df
VAR_TO_STAT_COLUMN = {v: k for k, v in STAT_COLUMN_TO_VAR.items()}

# ─── Alias mapping: attribute name aliases -> canonical names ────────────────
# Re-exports config's ALIAS_MAP for backward compatibility
ATTR_ALIASES = dict(ALIAS_MAP)


def clip_rating(value: float) -> int:
    return int(max(0, min(99, round(value))))


def _get_stat(stats: dict, key: str) -> float:
    """Get a stat value, falling back to league average if missing or None."""
    val = stats.get(key)
    if val is not None:
        return float(val)
    return float(LEAGUE_AVG.get(key, 0.0))


def _resolve_attr_name(attr: str) -> str:
    """Resolve abbreviated attribute names to canonical ones."""
    return ATTR_ALIASES.get(attr, attr)


def project_hitter_attribute(attr: str, stats: dict, coeffs: dict | None = None) -> int:
    """Project a hitter attribute rating from stats using a linear formula."""
    coeffs = coeffs or _load_active_coefficients()
    attr = _resolve_attr_name(attr)

    k_pct = _get_stat(stats, "k_pct")
    bb_pct = _get_stat(stats, "bb_pct")
    avg = _get_stat(stats, "avg")
    iso = _get_stat(stats, "iso")
    exit_velo = _get_stat(stats, "exit_velo")
    sprint = _get_stat(stats, "sprint_speed")

    if attr == "plate_vision":
        c = coeffs.get("plate_vision", DEFAULT_COEFFICIENTS["plate_vision"])
        return clip_rating(c["k_pct"] * k_pct * 100 + c["intercept"])

    if attr == "plate_discipline":
        c = coeffs.get("plate_discipline", DEFAULT_COEFFICIENTS["plate_discipline"])
        return clip_rating(c["bb_pct"] * bb_pct * 100 + c["intercept"])

    if attr in ("contact_right", "contact_left"):
        key = attr
        c = coeffs.get(key, DEFAULT_COEFFICIENTS.get(key, DEFAULT_COEFFICIENTS["contact_right"]))
        if "exit_velo" in c:
            return clip_rating(c["avg"] * avg + c["exit_velo"] * exit_velo + c["intercept"])
        return clip_rating(c["avg"] * avg + c["intercept"])

    if attr in ("power_right", "power_left"):
        key = attr
        c = coeffs.get(key, DEFAULT_COEFFICIENTS.get(key, DEFAULT_COEFFICIENTS["power_right"]))
        return clip_rating(c["iso"] * iso + c["intercept"])

    if attr == "speed":
        c = coeffs.get("speed", DEFAULT_COEFFICIENTS["speed"])
        if "sprint_speed" in c:
            return clip_rating(c["sprint_speed"] * sprint + c["intercept"])
        # Refitted coefficients may not have sprint_speed; use generic handler
        return _apply_linear_formula(c, stats)

    if attr == "batting_clutch":
        c = coeffs.get("batting_clutch", DEFAULT_COEFFICIENTS["batting_clutch"])
        if "exit_velo" in c:
            return clip_rating(c["avg"] * avg + c["iso"] * iso + c["exit_velo"] * exit_velo + c["intercept"])
        return clip_rating(c["avg"] * avg + c["iso"] * iso + c["intercept"])

    # Generic handler: try to use refitted coefficients if available
    if attr in coeffs and coeffs[attr] is not None:
        c = coeffs[attr]
        return _apply_linear_formula(c, stats)

    # Fallback: return league-average-ish rating
    return clip_rating(50 + avg * 40 + iso * 20)


def project_pitcher_attribute(attr: str, stats: dict, coeffs: dict | None = None) -> int:
    """Project a pitcher attribute rating from stats using a linear formula."""
    coeffs = coeffs or _load_active_coefficients()
    attr = _resolve_attr_name(attr)

    k_pct = _get_stat(stats, "k_pct")
    bb_pct = _get_stat(stats, "bb_pct")
    fb_velo = _get_stat(stats, "fb_velo")
    k9 = _get_stat(stats, "k9")
    bb9 = _get_stat(stats, "bb9")
    hr9 = _get_stat(stats, "hr9")

    if attr == "pitch_velocity":
        c = coeffs.get("pitch_velocity", DEFAULT_COEFFICIENTS["pitch_velocity"])
        if "fb_velo" in c:
            return clip_rating(c["fb_velo"] * fb_velo + c["intercept"])
        # Refitted coefficients may not have fb_velo; use generic handler
        return _apply_linear_formula(c, stats)

    if attr == "pitch_control":
        c = coeffs.get("pitch_control", DEFAULT_COEFFICIENTS["pitch_control"])
        if "bb_pct" in c:
            val = c["bb_pct"] * bb_pct * 100 + c["intercept"]
            if "k_pct" in c:
                val += c["k_pct"] * k_pct * 100
            return clip_rating(val)
        return _apply_linear_formula(c, stats)

    if attr == "pitch_movement":
        c = coeffs.get("pitch_movement", DEFAULT_COEFFICIENTS["pitch_movement"])
        if "k9" in c:
            val = c["k9"] * k9 + c["intercept"]
            if "bb9" in c:
                val += c["bb9"] * bb9
            return clip_rating(val)
        return _apply_linear_formula(c, stats)

    if attr == "k_per_9":
        c = coeffs.get("k_per_9", DEFAULT_COEFFICIENTS["k_per_9"])
        if "k_pct" in c:
            return clip_rating(c["k_pct"] * k_pct * 100 + c["intercept"])
        return _apply_linear_formula(c, stats)

    if attr == "hr_per_9":
        c = coeffs.get("hr_per_9", DEFAULT_COEFFICIENTS["hr_per_9"])
        if "hr9" in c or "k_pct" in c:
            val = c["intercept"]
            if "hr9" in c:
                val += c["hr9"] * hr9
            if "k_pct" in c:
                val += c["k_pct"] * k_pct * 100
            return clip_rating(val)
        return _apply_linear_formula(c, stats)

    if attr == "pitching_clutch":
        c = coeffs.get("pitching_clutch", DEFAULT_COEFFICIENTS.get("pitching_clutch", {"k_pct": 1.5, "bb_pct": -2.0, "intercept": 55.0}))
        if "k_pct" in c or "bb_pct" in c or "k9" in c:
            val = c["intercept"]
            if "k_pct" in c:
                val += c["k_pct"] * k_pct * 100
            if "bb_pct" in c:
                val += c["bb_pct"] * bb_pct * 100
            if "k9" in c:
                val += c["k9"] * k9
            return clip_rating(val)
        return _apply_linear_formula(c, stats)

    if attr == "stamina":
        ip = _get_stat(stats, "ip")
        gs = _get_stat(stats, "gamesStarted") or _get_stat(stats, "gs") or 1.0
        if gs < 1:
            gs = 1.0
        ip_per_gs = ip / gs
        # Reasonable stamina: baseline 45, +1.8 per IP/GS.  League avg starter
        # (~5.5 IP/start) ≈ 55, relievers (~1 IP) ≈ 47, aces (~7 IP) ≈ 58.
        return clip_rating(45.0 + ip_per_gs * 1.8)

    # k/9 variants (splits vs L/R)
    if attr in ("k_per_9_r", "k_per_9_l"):
        key = attr
        c = coeffs.get(key, DEFAULT_COEFFICIENTS.get(key, DEFAULT_COEFFICIENTS["k_per_9_r"]))
        val = c["k9"] * k9 + c["intercept"]
        if "bb9" in c:
            val += c["bb9"] * bb9
        return clip_rating(val)

    # h/9 variants (hits per 9, includes hr/9 and bb/9 terms)
    if attr in ("h_per_9_r", "h_per_9"):
        key = attr
        c = coeffs.get(key, DEFAULT_COEFFICIENTS.get(key, DEFAULT_COEFFICIENTS["h_per_9_r"]))
        val = c["intercept"]
        for var_name, multiplier in c.items():
            if var_name == "intercept":
                continue
            stat_val = _get_stat(stats, var_name)
            val += multiplier * stat_val
        return clip_rating(val)

    # bb/9 (walks per 9)
    if attr == "bb_per_9":
        c = coeffs.get("bb_per_9", DEFAULT_COEFFICIENTS["bb_per_9"])
        val = c["intercept"]
        if "bb9" in c:
            val += c["bb9"] * bb9
        if "k9" in c:
            val += c["k9"] * k9
        return clip_rating(val)

    # Generic handler: try to use refitted coefficients if available
    if attr in coeffs and coeffs[attr] is not None:
        c = coeffs[attr]
        return _apply_linear_formula(c, stats)

    # Fallback: reasonable pitcher rating based on K-BB profile
    return clip_rating(50 + k9 * 1.5 - bb9 * 1.0)


def _apply_linear_formula(c: dict, stats: dict) -> int:
    """Apply a generic linear formula from coefficient dict + stat values.

    Coefficient dict format: {"var1": coef1, "var2": coef2, "intercept": b}
    Variable names that end with _pct (k_pct, bb_pct) are multiplied by 100
    to convert from decimal to percentage before applying the coefficient,
    matching how the formula functions historically work.
    """
    val = c.get("intercept", 0.0)
    for var_name, coef in c.items():
        if var_name == "intercept":
            continue
        stat_val = _get_stat(stats, var_name)
        # K%/BB% are stored as decimals (0.22) but formulas use percentages (22)
        if var_name.endswith("_pct"):
            stat_val *= 100.0
        val += coef * stat_val
    return clip_rating(val)


def project_attribute(attr: str, stats: dict, is_hitter: bool, coeffs: dict | None = None) -> int:
    if is_hitter:
        return project_hitter_attribute(attr, stats, coeffs)
    return project_pitcher_attribute(attr, stats, coeffs)


# ─── Refitting ───────────────────────────────────────────────────────────────

# Mapping: attribute name -> list of (formula_var, stat_column) pairs
# This tells refit_coefficients which training columns to use for each attribute.
ATTR_FEATURE_MAP = {
    # Hitter attributes
    "plate_vision": [("k_pct", "stat_k_pct")],
    "plate_discipline": [("bb_pct", "stat_bb_pct")],
    "contact_right": [("avg", "stat_avg"), ("iso", "stat_iso")],
    "contact_left": [("avg", "stat_avg")],
    "power_right": [("iso", "stat_iso")],
    "power_left": [("iso", "stat_iso")],
    "speed": [],  # sprint_speed not in training data
    "batting_clutch": [("avg", "stat_avg"), ("iso", "stat_iso"), ("k_pct", "stat_k_pct")],
    # Pitcher attributes
    "pitch_velocity": [],  # fb_velo not in training data
    "pitch_control": [("k_pct", "stat_k_pct"), ("bb_pct", "stat_bb_pct")],
    "pitch_movement": [("k9", "stat_k9"), ("bb9", "stat_bb9")],
    "k_per_9": [("k_pct", "stat_k_pct")],
    "hr_per_9": [("k_pct", "stat_k_pct"), ("bb_pct", "stat_bb_pct")],
    "pitching_clutch": [("k_pct", "stat_k_pct"), ("bb_pct", "stat_bb_pct"), ("k9", "stat_k9")],
    "k_per_9_r": [("k9", "stat_k9"), ("bb9", "stat_bb9")],
    "k_per_9_l": [("k9", "stat_k9"), ("bb9", "stat_bb9")],
    "h_per_9_r": [("k9", "stat_k9"), ("bb9", "stat_bb9"), ("k_pct", "stat_k_pct")],
    "h_per_9": [("k9", "stat_k9"), ("bb9", "stat_bb9")],
    "bb_per_9": [("bb9", "stat_bb9")],
    "stamina": [("ip", "stat_ip")],
    # Abbreviated names that appear in training data
    "clt": [("avg", "stat_avg"), ("iso", "stat_iso"), ("k_pct", "stat_k_pct")],
    "vis": [("k_pct", "stat_k_pct")],
    "pclt": [("k_pct", "stat_k_pct"), ("bb_pct", "stat_bb_pct"), ("k9", "stat_k9")],
    "sta": [("ip", "stat_ip")],
}


def refit_coefficients(training_df, min_samples: int = 20) -> dict:
    """Refit linear coefficients per attribute from historical labels.

    This properly maps stat_ prefixed training columns to the variable names
    used in formula functions (k_pct, bb_pct, avg, iso, k9, bb9, etc.)
    using ATTR_FEATURE_MAP.

    Returns a dict of {attribute_name: {var_name: coefficient, ..., "intercept": b}}
    that matches the same format as DEFAULT_COEFFICIENTS so it can be used
    directly by project_attribute().
    """
    from sklearn.linear_model import Ridge

    refitted = {}
    for attr, group in training_df.groupby("attribute_name"):
        if len(group) < min_samples:
            continue

        # Resolve attribute aliases for feature lookup
        canonical = _resolve_attr_name(attr)

        # Determine which stat columns to use for this attribute
        feature_pairs = ATTR_FEATURE_MAP.get(attr, ATTR_FEATURE_MAP.get(canonical, []))
        if not feature_pairs:
            continue

        # Only keep pairs where the stat column exists in the dataframe
        available_pairs = [
            (var_name, col_name)
            for var_name, col_name in feature_pairs
            if col_name in group.columns
        ]
        if not available_pairs:
            continue

        # Build feature matrix
        # For _pct variables, multiply by 100 (matching formula convention)
        X_parts = []
        var_names = []
        for var_name, col_name in available_pairs:
            col_vals = group[col_name].fillna(LEAGUE_AVG.get(var_name, 0.0)).values
            X_parts.append(col_vals.reshape(-1, 1))
            var_names.append(var_name)

        X = np.hstack(X_parts)
        y = group["rating_after"].values

        if len(np.unique(y)) < 3:
            continue

        model = Ridge(alpha=1.0)
        model.fit(X, y)

        # Build coefficient dict in the same format as DEFAULT_COEFFICIENTS
        coeff_dict = {"intercept": float(model.intercept_)}
        for i, var_name in enumerate(var_names):
            coeff_dict[var_name] = float(model.coef_[i])

        refitted[attr] = coeff_dict

        # Also store under the canonical name if different
        if canonical != attr:
            refitted[canonical] = coeff_dict

    return refitted


# ─── Serialization ──────────────────────────────────────────────────────────

_COEFFICIENTS_CACHE: dict | None = None
_COEFFICIENTS_PATH = MODELS_DIR / "formula_coefficients.json"


def save_coefficients(coeffs: dict, path: Path | None = None) -> Path:
    """Serialize refitted coefficients to JSON so the prediction pipeline
    uses them instead of hardcoded defaults on next load.

    Normalizes attribute names to current canonical format so old names
    (h/9_r, bb_per_bf, etc.) are migrated to new names on save."""
    from src.models.registry import normalize_attr_name as _norm

    path = path or _COEFFICIENTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    clean = {}
    for attr, c in coeffs.items():
        if c is None:
            continue
        canonical = _norm(attr)
        clean[canonical] = {k: float(v) for k, v in c.items()}

    path.write_text(json.dumps(clean, indent=2, sort_keys=True), encoding="utf-8")
    global _COEFFICIENTS_CACHE
    _COEFFICIENTS_CACHE = clean
    return path


def load_coefficients(path: Path | None = None) -> dict | None:
    """Load refitted coefficients from JSON. Returns None if file doesn't exist."""
    global _COEFFICIENTS_CACHE
    if _COEFFICIENTS_CACHE is not None:
        return _COEFFICIENTS_CACHE

    path = path or _COEFFICIENTS_PATH
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _COEFFICIENTS_CACHE = data
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _load_active_coefficients() -> dict:
    """Load refitted coefficients if available, otherwise fall back to defaults.

    Normalizes old attribute names (h/9_r, bb_per_bf, etc.) to current
    canonical naming on load so that refits from previous training runs
    are applied to the correct attributes."""
    from src.models.registry import normalize_attr_name as _norm

    refitted = load_coefficients()
    if refitted:
        merged = dict(DEFAULT_COEFFICIENTS)
        for attr, c in refitted.items():
            if c is None:
                continue
            var_coeffs = [v for k, v in c.items() if k != "intercept"]
            if var_coeffs and all(abs(v) < 0.01 for v in var_coeffs):
                continue
            canonical = _norm(attr)
            if canonical != attr:
                attr = canonical
            merged[attr] = c
        return merged
    return DEFAULT_COEFFICIENTS


def refit_and_save(training_df, path: Path | None = None) -> dict:
    """Convenience: refit coefficients and save to disk in one call."""
    coeffs = refit_coefficients(training_df)
    if coeffs:
        save_coefficients(coeffs, path)
    return coeffs
