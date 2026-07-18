import pytest


def test_clip_rating():
    from src.formulas.ratings import clip_rating

    assert clip_rating(105) == 99
    assert clip_rating(-5) == 0
    assert clip_rating(75.4) == 75


def test_project_hitter_vision():
    from src.formulas.ratings import project_hitter_attribute

    stats = {"k_pct": 0.20, "bb_pct": 0.10, "avg": 0.280, "iso": 0.180}
    rating = project_hitter_attribute("plate_vision", stats)
    assert 0 <= rating <= 99


def test_change_label():
    from src.features.engineering import change_label

    assert change_label(3) == "upgrade"
    assert change_label(-2) == "downgrade"
    assert change_label(0) == "no_change"


def test_tier_distance():
    from src.features.engineering import tier_distance, tier_for_ovr

    assert tier_for_ovr(85) == "Gold"
    assert tier_distance(84, "up") == 1


def test_parse_delta():
    from src.ingest.sds_client import parse_delta
    from src.models.registry import normalize_attr_name

    assert parse_delta("+8") == 8
    assert parse_delta("-3") == -3
    assert normalize_attr_name("CTRL") == "pitch_control"
