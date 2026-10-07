import numpy as np
import pandas as pd
import pytest


def test_clip_rating():
    from src.formulas.ratings import clip_rating

    assert clip_rating(105) == 105
    assert clip_rating(140) == 125
    assert clip_rating(-5) == 0
    assert clip_rating(75.4) == 75


def test_project_hitter_vision():
    from src.formulas.ratings import project_hitter_attribute

    stats = {"k_pct": 0.20, "bb_pct": 0.10, "avg": 0.280, "iso": 0.180}
    rating = project_hitter_attribute("plate_vision", stats)
    assert 0 <= rating <= 125


def test_parse_delta():
    from src.ingest.sds_client import parse_delta
    from src.models.registry import normalize_attr_name

    assert parse_delta("+8") == 8
    assert parse_delta("-3") == -3
    assert normalize_attr_name("CTRL") == "pitch_control"


@pytest.mark.parametrize("label, canonical", [
    ("PCLT", "pitching_clutch"), ("CLT", "batting_clutch"), ("STA", "stamina"),
    ("H/9 R", "h_per_9_r"), ("K/9 L", "k_per_9_l"), ("h/9_r", "h_per_9_r"),
    ("vis", "plate_vision"), ("CON R", "contact_right"),
])
def test_normalize_sds_labels(label, canonical):
    from src.models.registry import normalize_attr_name

    assert normalize_attr_name(label, 26) == canonical


def test_quicksell_values():
    from src.config import quicksell_value

    assert quicksell_value(60) == 5
    assert quicksell_value(74) == 25
    assert quicksell_value(79) == 150
    assert quicksell_value(80) == 400
    assert quicksell_value(85) == 3000
    assert quicksell_value(99) == 10000


def test_windows_only_use_games_before_as_of():
    from src.features.windows import windows_as_of

    def game(day, hits, ab=4):
        return {"date": f"2026-06-{day:02d}", "stat": {"atBats": ab, "hits": hits, "plateAppearances": ab, "totalBases": hits}}

    games = [game(1, 0), game(5, 1), game(9, 4), game(10, 4)]
    w = windows_as_of(games, "2026-06-10", "hitting")
    assert w["ytd"]["g"] == 3          # the 10th itself is excluded
    assert w["7d"]["g"] == 2           # June 3-9
    assert w["ytd"]["avg"] == pytest.approx(5 / 12)


def test_rating_reconstruction_between_changes():
    from src.features.dataset import _rating_lookup

    changes = pd.DataFrame([
        # card A changed in May and July; June update should see May's rating_after
        {"card_uuid": "A", "attr": "contact_left", "update_date": "2026-05-08", "update_id": 1, "rating_before": 70, "rating_after": 74, "delta": 4},
        {"card_uuid": "A", "attr": "contact_left", "update_date": "2026-07-16", "update_id": 3, "rating_before": 74, "rating_after": 71, "delta": -3},
        # card B first changed in July; June should see July's rating_before
        {"card_uuid": "B", "attr": "contact_left", "update_date": "2026-07-16", "update_id": 3, "rating_before": 88, "rating_after": 90, "delta": 2},
    ])
    rating, hist, target = _rating_lookup(changes, "2026-06-12", ["contact_left"])
    assert rating[("A", "contact_left")] == 74
    assert rating[("B", "contact_left")] == 88
    assert hist.loc[("A", "contact_left"), "last_delta"] == 4
    assert ("A", "contact_left") not in target.index  # no change on June 12

    _, _, target = _rating_lookup(changes, "2026-07-16", ["contact_left"])
    assert target[("A", "contact_left")] == -3


def test_market_metrics_respect_ovr_cap():
    from src.models.train import OVR_MOVES, player_market_metrics

    probs = np.zeros((2, len(OVR_MOVES)))
    probs[:, OVR_MOVES.index(2)] = 1.0   # both cards "go up 2"
    mm = player_market_metrics(np.array([84, 99]), probs)
    assert mm.loc[0, "upgrade_probability"] == pytest.approx(1.0)
    assert mm.loc[0, "expected_qs_change"] == pytest.approx(3750 - 1500)
    assert mm.loc[0, "tier_up_probability"] == pytest.approx(1.0)  # 84 -> 86 crosses into Gold
    # A 99 can't go up: mass folds back onto "no change".
    assert mm.loc[1, "upgrade_probability"] == pytest.approx(0.0)
    assert mm.loc[1, "p_no_change"] == pytest.approx(1.0)
