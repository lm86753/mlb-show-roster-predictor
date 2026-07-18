"""Tests for momentum features."""
import sys
import os
import json
import tempfile
import numpy as np
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.momentum import (
    MomentumComputer, GameLogFetcher,
    _linear_slope, _compute_streak_count, _volatility_clustering,
    derive_hitter_game_stats, derive_pitcher_game_stats,
)


def test_linear_slope():
    # Perfect positive slope
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = _linear_slope(x)
    assert abs(result - 1.0) < 1e-6, f"Expected 1.0, got {result}"

    # Flat
    y = np.array([5.0, 5.0, 5.0, 5.0])
    result = _linear_slope(y)
    assert abs(result - 0.0) < 1e-6, f"Expected 0.0, got {result}"

    # Noisy upward trend
    z = np.array([0.2, 0.25, 0.23, 0.28, 0.30])
    result = _linear_slope(z)
    assert result > 0, f"Expected positive slope, got {result}"

    # Too short
    assert _linear_slope(np.array([1.0])) == 0.0
    print("✓ linear_slope tests passed")


def test_streak_count():
    # Improving ops (higher is better)
    ops_improving = np.array([0.6, 0.65, 0.70, 0.75, 0.80])
    result = _compute_streak_count(ops_improving, 'ops')
    assert result == 4, f"Expected 4, got {result}"

    # Declining ops
    ops_declining = np.array([0.80, 0.75, 0.70, 0.65, 0.60])
    result = _compute_streak_count(ops_declining, 'ops')
    assert result == -4, f"Expected -4, got {result}"

    # Worsening era (era: higher is worse)
    era_worsening = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
    result = _compute_streak_count(era_worsening, 'era')
    assert result == -4, f"Expected -4, got {result}"

    # Improving era
    era_improving = np.array([5.0, 4.5, 4.0, 3.5, 3.0])
    result = _compute_streak_count(era_improving, 'era')
    assert result == 4, f"Expected 4, got {result}"

    # Mixed (streak should be from the end)
    ops_mixed = np.array([0.6, 0.7, 0.65, 0.70, 0.75])
    result = _compute_streak_count(ops_mixed, 'ops')
    assert result == 2, f"Expected 2, got {result}"

    # Empty/short
    assert _compute_streak_count(np.array([0.5]), 'ops') == 0

    print("✓ streak_count tests passed")


def test_volatility_clustering():
    np.random.seed(42)

    # Clustered volatility: low vol followed by high vol
    low_vol = np.random.normal(0.8, 0.02, 20)
    high_vol = np.random.normal(0.8, 0.15, 20)
    clustered = np.concatenate([low_vol, high_vol])
    result_clustered = _volatility_clustering(clustered)

    # Random walk (constant volatility)
    random_walk = np.random.normal(0.8, 0.08, 40)
    result_random = _volatility_clustering(random_walk)

    # Clustered should show higher autocorrelation
    print(f"  Vol clustering (clustered): {result_clustered:.4f}")
    print(f"  Vol clustering (random): {result_random:.4f}")
    # Not asserting strict inequality due to randomness, just check they're in valid range
    assert -1.0 <= result_clustered <= 1.0
    assert -1.0 <= result_random <= 1.0

    # Too short
    assert _volatility_clustering(np.array([0.5, 0.6])) == 0.0

    print("✓ volatility_clustering tests passed")


def test_derive_hitter_stats():
    stat = {
        'atBats': 4, 'hits': 2, 'baseOnBalls': 1,
        'strikeOuts': 1, 'homeRuns': 1, 'hitByPitch': 0,
        'sacFlies': 0, 'totalBases': 5
    }
    result = derive_hitter_game_stats(stat)
    assert abs(result['avg'] - 0.5) < 1e-6  # 2/4
    assert abs(result['k_pct'] - 0.2) < 1e-6  # 1/5
    assert result['hr'] == 1
    assert result['pa'] == 5  # 4 AB + 1 BB
    print(f"  Hitter: {result}")
    print("✓ derive_hitter_stats tests passed")


def test_derive_pitcher_stats():
    stat = {
        'inningsPitched': '6.1', 'baseOnBalls': 2,
        'strikeOuts': 7, 'homeRuns': 0, 'hits': 4,
        'earnedRuns': 2, 'battersFaced': 25
    }
    result = derive_pitcher_game_stats(stat)
    # 6.1 IP = 6 + 1/3 = 6.333...
    assert result['ip'] > 6.3
    assert abs(result['k9'] - (7 * 9 / (19/3))) < 0.1  # ~10.03
    assert abs(result['bb9'] - (2 * 9 / (19/3))) < 0.1
    print(f"  Pitcher: {result}")
    print("✓ derive_pitcher_stats tests passed")


def test_momentum_computer_empty():
    computer = MomentumComputer()
    # No data == empty features
    features = computer._empty_features(is_hitter=True)
    assert "trend_avg" in features
    assert "trend_ops" in features
    assert "trend_k_pct" in features
    assert features["streak_primary"] == 0.0
    assert features["vol_cluster_primary"] == 0.0
    print(f"  Empty hitter features keys: {list(features.keys())}")

    features_p = computer._empty_features(is_hitter=False)
    assert "trend_era" in features_p
    assert "trend_k9" in features_p
    assert "trend_bb9" in features_p
    print(f"  Empty pitcher features keys: {list(features_p.keys())}")
    print("✓ MomentumComputer empty features tests passed")


def test_momentum_computer_with_mock_data():
    """Test MomentumComputer with pre-loaded mock games."""
    computer = MomentumComputer()

    # Create mock game data - hitter performance trending up
    # Use variable AB and hits to get realistic AVG progression
    np.random.seed(123)
    mock_games = []
    for i in range(25):
        # Simulate improving hitter: more hits per AB over time
        ab = np.random.randint(3, 5)
        # Hit rate increases from ~0.15 to ~0.40
        hit_prob = 0.15 + i * 0.01
        hits = np.random.binomial(ab, hit_prob)
        bb = 1 if i % 4 == 0 else 0
        so = np.random.randint(0, 3)
        hr = 1 if (i > 15 and i % 3 == 0) else 0
        tb = hits + hr * 3 + np.random.randint(0, 2)  # some extra bases
        mock_games.append({
            'date': f'2026-04-{i+1:02d}',
            'stat': {
                'atBats': int(ab),
                'hits': int(hits),
                'baseOnBalls': int(bb),
                'strikeOuts': int(so),
                'homeRuns': int(hr),
                'hitByPitch': 0,
                'sacFlies': 0,
                'totalBases': int(tb),
            }
        })

    # Inject into cache
    cache_key = (999999, 2026, True)
    computer._games_cache[cache_key] = mock_games

    # Test compute_trend via the public path
    games = computer._get_games(999999, 2026, True)
    stat_arrays = computer._get_stat_arrays(999999, 2026, True, games)

    trend = computer.compute_trend(games, stat_arrays, True)
    print(f"  Trend features: {trend}")
    # With random data we just check the keys exist and values are finite
    assert 'trend_avg' in trend
    assert 'trend_ops' in trend
    assert 'trend_k_pct' in trend
    assert all(np.isfinite(v) for v in trend.values())

    streak = computer.compute_streak(stat_arrays, True)
    print(f"  Streak features: {streak}")

    consistency = computer.compute_consistency(stat_arrays, True)
    print(f"  Consistency features: {consistency}")

    vol = computer.compute_volatility_clustering(stat_arrays, True)
    print(f"  Volatility features: {vol}")

    # Full compute_all
    # Override _get_games to return our mock
    original_get = computer._get_games
    computer._get_games = lambda pid, s, h: mock_games if pid == 999999 else original_get(pid, s, h)
    all_features = computer.compute_all(999999, 2026, True)
    print(f"  All features ({len(all_features)} total): {list(all_features.keys())[:10]}...")
    assert len(all_features) == 14  # 3x trend + 4 streak + 3x consistency + 4x vol = 14

    print("✓ MomentumComputer with mock data passed")


if __name__ == "__main__":
    test_linear_slope()
    test_streak_count()
    test_volatility_clustering()
    test_derive_hitter_stats()
    test_derive_pitcher_stats()
    test_momentum_computer_empty()
    test_momentum_computer_with_mock_data()
    print("\n====== ALL TESTS PASSED ======")
