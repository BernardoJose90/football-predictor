import numpy as np
import pandas as pd

from model.league_bridge import cl_goal_baselines, elo_bridge


def test_gamma_zero_is_a_noop():
    assert elo_bridge(1900, 1600, 0.0) == 1.0


def test_missing_elo_is_a_noop():
    assert elo_bridge(None, 1600, 1.0) == 1.0
    assert elo_bridge(1900, None, 1.0) == 1.0


def test_swapping_sides_gives_the_reciprocal():
    up = elo_bridge(1900, 1600, 0.7)
    down = elo_bridge(1600, 1900, 0.7)
    assert up > 1.0 > down
    assert up * down == np.float64(1.0) or abs(up * down - 1.0) < 1e-9


def test_stronger_home_lifts_the_multiplier_monotonically():
    weak = elo_bridge(1700, 1650, 0.7)
    strong = elo_bridge(2000, 1650, 0.7)
    assert strong > weak > 1.0


def test_goal_baselines_fall_back_when_history_is_thin():
    tiny = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "home_goals": [3], "away_goals": [0]})
    assert cl_goal_baselines(tiny, "2024-06-01") == (1.50, 1.20)


def test_goal_baselines_weight_recent_matches():
    dates = pd.date_range("2023-01-01", periods=80, freq="7D")
    df = pd.DataFrame({"date": dates, "home_goals": [2] * 80, "away_goals": [1] * 80})
    g_home, g_away = cl_goal_baselines(df, dates.max() + pd.Timedelta(days=7))
    assert round(g_home, 3) == 2.0
    assert round(g_away, 3) == 1.0


def test_goal_baselines_ignore_matches_on_or_after_as_of():
    dates = pd.date_range("2023-01-01", periods=60, freq="7D")
    df = pd.DataFrame({"date": list(dates) + [pd.Timestamp("2025-01-01")],
                       "home_goals": [1] * 60 + [9], "away_goals": [1] * 60 + [9]})
    g_home, _ = cl_goal_baselines(df, "2024-06-01")
    assert g_home < 2  # the 9-9 blowout is in the future and must not count
