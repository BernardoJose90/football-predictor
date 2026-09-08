import pytest

from model.league_bridge import elo_bridge
from model.predict import predict_cross_league
from model.ratings import build_ratings


@pytest.fixture
def two_leagues(synthetic_matches):
    """Same synthetic league used as two independent rating pools - stands in
    for 'home team's league' and 'away team's league'."""
    snap = build_ratings(synthetic_matches, as_of="2025-06-01", stat="goals", min_matches=8)
    return snap, snap


def test_probabilities_sum_to_one(two_leagues):
    snap_h, snap_a = two_leagues
    pred = predict_cross_league(snap_h, snap_a, "Alpha", "Gamma")
    assert pred["p_home"] + pred["p_draw"] + pred["p_away"] == pytest.approx(1.0, abs=1e-6)
    assert 0 < pred["home_pred"] < 15 and 0 < pred["away_pred"] < 15


def test_unrated_side_returns_none(two_leagues):
    snap_h, snap_a = two_leagues
    assert predict_cross_league(snap_h, snap_a, "Alpha", "Nobody FC") is None
    assert predict_cross_league(snap_h, snap_a, "Nobody FC", "Alpha") is None


def test_bridge_shifts_the_forecast_toward_the_stronger_side(two_leagues):
    snap_h, snap_a = two_leagues
    flat = predict_cross_league(snap_h, snap_a, "Gamma", "Beta", bridge=1.0)
    lifted = predict_cross_league(
        snap_h, snap_a, "Gamma", "Beta", bridge=elo_bridge(1950, 1600, 0.8))
    assert lifted["p_home"] > flat["p_home"]
    assert lifted["home_pred"] > flat["home_pred"]


def test_bridge_below_one_helps_the_away_side(two_leagues):
    snap_h, snap_a = two_leagues
    flat = predict_cross_league(snap_h, snap_a, "Beta", "Gamma", bridge=1.0)
    away_favoured = predict_cross_league(
        snap_h, snap_a, "Beta", "Gamma", bridge=elo_bridge(1600, 1950, 0.8))
    assert away_favoured["p_away"] > flat["p_away"]


def test_explicit_goal_baselines_are_used(two_leagues):
    snap_h, snap_a = two_leagues
    low = predict_cross_league(snap_h, snap_a, "Beta", "Delta", g_home=0.9, g_away=0.8)
    high = predict_cross_league(snap_h, snap_a, "Beta", "Delta", g_home=2.1, g_away=1.9)
    assert high["home_pred"] > low["home_pred"]
    assert high["p_over_2_5"] > low["p_over_2_5"]
