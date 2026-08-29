import pandas as pd
import pytest

from model.predict import predict_match, predict_fixtures
from model.ratings import build_ratings


@pytest.fixture
def snapshot(synthetic_matches):
    return build_ratings(synthetic_matches, as_of="2025-06-01", stat="goals", min_matches=8)


def test_probabilities_sum_to_one(snapshot):
    pred = predict_match(snapshot, "Alpha", "Gamma")
    total = pred["p_home"] + pred["p_draw"] + pred["p_away"]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_strong_home_team_is_favoured(snapshot):
    pred = predict_match(snapshot, "Alpha", "Gamma")
    assert pred["p_home"] > pred["p_away"]
    assert pred["p_home"] > 0.5


def test_attack_ratings_stay_in_a_sane_range(snapshot):
    # Design doc sanity check: no attack rating above ~2.5 for a reasonably
    # balanced synthetic league.
    for team in snapshot.teams:
        assert 0 < snapshot.attack(team) < 2.5


def test_unrated_team_returns_none(snapshot):
    assert predict_match(snapshot, "Alpha", "Nonexistent FC") is None
    assert predict_match(snapshot, "Nonexistent FC", "Alpha") is None


def test_predict_fixtures_flags_unrated_rows(snapshot):
    fixtures = pd.DataFrame([
        {"home_team": "Alpha", "away_team": "Gamma"},
        {"home_team": "Alpha", "away_team": "Nonexistent FC"},
    ])
    out = predict_fixtures(snapshot, fixtures)
    assert out.loc[0, "unrated"] == False
    assert out.loc[1, "unrated"] == True
    assert "p_home" in out.columns
    assert pd.isna(out.loc[1, "p_home"])


def test_likely_score_is_a_valid_scoreline(snapshot):
    pred = predict_match(snapshot, "Alpha", "Gamma")
    h, a = pred["likely_score"].split("-")
    assert int(h) >= 0 and int(a) >= 0
