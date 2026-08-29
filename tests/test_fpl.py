import pytest

from ingest.fpl import team_availability
from normalise import teams


def _bootstrap(elements, teams_list=None):
    teams_list = teams_list or [
        {"id": 1, "name": "Arsenal"},
        {"id": 2, "name": "Coventry City"},
    ]
    return {"teams": teams_list, "elements": elements}


@pytest.fixture(autouse=True)
def known_teams(monkeypatch, tmp_path):
    yaml_path = tmp_path / "teams.yaml"
    yaml_path.write_text("Arsenal: []\nCoventry: []\n")
    monkeypatch.setattr(teams, "_YAML", yaml_path)
    teams.reload_cache()
    yield
    teams.reload_cache()


def test_fully_fit_squad_gives_availability_one():
    elements = [
        {"team": 1, "now_cost": 60, "chance_of_playing_next_round": None, "status": "a"},
        {"team": 1, "now_cost": 55, "chance_of_playing_next_round": None, "status": "a"},
    ]
    out = team_availability(_bootstrap(elements), min_importance=1.0)
    assert out["Arsenal"]["availability"] == pytest.approx(1.0)
    assert out["Arsenal"]["n_flagged"] == 0


def test_injured_key_player_lowers_availability():
    elements = [
        {"team": 1, "now_cost": 60, "chance_of_playing_next_round": None, "status": "i"},  # out
        {"team": 1, "now_cost": 20, "chance_of_playing_next_round": None, "status": "a"},   # fit
    ]
    out = team_availability(_bootstrap(elements), min_importance=1.0)
    # 60's worth of price out of 80 total unavailable -> availability = 20/80 = 0.25
    assert out["Arsenal"]["availability"] == pytest.approx(0.25)
    assert out["Arsenal"]["n_flagged"] == 1


def test_season_long_absentee_with_zero_points_still_weighted_by_cost():
    # The actual bug this test guards against: a star player injured before
    # playing a single minute this season has total_points=0. Weighting by
    # points (the first version of this code) would silently treat him as
    # unimportant precisely because he's been out all season - the exact
    # opposite of what this feature needs to catch. now_cost doesn't zero
    # out just because a player hasn't played.
    elements = [
        {"team": 1, "now_cost": 60, "chance_of_playing_next_round": 0, "status": "i",
         "total_points": 0},                                                          # Saliba-like
        {"team": 1, "now_cost": 45, "chance_of_playing_next_round": None, "status": "a",
         "total_points": 12},
    ]
    out = team_availability(_bootstrap(elements), min_importance=1.0)
    assert out["Arsenal"]["availability"] == pytest.approx(45 / 105)
    assert out["Arsenal"]["n_flagged"] == 1


def test_chance_of_playing_used_when_present():
    elements = [
        {"team": 1, "now_cost": 60, "chance_of_playing_next_round": 75, "status": "d"},
    ]
    out = team_availability(_bootstrap(elements), min_importance=1.0)
    assert out["Arsenal"]["availability"] == pytest.approx(0.75)


def test_fpl_team_name_alias_resolves_to_canonical():
    elements = [
        {"team": 2, "now_cost": 45, "chance_of_playing_next_round": None, "status": "a"},
    ]
    out = team_availability(_bootstrap(elements), min_importance=1.0)
    assert "Coventry" in out   # "Coventry City" (FPL) -> "Coventry" (canonical)


def test_team_below_min_importance_is_omitted():
    elements = [
        {"team": 1, "now_cost": 2, "chance_of_playing_next_round": None, "status": "a"},
    ]
    out = team_availability(_bootstrap(elements), min_importance=5.0)
    assert "Arsenal" not in out


def test_zero_cost_players_ignored_entirely():
    elements = [
        {"team": 1, "now_cost": 0, "chance_of_playing_next_round": None, "status": "i"},
        {"team": 1, "now_cost": 45, "chance_of_playing_next_round": None, "status": "a"},
    ]
    out = team_availability(_bootstrap(elements), min_importance=1.0)
    assert out["Arsenal"]["availability"] == pytest.approx(1.0)


def test_unknown_team_id_skipped_not_guessed():
    elements = [
        {"team": 999, "now_cost": 60, "chance_of_playing_next_round": None, "status": "a"},
    ]
    out = team_availability(_bootstrap(elements), min_importance=1.0)
    assert out == {}
