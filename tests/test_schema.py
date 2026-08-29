import pandas as pd
import pytest

from normalise import schema, teams


def _raw_row(**overrides):
    row = {
        "Div": "E0", "Season": "2526", "Date": "15/08/2025", "Time": "20:00",
        "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
        "FTHG": "2", "FTAG": "1", "FTR": "H",
        "HST": "5", "AST": "3", "HS": "10", "AS": "8",
        "Referee": "M Oliver",
        "PSCH": "1.9", "PSCD": "3.6", "PSCA": "4.2",
        "AvgCH": "1.85", "AvgCD": "3.55", "AvgCA": "4.1",
        "B365CH": "1.9", "B365CD": "3.6", "B365CA": "4.3",
    }
    row.update(overrides)
    return row


@pytest.fixture(autouse=True)
def known_teams(monkeypatch, tmp_path):
    yaml_path = tmp_path / "teams.yaml"
    yaml_path.write_text("Arsenal: []\nChelsea: []\n")
    monkeypatch.setattr(teams, "_YAML", yaml_path)
    teams.reload_cache()
    yield
    teams.reload_cache()


def test_normalise_basic_row():
    raw = pd.DataFrame([_raw_row()])
    out = schema.normalise(raw)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["home_team"] == "Arsenal" and r["away_team"] == "Chelsea"
    assert r["home_goals"] == 2 and r["away_goals"] == 1
    assert r["result"] == "H"
    assert r["home_sot"] == 5 and r["away_sot"] == 3


def test_prefers_pinnacle_closing_odds():
    raw = pd.DataFrame([_raw_row()])
    out = schema.normalise(raw)
    r = out.iloc[0]
    assert r["close_home"] == pytest.approx(1.9)
    assert r["close_draw"] == pytest.approx(3.6)
    assert r["close_away"] == pytest.approx(4.2)


def test_falls_back_to_average_closing_when_pinnacle_missing():
    raw = pd.DataFrame([_raw_row(PSCH="", PSCD="", PSCA="")])
    out = schema.normalise(raw)
    r = out.iloc[0]
    assert r["close_home"] == pytest.approx(1.85)


def test_drops_unplayed_matches():
    raw = pd.DataFrame([
        _raw_row(),
        _raw_row(FTHG="", FTAG="", FTR="", HomeTeam="Chelsea", AwayTeam="Arsenal"),
    ])
    out = schema.normalise(raw)
    assert len(out) == 1


def test_unknown_team_name_raises():
    raw = pd.DataFrame([_raw_row(HomeTeam="Not A Real Club")])
    with pytest.raises(teams.UnknownTeamError):
        schema.normalise(raw)


def test_match_id_is_unique_and_stable():
    raw = pd.DataFrame([_raw_row()])
    out1 = schema.normalise(raw)
    out2 = schema.normalise(raw)
    assert out1.loc[0, "match_id"] == out2.loc[0, "match_id"]
