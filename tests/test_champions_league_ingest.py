import pandas as pd
import pytest

from ingest import champions_league as cl


def _match(home, away, utc, status, hg=None, ag=None, md=1, stage="LEAGUE_STAGE"):
    return {
        "matchday": md, "stage": stage, "utcDate": utc, "status": status,
        "homeTeam": {"name": f"{home} FC", "shortName": home, "tla": home[:3].upper()},
        "awayTeam": {"name": f"{away} FC", "shortName": away, "tla": away[:3].upper()},
        "score": {"fullTime": {"home": hg, "away": ag}},
    }


def test_normalise_parses_results_and_fixtures():
    payload = {"matches": [
        _match("Arsenal", "Barcelona", "2024-09-17T19:00:00Z", "FINISHED", 2, 1),
        _match("Benfica", "Bayern Munich", "2030-09-18T19:00:00Z", "SCHEDULED"),
    ]}
    df, unresolved = cl._normalise(payload)
    assert unresolved == []
    assert df.loc[0, "result"] == "H"
    assert pd.isna(df.loc[1, "result"])
    assert df.loc[0, "home_team"] == "Arsenal"
    assert df.loc[0, "competition"] == "CL"
    assert df.loc[0, "match_id"].startswith("CL_2425_20240917_")
    assert df.loc[0, "season"] == "2425"


def test_normalise_reports_unresolved_without_crashing():
    payload = {"matches": [{
        "matchday": 1, "stage": "LEAGUE_STAGE", "utcDate": "2024-09-17T19:00:00Z",
        "status": "FINISHED",
        "homeTeam": {"name": "Slovan Bratislava", "shortName": "Slovan", "tla": "SLB"},
        "awayTeam": {"name": "Arsenal FC", "shortName": "Arsenal", "tla": "ARS"},
        "score": {"fullTime": {"home": 0, "away": 3}},
    }]}
    df, unresolved = cl._normalise(payload)
    assert df.empty
    assert unresolved and unresolved[0][0] == "Slovan Bratislava"


def test_missing_token_raises_missing_token(monkeypatch):
    monkeypatch.setattr(cl.config, "FOOTBALL_DATA_ORG_TOKEN", "")
    with pytest.raises(cl.MissingToken):
        cl._get({"season": 2024})


def test_all_degrades_to_cache_when_refresh_fails(monkeypatch, tmp_path):
    cache = tmp_path / "ucl.csv"
    seed = pd.DataFrame([{
        "match_id": "CL_2425_20240917_arsenal_barcelona", "competition": "CL",
        "season": "2425", "date": "2024-09-17T19:00:00", "matchday": 1,
        "stage": "LEAGUE_STAGE", "home_team": "Arsenal", "away_team": "Barcelona",
        "home_goals": 2, "away_goals": 1, "result": "H", "status": "FINISHED",
    }])
    seed.to_csv(cache, index=False)
    monkeypatch.setattr(cl, "_CACHE", cache)

    def boom(*a, **k):
        raise cl.MissingToken("no token")

    monkeypatch.setattr(cl, "refresh", boom)
    out = cl._all(refresh_data=True)
    assert len(out) == 1 and out.loc[0, "home_team"] == "Arsenal"

    played = cl.results(refresh_data=False)
    assert list(played["result"]) == ["H"]
    assert played.loc[0, "home_goals"] == 2
