"""API contract tests. Skipped entirely if FastAPI isn't installed
(see api/requirements.txt) so the core test run stays dependency-light."""
import json
import warnings

import pytest

pytest.importorskip("fastapi")
with warnings.catch_warnings():  # starlette 1.6 "install httpx2" nudge, fires on import
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient  # noqa: E402

from api import data as api_data  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    preds = [
        {"match_id": "E0_2627_20260830_arsenal_chelsea", "league": "Premier League",
         "date": "2026-08-30T15:00", "home_team": "Arsenal", "away_team": "Chelsea", "unrated": False,
         "likely_score": "2-1", "p_home": 0.55, "p_draw": 0.25, "p_away": 0.20,
         "p_over_2_5": 0.6, "p_btts": 0.55, "stat_used": "xg",
         "market_p_home": 0.52, "market_p_draw": 0.26, "market_p_away": 0.22,
         "base_p_home": 0.5, "base_p_draw": 0.27, "base_p_away": 0.23,
         "home_pred": 1.8, "away_pred": 1.1, "base_home_pred": 1.7, "base_away_pred": 1.1,
         "home_attack": 1.3, "home_defence": 0.9, "away_attack": 1.0, "away_defence": 1.0,
         "home_matches_used": 40, "away_matches_used": 38,
         "adjustments": [{"kind": "rest", "detail": "6d/4d", "home_factor": 1.0, "away_factor": 0.98}],
         "adj_note": "rest=6d/4d"},
        {"match_id": "SP1_2627_20260830_madrid_malaga", "league": "La Liga",
         "date": "2026-08-30T18:00", "home_team": "Real Madrid", "away_team": "Malaga", "unrated": True},
    ]
    why = [{
        "match_id": "E0_2627_20260830_arsenal_chelsea", "league": "Premier League",
        "date": "2026-08-30T15:00", "home": "Arsenal", "away": "Chelsea",
        "primary": "home", "direction": "higher", "gap": 8.0, "moved": 1.0, "attribution": "ratings",
        "pHome": 55.0, "pDraw": 25.0, "pAway": 20.0, "mHome": 47.0, "mDraw": 28.0, "mAway": 25.0,
        "pureHome": 54.0, "pureDraw": 25.5, "pureAway": 20.5, "score": "2-1", "stat": "xg",
        "adjustments": [], "homeAtk": 1.3, "homeDef": 0.9, "awayAtk": 1.0, "awayDef": 1.0,
        "homeMatches": 40, "awayMatches": 38, "baseGoals": [1.7, 1.1], "finalGoals": [1.8, 1.1],
    }]
    track = {"generated": "2026-08-29 10:00 UTC",
             "live": {"n_logged": 3, "n_scored": 1, "n_pending": 2,
                      "fixtures": [{"match_id": "x", "home": "A", "away": "B", "result": "H"}]},
            "validation": {"headline": {"model_rps": 0.2}}}

    (tmp_path / "upcoming_predictions.json").write_text(json.dumps(preds))
    (tmp_path / "why.json").write_text(json.dumps(why))
    (tmp_path / "track_record.json").write_text(json.dumps(track))
    monkeypatch.setattr(api_data, "PREDICTIONS", tmp_path / "upcoming_predictions.json")
    monkeypatch.setattr(api_data, "WHY", tmp_path / "why.json")
    monkeypatch.setattr(api_data, "TRACK_RECORD", tmp_path / "track_record.json")
    api_data.clear_cache()

    from api.app import app
    return TestClient(app)


def test_health_lists_artefacts(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["artefacts"] == {"predictions": True, "disagreements": True, "track_record": True}


def test_leagues(client):
    r = client.get("/v1/leagues")
    assert r.status_code == 200
    codes = {x["code"] for x in r.json()}
    assert {"E0", "SP1", "D1"} <= codes


def test_predictions_default_drops_unrated(client):
    r = client.get("/v1/predictions")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["home_team"] == "Arsenal"
    assert body[0]["model"] == {"home": 55.0, "draw": 25.0, "away": 20.0}
    assert body[0]["market"]["home"] == 52.0


def test_predictions_include_unrated(client):
    r = client.get("/v1/predictions?rated_only=false")
    assert len(r.json()) == 2
    assert any(f["unrated"] for f in r.json())


def test_predictions_league_filter_by_code_and_name(client):
    assert len(client.get("/v1/predictions?league=E0").json()) == 1
    assert len(client.get("/v1/predictions?league=premier").json()) == 1
    assert client.get("/v1/predictions?league=I1").json() == []


def test_prediction_detail_has_breakdown(client):
    r = client.get("/v1/predictions/E0_2627_20260830_arsenal_chelsea")
    assert r.status_code == 200
    d = r.json()
    assert d["base_model"] == {"home": 50.0, "draw": 27.0, "away": 23.0}
    assert d["home_attack"] == 1.3
    assert d["adjustments"][0]["kind"] == "rest"


def test_prediction_detail_404(client):
    assert client.get("/v1/predictions/nope").status_code == 404


def test_disagreements_min_gap(client):
    assert len(client.get("/v1/disagreements?min_gap=5").json()) == 1
    assert client.get("/v1/disagreements?min_gap=20").json() == []
    row = client.get("/v1/disagreements").json()[0]
    assert row["attribution"] == "ratings"
    assert row["gap_points"] == 8.0
    assert row["ratings_only"]["home"] == 54.0


def test_track_record_and_fixtures(client):
    assert client.get("/v1/track-record").json()["validation"]["headline"]["model_rps"] == 0.2
    assert client.get("/v1/track-record/fixtures").json()[0]["result"] == "H"


def test_503_when_artefact_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api_data, "PREDICTIONS", tmp_path / "gone.json")
    api_data.clear_cache()
    r = client.get("/v1/predictions")
    assert r.status_code == 503
    assert "artefact" in r.json()["detail"]


def test_cache_control_header_on_get(client):
    assert "max-age" in client.get("/v1/leagues").headers.get("cache-control", "")


def test_openapi_schema_builds(client):
    assert client.get("/openapi.json").status_code == 200
