"""The Champions League block of scripts.predict_upcoming, with the two
network calls (fixtures + Club Elo) stubbed."""
from types import SimpleNamespace

import pandas as pd
import pytest

from ingest import champions_league, club_elo
from scripts import predict_upcoming


@pytest.fixture
def domestic(synthetic_matches):
    l1 = synthetic_matches.copy()
    l1["div"] = "L1"
    l2 = synthetic_matches.copy()
    l2["div"] = "L2"
    for c in ("home_team", "away_team"):
        l2[c] = l2[c] + " B"
    return pd.concat([l1, l2], ignore_index=True)


@pytest.fixture
def args():
    return SimpleNamespace(refresh=False, stat="goals", xi=0.0035, rho=0.10,
                           delta=0.20, cl_gamma=0.7)


def _stub(monkeypatch, fixtures, elo):
    monkeypatch.setattr(champions_league, "upcoming", lambda **k: fixtures)
    monkeypatch.setattr(champions_league, "results",
                        lambda **k: pd.DataFrame(columns=["date", "home_goals", "away_goals"]))
    monkeypatch.setattr(club_elo, "elo_by_date", lambda dates, **k: elo)


def test_block_predicts_rated_ties_and_flags_unknown_clubs(domestic, args, monkeypatch):
    day = pd.Timestamp("2025-08-06 20:00")
    fixtures = pd.DataFrame([
        {"date": day, "home_team": "Alpha", "away_team": "Gamma B",
         "match_id": "CL_1", "matchday": 1, "stage": "LEAGUE_STAGE"},
        {"date": day, "home_team": "Beta", "away_team": "Nowhere FC",
         "match_id": "CL_2", "matchday": 1, "stage": "LEAGUE_STAGE"},
    ])
    # elo_by_date is called for "today", not the fixture date
    elo = {pd.Timestamp("2025-08-01"): {"Alpha": 1950, "Gamma B": 1500, "Beta": 1750}}
    _stub(monkeypatch, fixtures, elo)

    out = predict_upcoming._champions_league_block(domestic, args, pd.Timestamp("2025-08-01"), 14)
    assert len(out) == 2
    rated = [r for r in out if not r["unrated"]]
    assert len(rated) == 1
    r = rated[0]
    assert r["league"] == "Champions League"
    assert r["competition"] == "CL"
    assert abs(r["p_home"] + r["p_draw"] + r["p_away"] - 1.0) < 1e-6
    assert r["elo_p_home"] > r["elo_p_away"]        # 1950 vs 1500
    assert r["bridge"] > 1.0
    assert out[1]["unrated"] is True


def test_block_runs_without_club_elo(domestic, args, monkeypatch):
    day = pd.Timestamp("2025-08-06 20:00")
    fixtures = pd.DataFrame([{
        "date": day, "home_team": "Alpha", "away_team": "Beta B",
        "match_id": "CL_1", "matchday": 1, "stage": "LEAGUE_STAGE"}])
    _stub(monkeypatch, fixtures, {})           # no Elo at all

    out = predict_upcoming._champions_league_block(domestic, args, pd.Timestamp("2025-08-01"), 14)
    assert len(out) == 1 and out[0]["unrated"] is False
    assert out[0]["bridge"] == 1.0
    assert "no Club Elo" in out[0]["adj_note"]
    assert "elo_p_home" not in out[0]


def test_block_skips_cleanly_when_fixtures_unavailable(domestic, args, monkeypatch):
    def boom(**k):
        raise champions_league.MissingToken("no token")

    monkeypatch.setattr(champions_league, "upcoming", boom)
    out = predict_upcoming._champions_league_block(domestic, args, pd.Timestamp("2025-08-01"), 14)
    assert out == []
