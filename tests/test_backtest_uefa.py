import pandas as pd
import pytest

from evaluate.backtest import BacktestConfig, backtest_uefa, report


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
def cl_results():
    day = pd.Timestamp("2025-08-06")
    rows = [
        ("Alpha", "Gamma B", 2, 0),
        ("Beta B", "Delta", 1, 1),
        ("Gamma", "Alpha B", 0, 3),
    ]
    return pd.DataFrame([
        {"match_id": f"CL_2526_20250806_{i}", "date": day, "season": "2526",
         "matchday": 1, "stage": "LEAGUE_STAGE", "home_team": h, "away_team": a,
         "home_goals": hg, "away_goals": ag,
         "result": "H" if hg > ag else "A" if ag > hg else "D"}
        for i, (h, a, hg, ag) in enumerate(rows)
    ])


@pytest.fixture
def elo_lookup():
    return {pd.Timestamp("2025-08-06"): {
        "Alpha": 1950, "Beta": 1750, "Gamma": 1650, "Delta": 1700,
        "Alpha B": 1600, "Beta B": 1550, "Gamma B": 1500, "Delta B": 1520,
    }}


def test_backtest_uefa_scores_every_rated_tie(domestic, cl_results, elo_lookup):
    cfg = BacktestConfig(stat="goals", min_matches=8, cl_gamma=0.6)
    preds = backtest_uefa(cl_results, domestic, "2025-01-01", None, cfg, elo_lookup=elo_lookup)
    assert len(preds) == 3
    assert (~preds["unrated"]).all()
    assert preds["p_home"].between(0, 1).all()
    assert preds["elo_p_home"].notna().all()
    assert preds.loc[0, "home_div"] == "L1"
    assert preds.loc[0, "away_div"] == "L2"

    rep = report(preds)
    assert rep["coverage"] == 1.0
    assert rep["rated"] == 3


def test_unknown_club_makes_the_tie_unrated(domestic, elo_lookup):
    day = pd.Timestamp("2025-08-06")
    cl = pd.DataFrame([{
        "match_id": "CL_x", "date": day, "season": "2526", "matchday": 1,
        "stage": "LEAGUE_STAGE", "home_team": "Alpha", "away_team": "Nowhere United",
        "home_goals": 1, "away_goals": 0, "result": "H",
    }])
    cfg = BacktestConfig(stat="goals", min_matches=8)
    preds = backtest_uefa(cl, domestic, "2025-01-01", None, cfg, elo_lookup=elo_lookup)
    assert len(preds) == 1
    assert bool(preds.loc[0, "unrated"]) is True
    assert pd.isna(preds.loc[0, "p_home"])


def test_gamma_moves_the_forecast(domestic, cl_results, elo_lookup):
    flat = backtest_uefa(cl_results, domestic, "2025-01-01", None,
                         BacktestConfig(stat="goals", min_matches=8, cl_gamma=0.0),
                         elo_lookup=elo_lookup)
    bridged = backtest_uefa(cl_results, domestic, "2025-01-01", None,
                            BacktestConfig(stat="goals", min_matches=8, cl_gamma=1.0),
                            elo_lookup=elo_lookup)
    # Alpha (1950) at home to Gamma B (1500): a positive gamma should lift P(home).
    assert bridged.loc[0, "p_home"] > flat.loc[0, "p_home"]
    assert (flat["bridge"] == 1.0).all()
