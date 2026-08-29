import pandas as pd
import pytest


@pytest.fixture
def synthetic_matches():
    """Small hand-built league: Alpha are clearly stronger than Beta and Gamma.

    Enough matches per team to clear the default min_matches=8 guard, spread
    over a year so time-weighting has something to bite on.
    """
    teams = ["Alpha", "Beta", "Gamma", "Delta"]
    rows = []
    start = pd.Timestamp("2024-08-01")
    rng_goals = {
        # (home, away) -> (home_goals, away_goals, home_sot, away_sot)
        "Alpha": dict(attack=2.2, defence=0.6),
        "Beta": dict(attack=1.1, defence=1.1),
        "Gamma": dict(attack=0.9, defence=1.3),
        "Delta": dict(attack=1.0, defence=1.0),
    }
    day = start
    match_no = 0
    for round_no in range(10):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                match_no += 1
                day = start + pd.Timedelta(days=match_no * 3)
                hg = max(0, round(rng_goals[home]["attack"] * rng_goals[away]["defence"]))
                ag = max(0, round(rng_goals[away]["attack"] * rng_goals[home]["defence"] * 0.8))
                result = "H" if hg > ag else "A" if ag > hg else "D"
                rows.append({
                    "match_id": f"m{match_no}",
                    "date": day,
                    "div": "T0",
                    "league": "Test League",
                    "season": "TEST",
                    "home_team": home,
                    "away_team": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "result": result,
                    "home_sot": hg + 3,
                    "away_sot": ag + 3,
                    "home_goals_stat": hg,
                    "away_goals_stat": ag,
                    "home_xg": float(hg) + 0.3,
                    "away_xg": float(ag) + 0.3,
                    "close_home": 1.5, "close_draw": 4.0, "close_away": 6.0,
                    "referee": "Ref A" if match_no % 2 == 0 else "Ref B",
                })
    return pd.DataFrame(rows)
