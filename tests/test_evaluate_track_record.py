import pandas as pd

from evaluate.track_record import join_results, score


def _log_row(match_id, home, away, mp_home=0.5, mp_draw=0.25, mp_away=0.25,
             mkt_home=None, mkt_draw=None, mkt_away=None, league="Premier League",
             kickoff="2026-08-30 15:00", adj_note=""):
    return {
        "logged_at": "2026-08-28T09:00:00Z", "match_id": match_id, "league": league,
        "kickoff": kickoff, "home_team": home, "away_team": away, "likely_score": "1-0",
        "model_p_home": mp_home, "model_p_draw": mp_draw, "model_p_away": mp_away,
        "market_p_home": mkt_home, "market_p_draw": mkt_draw, "market_p_away": mkt_away,
        "adj_note": adj_note,
    }


def _match_row(match_id, result, hg, ag):
    return {"match_id": match_id, "result": result, "home_goals": hg, "away_goals": ag}


def test_score_on_empty_log_returns_zeroed_report():
    rep = score(pd.DataFrame(columns=[
        "logged_at", "match_id", "league", "kickoff", "home_team", "away_team",
        "likely_score", "model_p_home", "model_p_draw", "model_p_away",
        "market_p_home", "market_p_draw", "market_p_away", "adj_note",
    ]), pd.DataFrame(columns=["match_id", "result", "home_goals", "away_goals"]))
    assert rep["n_logged"] == 0
    assert rep["n_scored"] == 0
    assert rep["model"] is None
    assert rep["fixtures"] == []


def test_unplayed_fixture_is_pending_not_scored():
    log = pd.DataFrame([_log_row("m1", "Arsenal", "Chelsea")])
    matches = pd.DataFrame(columns=["match_id", "result", "home_goals", "away_goals"])
    rep = score(log, matches)
    assert rep["n_logged"] == 1
    assert rep["n_scored"] == 0
    assert rep["n_pending"] == 1
    assert rep["model"] is None


def test_played_fixture_is_scored_with_correct_rps():
    log = pd.DataFrame([_log_row("m1", "Arsenal", "Chelsea", mp_home=0.6, mp_draw=0.25, mp_away=0.15)])
    matches = pd.DataFrame([_match_row("m1", "H", 2, 1)])
    rep = score(log, matches)
    assert rep["n_scored"] == 1
    assert rep["n_pending"] == 0
    assert rep["model"]["n"] == 1
    assert rep["model"]["hit_rate"] == 1.0  # model called H (highest prob) and H happened
    assert rep["fixtures"][0]["result"] == "H"
    assert rep["fixtures"][0]["model_called"] is True


def test_market_summary_only_covers_fixtures_with_market_odds():
    log = pd.DataFrame([
        _log_row("m1", "Arsenal", "Chelsea", mkt_home=0.55, mkt_draw=0.25, mkt_away=0.20),
        _log_row("m2", "Liverpool", "Everton"),  # no market odds logged
    ])
    matches = pd.DataFrame([_match_row("m1", "H", 2, 0), _match_row("m2", "D", 1, 1)])
    rep = score(log, matches)
    assert rep["n_scored"] == 2
    assert rep["n_with_market"] == 1
    assert rep["market"]["n"] == 1
    assert rep["model_on_market_set"]["n"] == 1  # model restricted to the same fixture


def test_cumulative_rps_is_in_kickoff_order():
    log = pd.DataFrame([
        _log_row("m1", "Arsenal", "Chelsea", kickoff="2026-08-30 15:00"),
        _log_row("m2", "Liverpool", "Everton", kickoff="2026-08-23 15:00"),
    ])
    matches = pd.DataFrame([_match_row("m1", "H", 1, 0), _match_row("m2", "D", 1, 1)])
    rep = score(log, matches)
    dates = [c["date"] for c in rep["cumulative"]]
    assert dates == sorted(dates)


def test_calibration_pools_all_three_outcomes():
    log = pd.DataFrame([_log_row("m1", "A", "B"), _log_row("m2", "C", "D", kickoff="2026-08-31 15:00")])
    matches = pd.DataFrame([_match_row("m1", "H", 2, 0), _match_row("m2", "A", 0, 2)])
    rep = score(log, matches)
    assert len(rep["calibration"]) > 0
    assert sum(b["n"] for b in rep["calibration"]) == 2 * 3  # 2 fixtures x 3 outcome rows each


def test_join_results_keeps_pending_rows_with_nan_result():
    log = pd.DataFrame([_log_row("m1", "Arsenal", "Chelsea")])
    matches = pd.DataFrame(columns=["match_id", "result", "home_goals", "away_goals"])
    joined = join_results(log, matches)
    assert len(joined) == 1
    assert pd.isna(joined.loc[0, "result"])
