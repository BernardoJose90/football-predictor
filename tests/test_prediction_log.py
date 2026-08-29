import pandas as pd

from evaluate import prediction_log


def _rec(match_id="E0_2526_20260830_arsenal_chelsea", **overrides):
    rec = {
        "league": "Premier League", "date": pd.Timestamp("2026-08-30 15:00"),
        "home_team": "Arsenal", "away_team": "Chelsea", "unrated": False,
        "match_id": match_id, "likely_score": "2-1",
        "p_home": 0.55, "p_draw": 0.25, "p_away": 0.20,
        "adj_note": "rest=6d/6d",
        "market_p_home": 0.5, "market_p_draw": 0.27, "market_p_away": 0.23,
    }
    rec.update(overrides)
    return rec


def test_load_missing_file_returns_empty_shaped_frame(tmp_path):
    df = prediction_log.load(tmp_path / "no_such_log.csv")
    assert list(df.columns) == prediction_log.COLUMNS
    assert df.empty


def test_append_writes_new_rows(tmp_path):
    path = tmp_path / "log.csv"
    n = prediction_log.append([_rec()], path=path, logged_at="2026-08-28T09:00:00Z")
    assert n == 1
    df = prediction_log.load(path)
    assert len(df) == 1
    assert df.loc[0, "match_id"] == "E0_2526_20260830_arsenal_chelsea"
    assert df.loc[0, "model_p_home"] == 0.55
    assert df.loc[0, "market_p_home"] == 0.5


def test_unrated_fixtures_are_never_logged(tmp_path):
    path = tmp_path / "log.csv"
    n = prediction_log.append([_rec(unrated=True, p_home=None)], path=path)
    assert n == 0
    assert prediction_log.load(path).empty


def test_first_prediction_wins_second_run_never_overwrites(tmp_path):
    # Thursday's run logs the fixture at 55% home...
    path = tmp_path / "log.csv"
    prediction_log.append([_rec(p_home=0.55)], path=path, logged_at="2026-08-28T09:00:00Z")
    # ...Saturday's run has a better-informed 60% for the same fixture, but
    # the anti-cherry-pick rule says the original number stands.
    n_second = prediction_log.append([_rec(p_home=0.60)], path=path, logged_at="2026-08-30T08:00:00Z")
    assert n_second == 0
    df = prediction_log.load(path)
    assert len(df) == 1
    assert df.loc[0, "model_p_home"] == 0.55


def test_different_fixtures_both_get_logged(tmp_path):
    path = tmp_path / "log.csv"
    prediction_log.append([_rec(match_id="m1", home_team="Arsenal", away_team="Chelsea")], path=path)
    n = prediction_log.append([_rec(match_id="m2", home_team="Liverpool", away_team="Everton")], path=path)
    assert n == 1
    assert len(prediction_log.load(path)) == 2
