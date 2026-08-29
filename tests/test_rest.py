import pandas as pd

from model.rest import add_rest_days, rest_factor


def _matches(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_rest_days_computed_from_teams_previous_match():
    m = _matches([
        {"match_id": "m1", "date": "2024-08-01", "home_team": "Alpha", "away_team": "Beta"},
        {"match_id": "m2", "date": "2024-08-08", "home_team": "Beta", "away_team": "Alpha"},
    ])
    out = add_rest_days(m)
    row2 = out[out.match_id == "m2"].iloc[0]
    assert row2["home_rest_days"] == 7   # Beta: away in m1 (Aug 1) -> home in m2 (Aug 8)
    assert row2["away_rest_days"] == 7   # Alpha: same gap


def test_first_match_on_record_has_nan_rest_days():
    m = _matches([
        {"match_id": "m1", "date": "2024-08-01", "home_team": "Alpha", "away_team": "Beta"},
    ])
    out = add_rest_days(m)
    row = out.iloc[0]
    assert pd.isna(row["home_rest_days"])
    assert pd.isna(row["away_rest_days"])


def test_rest_days_crosses_divisions_for_the_same_team():
    # Alpha's previous match was in E1; this match is in E0 (promotion). The
    # gap must still be measured, not reset just because the division changed.
    m = _matches([
        {"match_id": "m1", "date": "2024-05-01", "home_team": "Alpha", "away_team": "Zeta", "div": "E1"},
        {"match_id": "m2", "date": "2024-08-15", "home_team": "Alpha", "away_team": "Gamma", "div": "E0"},
    ])
    out = add_rest_days(m)
    row2 = out[out.match_id == "m2"].iloc[0]
    assert row2["home_rest_days"] == (pd.Timestamp("2024-08-15") - pd.Timestamp("2024-05-01")).days


def test_rest_factor_neutral_at_reference():
    assert rest_factor(6, reference=6.0) == 1.0


def test_rest_factor_penalises_short_rest():
    assert rest_factor(2, k=0.02, reference=6.0) < 1.0


def test_rest_factor_rewards_long_rest_but_caps():
    assert rest_factor(30, k=0.02, reference=6.0, ceiling=1.05) == 1.05


def test_rest_factor_floors_out():
    assert rest_factor(0, k=0.05, reference=6.0, floor=0.85) == 0.85


def test_rest_factor_nan_is_neutral():
    assert rest_factor(float("nan")) == 1.0
    assert rest_factor(None) == 1.0
