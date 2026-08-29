import pandas as pd
import pytest

from model.ratings import build_ratings, time_weights


def test_time_weights_decay_and_bounds():
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-06-01", "2025-01-01"]))
    w = time_weights(dates, reference="2025-01-01", xi=0.0018)
    assert (w > 0).all() and (w <= 1).all()
    assert w[2] == pytest.approx(1.0)   # zero days ago
    assert w[0] < w[1] < w[2]           # older matches weighted less


def test_time_weights_future_dates_get_zero_weight():
    dates = pd.Series(pd.to_datetime(["2025-06-01"]))
    w = time_weights(dates, reference="2025-01-01", xi=0.0018)
    assert w[0] == 0.0


def test_leakage_guard_ignores_matches_on_or_after_as_of(synthetic_matches):
    as_of = pd.Timestamp("2025-01-01")

    before = synthetic_matches[synthetic_matches.date < as_of]
    snap_clean = build_ratings(before, as_of=as_of, stat="goals", min_matches=8)

    # Plant an absurd result *after* as_of for Alpha - if it leaked in,
    # Alpha's attack rating would move a lot.
    poisoned = synthetic_matches.copy()
    future_rows = poisoned[poisoned.date >= as_of].index
    poisoned.loc[future_rows, "home_goals_stat"] = 0
    poisoned.loc[future_rows, "away_goals_stat"] = 0
    # also poison a specific future match involving Alpha at home with an
    # extreme scoreline, to make any leak obvious
    future_alpha = poisoned[(poisoned.date >= as_of) & (poisoned.home_team == "Alpha")].index
    poisoned.loc[future_alpha, "home_goals_stat"] = 99

    snap_poisoned = build_ratings(poisoned, as_of=as_of, stat="goals", min_matches=8)

    assert snap_clean.attack("Alpha") == pytest.approx(snap_poisoned.attack("Alpha"))
    assert set(snap_clean.teams) == set(snap_poisoned.teams)
    for team in snap_clean.teams:
        assert snap_clean.attack(team) == pytest.approx(snap_poisoned.attack(team))
        assert snap_clean.defence(team) == pytest.approx(snap_poisoned.defence(team))


def test_min_matches_guard_excludes_sparse_teams(synthetic_matches):
    as_of = pd.Timestamp("2024-08-20")  # only a handful of matches exist yet
    snap = build_ratings(synthetic_matches, as_of=as_of, stat="goals", min_matches=8)
    early = synthetic_matches[synthetic_matches.date < as_of]
    counts = pd.concat([early.home_team, early.away_team]).value_counts()
    for team, n in counts.items():
        if n < 8:
            assert team not in snap.teams


def test_stronger_team_gets_higher_attack_and_lower_defence(synthetic_matches):
    as_of = pd.Timestamp("2025-06-01")
    snap = build_ratings(synthetic_matches, as_of=as_of, stat="goals", min_matches=8)
    assert snap.attack("Alpha") > snap.attack("Gamma")
    assert snap.defence("Alpha") < snap.defence("Gamma")


def test_unknown_stat_rejected(synthetic_matches):
    with pytest.raises(ValueError):
        build_ratings(synthetic_matches, as_of="2025-06-01", stat="nonsense")


def test_no_history_before_as_of_raises(synthetic_matches):
    with pytest.raises(ValueError):
        build_ratings(synthetic_matches, as_of="2000-01-01", stat="goals")


# ---- squad-value fallback (section 10.1 rank 4) ----------------------------

def _squad_values_for(synthetic_matches, as_of):
    """Made-up but internally consistent values: Alpha (known strongest)
    highest, Gamma (weakest) lowest - enough for fit_prior to find a real
    relationship, plus a brand-new team with zero matches on record."""
    return {"Alpha": 900_000_000, "Beta": 200_000_000,
           "Gamma": 60_000_000, "Delta": 150_000_000,
           "NewlyPromoted FC": 40_000_000}


def test_unrated_team_gets_a_prior_when_squad_value_available(synthetic_matches):
    as_of = pd.Timestamp("2025-06-01")
    values = _squad_values_for(synthetic_matches, as_of)
    snap = build_ratings(synthetic_matches, as_of=as_of, stat="goals",
                         min_matches=8, squad_values=values, value_prior_min_points=4)

    assert "NewlyPromoted FC" not in snap.teams          # no real history
    assert snap.has("NewlyPromoted FC")                  # but a prior is available
    assert snap.is_prior("NewlyPromoted FC")
    assert not snap.is_prior("Alpha")                    # Alpha has real history

    attack = snap.attack("NewlyPromoted FC")
    defence = snap.defence("NewlyPromoted FC")
    assert 0.3 <= attack <= 2.5
    assert 0.3 <= defence <= 2.5


def test_no_prior_without_squad_values(synthetic_matches):
    as_of = pd.Timestamp("2025-06-01")
    snap = build_ratings(synthetic_matches, as_of=as_of, stat="goals", min_matches=8)
    assert not snap.has("NewlyPromoted FC")


def test_prior_reflects_relative_squad_value(synthetic_matches):
    as_of = pd.Timestamp("2025-06-01")
    values = _squad_values_for(synthetic_matches, as_of)
    values["Rich Newcomer"] = 950_000_000   # even more valuable than Alpha
    values["Poor Newcomer"] = 10_000_000    # least valuable of anyone
    snap = build_ratings(synthetic_matches, as_of=as_of, stat="goals",
                         min_matches=8, squad_values=values, value_prior_min_points=4)

    assert snap.attack("Rich Newcomer") > snap.attack("Poor Newcomer")
    assert snap.defence("Rich Newcomer") < snap.defence("Poor Newcomer")


def test_rated_team_ignores_its_own_squad_value(synthetic_matches):
    # A team with real history uses its form-based rating even if it also
    # has a squad value on file - the prior is a fallback, never an override.
    as_of = pd.Timestamp("2025-06-01")
    values = _squad_values_for(synthetic_matches, as_of)
    without_values = build_ratings(synthetic_matches, as_of=as_of, stat="goals", min_matches=8)
    with_values = build_ratings(synthetic_matches, as_of=as_of, stat="goals",
                                min_matches=8, squad_values=values)
    assert without_values.attack("Alpha") == with_values.attack("Alpha")
