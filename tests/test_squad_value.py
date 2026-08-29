import pytest

from model.squad_value import fit_prior, predict_prior


def _synthetic_teams_and_values():
    # attack/defence roughly tracks log(value) by construction, so the fit
    # should recover a clearly positive attack slope and negative defence
    # slope (higher value -> better defence -> LOWER defence number, since
    # lower is better in this project's convention).
    teams = {
        "Big":    {"attack": 1.8, "defence": 0.6, "matches": 20},
        "Medium": {"attack": 1.1, "defence": 1.0, "matches": 20},
        "Small":  {"attack": 0.7, "defence": 1.3, "matches": 20},
        "Tiny":   {"attack": 0.5, "defence": 1.5, "matches": 20},
        "Huge":   {"attack": 2.0, "defence": 0.5, "matches": 20},
    }
    values = {
        "Big": 500_000_000, "Medium": 100_000_000, "Small": 20_000_000,
        "Tiny": 5_000_000, "Huge": 900_000_000,
    }
    return teams, values


def test_fit_prior_returns_none_with_too_few_points():
    teams, values = _synthetic_teams_and_values()
    small_teams = dict(list(teams.items())[:2])
    small_values = {k: values[k] for k in small_teams}
    assert fit_prior(small_teams, small_values, min_points=5) is None


def test_fit_prior_succeeds_with_enough_points():
    teams, values = _synthetic_teams_and_values()
    coeffs = fit_prior(teams, values, min_points=5)
    assert coeffs is not None
    assert coeffs["n"] == 5


def test_fit_prior_ignores_teams_without_a_value():
    teams, values = _synthetic_teams_and_values()
    values = {k: v for k, v in values.items() if k != "Tiny"}  # one team has no value
    coeffs = fit_prior(teams, values, min_points=4)
    assert coeffs["n"] == 4


def test_predict_prior_higher_value_means_more_attack_less_defence():
    teams, values = _synthetic_teams_and_values()
    coeffs = fit_prior(teams, values)
    small_attack, small_defence = predict_prior(10_000_000, coeffs)
    big_attack, big_defence = predict_prior(700_000_000, coeffs)
    assert big_attack > small_attack
    assert big_defence < small_defence  # lower defence number = better


def test_predict_prior_respects_bounds():
    teams, values = _synthetic_teams_and_values()
    coeffs = fit_prior(teams, values)
    # absurdly large value shouldn't blow past the sanity bounds
    attack, defence = predict_prior(1e15, coeffs, attack_bounds=(0.3, 2.5), defence_bounds=(0.3, 2.5))
    assert 0.3 <= attack <= 2.5
    assert 0.3 <= defence <= 2.5
