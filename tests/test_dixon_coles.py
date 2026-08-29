import numpy as np
import pytest

from model import dixon_coles as dc


def test_grid_sums_to_one():
    grid = dc.score_grid(1.6, 1.1, rho=0.1)
    assert grid.sum() == pytest.approx(1.0, abs=1e-9)


def test_grid_all_nonnegative():
    grid = dc.score_grid(1.6, 1.1, rho=0.15)
    assert (grid >= 0).all()


def test_outcome_probabilities_sum_to_one():
    grid = dc.score_grid(1.8, 0.9, rho=0.1)
    probs = dc.outcome_probabilities(grid)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)


def test_rho_zero_is_plain_independent_poisson():
    from scipy.stats import poisson
    lam, mu, max_goals = 1.4, 1.1, 10
    grid = dc.score_grid(lam, mu, rho=0.0, delta=0.0, max_goals=max_goals)
    h = poisson.pmf(np.arange(max_goals + 1), lam)
    a = poisson.pmf(np.arange(max_goals + 1), mu)
    expected = np.outer(h, a)
    expected /= expected.sum()
    np.testing.assert_allclose(grid, expected, atol=1e-10)


def test_tau_matches_dixon_coles_formula_on_four_cells():
    lam, mu, rho = 1.5, 1.2, 0.1
    grid = np.ones((3, 3))
    out = dc.dc_tau(grid.copy(), lam, mu, rho)
    assert out[0, 0] == pytest.approx(1 - lam * mu * rho)
    assert out[0, 1] == pytest.approx(1 + lam * rho)
    assert out[1, 0] == pytest.approx(1 + mu * rho)
    assert out[1, 1] == pytest.approx(1 - rho)
    # untouched elsewhere
    assert out[2, 2] == pytest.approx(1.0)
    assert out[0, 2] == pytest.approx(1.0)


def test_higher_lambda_favours_home_win():
    grid = dc.score_grid(2.2, 0.8, rho=0.1)
    probs = dc.outcome_probabilities(grid)
    assert probs["p_home"] > probs["p_away"]
    assert probs["p_home"] > probs["p_draw"]


def test_market_probabilities_consistent_with_grid():
    grid = dc.score_grid(1.6, 1.3, rho=0.1)
    m = dc.market_probabilities(grid)
    assert 0 <= m["p_over_2_5"] <= 1
    assert 0 <= m["p_btts"] <= 1


def test_negative_lambda_rejected():
    with pytest.raises(ValueError):
        dc.score_grid(-1.0, 1.0)


# ---- diagonal inflation (Karlis & Ntzoufras 2003; Egidi et al. 2026) -------

def test_diagonal_inflate_zero_delta_is_a_noop():
    grid = dc.score_grid(1.6, 1.1, rho=0.1, delta=0.0)
    inflated = dc.diagonal_inflate(grid, 0.0)
    np.testing.assert_allclose(inflated, grid)


def test_diagonal_inflate_sums_to_one():
    grid = dc.score_grid(1.6, 1.1, rho=0.1, delta=0.0)
    inflated = dc.diagonal_inflate(grid, 0.2)
    assert inflated.sum() == pytest.approx(1.0, abs=1e-9)


def test_diagonal_inflate_boosts_every_draw_cell_not_just_low_scores():
    grid = dc.score_grid(1.6, 1.6, rho=0.1, delta=0.0)
    inflated = dc.diagonal_inflate(grid, 0.5)
    # every diagonal cell's SHARE of the total should have grown, including
    # high-scoring draws (2-2, 3-3) that plain Dixon-Coles never touches -
    # that's the whole point versus dc_tau's 4-cell-only correction.
    for i in [2, 3]:
        assert inflated[i, i] / inflated.sum() > grid[i, i] / grid.sum()


def test_diagonal_inflate_raises_more_probability_of_a_draw_overall():
    grid = dc.score_grid(1.5, 1.3, rho=0.1, delta=0.0)
    before = dc.outcome_probabilities(grid)["p_draw"]
    inflated = dc.diagonal_inflate(grid, 0.3)
    after = dc.outcome_probabilities(inflated)["p_draw"]
    assert after > before


def test_diagonal_inflate_rejects_nonsquare_grid():
    with pytest.raises(ValueError):
        dc.diagonal_inflate(np.ones((3, 4)), 0.1)


def test_score_grid_delta_zero_matches_plain_dixon_coles():
    # delta=0.0 explicitly must reproduce dc_tau's output with no further
    # correction - score_grid's own *default* delta is config.DEFAULT_DELTA
    # (0.20, tuned - see config.py), same pattern as rho, so this test
    # reconstructs "plain Dixon-Coles" independently rather than relying on
    # score_grid's bare-default call, which is no longer delta=0.
    from scipy.stats import poisson
    lam, mu, rho, max_goals = 1.6, 1.1, 0.1, 10
    h = poisson.pmf(np.arange(max_goals + 1), lam)
    a_arr = poisson.pmf(np.arange(max_goals + 1), mu)
    expected = dc.dc_tau(np.outer(h, a_arr), lam, mu, rho)
    expected = np.clip(expected, 0.0, None)
    expected /= expected.sum()
    got = dc.score_grid(lam, mu, rho=rho, delta=0.0, max_goals=max_goals)
    np.testing.assert_allclose(got, expected)


def test_score_grid_with_delta_still_sums_to_one():
    grid = dc.score_grid(1.6, 1.1, rho=0.1, delta=0.25)
    assert grid.sum() == pytest.approx(1.0, abs=1e-9)
