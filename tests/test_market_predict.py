import numpy as np
import pytest

from model import dixon_coles
from model.market_predict import (
    devig_over_under,
    implied_goals,
    market_prediction,
    market_prediction_from_odds,
)


@pytest.mark.parametrize("lam,mu", [(1.6, 1.1), (2.3, 0.7), (0.9, 1.8), (1.3, 1.3)])
def test_implied_goals_round_trips_a_known_grid(lam, mu):
    # A grid built from (lam, mu) has 1X2 probabilities that implied_goals()
    # must invert back to (lam, mu) - with rho/delta fixed this is exact.
    grid = dixon_coles.score_grid(lam, mu)
    oc = dixon_coles.outcome_probabilities(grid)
    lam_hat, mu_hat, sol = implied_goals(oc["p_home"], oc["p_draw"], oc["p_away"])
    assert lam_hat == pytest.approx(lam, abs=1e-3)
    assert mu_hat == pytest.approx(mu, abs=1e-3)
    assert np.sqrt(np.mean(np.square(sol.fun))) < 1e-6


def test_market_prediction_publishes_the_devigged_1x2_verbatim():
    pred = market_prediction("H", "A", 0.55, 0.25, 0.20)
    assert (pred["p_home"], pred["p_draw"], pred["p_away"]) == (0.55, 0.25, 0.20)
    # and the grid it fitted reproduces them closely
    assert pred["grid_p_home"] == pytest.approx(0.55, abs=5e-3)
    assert pred["fit_residual"] < 5e-3


def test_market_prediction_has_the_same_shape_as_predict_match():
    pred = market_prediction("H", "A", 0.5, 0.27, 0.23, p_over_2_5=0.55)
    for key in ("home_team", "away_team", "home_pred", "away_pred", "likely_score",
                "p_home", "p_draw", "p_away", "p_over_2_5", "p_btts"):
        assert key in pred
    h, a = (int(x) for x in pred["likely_score"].split("-"))
    assert 0 <= h <= 10 and 0 <= a <= 10


def test_stronger_favourite_gets_the_higher_expected_goals():
    home_fav = market_prediction("H", "A", 0.70, 0.20, 0.10)
    away_fav = market_prediction("H", "A", 0.15, 0.22, 0.63)
    assert home_fav["home_pred"] > home_fav["away_pred"]
    assert away_fav["away_pred"] > away_fav["home_pred"]


def test_over_under_line_moves_the_total_not_the_1x2():
    low = market_prediction("H", "A", 0.45, 0.27, 0.28, p_over_2_5=0.35)
    high = market_prediction("H", "A", 0.45, 0.27, 0.28, p_over_2_5=0.65)
    assert high["home_pred"] + high["away_pred"] > low["home_pred"] + low["away_pred"]
    assert high["p_over_2_5"] > low["p_over_2_5"]
    # the who-wins number is the market's regardless of the total
    assert low["p_home"] == high["p_home"] == 0.45


def test_devig_over_under_strips_the_margin():
    # a book with a ~5% overround on a pick'em total
    p_over = devig_over_under(1.90, 1.90)
    assert p_over == pytest.approx(0.5, abs=1e-9)
    assert 0.0 < devig_over_under(1.5, 2.6) < 1.0


def test_from_odds_matches_manual_devig():
    from evaluate.baselines import devig
    p = devig(2.0, 3.5, 4.0)
    a = market_prediction_from_odds("H", "A", 2.0, 3.5, 4.0)
    b = market_prediction("H", "A", p["p_home"], p["p_draw"], p["p_away"])
    assert a["likely_score"] == b["likely_score"]
    assert a["p_home"] == pytest.approx(b["p_home"])


def test_bad_over_under_odds_are_ignored_not_fatal():
    pred = market_prediction_from_odds("H", "A", 2.0, 3.5, 4.0,
                                       odds_over="n/a", odds_under=None)
    assert pred["p_home"] > 0
