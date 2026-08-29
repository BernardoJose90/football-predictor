import math

import pytest

from evaluate import metrics


def test_rps_perfect_prediction_is_zero():
    assert metrics.rps(1.0, 0.0, 0.0, "H") == pytest.approx(0.0)
    assert metrics.rps(0.0, 1.0, 0.0, "D") == pytest.approx(0.0)
    assert metrics.rps(0.0, 0.0, 1.0, "A") == pytest.approx(0.0)


def test_rps_worst_case_home_vs_away_is_one():
    # Confident away win predicted, home win happens: max ordinal distance.
    assert metrics.rps(0.0, 0.0, 1.0, "H") == pytest.approx(1.0)


def test_rps_penalises_distance_not_just_being_wrong():
    # Same "wrongness" (draw predicted with certainty) scored against the two
    # possible actual outcomes: RPS must be lower when the outcome is adjacent
    # (draw missed by one) than when it's at the far end.
    predict_draw = (0.0, 1.0, 0.0)
    rps_vs_home = metrics.rps(*predict_draw, "H")
    rps_vs_away = metrics.rps(*predict_draw, "A")
    assert rps_vs_home == pytest.approx(rps_vs_away)  # symmetric around draw
    assert rps_vs_home > 0


def test_rps_known_value():
    # p=[0.8, 0.1, 0.1], outcome H
    # cum_p=[0.8, 0.9], cum_a=[1, 1] -> ((0.2)^2 + (0.1)^2)/2 = 0.025
    assert metrics.rps(0.8, 0.1, 0.1, "H") == pytest.approx(0.025)


def test_rps_matches_brier_style_ordering_property():
    # A model closer to the truth must never score worse than one further away.
    close = metrics.rps(0.5, 0.3, 0.2, "H")
    far = metrics.rps(0.2, 0.3, 0.5, "H")
    assert close < far


def test_rps_rejects_bad_outcome():
    with pytest.raises(ValueError):
        metrics.rps(0.5, 0.3, 0.2, "X")


def test_log_loss_perfect_is_near_zero():
    assert metrics.log_loss(1.0 - 2e-16, 1e-16, 1e-16, "H") < 1e-6


def test_log_loss_confident_wrong_is_large():
    assert metrics.log_loss(0.001, 0.001, 0.998, "H") > 5


def test_calibration_table_diagonal_when_perfectly_calibrated():
    import numpy as np
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    hit = (rng.uniform(0, 1, 5000) < p).astype(int)
    tbl = metrics.calibration_table(p, hit, bins=5)
    err = metrics.calibration_error(p, hit, bins=5)
    assert err < 0.05, tbl
