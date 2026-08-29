import pytest

from model.injuries import injury_factor


def test_full_availability_is_a_noop():
    assert injury_factor(1.0) == 1.0


def test_none_availability_is_a_noop():
    assert injury_factor(None) == 1.0


def test_nan_availability_is_a_noop():
    assert injury_factor(float("nan")) == 1.0


def test_lower_availability_gives_lower_factor():
    full = injury_factor(1.0)
    half = injury_factor(0.5)
    hurt = injury_factor(0.2)
    assert full > half > hurt


def test_factor_floors_out():
    assert injury_factor(0.0, k=1.0, floor=0.6) == 0.6


def test_factor_never_exceeds_ceiling():
    assert injury_factor(1.0, ceiling=1.0) <= 1.0
