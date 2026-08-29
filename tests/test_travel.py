import pytest

from model.travel import haversine_km, trip_distance_km, travel_factor
from model.stadiums import COORDS


def test_haversine_known_distance_london_to_paris():
    london = (51.5074, -0.1278)
    paris = (48.8566, 2.3522)
    d = haversine_km(london, paris)
    assert 340 < d < 350  # actual great-circle distance ~344km


def test_haversine_zero_for_same_point():
    p = (51.5, -0.1)
    assert haversine_km(p, p) == pytest.approx(0.0, abs=1e-9)


def test_trip_distance_same_city_derby_is_small():
    # Milan and Inter share a stadium/city in our table.
    d = trip_distance_km("Milan", "Inter")
    assert d == pytest.approx(0.0, abs=1.0)


def test_trip_distance_long_haul():
    # Real Madrid hosting Barcelona is a genuinely long trip within Spain.
    d = trip_distance_km("Real Madrid", "Barcelona")
    assert d > 400


def test_trip_distance_none_for_unknown_team():
    assert trip_distance_km("Real Madrid", "Definitely Not A Real Club") is None
    assert trip_distance_km("Definitely Not A Real Club", "Real Madrid") is None


def test_all_canonical_stadium_coords_are_valid_lat_lon():
    for team, (lat, lon) in COORDS.items():
        assert -90 <= lat <= 90, team
        assert -180 <= lon <= 180, team


def test_travel_factor_none_distance_is_neutral():
    assert travel_factor(None) == 1.0


def test_travel_factor_penalises_long_trip():
    assert travel_factor(2000, rest_days=6) < 1.0


def test_travel_factor_short_trip_barely_moves():
    short = travel_factor(50, rest_days=6)
    long = travel_factor(2000, rest_days=6)
    assert short > long
    assert short == pytest.approx(1.0, abs=0.01)


def test_travel_factor_amplified_by_short_rest():
    plenty_of_rest = travel_factor(1500, rest_days=6)
    no_rest = travel_factor(1500, rest_days=1)
    assert no_rest < plenty_of_rest  # same trip, worse when rest is short too


def test_travel_factor_floors_out():
    assert travel_factor(100_000, rest_days=0, k=0.01, floor=0.85) == 0.85


def test_travel_factor_nan_rest_treated_as_no_amplification():
    assert travel_factor(1000, rest_days=float("nan")) == travel_factor(1000, rest_days=None)
