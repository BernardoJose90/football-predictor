"""Away-team travel distance as a fatigue adjustment (design doc section
10.1, rank 3).

An XGBoost study on European club fixtures ranked travel distance above rest
days for outcome importance, with the explicit finding that distance matters
most *combined with* short rest, not alone - so this doesn't apply a flat
per-km penalty. It scales the penalty by how little rest the travelling team
had, using the same short-rest reference point as model/rest.py.

Pure geometry from model/stadiums.py - no as_of parameter, no leakage risk by
construction, same as rest days.
"""
from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

import numpy as np

from model.stadiums import coords

EARTH_RADIUS_KM = 6371.0


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(h))


def trip_distance_km(home_team: str, away_team: str) -> float | None:
    """Distance the away side travels for this fixture, or None if either
    team's city is missing from model/stadiums.py."""
    h, a = coords(home_team), coords(away_team)
    if h is None or a is None:
        return None
    return haversine_km(h, a)


def travel_factor(
    distance_km: float | None,
    rest_days: float | None = None,
    k: float = 0.00006,
    short_rest_days: float = 5.0,
    floor: float = 0.85,
    ceiling: float = 1.02,
) -> float:
    """Multiplier on the AWAY team's expected goals for a trip of this length.

    Only the away side travels, so this is never applied to the home side.
    Distance alone applies a small flat penalty; when rest_days is also given
    and is under ``short_rest_days``, the penalty is amplified in proportion
    to how much rest was missed - the interaction the source study found,
    rather than treating distance as independent of rest.

    None distance (city not in the lookup table) is a no-op, same pattern as
    an unknown referee: this feature can't be silently guessed at.
    """
    if distance_km is None or (isinstance(distance_km, float) and np.isnan(distance_km)):
        return 1.0

    urgency = 1.0
    if rest_days is not None and not (isinstance(rest_days, float) and np.isnan(rest_days)):
        urgency = 1.0 + max(0.0, short_rest_days - rest_days)

    penalty = k * distance_km * urgency
    return float(np.clip(1.0 - penalty, floor, ceiling))
