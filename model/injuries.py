"""Premier-League-only injury/availability adjustment (ingest/fpl.py).

Live-only, same category as model/squad_value.py: there is no free
historical archive of pre-match availability to test against (see
ingest/fpl.py's docstring), so this cannot be walk-forward tested against
RPS the way referee/rest/travel/diagonal-inflation were - there is no
`evaluate.backtest` wiring for it, deliberately, same as squad value.
Default OFF (see scripts/predict_upcoming.py's --use-injuries flag) until
there's a real signal one way or the other. This is an experiment, not yet
a decision.
"""
from __future__ import annotations

import numpy as np


def injury_factor(
    availability: float | None,
    k: float = 0.3,
    floor: float = 0.6,
    ceiling: float = 1.0,
) -> float:
    """Multiplier on a team's own expected goals, from its FPL-derived
    availability fraction (1.0 = full-strength squad by points-weighting).

    availability=None (team not found - a non-Premier-League fixture, or too
    early in the season to weight by points yet) is a no-op, same pattern as
    an unknown referee or a team missing from the stadium-coordinates table.
    """
    if availability is None or (isinstance(availability, float) and np.isnan(availability)):
        return 1.0
    penalty = k * (1.0 - availability)
    return float(np.clip(1.0 - penalty, floor, ceiling))
