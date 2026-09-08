"""Club Elo -> implied 1X2, the comparison price for a Champions League tie.

The domestic model is scored against the devigged bookmaker closing line. No
free closing line exists for the CL, so its stand-in is the forecast implied by
Club Elo (clubelo.com) - the "hard to beat free benchmark" the design doc
already names (see evaluate/baselines.py).

The maths is the logistic Elo expectation with a rating-gap-dependent draw
share, identical in form to ``evaluate.baselines.EloModel.probabilities`` - the
only differences are that the ratings come in from outside (clubelo, not a
self-fit) and the home-field advantage is a touch smaller, since CL home
advantage runs below a typical domestic league.
"""
from __future__ import annotations

import math

# clubelo.com quotes home advantage at roughly +65 Elo for domestic games;
# Champions League home advantage is measurably smaller, so 55 is the default
# here. Overridable, and the CL backtest will flag it if it is wrong.
CL_ELO_HFA = 55.0
_DRAW_BASE = 0.28
_DRAW_DECAY = 0.0009


def implied_1x2(elo_home: float, elo_away: float, *, hfa: float = CL_ELO_HFA) -> dict[str, float]:
    """``{p_home, p_draw, p_away}`` implied by two Club Elo ratings."""
    diff = float(elo_home) + hfa - float(elo_away)
    e_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))          # P(home not-loss), roughly
    p_draw = max(0.06, _DRAW_BASE * math.exp(-_DRAW_DECAY * abs(diff)))
    p_home = e_home * (1.0 - p_draw)
    p_away = (1.0 - e_home) * (1.0 - p_draw)
    s = p_home + p_draw + p_away
    return {"p_home": p_home / s, "p_draw": p_draw / s, "p_away": p_away / s}
