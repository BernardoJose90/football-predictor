"""Scoring rules and calibration.

RPS is primary (outcomes are ordinal: a home win is closer to a draw than to an
away win). Log loss is reported alongside it because Wheatcroft (2021) disputes
that distance-sensitivity belongs in a scoring rule and prefers the Ignorance
(log) score.

Outcomes are ordered (home, draw, away).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}
_ONEHOT = {
    "H": np.array([1.0, 0.0, 0.0]),
    "D": np.array([0.0, 1.0, 0.0]),
    "A": np.array([0.0, 0.0, 1.0]),
}


def _as_probs(p_home, p_draw, p_away) -> np.ndarray:
    p = np.asarray([p_home, p_draw, p_away], dtype=float)
    if np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError(f"invalid probabilities: {p}")
    s = p.sum()
    if s <= 0:
        raise ValueError("probabilities sum to zero")
    return p / s


def rps(p_home: float, p_draw: float, p_away: float, outcome: str) -> float:
    """Ranked Probability Score for one prediction. Lower is better, range [0, 1].

    RPS = (1/2) * sum_{i=1..2} ( (sum_{j<=i} p_j) - (sum_{j<=i} a_j) )^2
    """
    if outcome not in _OUTCOME_INDEX:
        raise ValueError(f"outcome must be H/D/A, got {outcome!r}")
    p = _as_probs(p_home, p_draw, p_away)
    a = _ONEHOT[outcome]
    cum_p = np.cumsum(p)
    cum_a = np.cumsum(a)
    return float(np.sum((cum_p[:-1] - cum_a[:-1]) ** 2) / 2.0)


def log_loss(p_home: float, p_draw: float, p_away: float, outcome: str,
             eps: float = 1e-15) -> float:
    """Negative log-likelihood of the observed outcome (a.k.a. Ignorance score)."""
    p = _as_probs(p_home, p_draw, p_away)
    return float(-np.log(np.clip(p[_OUTCOME_INDEX[outcome]], eps, 1.0)))


def rps_series(df: pd.DataFrame,
               cols=("p_home", "p_draw", "p_away"),
               outcome_col: str = "result") -> pd.Series:
    ph, pd_, pa = cols
    return df.apply(
        lambda r: rps(r[ph], r[pd_], r[pa], r[outcome_col]), axis=1
    )


def log_loss_series(df: pd.DataFrame,
                    cols=("p_home", "p_draw", "p_away"),
                    outcome_col: str = "result") -> pd.Series:
    ph, pd_, pa = cols
    return df.apply(
        lambda r: log_loss(r[ph], r[pd_], r[pa], r[outcome_col]), axis=1
    )


def summary(df: pd.DataFrame,
            cols=("p_home", "p_draw", "p_away"),
            outcome_col: str = "result") -> dict:
    """Mean RPS, mean log loss and n over a predictions frame."""
    scored = df.dropna(subset=list(cols) + [outcome_col])
    return {
        "n": int(len(scored)),
        "rps": float(rps_series(scored, cols, outcome_col).mean()),
        "log_loss": float(log_loss_series(scored, cols, outcome_col).mean()),
    }


def calibration_table(preds, outcomes, bins: int = 10) -> pd.DataFrame:
    """Predicted probability vs observed frequency, in equal-width bins.

    ``preds`` is a 1-D array of predicted probabilities for a single event
    (e.g. P(home win)); ``outcomes`` is the matching 0/1 array of whether it
    happened. This is the plot you show people (design doc section 12).
    """
    df = pd.DataFrame({"p": np.asarray(preds, dtype=float),
                       "hit": np.asarray(outcomes, dtype=float)})
    edges = np.linspace(0.0, 1.0, bins + 1)
    df["bin"] = pd.cut(df["p"], edges, include_lowest=True)
    out = df.groupby("bin", observed=True).agg(
        predicted=("p", "mean"),
        observed=("hit", "mean"),
        n=("p", "size"),
    ).reset_index()
    return out


def calibration_error(preds, outcomes, bins: int = 10) -> float:
    """Sample-size-weighted mean |predicted - observed| across bins (ECE)."""
    tbl = calibration_table(preds, outcomes, bins)
    if tbl["n"].sum() == 0:
        return float("nan")
    return float(np.average((tbl["predicted"] - tbl["observed"]).abs(), weights=tbl["n"]))
