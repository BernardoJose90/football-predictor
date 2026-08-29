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
    """Mean RPS, mean log loss and n over a predictions frame.

    Safe on a frame with zero scored rows (e.g. a baseline's warmup period
    consuming the whole window) - `.apply(axis=1)` on an empty DataFrame
    returns the frame itself rather than an empty Series, so a downstream
    `.mean()` would otherwise blow up on any non-numeric column instead of
    just reporting nothing to average.
    """
    scored = df.dropna(subset=list(cols) + [outcome_col])
    if scored.empty:
        return {"n": 0, "rps": float("nan"), "log_loss": float("nan")}
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


# --------------------------------------------------------------------------
# Extended evaluation battery (Gneiting et al. 2007 on calibration/sharpness;
# Murphy 1973 on the Brier decomposition; Diebold & Mariano 1995 on comparing
# two forecasters). All operate on the same (p_home, p_draw, p_away) + result
# frame the rest of this module uses.
# --------------------------------------------------------------------------

def _probs_matrix(df: pd.DataFrame, cols=("p_home", "p_draw", "p_away")) -> np.ndarray:
    p = df.loc[:, list(cols)].to_numpy(dtype=float)
    return p / p.sum(axis=1, keepdims=True)


def _onehot_matrix(outcomes: pd.Series) -> np.ndarray:
    return np.array([_ONEHOT[o] for o in outcomes], dtype=float)


def brier_multiclass(df: pd.DataFrame,
                     cols=("p_home", "p_draw", "p_away"),
                     outcome_col: str = "result") -> float:
    """Mean multi-class Brier score: mean over matches of sum_k (p_k - o_k)^2.

    Range [0, 2]; lower is better. A second strictly proper scoring rule to
    sit beside RPS and log loss (RPS is distance-sensitive, Brier and log
    loss are not - reporting all three is the standard hedge)."""
    scored = df.dropna(subset=list(cols) + [outcome_col])
    if scored.empty:
        return float("nan")
    p = _probs_matrix(scored, cols)
    o = _onehot_matrix(scored[outcome_col])
    return float(((p - o) ** 2).sum(axis=1).mean())


def brier_decomposition(df: pd.DataFrame,
                        cols=("p_home", "p_draw", "p_away"),
                        outcome_col: str = "result",
                        bins: int = 10) -> dict:
    """Murphy's reliability / resolution / uncertainty decomposition of the
    multi-class Brier score, summed over the three one-vs-rest problems.

        brier ~= reliability - resolution + uncertainty

    reliability  : calibration error (lower better; 0 = perfectly calibrated)
    resolution   : how much the forecasts separate high- from low-probability
                   events (higher better)
    uncertainty  : variance of the outcome itself - a property of the data,
                   not the forecaster, and the same for every model on this
                   sample (so it's the yardstick resolution is judged against)
    """
    scored = df.dropna(subset=list(cols) + [outcome_col])
    if scored.empty:
        return {"reliability": float("nan"), "resolution": float("nan"),
                "uncertainty": float("nan"), "brier": float("nan"), "n": 0}
    p = _probs_matrix(scored, cols)
    o = _onehot_matrix(scored[outcome_col])
    n = len(scored)
    edges = np.linspace(0.0, 1.0, bins + 1)

    reliability = resolution = uncertainty = 0.0
    for k in range(3):
        pk, ok = p[:, k], o[:, k]
        base = ok.mean()
        uncertainty += base * (1.0 - base)
        idx = np.clip(np.digitize(pk, edges[1:-1]), 0, bins - 1)
        for b in range(bins):
            m = idx == b
            nb = int(m.sum())
            if nb == 0:
                continue
            pbar, obar = pk[m].mean(), ok[m].mean()
            reliability += nb / n * (pbar - obar) ** 2
            resolution += nb / n * (obar - base) ** 2
    return {
        "reliability": round(float(reliability), 5),
        "resolution": round(float(resolution), 5),
        "uncertainty": round(float(uncertainty), 5),
        "brier": round(float(((p - o) ** 2).sum(axis=1).mean()), 5),
        "n": n,
    }


def sharpness(df: pd.DataFrame, cols=("p_home", "p_draw", "p_away")) -> dict:
    """How committed the forecasts are, independent of whether they're right.

    ``entropy`` is the mean Shannon entropy (nats) of the 1X2 vector - lower
    is sharper, 0 = always certain, ln(3)=1.0986 = always 33/33/33.
    ``mean_max_prob`` is the average probability put on the most likely
    outcome. Sharpness only counts as a virtue once calibration holds
    (Gneiting et al. 2007): a sharp but miscalibrated model is just
    confidently wrong."""
    p = _probs_matrix(df.loc[:, list(cols)].dropna(), cols)
    if len(p) == 0:
        return {"entropy": float("nan"), "mean_max_prob": float("nan"), "n": 0}
    ent = -(p * np.log(np.clip(p, 1e-15, 1.0))).sum(axis=1)
    return {
        "entropy": round(float(ent.mean()), 4),
        "mean_max_prob": round(float(p.max(axis=1).mean()), 4),
        "n": int(len(p)),
    }


def forecast_comparison(loss_a, loss_b, n_boot: int = 10000, seed: int = 0) -> dict:
    """Is forecaster A's per-match loss (RPS, say) really lower than B's?

    ``loss_a`` / ``loss_b`` are aligned per-match loss arrays for the SAME
    fixtures. Returns the mean difference (a - b; negative => A better), a
    Diebold-Mariano statistic (Diebold & Mariano 1995 - no autocovariance
    term, valid here because match forecasts are one-step and independent
    across fixtures), its two-sided p-value, and a paired bootstrap 95% CI on
    the mean difference (same non-parametric check the README already uses
    for the xg-vs-sot comparison)."""
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"loss arrays must align: {a.shape} vs {b.shape}")
    d = a - b
    n = len(d)
    if n < 2 or np.allclose(d, 0.0):
        return {"n": n, "mean_diff": float(d.mean()) if n else float("nan"),
                "dm_stat": 0.0, "p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0}

    from scipy import stats

    se = np.sqrt(d.var(ddof=1) / n)
    dm = float(d.mean() / se)
    p = float(2 * stats.t.sf(abs(dm), df=n - 1))

    rng = np.random.default_rng(seed)
    boot = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": n,
        "mean_diff": round(float(d.mean()), 5),
        "dm_stat": round(dm, 3),
        "p_value": round(p, 5),
        "ci_low": round(float(lo), 5),
        "ci_high": round(float(hi), 5),
    }
