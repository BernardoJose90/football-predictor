"""xi sweep: RPS vs the time-decay parameter, and the input-stat comparison.

Produces the interview artefact (design doc section 12): a U-shaped curve of
RPS against xi with the selected minimum marked.

Tuning must run on a *separate* window from the one you report on. This module
only sweeps and plots; run_backtest.py holds the final untouched evaluation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
from evaluate import metrics
from evaluate.backtest import BacktestConfig, backtest, backtest_uefa, report


def sweep_xi(
    matches: pd.DataFrame,
    start,
    end,
    xis=None,
    stat: str = config.DEFAULT_STAT,
    rho: float = config.DEFAULT_RHO,
) -> pd.DataFrame:
    if xis is None:
        xis = np.round(np.linspace(0.0, 0.010, 21), 5)
    rows = []
    for xi in xis:
        cfg = BacktestConfig(stat=stat, xi=float(xi), rho=rho)
        preds = backtest(matches, start=start, end=end, cfg=cfg)
        rep = report(preds)
        rep["xi"] = float(xi)
        rows.append(rep)
        print(f"  xi={xi:.5f}  rps={rep.get('rps', float('nan')):.4f}  n={rep.get('rated', 0)}")
    return pd.DataFrame(rows).sort_values("xi").reset_index(drop=True)


def sweep_stat(matches: pd.DataFrame, start, end,
               stats=("xg", "sot", "goals"), xi: float = config.DEFAULT_XI) -> pd.DataFrame:
    rows = []
    for stat in stats:
        cfg = BacktestConfig(stat=stat, xi=xi)
        try:
            preds = backtest(matches, start=start, end=end, cfg=cfg)
            rep = report(preds)
        except ValueError as exc:
            print(f"  stat={stat}: skipped ({exc})")
            continue
        if rep["rated"] == 0:
            print(f"  stat={stat}: skipped (no rated matches - "
                  f"data not populated for this stat in the window)")
            continue
        rep["stat"] = stat
        rows.append(rep)
        print(f"  stat={stat:5s}  rps={rep.get('rps', float('nan')):.4f}  n={rep.get('rated', 0)}")
    return pd.DataFrame(rows)


def _elo_baseline_rps(preds: pd.DataFrame) -> dict:
    """RPS of the Club-Elo-implied 1X2 on the fixtures the model rated - the
    thing the CL model has to beat to be worth the extra machinery."""
    rated = preds[~preds["unrated"] & preds["p_home"].notna()]
    cols = {"elo_p_home": "p_home", "elo_p_draw": "p_draw", "elo_p_away": "p_away"}
    base = rated.dropna(subset=list(cols))[list(cols) + ["result"]].rename(columns=cols)
    return metrics.summary(base)


def sweep_cl_gamma(
    cl_results: pd.DataFrame,
    domestic_matches: pd.DataFrame,
    start,
    end,
    gammas=None,
    *,
    elo_lookup=None,
    stat: str = config.DEFAULT_STAT,
    xi: float = config.DEFAULT_XI,
) -> pd.DataFrame:
    """RPS vs the Champions League Elo-bridge exponent gamma, walk-forward.

    gamma=0 is "domestic form only"; the row also carries the Club Elo baseline
    RPS on the same rated fixtures so the sweep shows whether *any* gamma beats
    just using Club Elo directly.
    """
    if gammas is None:
        gammas = np.round(np.linspace(0.0, 2.0, 11), 3)
    rows = []
    for g in gammas:
        cfg = BacktestConfig(stat=stat, xi=xi, cl_gamma=float(g))
        preds = backtest_uefa(cl_results, domestic_matches, start, end, cfg,
                              elo_lookup=elo_lookup)
        rep = report(preds)
        rep["cl_gamma"] = float(g)
        rep["elo_baseline_rps"] = _elo_baseline_rps(preds).get("rps", float("nan"))
        rows.append(rep)
        print(f"  gamma={g:.3f}  rps={rep.get('rps', float('nan')):.4f}  "
              f"(elo {rep['elo_baseline_rps']:.4f})  n={rep.get('rated', 0)}")
    return pd.DataFrame(rows).sort_values("cl_gamma").reset_index(drop=True)


def plot_cl_gamma_curve(curve: pd.DataFrame, out: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out) if out else config.ARTEFACTS / "cl_gamma_sweep.png"
    best = curve.loc[curve["rps"].idxmin()]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve["cl_gamma"], curve["rps"], marker="o", lw=1.5, label="model")
    if curve["elo_baseline_rps"].notna().any():
        ax.axhline(curve["elo_baseline_rps"].dropna().iloc[0], color="grey", ls=":",
                   lw=1, label="Club Elo implied")
    ax.axvline(best["cl_gamma"], color="crimson", ls="--", lw=1)
    ax.scatter([best["cl_gamma"]], [best["rps"]], color="crimson", zorder=5)
    ax.annotate(f"min RPS {best['rps']:.4f}\nat gamma={best['cl_gamma']:.3f}",
                xy=(best["cl_gamma"], best["rps"]),
                xytext=(0.55, 0.75), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="crimson"))
    ax.set_xlabel("Club Elo bridge exponent  gamma")
    ax.set_ylabel("mean RPS  (lower is better)")
    ax.set_title("Champions League gamma sweep - walk-forward tuning window")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_xi_curve(curve: pd.DataFrame, out: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out) if out else config.ARTEFACTS / "xi_sweep.png"
    best = curve.loc[curve["rps"].idxmin()]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve["xi"], curve["rps"], marker="o", lw=1.5)
    ax.axvline(best["xi"], color="crimson", ls="--", lw=1)
    ax.scatter([best["xi"]], [best["rps"]], color="crimson", zorder=5)
    ax.annotate(
        f"min RPS {best['rps']:.4f}\nat xi={best['xi']:.5f}",
        xy=(best["xi"], best["rps"]),
        xytext=(0.55, 0.75), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="crimson"),
    )
    ax.set_xlabel("time-decay xi  (per day)")
    ax.set_ylabel("mean RPS  (lower is better)")
    ax.set_title("xi sweep - RPS vs time-decay, walk-forward tuning window")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
