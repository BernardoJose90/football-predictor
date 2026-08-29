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
from evaluate.backtest import BacktestConfig, backtest, report


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
