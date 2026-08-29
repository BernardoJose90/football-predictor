"""Milestone 2 + 4: baselines, then the model, on the untouched evaluation window.

    python -m scripts.run_backtest --eval-start 2025-08-01 [--stat sot] [--xi 0.0018]

Prints:
  * devigged closing-line RPS   (baseline 1)
  * Elo RPS                     (baseline 2)
  * model RPS / log loss / coverage / calibration error
  * whether the model beats Elo and lands within 0.01 of the closing line

Writes artefacts/calibration.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
from evaluate import baselines, metrics
from evaluate.backtest import BacktestConfig, backtest, report


def _load_matches() -> pd.DataFrame:
    pq = config.DATA_PROCESSED / "matches.parquet"
    csv = config.DATA_PROCESSED / "matches.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise SystemExit("no dataset - run `python -m scripts.build_dataset --download` first")
    df["date"] = pd.to_datetime(df["date"])
    return df


def _plot_calibration(preds: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect")
    for outcome, col, colour in [("H", "p_home", "tab:blue"),
                                 ("D", "p_draw", "tab:orange"),
                                 ("A", "p_away", "tab:green")]:
        tbl = metrics.calibration_table(preds[col], (preds["result"] == outcome).astype(int), bins=10)
        ax.plot(tbl["predicted"], tbl["observed"], marker="o", color=colour, label=outcome)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration - walk-forward evaluation window")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-start", default="2025-08-01", help="first date in the report window")
    ap.add_argument("--eval-end", default=None)
    ap.add_argument("--stat", choices=config.STAT_CHOICES, default=config.DEFAULT_STAT)
    ap.add_argument("--xi", type=float, default=config.DEFAULT_XI)
    ap.add_argument("--rho", type=float, default=config.DEFAULT_RHO)
    ap.add_argument("--delta", type=float, default=config.DEFAULT_DELTA,
                    help="diagonal-inflation strength on the Dixon-Coles grid (0 = off)")
    args = ap.parse_args(argv)

    matches = _load_matches()
    start = pd.Timestamp(args.eval_start)
    end = pd.Timestamp(args.eval_end) if args.eval_end else None

    window = matches[matches["date"] >= start]
    if end is not None:
        window = window[window["date"] < end]

    # ---- baseline 1: devigged closing line -------------------------------
    dv = baselines.devig_frame(window)
    dv_scored = dv.dropna(subset=["p_home", "p_draw", "p_away"])
    dv_summary = metrics.summary(dv_scored)

    # ---- baseline 2: Elo (fit on full history, scored on the window) -----
    elo = baselines.elo_walk_forward(matches)
    elo_window = elo[elo["date"] >= start]
    if end is not None:
        elo_window = elo_window[elo_window["date"] < end]
    elo_scored = elo_window.dropna(subset=["p_home", "p_draw", "p_away"])
    elo_summary = metrics.summary(elo_scored)

    # ---- model ----------------------------------------------------------
    cfg = BacktestConfig(stat=args.stat, xi=args.xi, rho=args.rho, delta=args.delta)
    preds = backtest(matches, start=start, end=end, cfg=cfg)
    rated = preds[~preds["unrated"] & preds["p_home"].notna()]
    rep = report(preds)

    # align: score all three on the SAME fixtures the model rated
    common_ids = set(rated["match_id"])
    dv_common = metrics.summary(dv_scored[dv_scored["match_id"].isin(common_ids)])
    elo_common = metrics.summary(elo_scored[elo_scored["match_id"].isin(common_ids)])

    out = config.ARTEFACTS / "calibration.png"
    if len(rated):
        _plot_calibration(rated, out)

    print("\n==================  EVALUATION  ==================")
    print(f"window            : {start.date()} .. "
          f"{(end.date() if end is not None else matches['date'].max().date())}")
    print(f"stat / xi / rho / delta : {args.stat} / {args.xi} / {args.rho} / {args.delta}")
    print(f"model coverage    : {rep['rated']}/{rep['matches']}  ({rep['coverage']:.1%})")
    print("-------------------------------------------------")
    print(f"{'':22s}{'RPS':>9s}{'log loss':>11s}{'n':>8s}")
    print(f"{'devig closing line':22s}{dv_common['rps']:>9.4f}{dv_common['log_loss']:>11.4f}{dv_common['n']:>8d}")
    print(f"{'Club Elo (proxy)':22s}{elo_common['rps']:>9.4f}{elo_common['log_loss']:>11.4f}{elo_common['n']:>8d}")
    print(f"{'this model':22s}{rep['rps']:>9.4f}{rep['log_loss']:>11.4f}{rep['rated']:>8d}")
    print("-------------------------------------------------")
    print(f"full-sample devig RPS : {dv_summary['rps']:.4f}  (n={dv_summary['n']})")
    print(f"full-sample Elo   RPS : {elo_summary['rps']:.4f}  (n={elo_summary['n']})")
    print(f"calibration error (P home win): {rep.get('calibration_error_home', float('nan')):.4f}")
    print("-------------------------------------------------")
    beats_elo = rep["rps"] < elo_common["rps"]
    near_line = abs(rep["rps"] - dv_common["rps"]) <= 0.01
    under_021 = rep["rps"] < 0.21
    print(f"RPS < 0.21                 : {'YES' if under_021 else 'no'}  ({rep['rps']:.4f})")
    print(f"beats Elo on same fixtures : {'YES' if beats_elo else 'no'}")
    print(f"within 0.01 of closing line: {'YES' if near_line else 'no'}  "
          f"(gap {rep['rps'] - dv_common['rps']:+.4f})")
    if len(rated):
        print(f"\ncalibration plot -> {out}")
    print("=================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
