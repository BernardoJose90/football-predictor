"""Milestone 4: sweep xi and compare input stats on the TUNING window only.

    python -m scripts.run_tune --tune-start 2024-08-01 --tune-end 2025-06-30

Writes artefacts/xi_sweep.png and artefacts/xi_sweep.csv. Keep the tuning
window strictly earlier than run_backtest.py's --eval-start.
"""
from __future__ import annotations

import argparse

import pandas as pd

import config
from evaluate import tune
from scripts.run_backtest import _load_matches


def _tune_cl(args) -> int:
    from ingest import champions_league

    domestic = _load_matches(coupon_only=False)
    try:
        cl_results = champions_league.results()
    except Exception as exc:  # noqa: BLE001 - missing token / no mirror
        print(f"no CL results available ({exc}) - run "
              f"`python -m ingest.champions_league --refresh` with FOOTBALL_DATA_ORG_TOKEN set")
        return 1
    if cl_results.empty:
        print("CL mirror has no played matches yet - run `python -m ingest.champions_league --refresh`")
        return 1
    start, end = pd.Timestamp(args.tune_start), pd.Timestamp(args.tune_end)
    print(f"\nCL gamma sweep  ({start.date()} .. {end.date()}, stat={args.stat})")
    curve = tune.sweep_cl_gamma(cl_results, domestic, start=start, end=end, stat=args.stat)
    curve.to_csv(config.ARTEFACTS / "cl_gamma_sweep.csv", index=False)
    png = tune.plot_cl_gamma_curve(curve)
    best = curve.loc[curve["rps"].idxmin()]
    print(f"\nbest gamma = {best['cl_gamma']:.3f}  (RPS {best['rps']:.4f}, "
          f"Club Elo baseline {best['elo_baseline_rps']:.4f})")
    print(f"curve      -> {png}")
    print("Set config.DEFAULT_CL_GAMMA to this value if it beats the baseline.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tune-start", default="2024-08-01")
    ap.add_argument("--tune-end", default="2025-06-30")
    ap.add_argument("--stat", choices=config.STAT_CHOICES, default=config.DEFAULT_STAT)
    ap.add_argument("--skip-stat-sweep", action="store_true")
    ap.add_argument("--competition", choices=("domestic", "cl"), default="domestic",
                    help="'cl' sweeps the Champions League Elo-bridge exponent gamma instead")
    args = ap.parse_args(argv)

    if args.competition == "cl":
        return _tune_cl(args)

    matches = _load_matches()
    start, end = pd.Timestamp(args.tune_start), pd.Timestamp(args.tune_end)

    print(f"\nxi sweep  ({start.date()} .. {end.date()}, stat={args.stat})")
    curve = tune.sweep_xi(matches, start=start, end=end, stat=args.stat)
    curve.to_csv(config.ARTEFACTS / "xi_sweep.csv", index=False)
    png = tune.plot_xi_curve(curve)
    best = curve.loc[curve["rps"].idxmin()]
    print(f"\nbest xi = {best['xi']:.5f}  (RPS {best['rps']:.4f})")
    print(f"curve   -> {png}")

    if not args.skip_stat_sweep:
        print(f"\ninput-stat comparison  (xi={best['xi']:.5f})")
        stat_tbl = tune.sweep_stat(matches, start=start, end=end, xi=float(best["xi"]))
        if not stat_tbl.empty:
            stat_tbl.to_csv(config.ARTEFACTS / "stat_sweep.csv", index=False)
            print(stat_tbl[["stat", "rps", "log_loss", "rated"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
