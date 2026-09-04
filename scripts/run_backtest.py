"""Milestone 2 + 4: baselines, then the model, on the untouched evaluation window.

    python -m scripts.run_backtest --eval-start 2025-08-01 [--stat sot] [--xi 0.0018]

Prints:
  * devigged closing-line RPS   (baseline 1)
  * Elo RPS                     (baseline 2)
  * model RPS / log loss / coverage / calibration error
  * whether the model beats Elo and lands within 0.01 of the closing line
  * extended battery: multiclass Brier + Murphy reliability/resolution/
    uncertainty decomposition, sharpness (predictive entropy), a paired
    Diebold-Mariano test of the model-vs-market and model-vs-Elo RPS gaps
    (are they real or noise?), and per-season RPS

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

    # ---- extended battery ---------------------------------------------------
    if len(rated) >= 2:
        _print_extended(rated, dv_scored, elo_scored)

    # ---- market scoreline reconstruction ----------------------------------
    # The market's 1X2 RPS is already the "devig closing line" row above. This
    # block scores the *scoreline grid* fitted to that price (+ the closing
    # Over/Under 2.5 line): correct-score hit rate and Over/Under 2.5 Brier,
    # next to the model's, on the same fixtures the model rated.
    if len(rated):
        _print_market_grid(window, rated)

    print("=================================================\n")
    return 0


def _score_hit_rate(df: pd.DataFrame) -> float:
    got = (df["home_goals"].astype(int).astype(str) + "-"
           + df["away_goals"].astype(int).astype(str))
    return float((df["likely_score"].astype(str) == got).mean())


def _ou_brier(df: pd.DataFrame) -> float:
    actual_over = ((df["home_goals"] + df["away_goals"]) > 2.5).astype(float)
    return float(((df["p_over_2_5"] - actual_over) ** 2).mean())


def _print_market_grid(window: pd.DataFrame, rated: pd.DataFrame) -> None:
    from evaluate import baselines

    if not {"close_over_2_5", "close_under_2_5"}.issubset(window.columns):
        print("\n----------  market scoreline grid  -------------")
        print("  dataset has no closing Over/Under columns - rebuild with "
              "`python -m scripts.build_dataset` to enable this scorecard")
        return

    grid = baselines.market_grid_frame(window)
    grid = grid[grid["p_home"].notna() & grid["match_id"].isin(set(rated["match_id"]))]
    m = rated[rated["p_over_2_5"].notna()]
    m = m[m["match_id"].isin(set(grid["match_id"]))]
    grid = grid[grid["match_id"].isin(set(m["match_id"]))]
    print("\n----------  market scoreline grid  -------------")
    if len(grid) < 2:
        print("  not enough fixtures with a closing Over/Under line to score")
        return
    print(f"n (model & market grid, same fixtures): {len(grid)}")
    print(f"correct-score hit rate : market {_score_hit_rate(grid):.3f}   "
          f"model {_score_hit_rate(m):.3f}")
    print(f"Over/Under 2.5 Brier   : market {_ou_brier(grid):.4f}   "
          f"model {_ou_brier(m):.4f}   (lower better)")
    ou_cov = window[["close_over_2_5", "close_under_2_5"]].notna().all(axis=1).mean()
    print(f"closing O/U 2.5 coverage in window: {ou_cov:.1%}")


def _aligned_rps(rated: pd.DataFrame, other: pd.DataFrame) -> tuple:
    """Per-match RPS for the model and for `other` (devig or Elo), on exactly
    the fixtures both rated, in the same order - so forecast_comparison can
    pair them."""
    o = other.dropna(subset=["p_home", "p_draw", "p_away"])[
        ["match_id", "p_home", "p_draw", "p_away"]]
    m = rated[["match_id", "p_home", "p_draw", "p_away", "result"]].merge(
        o, on="match_id", suffixes=("_m", "_o"), how="inner")
    if m.empty:
        return None, None
    model_rps = metrics.rps_series(
        m, cols=("p_home_m", "p_draw_m", "p_away_m"), outcome_col="result")
    other_rps = metrics.rps_series(
        m, cols=("p_home_o", "p_draw_o", "p_away_o"), outcome_col="result")
    return model_rps.to_numpy(), other_rps.to_numpy()


def _print_extended(rated: pd.DataFrame, dv_scored: pd.DataFrame, elo_scored: pd.DataFrame) -> None:
    print("\n----------  extended battery  -------------------")

    dec = metrics.brier_decomposition(rated)
    shp = metrics.sharpness(rated)
    print(f"Brier (multiclass)        : {metrics.brier_multiclass(rated):.4f}")
    print(f"  reliability (cal, lower) : {dec['reliability']:.4f}")
    print(f"  resolution  (info, higher): {dec['resolution']:.4f}")
    print(f"  uncertainty (data const): {dec['uncertainty']:.4f}")
    print(f"sharpness: entropy {shp['entropy']:.3f} nats (1.099 = flat 33/33/33), "
          f"mean top prob {shp['mean_max_prob']:.3f}")

    for name, other in (("devig line", dv_scored), ("Elo proxy", elo_scored)):
        m_rps, o_rps = _aligned_rps(rated, other)
        if m_rps is None:
            continue
        cmp = metrics.forecast_comparison(m_rps, o_rps)
        verdict = ("model better" if cmp["mean_diff"] < 0 else "model worse") if cmp["p_value"] < 0.05 \
            else "no significant difference"
        print(f"vs {name:10s}: ΔRPS {cmp['mean_diff']:+.4f} "
              f"[{cmp['ci_low']:+.4f}, {cmp['ci_high']:+.4f}]  "
              f"DM {cmp['dm_stat']:+.2f}  p={cmp['p_value']:.4f}  -> {verdict}")

    by_season = (rated.assign(_rps=metrics.rps_series(rated))
                 .groupby("season")["_rps"].agg(["mean", "size"]))
    print("per-season model RPS:")
    for season, row in by_season.iterrows():
        print(f"  {season}: {row['mean']:.4f}  (n={int(row['size'])})")


if __name__ == "__main__":
    raise SystemExit(main())
