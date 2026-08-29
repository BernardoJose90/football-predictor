"""Score the live prediction log, plus a walk-forward validation backtest,
into one JSON for the public track-record page (#5).

    python -m scripts.track_record [--eval-start 2025-08-01] [--out artefacts/track_record.json]

Two distinct kinds of evidence, kept separate rather than blended together:

  * "live" - every fixture actually published on the coupon (see
    scripts.predict_upcoming --log-predictions and evaluate.prediction_log),
    scored against results as they resolve. This is the real, un-cherry-picked
    track record described there: first prediction per fixture wins, nothing
    is restated with hindsight. It starts thin - it can only ever cover
    fixtures predicted *since* logging began - and fills in one matchday at a
    time; most entries are "pending" (not yet played) at first.
  * "validation" - the model's own walk-forward backtest over the untouched
    evaluation window, run with its current production config (same defaults
    as scripts.run_backtest). This is long-run evidence that doesn't depend
    on how many weeks the live log has been running, shown alongside the live
    numbers rather than instead of them.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

import config
from evaluate import baselines, metrics, prediction_log
from evaluate import track_record as live_tr
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


def _monthly_rps(df: pd.DataFrame, cols=("p_home", "p_draw", "p_away")) -> pd.DataFrame:
    """Mean RPS per calendar month, plus a running (expanding) mean.

    Safe on a frame with zero scored rows (e.g. Elo's warmup period eating
    the whole window on a short history) - pandas' `.apply(axis=1)` returns
    the frame itself, not an empty Series, when there are no rows to call the
    function on, so that case needs its own early return here.
    """
    scored = df.dropna(subset=list(cols) + ["result"]).sort_values("date").copy()
    if scored.empty:
        return pd.DataFrame(columns=["month", "rps", "n", "running_rps"])
    scored["rps"] = metrics.rps_series(scored, cols=cols)
    scored["month"] = scored["date"].dt.to_period("M").dt.to_timestamp()
    monthly = scored.groupby("month").agg(rps=("rps", "mean"), n=("rps", "size")).reset_index()
    scored_sorted = scored.sort_values("date")
    # "month" is each bucket's first-of-month midnight; compare against the
    # *next* month's start (not that month's own midnight-31st) so a late
    # kickoff on the last day of the month - anything after 00:00 - isn't
    # dropped from its own running mean.
    next_month_start = monthly["month"] + pd.DateOffset(months=1)
    monthly["running_rps"] = [
        scored_sorted.loc[scored_sorted["date"] < nm, "rps"].mean() for nm in next_month_start
    ]
    return monthly


def _validation(matches: pd.DataFrame, eval_start: str, eval_end: str | None) -> dict:
    start = pd.Timestamp(eval_start)
    end = pd.Timestamp(eval_end) if eval_end else None

    window = matches[matches["date"] >= start]
    if end is not None:
        window = window[window["date"] < end]

    dv = baselines.devig_frame(window)
    dv_scored = dv.dropna(subset=["p_home", "p_draw", "p_away"])

    elo = baselines.elo_walk_forward(matches)
    elo_window = elo[elo["date"] >= start]
    if end is not None:
        elo_window = elo_window[elo_window["date"] < end]
    elo_scored = elo_window.dropna(subset=["p_home", "p_draw", "p_away"])

    cfg = BacktestConfig()  # production defaults, unmodified
    preds = backtest(matches, start=start, end=end, cfg=cfg)
    rated = preds[~preds["unrated"] & preds["p_home"].notna()].copy()
    rep = report(preds)

    common_ids = set(rated["match_id"])
    dv_common = dv_scored[dv_scored["match_id"].isin(common_ids)]
    elo_common = elo_scored[elo_scored["match_id"].isin(common_ids)]

    def series(monthly: pd.DataFrame) -> list[dict]:
        return [
            {"month": r.month.strftime("%Y-%m"), "rps": round(r.rps, 4),
             "running": round(r.running_rps, 4), "n": int(r.n)}
            for r in monthly.itertuples(index=False)
        ]

    calib = {}
    for outcome, col in (("home", "p_home"), ("draw", "p_draw"), ("away", "p_away")):
        tbl = metrics.calibration_table(rated[col], (rated["result"] == outcome[0].upper()).astype(int), bins=10)
        calib[outcome] = [
            {"predicted": round(float(r.predicted), 4), "observed": round(float(r.observed), 4), "n": int(r.n)}
            for r in tbl.itertuples(index=False) if r.n > 0
        ]

    leagues = []
    if not rated.empty:
        per_league = (
            rated.groupby("league").apply(lambda g: metrics.summary(g), include_groups=False)
            .apply(pd.Series).reset_index().sort_values("rps")
        )
        leagues = [{"league": r.league, "rps": round(r.rps, 4), "n": int(r.n)}
                  for r in per_league.itertuples(index=False)]

    dv_summary = metrics.summary(dv_common)
    elo_summary = metrics.summary(elo_common)

    return {
        "window": {"start": start.strftime("%Y-%m-%d"),
                  "end": (end.strftime("%Y-%m-%d") if end is not None else matches["date"].max().strftime("%Y-%m-%d"))},
        "config": {"stat": cfg.stat, "xi": cfg.xi, "rho": cfg.rho, "delta": cfg.delta,
                  "referee": cfg.use_referee, "rest": cfg.use_rest, "travel": cfg.use_travel},
        "headline": {
            "model_rps": round(rep["rps"], 4), "model_log_loss": round(rep["log_loss"], 4),
            "model_n": rep["rated"], "coverage": rep["coverage"],
            "devig_rps": round(dv_summary["rps"], 4), "devig_n": dv_summary["n"],
            "elo_rps": round(elo_summary["rps"], 4), "elo_n": elo_summary["n"],
            "calibration_error": rep.get("calibration_error_home", float("nan")),
            "beats_elo": bool(rep["rps"] < elo_summary["rps"]),
            "gap_to_market": round(rep["rps"] - dv_summary["rps"], 4),
        },
        "monthly": {"model": series(_monthly_rps(rated)), "devig": series(_monthly_rps(dv_common)),
                   "elo": series(_monthly_rps(elo_common))},
        "calibration": calib,
        "leagues": leagues,
    }


def build(eval_start: str, eval_end: str | None) -> dict:
    matches = _load_matches()
    log = prediction_log.load()
    return {
        "generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "live": live_tr.score(log, matches),
        "validation": _validation(matches, eval_start, eval_end),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-start", default="2025-08-01",
                    help="start of the validation backtest window (the live log has no window - "
                         "it's whatever has been logged)")
    ap.add_argument("--eval-end", default=None)
    ap.add_argument("--out", default=str(config.ARTEFACTS / "track_record.json"))
    args = ap.parse_args(argv)

    payload = build(args.eval_start, args.eval_end)
    from pathlib import Path
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    live = payload["live"]
    val = payload["validation"]["headline"]
    print(f"live log   : {live['n_scored']} scored, {live['n_pending']} pending, "
          f"{live['n_logged']} logged total")
    if live["model"]:
        print(f"             model RPS {live['model']['rps']:.4f}")
    if live.get("scoreline"):
        sl = live["scoreline"]
        print(f"             scoreline: {sl['exact']:.1%} exact, {sl['goal_diff']:.1%} margin, "
              f"{sl['result_and_within_1']:.1%} result+1  (1-1 baseline {sl['baseline_always_1_1']:.1%}, n={sl['n']})")
    print(f"validation : model RPS {val['model_rps']:.4f} vs devig {val['devig_rps']:.4f} "
          f"vs Elo {val['elo_rps']:.4f}  (n={val['model_n']})")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
