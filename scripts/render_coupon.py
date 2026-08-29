"""Regenerate this weekend's prediction page from a fresh model run.

    python -m scripts.render_coupon [--days 4] [--no-refresh]

Runs `scripts.predict_upcoming` (refreshing data by default), then renders
`web/coupon_template.html` with the result into `docs/index.html` - the file
GitHub Pages serves (see .github/workflows/weekly-predictions.yml). This is
the one script both the manual workflow and the CI schedule call, so the
auto-updating page and any local run always go through the same code path.

Always passes --log-predictions through to scripts.predict_upcoming, so every
published fixture is appended to artefacts/prediction_log.csv (first
prediction wins - see evaluate.prediction_log) - the standing record
scripts.track_record scores for the public track-record page (#5).
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

import config
from scripts import predict_upcoming

TEMPLATE = config.ROOT / "web" / "coupon_template.html"
OUT = config.ROOT / "docs" / "index.html"


def build_payload(csv_path) -> list[dict]:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    recs = []
    for r in df.itertuples(index=False):
        rec = {
            "league": r.league, "date": r.date.strftime("%Y-%m-%dT%H:%M"),
            "home": r.home_team, "away": r.away_team, "unrated": bool(r.unrated),
        }
        if not r.unrated:
            rec.update({
                "homePred": round(float(r.home_pred), 2), "awayPred": round(float(r.away_pred), 2),
                "score": r.likely_score,
                "pHome": round(float(r.p_home) * 100, 1), "pDraw": round(float(r.p_draw) * 100, 1),
                "pAway": round(float(r.p_away) * 100, 1),
                "over25": round(float(r.p_over_2_5) * 100, 1), "btts": round(float(r.p_btts) * 100, 1),
                "note": r.adj_note if isinstance(r.adj_note, str) and r.adj_note else "",
            })
            if pd.notna(r.market_p_home):
                rec.update({
                    "mHome": round(float(r.market_p_home) * 100, 1),
                    "mDraw": round(float(r.market_p_draw) * 100, 1),
                    "mAway": round(float(r.market_p_away) * 100, 1),
                })
        recs.append(rec)
    return recs


def render(payload: list[dict]) -> str:
    from scripts import render_common
    return render_common.finalize(
        TEMPLATE.read_text(encoding="utf-8"), payload,
        render_common.utc_now_str(), where=str(TEMPLATE),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--no-refresh", dest="refresh", action="store_false", default=True,
                    help="skip re-downloading historical data (use whatever's already on disk)")
    ap.add_argument("--no-injuries", dest="injuries", action="store_false", default=True,
                    help="matches Weekend Coupon's current live state (injuries ON); "
                         "pass this to render without the experimental PL injuries adjustment")
    args = ap.parse_args(argv)

    predict_argv = ["--days", str(args.days), "--log-predictions"]
    if args.refresh:
        predict_argv.append("--refresh")
    if args.injuries:
        predict_argv.append("--use-injuries")
    rc = predict_upcoming.main(predict_argv)
    if rc != 0:
        print(f"predict_upcoming exited {rc}, rendering with whatever it produced", file=sys.stderr)

    csv_path = config.ARTEFACTS / "upcoming_predictions.csv"
    if not csv_path.exists():
        print(f"FAIL: {csv_path} was not produced", file=sys.stderr)
        return 1

    payload = build_payload(csv_path)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(payload), encoding="utf-8")
    print(f"wrote {len(payload)} fixtures -> {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
