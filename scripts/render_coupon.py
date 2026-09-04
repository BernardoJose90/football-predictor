"""Regenerate this weekend's prediction page from a fresh model run.

    python -m scripts.render_coupon [--days 4] [--no-refresh] [--source market]

Runs `scripts.predict_upcoming` (refreshing data by default), then renders
`web/coupon_template.html` with the result into `docs/index.html` - the file
GitHub Pages serves (see .github/workflows/weekly-predictions.yml). This is
the one script both the manual workflow and the CI schedule call, so the
auto-updating page and any local run always go through the same code path.

`--source` picks which forecast is the page's headline:
  * `model`  (default) - this repo's attack/defence ratings + Dixon-Coles,
    with the bookmakers' devigged price shown underneath for comparison.
  * `market` - the bookmakers' devigged price is the headline (who wins, and
    a scoreline grid fitted to it for the likely score / over / BTTS), with
    the model shown underneath instead. predict_upcoming still runs in model
    mode either way, so The Working / The Angles / The Ledger are unaffected;
    only this page's emphasis changes.

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


def _pct(x) -> float:
    return round(float(x) * 100, 1)


def _model_block(r: dict) -> dict | None:
    """The ratings model's forecast for one fixture row, or None if absent.

    On a normal card that's ``p_*`` / ``likely_score``; on a --source market
    card predict_upcoming carries it separately as ``model_p_*`` so the
    headline ``p_*`` can be the market price.
    """
    if pd.notna(r.get("model_p_home")):
        return {
            "pHome": _pct(r["model_p_home"]), "pDraw": _pct(r["model_p_draw"]),
            "pAway": _pct(r["model_p_away"]),
            "score": r.get("model_likely_score") or r.get("likely_score"),
            "homePred": round(float(r.get("model_home_pred", r.get("home_pred"))), 2),
            "awayPred": round(float(r.get("model_away_pred", r.get("away_pred"))), 2),
        }
    if pd.notna(r.get("p_home")):
        return {
            "pHome": _pct(r["p_home"]), "pDraw": _pct(r["p_draw"]), "pAway": _pct(r["p_away"]),
            "score": r.get("likely_score"),
            "homePred": round(float(r["home_pred"]), 2),
            "awayPred": round(float(r["away_pred"]), 2),
        }
    return None


def _market_block(r: dict) -> dict | None:
    """The bookmakers' devigged forecast for one fixture row, or None."""
    if not pd.notna(r.get("market_p_home")):
        return None
    block = {
        "pHome": _pct(r["market_p_home"]), "pDraw": _pct(r["market_p_draw"]),
        "pAway": _pct(r["market_p_away"]),
    }
    if pd.notna(r.get("market_likely_score")):
        block["score"] = r["market_likely_score"]
        block["homePred"] = round(float(r["market_home_pred"]), 2)
        block["awayPred"] = round(float(r["market_away_pred"]), 2)
        block["over25"] = _pct(r["market_p_over_2_5"])
        block["btts"] = _pct(r["market_p_btts"])
    return block


def build_payload(csv_path, source: str = "model") -> list[dict]:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    recs = []
    for r in df.to_dict("records"):
        rec = {
            "league": r["league"], "date": r["date"].strftime("%Y-%m-%dT%H:%M"),
            "home": r["home_team"], "away": r["away_team"], "unrated": bool(r["unrated"]),
        }
        if not rec["unrated"]:
            model = _model_block(r)
            market = _market_block(r)
            if source == "market" and market is not None:
                headline, compare = market, model
            else:
                headline, compare = model, market
            if headline is None:
                rec["unrated"] = True
                recs.append(rec)
                continue

            # Over/Under 2.5 + BTTS come from the headline forecast's own grid
            # when it has them, else the model's (the market block only carries
            # them when predict_upcoming fitted a market grid).
            over25 = headline.get("over25")
            btts = headline.get("btts")
            if over25 is None and pd.notna(r.get("p_over_2_5")):
                over25, btts = _pct(r["p_over_2_5"]), _pct(r["p_btts"])

            rec.update({
                "source": source,
                "homePred": headline["homePred"], "awayPred": headline["awayPred"],
                "score": headline["score"],
                "pHome": headline["pHome"], "pDraw": headline["pDraw"], "pAway": headline["pAway"],
                "over25": over25, "btts": btts,
                "note": r["adj_note"] if isinstance(r.get("adj_note"), str) and r["adj_note"] else "",
            })
            if compare is not None:
                rec.update({
                    "mHome": compare["pHome"], "mDraw": compare["pDraw"], "mAway": compare["pAway"],
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
    ap.add_argument("--source", choices=("model", "market"), default="model",
                    help="which forecast is the page's headline (see module docstring)")
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

    payload = build_payload(csv_path, source=args.source)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(payload), encoding="utf-8")
    print(f"wrote {len(payload)} fixtures ({args.source} headline) -> {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
