"""Build "The Working" - why the model disagrees with the market (#3).

    python -m scripts.render_why [--in artefacts/upcoming_predictions.json]
                                 [--min-gap 5] [--json-out ...] [--html-out ...]

Reads the rich per-fixture JSON ``scripts.predict_upcoming`` always writes
alongside its CSV (ratings, the structured ``adjustments`` list, and
``base_p_*`` - the same fixture priced with every section-10.1 adjustment
turned off). For each rated fixture with a market price, this splits the
model/market gap into two numbers:

  * ``gap``   - how far the model's *final*, published number is from the
    market's, in percentage points (the headline disagreement).
  * ``moved`` - how far the adjustments (referee/rest/travel/injuries) pushed
    the number away from the plain "ratings + Dixon-Coles" base prediction.

A big ``gap`` with a small ``moved`` means the core attack/defence ratings
disagree with the market on their own; a big ``moved`` close to the ``gap``
means the adjustments are doing most of the disagreeing. Nothing here reruns
the model - it only reads what scripts.predict_upcoming already computed, so
running this twice on the same input is always a no-op.

Writes both a JSON dataset (artefacts/why.json) and the rendered page
(docs/why.html, from web/why_template.html) so CI can commit the page
directly - see .github/workflows/weekly-predictions.yml.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config

TEMPLATE = config.ROOT / "web" / "why_template.html"
JSON_OUT = config.ARTEFACTS / "why.json"
HTML_OUT = config.ROOT / "docs" / "why.html"
SRC = config.ARTEFACTS / "upcoming_predictions.json"


def _max_gap(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b)) * 100


def _primary(final: tuple, market: tuple) -> int:
    diffs = [f - m for f, m in zip(final, market)]
    return max(range(3), key=lambda i: abs(diffs[i]))


def build(records: list[dict], min_gap: float) -> list[dict]:
    out = []
    for r in records:
        if r.get("unrated") or r.get("market_p_home") is None:
            continue
        if "base_p_home" not in r:
            continue
        final = (r["p_home"], r["p_draw"], r["p_away"])
        market = (r["market_p_home"], r["market_p_draw"], r["market_p_away"])
        base = (r["base_p_home"], r["base_p_draw"], r["base_p_away"])
        gap = _max_gap(final, market)
        if gap < min_gap:
            continue
        moved = _max_gap(final, base)
        pi = _primary(final, market)
        # attribution: how much of the gap on the most-divergent outcome is
        # already in the ratings (base vs market) vs added by the nudges
        ratings_part = abs(base[pi] - market[pi]) * 100
        adj_part = abs(final[pi] - base[pi]) * 100
        if gap < 0.05:
            attribution = "even"
        elif ratings_part >= 0.6 * gap:
            attribution = "ratings"
        elif adj_part >= 0.6 * gap:
            attribution = "adjustments"
        else:
            attribution = "mixed"
        out.append({
            "league": r["league"], "date": r["date"],
            "home": r["home_team"], "away": r["away_team"],
            "score": r["likely_score"],
            "primary": ["home", "draw", "away"][pi],
            "direction": "higher" if (final[pi] - market[pi]) > 0 else "lower",
            "pureHome": round(base[0] * 100, 1), "pureDraw": round(base[1] * 100, 1),
            "pureAway": round(base[2] * 100, 1),
            "pHome": round(final[0] * 100, 1), "pDraw": round(final[1] * 100, 1),
            "pAway": round(final[2] * 100, 1),
            "mHome": round(market[0] * 100, 1), "mDraw": round(market[1] * 100, 1),
            "mAway": round(market[2] * 100, 1),
            "homeAtk": r.get("home_attack"), "homeDef": r.get("home_defence"),
            "awayAtk": r.get("away_attack"), "awayDef": r.get("away_defence"),
            "homeMatches": r.get("home_matches_used"), "awayMatches": r.get("away_matches_used"),
            "baseGoals": [r.get("base_home_pred"), r.get("base_away_pred")],
            "finalGoals": [r.get("home_pred"), r.get("away_pred")],
            "stat": r.get("stat_used") or r.get("stat"),
            "adjustments": r.get("adjustments") or [],
            "gap": round(gap, 1),
            "moved": round(moved, 1),
            "attribution": attribution,
        })
    out.sort(key=lambda d: -d["gap"])
    return out


def render(payload: list[dict], template: Path | None = None) -> str:
    from scripts import render_common
    tmpl = template or TEMPLATE
    return render_common.finalize(
        tmpl.read_text(encoding="utf-8"), payload,
        render_common.utc_now_str(), where=str(tmpl),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", default=str(SRC))
    ap.add_argument("--min-gap", type=float, default=5.0,
                    help="only keep fixtures where the model and market differ by at least "
                         "this many points on some outcome (default 5)")
    ap.add_argument("--json-out", default=str(JSON_OUT))
    ap.add_argument("--html-out", default=str(HTML_OUT))
    ap.add_argument("--no-html", action="store_true", help="skip rendering docs/why.html")
    args = ap.parse_args(argv)

    with open(args.in_path, encoding="utf-8") as fh:
        records = json.load(fh)

    payload = build(records, args.min_gap)
    Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{len(payload)}/{len(records)} fixtures at >= {args.min_gap}pt gap -> {args.json_out}")

    if not args.no_html:
        html_out = Path(args.html_out)
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(render(payload), encoding="utf-8")
        print(f"-> {html_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
