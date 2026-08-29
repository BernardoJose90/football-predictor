"""Build "The Angles" - the model's best disagreements with the market price.

    python -m scripts.render_angles [--min-edge 5] [--min-prob 35]

For every rated fixture, this looks at all three outcomes (home / draw / away)
and picks the one where the model's probability is *furthest above* the
market's - i.e. the outcome the model thinks the price has underpriced. That
gap is the "edge". Fixtures are kept only where the edge clears ``--min-edge``
percentage points AND the model gives the pick a real chance
(``--min-prob``), then sorted biggest edge first.

This is deliberately NOT a list of short favourites: a team the model rates
80% but the market already prices at 80% has no edge and doesn't appear. It
reads the rich per-fixture JSON ``scripts.predict_upcoming`` writes - nothing
is recomputed - and renders ``web/angles_template.html`` into
``docs/angles.html`` alongside ``artefacts/angles.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config

TEMPLATE = config.ROOT / "web" / "angles_template.html"
JSON_OUT = config.ARTEFACTS / "angles.json"
HTML_OUT = config.ROOT / "docs" / "angles.html"
SRC = config.ARTEFACTS / "upcoming_predictions.json"

_SIDES = ("home", "draw", "away")


def build(records: list[dict], min_edge: float = 5.0, min_prob: float = 35.0) -> list[dict]:
    out = []
    for r in records:
        if r.get("unrated") or r.get("market_p_home") is None or "p_home" not in r:
            continue
        model = [r["p_home"] * 100, r["p_draw"] * 100, r["p_away"] * 100]
        market = [r["market_p_home"] * 100, r["market_p_draw"] * 100, r["market_p_away"] * 100]
        edges = [m - k for m, k in zip(model, market)]
        i = max(range(3), key=lambda j: edges[j])
        if edges[i] < min_edge or model[i] < min_prob:
            continue

        side = _SIDES[i]
        if side == "draw":
            pick, category = "Draw", "draw"
        else:
            pick = r["home_team"] if side == "home" else r["away_team"]
            # underdog = the market makes the picked team the outsider
            picked_mkt = market[0] if side == "home" else market[2]
            other_mkt = market[2] if side == "home" else market[0]
            category = "underdog" if picked_mkt < other_mkt else "favourite"

        out.append({
            "league": r["league"], "date": r["date"],
            "home": r["home_team"], "away": r["away_team"],
            "pick": pick, "side": side, "category": category,
            "model": round(model[i], 1), "market": round(market[i], 1),
            "edge": round(edges[i], 1),
            "modelProbs": [round(x, 1) for x in model],
            "marketProbs": [round(x, 1) for x in market],
            "likelyScore": r.get("likely_score"),
        })
    out.sort(key=lambda d: -d["edge"])
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
    ap.add_argument("--min-edge", type=float, default=5.0,
                    help="minimum model-minus-market gap on the picked outcome, in points")
    ap.add_argument("--min-prob", type=float, default=35.0,
                    help="the model must give the picked outcome at least this probability")
    ap.add_argument("--json-out", default=str(JSON_OUT))
    ap.add_argument("--html-out", default=str(HTML_OUT))
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args(argv)

    with open(args.in_path, encoding="utf-8") as fh:
        records = json.load(fh)

    payload = build(records, min_edge=args.min_edge, min_prob=args.min_prob)
    Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{len(payload)} value pick(s) at >= {args.min_edge}pt edge -> {args.json_out}")

    if not args.no_html:
        out = Path(args.html_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render(payload), encoding="utf-8")
        print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
