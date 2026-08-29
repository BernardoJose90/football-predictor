"""Build the "why does the model disagree with the market" dataset (#3).

    python -m scripts.render_why [--in artefacts/upcoming_predictions.json] [--min-gap 5] [--out artefacts/why.json]

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
"""
from __future__ import annotations

import argparse
import json

import config


def _max_gap(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b)) * 100


def build(records: list[dict], min_gap: float) -> list[dict]:
    out = []
    for r in records:
        if r.get("unrated") or r.get("market_p_home") is None:
            continue
        final = (r["p_home"], r["p_draw"], r["p_away"])
        market = (r["market_p_home"], r["market_p_draw"], r["market_p_away"])
        base = (r["base_p_home"], r["base_p_draw"], r["base_p_away"])
        gap = _max_gap(final, market)
        if gap < min_gap:
            continue
        out.append({
            "league": r["league"], "date": r["date"],
            "home": r["home_team"], "away": r["away_team"],
            "pureHome": round(base[0] * 100, 1), "pureDraw": round(base[1] * 100, 1),
            "pureAway": round(base[2] * 100, 1),
            "pHome": round(final[0] * 100, 1), "pDraw": round(final[1] * 100, 1),
            "pAway": round(final[2] * 100, 1), "score": r["likely_score"],
            "mHome": round(market[0] * 100, 1), "mDraw": round(market[1] * 100, 1),
            "mAway": round(market[2] * 100, 1),
            "note": r.get("adj_note") or "",
            "adjustments": r.get("adjustments") or [],
            "gap": round(gap, 1),
            "moved": round(_max_gap(final, base), 1),
        })
    out.sort(key=lambda d: -d["gap"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", default=str(config.ARTEFACTS / "upcoming_predictions.json"))
    ap.add_argument("--min-gap", type=float, default=5.0,
                    help="only keep fixtures where the model and market differ by at least "
                         "this many points on some outcome (default 5)")
    ap.add_argument("--out", default=str(config.ARTEFACTS / "why.json"))
    args = ap.parse_args(argv)

    with open(args.in_path, encoding="utf-8") as fh:
        records = json.load(fh)

    payload = build(records, args.min_gap)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"{len(payload)}/{len(records)} fixtures at >= {args.min_gap}pt gap -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
