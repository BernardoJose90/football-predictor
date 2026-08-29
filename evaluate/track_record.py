"""Score the published prediction log against results as they come in.

``evaluate.prediction_log`` records what the site said, before kick-off, for
every fixture it published. This module joins that log to the normalised
results table and turns it into the numbers a track-record page shows:

  * headline RPS / log loss / hit rate for the model and, on the identical
    fixture set, for the bookmaker's pre-match price - the only fair yardstick
  * a cumulative RPS curve (how the average has moved as fixtures resolved)
  * a calibration table (predicted probability vs how often it actually
    happened), pooled over home/draw/away - the same plot the backtest draws,
    but on live predictions
  * a scoreline scorecard - how often the single "likely score" was exactly
    right / had the right margin / was within a goal each side, next to the
    dumb "always guess 1-1" baseline
  * the per-fixture "here's what we said, here's what happened" list

Nothing here rebuilds or reruns the model. It only reads the log and the
results feed, so it's cheap to run on every refresh and can't accidentally
restate a prediction with hindsight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluate import metrics

_PCOLS = ("p_home", "p_draw", "p_away")


def _argmax_outcome(row, prefix: str) -> str:
    vals = [row[f"{prefix}_p_home"], row[f"{prefix}_p_draw"], row[f"{prefix}_p_away"]]
    return "HDA"[int(np.argmax(vals))]


def join_results(log: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Log rows left-joined to their actual result. A row with no result yet
    (fixture not in ``matches``) keeps NaN in ``result`` - that's "pending"."""
    res = matches[["match_id", "result", "home_goals", "away_goals"]].drop_duplicates("match_id")
    out = log.merge(res, on="match_id", how="left")
    out["kickoff"] = pd.to_datetime(out["kickoff"], errors="coerce")
    return out


def _calibration(scored: pd.DataFrame, bins: int = 10) -> list[dict]:
    """Pooled reliability table over all three outcomes."""
    preds, hits = [], []
    for outcome, col in zip("HDA", _PCOLS):
        preds.append(scored[f"model_{col}"].to_numpy(dtype=float))
        hits.append((scored["result"] == outcome).to_numpy(dtype=float))
    tbl = metrics.calibration_table(np.concatenate(preds), np.concatenate(hits), bins=bins)
    return [
        {"predicted": round(float(r.predicted), 4),
         "observed": round(float(r.observed), 4),
         "n": int(r.n)}
        for r in tbl.itertuples(index=False)
    ]


def _parse_score(s) -> tuple[int, int] | None:
    """'2-1' -> (2, 1). None for anything that doesn't parse."""
    try:
        h, a = str(s).split("-")
        return int(h), int(a)
    except (ValueError, AttributeError):
        return None


def _scoreline_block(scored: pd.DataFrame) -> dict | None:
    """How the model's single 'likely score' did against the actual score.

    Kept separate from the RPS/outcome numbers on purpose: the likely score is
    just the most probable cell of the Poisson grid, not what the model is
    built to predict, and exact-score accuracy tops out around 11-13% for
    anyone (that's why 'baseline_always_1_1' is reported next to it - the
    dumbest possible fixed guess). 'result_and_within_1' = got the H/D/A right
    AND each side's goals within one; 'goal_diff' = predicted margin exactly
    right (which already implies the result).
    """
    rows = scored.dropna(subset=["home_goals", "away_goals"])
    hits_exact = hits_gd = hits_close = base_11 = n = 0
    for r in rows.itertuples(index=False):
        pred = _parse_score(r.likely_score)
        if pred is None:
            continue
        ph, pa = pred
        ah, aa = int(r.home_goals), int(r.away_goals)
        n += 1
        hits_exact += (ph == ah and pa == aa)
        hits_gd += ((ph - pa) == (ah - aa))
        pred_res = "H" if ph > pa else "A" if pa > ph else "D"
        hits_close += (pred_res == r.result and abs(ph - ah) <= 1 and abs(pa - aa) <= 1)
        base_11 += (ah == 1 and aa == 1)
    if n == 0:
        return None
    return {
        "n": n,
        "exact": round(hits_exact / n, 4),
        "goal_diff": round(hits_gd / n, 4),
        "result_and_within_1": round(hits_close / n, 4),
        "baseline_always_1_1": round(base_11 / n, 4),
    }


def _summary_block(scored: pd.DataFrame, prefix: str) -> dict:
    df = scored.rename(columns={f"{prefix}_{c}": c for c in _PCOLS})
    s = metrics.summary(df, cols=_PCOLS, outcome_col="result")
    called = scored.apply(lambda r: _argmax_outcome(r, prefix) == r["result"], axis=1)
    s["hit_rate"] = round(float(called.mean()), 4) if len(scored) else float("nan")
    s["rps"] = round(s["rps"], 4)
    s["log_loss"] = round(s["log_loss"], 4)
    return s


def score(log: pd.DataFrame, matches: pd.DataFrame, bins: int = 10) -> dict:
    joined = join_results(log, matches)
    has_market = joined["market_p_home"].notna()
    scored = joined[joined["result"].isin(["H", "D", "A"])].copy()
    scored_mkt = scored[scored["market_p_home"].notna()].copy()
    pending = joined[~joined["result"].isin(["H", "D", "A"])].copy()

    report: dict = {
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "n_logged": int(len(joined)),
        "n_scored": int(len(scored)),
        "n_pending": int(len(pending)),
        "n_with_market": int(has_market.sum()),
        "first_logged": (joined["logged_at"].min() if len(joined) else None),
        "span": None,
        "model": None,
        "market": None,
        "cumulative": [],
        "calibration": [],
        "scoreline": None,
        "by_league": [],
        "fixtures": [],
    }
    if scored.empty:
        return report

    scored = scored.sort_values("kickoff").reset_index(drop=True)
    report["span"] = [scored["kickoff"].min().strftime("%Y-%m-%d"),
                      scored["kickoff"].max().strftime("%Y-%m-%d")]
    report["model"] = _summary_block(scored, "model")
    report["calibration"] = _calibration(scored, bins=bins)
    report["scoreline"] = _scoreline_block(scored)
    if not scored_mkt.empty:
        report["market"] = _summary_block(scored_mkt, "market")
        # model, restricted to the exact fixtures the market row covers, so the
        # two headline numbers are comparable rather than on different samples.
        report["model_on_market_set"] = _summary_block(scored_mkt, "model")

    # cumulative RPS as fixtures resolved, in kick-off order
    scored["model_rps_val"] = metrics.rps_series(
        scored.rename(columns={f"model_{c}": c for c in _PCOLS}), _PCOLS, "result").to_numpy()
    scored["cum_model_rps"] = scored["model_rps_val"].expanding().mean()
    cum_mkt: dict[str, float] = {}
    if not scored_mkt.empty:
        m = scored_mkt.sort_values("kickoff").reset_index(drop=True)
        m["mkt_rps_val"] = metrics.rps_series(
            m.rename(columns={f"market_{c}": c for c in _PCOLS}), _PCOLS, "result").to_numpy()
        m["cum_mkt_rps"] = m["mkt_rps_val"].expanding().mean()
        cum_mkt = dict(zip(m["match_id"], m["cum_mkt_rps"]))
    for i, r in enumerate(scored.itertuples(index=False), start=1):
        report["cumulative"].append({
            "date": r.kickoff.strftime("%Y-%m-%d"),
            "n": i,
            "model": round(float(r.cum_model_rps), 4),
            "market": (round(float(cum_mkt[r.match_id]), 4)
                       if r.match_id in cum_mkt else None),
        })

    # per-league breakdown
    for lg, g in scored.groupby("league"):
        blk = _summary_block(g, "model")
        report["by_league"].append({
            "league": lg, "n": int(len(g)),
            "rps": blk["rps"], "hit_rate": blk["hit_rate"],
        })
    report["by_league"].sort(key=lambda d: -d["n"])

    # per-fixture list, most recent first
    for r in scored.sort_values("kickoff", ascending=False).itertuples(index=False):
        rec = {
            "match_id": r.match_id,
            "league": r.league,
            "date": r.kickoff.strftime("%Y-%m-%d"),
            "home": r.home_team, "away": r.away_team,
            "result": r.result,
            "score": (f"{int(r.home_goals)}-{int(r.away_goals)}"
                      if pd.notna(r.home_goals) else None),
            "likely_score": r.likely_score,
            "scoreline_exact": bool(
                pd.notna(r.home_goals)
                and _parse_score(r.likely_score) == (int(r.home_goals), int(r.away_goals))
            ),
            "model": [round(float(r.model_p_home) * 100, 1),
                      round(float(r.model_p_draw) * 100, 1),
                      round(float(r.model_p_away) * 100, 1)],
            "model_rps": round(float(r.model_rps_val), 4),
            "model_called": _argmax_outcome(r._asdict(), "model") == r.result,
            "adj_note": r.adj_note if isinstance(r.adj_note, str) else "",
        }
        if pd.notna(r.market_p_home):
            rec["market"] = [round(float(r.market_p_home) * 100, 1),
                             round(float(r.market_p_draw) * 100, 1),
                             round(float(r.market_p_away) * 100, 1)]
            rec["market_called"] = _argmax_outcome(r._asdict(), "market") == r.result
        report["fixtures"].append(rec)

    return report
