"""Walk-forward validation.

For each league, step through matchdays in chronological order. On each
matchday, build ratings from that league's matches *strictly before* the
matchday, predict every fixture, and record the prediction next to the actual
result. Ratings are league-relative, so leagues never share a rating pool.

Leakage guards:
  * build_ratings only ever sees matches with date < as_of.
  * The "as_of" for a matchday is that matchday's own date, so a match cannot
    inform its own prediction.
  * Splitting into a tuning window and a later untouched evaluation window is
    the caller's job (see run_backtest.py / tune.py) - the xi sweep is model
    selection and must not touch the reporting period.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import config
from model import predict as predict_mod
from model.ratings import build_ratings
from model.referee import build_referee_factors
from model.rest import add_rest_days, rest_factor
from model.travel import travel_factor, trip_distance_km
from evaluate import metrics


@dataclass
class BacktestConfig:
    stat: str = config.DEFAULT_STAT
    xi: float = config.DEFAULT_XI
    rho: float = config.DEFAULT_RHO
    delta: float = config.DEFAULT_DELTA   # diagonal inflation, see model/dixon_coles.py
    min_matches: int = config.DEFAULT_MIN_MATCHES
    max_goals: int = config.MAX_GOALS
    rebuild_every_days: int = 7          # reuse ratings within this many days
    # Section 10.1 feature candidates - referee and rest default ON per an
    # explicit product decision (see config.py); they raised RPS in this
    # repo's own walk-forward test, kept anyway for the real-world signal.
    use_referee: bool = config.DEFAULT_USE_REFEREE
    referee_min_matches: int = config.REFEREE_MIN_MATCHES
    referee_xi: float | None = None      # None => reuse cfg.xi
    use_rest: bool = config.DEFAULT_USE_REST
    rest_k: float = config.REST_K
    use_travel: bool = config.DEFAULT_USE_TRAVEL
    travel_k: float = config.TRAVEL_K
    # Squad value (rank 4) defaults OFF here specifically, unlike the live
    # prediction path (config.DEFAULT_USE_SQUAD_VALUE=True): the only value
    # data available is a single current-day snapshot, so using it to rate a
    # team in a 2023/24 match is look-ahead bias, not a fair walk-forward
    # test. Opt in deliberately (e.g. to sanity-check very recent matchdays
    # only), never as a default backtest setting.
    use_squad_value: bool = False
    squad_values: dict[str, float] | None = None
    value_prior_min_points: int = config.VALUE_PRIOR_MIN_POINTS


def _iter_matchdays(league_df: pd.DataFrame):
    for day, chunk in league_df.groupby(league_df["date"].dt.normalize(), sort=True):
        yield day, chunk


def backtest_league(
    league_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    league_df = league_df.sort_values("date").reset_index(drop=True)
    end = pd.Timestamp(end) if end is not None else league_df["date"].max() + pd.Timedelta(days=1)
    start = pd.Timestamp(start)

    rows: list[dict] = []
    snap = None
    snap_day = None
    ref_factors = None

    for day, chunk in _iter_matchdays(league_df):
        if day < start or day >= end:
            continue
        need_rebuild = (
            snap is None
            or snap_day is None
            or (day - snap_day).days >= cfg.rebuild_every_days
        )
        if need_rebuild:
            # stat="auto" tries the primary stat (xg) first, falling back to
            # sot for a division that doesn't have it (e.g. E1/SC0/P1) -
            # resolved here, per division, never passed into build_ratings
            # directly. A concrete stat is tried once, same as before.
            stats_to_try = (
                (config.AUTO_STAT_PRIMARY, config.AUTO_STAT_FALLBACK)
                if cfg.stat == "auto" else (cfg.stat,)
            )
            snap = None
            for candidate_stat in stats_to_try:
                try:
                    snap = build_ratings(
                        league_df, as_of=day, stat=candidate_stat, xi=cfg.xi,
                        min_matches=cfg.min_matches,
                        squad_values=cfg.squad_values if cfg.use_squad_value else None,
                        value_prior_min_points=cfg.value_prior_min_points,
                    )
                    break
                except ValueError:
                    continue
            if snap is None:
                continue  # not enough history yet, in any candidate stat
            snap_day = day
            if cfg.use_referee:
                ref_factors = build_referee_factors(
                    league_df, as_of=day, xi=cfg.referee_xi or cfg.xi,
                    min_matches=cfg.referee_min_matches,
                )

        for row in chunk.itertuples(index=False):
            lam_mult, mu_mult = (
                ref_factors.factor(getattr(row, "referee", None))
                if cfg.use_referee and ref_factors is not None else (1.0, 1.0)
            )
            away_rest = getattr(row, "away_rest_days", None)
            if cfg.use_rest:
                lam_mult *= rest_factor(getattr(row, "home_rest_days", None), k=cfg.rest_k)
                mu_mult *= rest_factor(away_rest, k=cfg.rest_k)
            if cfg.use_travel:
                dist = trip_distance_km(row.home_team, row.away_team)
                mu_mult *= travel_factor(dist, rest_days=away_rest, k=cfg.travel_k)
            pred = predict_mod.predict_match(
                snap, row.home_team, row.away_team, rho=cfg.rho, delta=cfg.delta,
                max_goals=cfg.max_goals, lam_mult=lam_mult, mu_mult=mu_mult,
            )
            rec = {
                "match_id": row.match_id,
                "date": row.date,
                "div": row.div,
                "league": row.league,
                "season": row.season,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "result": row.result,
                "home_goals": row.home_goals,
                "away_goals": row.away_goals,
                "ratings_as_of": snap.as_of,
                "unrated": pred is None,
            }
            if pred is not None:
                for k in ("home_pred", "away_pred", "likely_score",
                          "p_home", "p_draw", "p_away", "p_over_2_5", "p_btts"):
                    rec[k] = pred[k]
            rows.append(rec)

    return pd.DataFrame(rows)


# Columns backtest_league emits per row. Kept here so an all-empty result
# (e.g. stat="xg" with no xG data joined yet) still has the right shape rather
# than being a bare `pd.DataFrame()` with no columns at all - report() and
# any caller indexing by column name would otherwise KeyError on it.
_RESULT_COLUMNS = [
    "match_id", "date", "div", "league", "season", "home_team", "away_team",
    "result", "home_goals", "away_goals", "ratings_as_of", "unrated",
    "home_pred", "away_pred", "likely_score",
    "p_home", "p_draw", "p_away", "p_over_2_5", "p_btts",
]


def backtest(
    matches: pd.DataFrame,
    start,
    end=None,
    cfg: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Run the walk-forward backtest across every league in ``matches``.

    Returns an empty-but-correctly-shaped frame (see _RESULT_COLUMNS) if no
    league had enough history/data for the requested stat and window - this
    happens for stat="xg" before ingest.understat has been run, for example.
    """
    cfg = cfg or BacktestConfig()
    # Rest days are computed across ALL divisions at once (a team's previous
    # match may have been in a different division after promotion/relegation),
    # so this has to happen before the per-division split below, not inside it.
    # Travel also needs the away side's rest days (distance matters most
    # combined with short rest - see model/travel.py), so either flag triggers it.
    if cfg.use_rest or cfg.use_travel:
        matches = add_rest_days(matches)
    frames = [
        backtest_league(g, pd.Timestamp(start), end, cfg)
        for _, g in matches.groupby("div")
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=_RESULT_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def report(preds: pd.DataFrame) -> dict:
    """Headline numbers for a backtest frame: coverage, RPS, log loss.

    Safe to call on an empty frame (e.g. no data for the requested stat) -
    returns zeroed-out / NaN numbers rather than raising.
    """
    total = len(preds)
    if total == 0 or "unrated" not in preds.columns:
        return {
            "matches": 0, "rated": 0, "coverage": 0.0,
            "n": 0, "rps": float("nan"), "log_loss": float("nan"),
        }
    rated = preds[~preds["unrated"] & preds["p_home"].notna()]
    out = {
        "matches": total,
        "rated": len(rated),
        "coverage": round(len(rated) / total, 4) if total else 0.0,
    }
    out.update(metrics.summary(rated))
    # calibration error on P(home win)
    if len(rated):
        out["calibration_error_home"] = round(
            metrics.calibration_error(rated["p_home"], (rated["result"] == "H").astype(int)), 4
        )
    return out
