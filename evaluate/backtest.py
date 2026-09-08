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
from model.club_elo_implied import implied_1x2
from model.league_bridge import cl_goal_baselines, elo_bridge
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
    # Champions League cross-league bridge exponent (model/league_bridge.py).
    # Only backtest_uefa reads it; 0.0 = domestic form only, no Elo bridge.
    cl_gamma: float = config.DEFAULT_CL_GAMMA
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


def build_snapshot(league_df, as_of, cfg: BacktestConfig, *, squad_values=None):
    """Walk cfg's stat chain (xg -> sot -> goals for "auto") and return the
    first RatingSnapshot that has enough history, or None. Shared by the
    domestic and the Champions League walk-forwards."""
    stats_to_try = config.AUTO_STAT_CHAIN if cfg.stat == "auto" else (cfg.stat,)
    for candidate_stat in stats_to_try:
        try:
            return build_ratings(
                league_df, as_of=as_of, stat=candidate_stat, xi=cfg.xi,
                min_matches=cfg.min_matches, squad_values=squad_values,
                value_prior_min_points=cfg.value_prior_min_points,
            )
        except ValueError:
            continue
    return None


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
            # stat="auto" walks xg -> sot -> goals, per division, using the
            # first the division has usable data for - resolved here, never
            # passed into build_ratings directly.
            snap = build_snapshot(
                league_df, day, cfg,
                squad_values=cfg.squad_values if cfg.use_squad_value else None,
            )
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


_UEFA_RESULT_COLUMNS = [
    "match_id", "date", "competition", "season", "matchday", "stage",
    "home_team", "away_team", "home_div", "away_div", "result",
    "home_goals", "away_goals", "home_elo", "away_elo", "bridge",
    "ratings_as_of", "unrated",
    "home_pred", "away_pred", "likely_score",
    "p_home", "p_draw", "p_away", "p_over_2_5", "p_btts",
    "elo_p_home", "elo_p_draw", "elo_p_away",
]


def _division_as_of(appearances: dict[str, list[tuple]], team: str, as_of) -> str | None:
    """The division of ``team``'s most recent domestic match before ``as_of``.

    ``appearances[team]`` is a date-sorted list of ``(date, div)`` tuples built
    once by the caller.
    """
    hist = appearances.get(team)
    if not hist:
        return None
    as_of = pd.Timestamp(as_of)
    prev = [div for d, div in hist if d < as_of]
    return prev[-1] if prev else None


def backtest_uefa(
    cl_results: pd.DataFrame,
    domestic_matches: pd.DataFrame,
    start,
    end=None,
    cfg: BacktestConfig | None = None,
    *,
    elo_lookup: dict | None = None,
) -> pd.DataFrame:
    """Walk-forward over Champions League matchdays.

    For each tie: find each club's current domestic division, build that
    league's ratings as of the matchday, bridge the two scales with Club Elo
    (model/league_bridge.py), and score the cross-league prediction against the
    actual result. Also records the Club-Elo-implied 1X2 as the baseline the
    model is measured against (no free CL closing line exists).

    ``cl_results`` needs: match_id, date, season, home_team, away_team, result,
    home_goals, away_goals (matchday / stage optional). ``domestic_matches`` is
    the full multi-division match table. ``elo_lookup`` (``{matchday ->
    {team -> elo}}``) is pulled from ingest.club_elo when not supplied - tests
    pass a synthetic one.
    """
    cfg = cfg or BacktestConfig()
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else None

    dm = domestic_matches.sort_values("date")
    appearances: dict[str, list[tuple]] = {}
    for r in dm.itertuples(index=False):
        appearances.setdefault(r.home_team, []).append((r.date, r.div))
        appearances.setdefault(r.away_team, []).append((r.date, r.div))

    cl = cl_results.sort_values("date").reset_index(drop=True)
    cl = cl[cl["date"] >= start]
    if end is not None:
        cl = cl[cl["date"] < end]
    if cl.empty:
        return pd.DataFrame(columns=_UEFA_RESULT_COLUMNS)

    if elo_lookup is None:
        from ingest import club_elo
        elo_lookup = club_elo.elo_by_date(cl["date"].dt.normalize().unique())

    snap_cache: dict[tuple, object] = {}

    def _snap(div, day):
        if div is None:
            return None
        key = (div, pd.Timestamp(day).normalize())
        if key not in snap_cache:
            league_df = domestic_matches[domestic_matches["div"] == div]
            snap_cache[key] = build_snapshot(league_df, day, cfg)
        return snap_cache[key]

    rows: list[dict] = []
    for day, chunk in _iter_matchdays(cl):
        emap = elo_lookup.get(pd.Timestamp(day).normalize(), {}) or {}
        g_home, g_away = cl_goal_baselines(cl_results, day, xi=cfg.xi)
        for row in chunk.itertuples(index=False):
            hdiv = _division_as_of(appearances, row.home_team, day)
            adiv = _division_as_of(appearances, row.away_team, day)
            snap_h, snap_a = _snap(hdiv, day), _snap(adiv, day)
            elo_h, elo_a = emap.get(row.home_team), emap.get(row.away_team)
            bridge = elo_bridge(elo_h, elo_a, cfg.cl_gamma)

            pred = None
            if snap_h is not None and snap_a is not None:
                pred = predict_mod.predict_cross_league(
                    snap_h, snap_a, row.home_team, row.away_team,
                    bridge=bridge, g_home=g_home, g_away=g_away,
                    rho=cfg.rho, delta=cfg.delta, max_goals=cfg.max_goals,
                )

            rec = {
                "match_id": getattr(row, "match_id", None),
                "date": row.date,
                "competition": "CL",
                "season": getattr(row, "season", None),
                "matchday": getattr(row, "matchday", None),
                "stage": getattr(row, "stage", None),
                "home_team": row.home_team,
                "away_team": row.away_team,
                "home_div": hdiv,
                "away_div": adiv,
                "result": row.result,
                "home_goals": row.home_goals,
                "away_goals": row.away_goals,
                "home_elo": elo_h,
                "away_elo": elo_a,
                "bridge": round(float(bridge), 4),
                "ratings_as_of": pd.Timestamp(day).normalize(),
                "unrated": pred is None,
            }
            if pred is not None:
                for k in ("home_pred", "away_pred", "likely_score",
                          "p_home", "p_draw", "p_away", "p_over_2_5", "p_btts"):
                    rec[k] = pred[k]
            if elo_h is not None and elo_a is not None:
                imp = implied_1x2(elo_h, elo_a)
                rec["elo_p_home"] = round(imp["p_home"], 4)
                rec["elo_p_draw"] = round(imp["p_draw"], 4)
                rec["elo_p_away"] = round(imp["p_away"], 4)
            rows.append(rec)

    return pd.DataFrame(rows, columns=_UEFA_RESULT_COLUMNS)


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
