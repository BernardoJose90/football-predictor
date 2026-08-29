"""Predict real, not-yet-played fixtures for the next few days.

    python -m scripts.predict_upcoming [--days 4] [--stat sot] [--refresh]

Pulls this weekend's card from football-data.co.uk's free fixtures.csv
(ingest/fixtures.py), builds each league's ratings as of today from
data/processed/matches (rebuilding it first with --refresh), and prints a
prediction next to that same fixture's own pre-match market price for
reference. A team with no rating yet (newly promoted, or too few matches this
season - the min_matches guard) is shown as unrated rather than guessed at,
per the design doc's rule against defaulting to a league-average prior.

Referee identity, rest days, and travel distance (section 10.1 ranks 1-3) are
ON by default here, matching config.py / evaluate.backtest.BacktestConfig -
each individually raised RPS in this repo's own walk-forward test (see the
README's Milestone 4 section), kept on anyway per an explicit product
decision. Use --no-referee / --no-rest / --no-travel to get the plain,
better-tested model back.

Squad market value (rank 4) is OFF by default here (see config.py) - it
worked (turned every "UNRATED" fixture into a rated one) but was switched off
by an explicit product decision: a team without enough history goes back to
being excluded rather than priced from money. --use-squad-value opts back in.

Premier-League-only injury/availability data (Fantasy Premier League's own
free API, ingest/fpl.py) is OFF by default and, unlike everything else in
this file, CANNOT be walk-forward tested against RPS at all - there's no free
historical archive of pre-match availability to test it against, only ever
live/current data. --use-injuries opts in as a live-only experiment; it has
no accuracy claim behind it the way diagonal inflation or even referee/rest/
travel do.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import config
from evaluate.baselines import devig
from ingest import fixtures as fixtures_mod
from ingest import fpl as fpl_mod
from ingest import historical
from ingest import squad_value as squad_value_mod
from model.injuries import injury_factor
from model.predict import expected_goals, predict_match
from model.ratings import build_ratings
from model.referee import build_referee_factors
from model.rest import rest_factor
from model.travel import travel_factor, trip_distance_km
from normalise import schema
from normalise.teams import UnknownTeamError, resolve


def _rebuild_dataset() -> pd.DataFrame:
    """Download, re-seed team aliases, THEN normalise.

    Order matters: normalise.schema.normalise() raises on any team name not
    yet in teams.yaml, and a promoted/relegated team's name only appears once
    its season's CSV has been downloaded. Aliases must be re-seeded from the
    fresh CSVs before normalising, not after.
    """
    print("Refreshing historical data (incl. in-progress season)...", file=sys.stderr)
    historical.download_all(force=True)

    from normalise import build_aliases
    from normalise.teams import reload_cache
    build_aliases.main()
    reload_cache()

    raw = historical.load_all()
    matches = schema.normalise(raw)
    # Join whatever xG has already been pulled (python -m ingest.understat) -
    # a no-op if that file doesn't exist yet. Without this, --refresh would
    # silently wipe home_xg/away_xg back to NaN on every run, breaking
    # stat="auto"'s xg-first behaviour right after the data that made it work.
    from ingest import understat
    matches = understat.join(matches)
    matches.to_csv(config.DATA_PROCESSED / "matches.csv", index=False)
    try:
        matches.to_parquet(config.DATA_PROCESSED / "matches.parquet", index=False)
    except Exception:
        pass
    return matches


def _load_matches() -> pd.DataFrame:
    pq = config.DATA_PROCESSED / "matches.parquet"
    csv = config.DATA_PROCESSED / "matches.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise SystemExit("no dataset - run with --refresh first")
    df["date"] = pd.to_datetime(df["date"])
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=4, help="lookahead window from --start")
    ap.add_argument("--start", default=None,
                    help="override the window start (default: today). Ratings are also built "
                         "as_of this date, so a past date still only sees strictly-earlier "
                         "matches - useful for re-predicting a specific past matchday, e.g. "
                         "one the results feed hasn't posted yet even though the fixtures.csv "
                         "listing still carries it")
    ap.add_argument("--stat", choices=config.STAT_CHOICES, default=config.DEFAULT_STAT)
    ap.add_argument("--xi", type=float, default=config.DEFAULT_XI)
    ap.add_argument("--rho", type=float, default=config.DEFAULT_RHO)
    ap.add_argument("--delta", type=float, default=config.DEFAULT_DELTA,
                    help="diagonal-inflation strength on the Dixon-Coles grid (0 = off)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download historical CSVs (incl. current season) and rebuild teams.yaml first")
    ap.add_argument("--no-referee", dest="use_referee", action="store_false",
                    default=config.DEFAULT_USE_REFEREE,
                    help="disable the referee-identity adjustment (section 10.1 rank 1 - "
                         "raised RPS in testing; ON by default per product decision, see README)")
    ap.add_argument("--no-rest", dest="use_rest", action="store_false",
                    default=config.DEFAULT_USE_REST,
                    help="disable the days-since-last-match fatigue adjustment (section 10.1 rank 2 - "
                         "raised RPS in testing; ON by default per product decision)")
    ap.add_argument("--no-travel", dest="use_travel", action="store_false",
                    default=config.DEFAULT_USE_TRAVEL,
                    help="disable the away-travel-distance adjustment (section 10.1 rank 3 - "
                         "raised RPS in testing; ON by default per product decision)")
    ap.add_argument("--use-squad-value", dest="use_squad_value", action="store_true",
                    default=config.DEFAULT_USE_SQUAD_VALUE,
                    help="enable the squad-market-value prior for unrated teams (section 10.1 "
                         "rank 4 - OFF by default per product decision; a team with too little "
                         "history is excluded (UNRATED) rather than priced from squad value)")
    ap.add_argument("--no-squad-value", dest="use_squad_value", action="store_false",
                    help="explicitly disable it (redundant while the default is off, kept for symmetry)")
    ap.add_argument("--refresh-squad-value", action="store_true",
                    help="re-pull squad values from Transfermarkt first (python -m ingest.squad_value)")
    ap.add_argument("--use-injuries", action="store_true", default=config.DEFAULT_USE_INJURIES,
                    help="enable the Fantasy-Premier-League injury/availability adjustment - "
                         "Premier League only, OFF by default, CANNOT be RPS-tested (no free "
                         "historical archive of pre-match availability exists) - a live-only "
                         "experiment, not an evidence-backed feature like the others")
    ap.add_argument("--no-injuries", dest="use_injuries", action="store_false",
                    help="explicitly disable it (redundant while the default is off)")
    ap.add_argument("--injury-k", type=float, default=config.INJURY_K)
    ap.add_argument("--referee-min-matches", type=int, default=config.REFEREE_MIN_MATCHES)
    ap.add_argument("--rest-k", type=float, default=config.REST_K)
    ap.add_argument("--travel-k", type=float, default=config.TRAVEL_K)
    ap.add_argument("--value-prior-min-points", type=int, default=config.VALUE_PRIOR_MIN_POINTS)
    ap.add_argument("--out", default=None,
                    help="output CSV path (default: artefacts/upcoming_predictions.csv). "
                         "A richer per-fixture .json (ratings, per-adjustment breakdown, "
                         "pre-adjustment probabilities) is always written alongside it, "
                         "with the same stem - that's what scripts.render_why reads.")
    ap.add_argument("--log-predictions", action="store_true",
                    help="append this run's rated fixtures to artefacts/prediction_log.csv "
                         "(first prediction per fixture wins, never overwritten) - the standing "
                         "record scripts.track_record scores. scripts.render_coupon passes this "
                         "so the published page and the track record stay in lockstep.")
    args = ap.parse_args(argv)

    today = pd.Timestamp(args.start).normalize() if args.start else pd.Timestamp.now().normalize()
    end = today + pd.Timedelta(days=args.days)

    squad_values: dict[str, float] = {}
    if args.use_squad_value:
        if args.refresh_squad_value:
            squad_value_mod.build()
        squad_values = squad_value_mod.load()
        if not squad_values:
            print("note: no squad-value data on disk - run `python -m ingest.squad_value` "
                  "or pass --refresh-squad-value. Continuing without it.", file=sys.stderr)

    pl_availability: dict[str, dict] = {}
    if args.use_injuries:
        try:
            pl_availability = fpl_mod.team_availability(
                fpl_mod.fetch_bootstrap(), min_importance=config.INJURY_MIN_IMPORTANCE)
        except Exception as exc:  # noqa: BLE001 - a live third-party API call; degrade, don't crash
            print(f"note: could not fetch FPL data ({exc}). Continuing without injuries.", file=sys.stderr)

    matches = _rebuild_dataset() if args.refresh else _load_matches()

    raw_fixtures = fixtures_mod.upcoming(leagues=set(config.LEAGUES), start=today, end=end)
    if raw_fixtures.empty:
        print(f"No fixtures found for {today.date()}..{end.date()} in {sorted(config.LEAGUES)}.",
              file=sys.stderr)
        return 1

    # Resolve team names and pull each fixture's own pre-match market price in
    # one pass. An unresolved name is reported, not fatal - a single unmapped
    # promoted-team spelling shouldn't block every other prediction this
    # weekend, and --refresh (which re-seeds teams.yaml from the fresh CSVs)
    # is almost always the actual fix.
    odds_cols = [("PSH", "PSD", "PSA"), ("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA")]
    resolved_rows = []
    market = {}
    unresolved = []
    for row in raw_fixtures.itertuples(index=False):
        try:
            home = resolve(row.HomeTeam)
            away = resolve(row.AwayTeam)
        except UnknownTeamError as exc:
            unresolved.append((row.Div, row.HomeTeam, row.AwayTeam, str(exc)))
            continue
        resolved_rows.append({
            "div": row.Div, "league": row.league, "date": row.date,
            "home_team": home, "away_team": away,
            "referee": getattr(row, "Referee", None),
        })
        for h, d, a in odds_cols:
            if hasattr(row, h) and hasattr(row, d) and hasattr(row, a):
                try:
                    oh, od, oa = float(getattr(row, h)), float(getattr(row, d)), float(getattr(row, a))
                    market[(row.Div, home, away)] = devig(oh, od, oa)
                    break
                except (TypeError, ValueError):
                    continue

    if unresolved:
        print(f"\n{len(unresolved)} fixture(s) skipped - unresolved team name "
              f"(run with --refresh, or add to normalise/teams.yaml):", file=sys.stderr)
        for div, h, a, _ in unresolved:
            print(f"  [{div}] {h} vs {a}", file=sys.stderr)

    fx = pd.DataFrame(resolved_rows)
    if fx.empty:
        print("No resolvable fixtures.", file=sys.stderr)
        return 1

    adjustments_off = not (args.use_referee or args.use_rest or args.use_travel
                          or args.use_squad_value or args.use_injuries)
    print(f"\nUpcoming fixtures {today.date()}..{end.date()}  "
          f"(stat={args.stat}, xi={args.xi}, rho={args.rho}, delta={args.delta}"
          f"{', referee ON' if args.use_referee else ', referee off'}"
          f"{', rest ON (k=' + str(args.rest_k) + ')' if args.use_rest else ', rest off'}"
          f"{', travel ON (k=' + str(args.travel_k) + ')' if args.use_travel else ', travel off'}"
          f"{', squad-value ON (' + str(len(squad_values)) + ' clubs)' if args.use_squad_value else ', squad-value off'}"
          f"{', injuries ON (' + str(len(pl_availability)) + ' PL clubs, k=' + str(args.injury_k) + ')' if args.use_injuries else ', injuries off'})\n")

    # Rest days: each team's most recent PLAYED match across ALL divisions
    # (a team's last match may have been in a different division after
    # promotion/relegation). Every row in `matches` is already in the past by
    # construction, so this needs no as_of filtering of its own. Travel also
    # needs the away side's rest days to weight the distance penalty, so
    # either flag triggers this.
    last_match_by_team = {}
    if args.use_rest or args.use_travel:
        long = pd.concat([
            matches[["date", "home_team"]].rename(columns={"home_team": "team"}),
            matches[["date", "away_team"]].rename(columns={"away_team": "team"}),
        ])
        last_match_by_team = long.groupby("team")["date"].max().to_dict()

    results = []
    for div, group in fx.groupby("div"):
        league_matches = matches[matches["div"] == div]
        # stat="auto": try xg first, fall back to sot for a division that
        # doesn't have it (e.g. E1/SC0/P1) - see config.py for why this is
        # the default now (matched, paired, significant comparison).
        stats_to_try = ((config.AUTO_STAT_PRIMARY, config.AUTO_STAT_FALLBACK)
                        if args.stat == "auto" else (args.stat,))
        snap, last_exc, stat_used = None, None, None
        for candidate_stat in stats_to_try:
            try:
                snap = build_ratings(league_matches, as_of=today, stat=candidate_stat,
                                     xi=args.xi, min_matches=config.DEFAULT_MIN_MATCHES,
                                     squad_values=squad_values if args.use_squad_value else None,
                                     value_prior_min_points=args.value_prior_min_points)
                stat_used = candidate_stat
                break
            except ValueError as exc:
                last_exc = exc
        if snap is None:
            print(f"[{config.LEAGUES.get(div, div)}] no ratings available ({last_exc})")
            continue
        if args.stat == "auto":
            print(f"[{config.LEAGUES.get(div, div)}] using stat={stat_used}")

        ref_factors = None
        if args.use_referee:
            ref_factors = build_referee_factors(
                league_matches, as_of=today, xi=args.xi, min_matches=args.referee_min_matches,
            )

        print(f"--- {config.LEAGUES.get(div, div)} ---")
        for row in group.sort_values("date").itertuples(index=False):
            lam_mult, mu_mult = 1.0, 1.0
            adj_note = []
            # Structured, machine-readable twin of adj_note: one entry per
            # adjustment that actually moved the number, with its direction and
            # size, so scripts.render_why can narrate the model/market gap
            # instead of re-deriving it. Each factor > 1 lifts that side's
            # expected goals, < 1 suppresses it.
            adjustments = []
            if args.use_referee and ref_factors is not None:
                rh, ra = ref_factors.factor(row.referee)
                if (rh, ra) != (1.0, 1.0):
                    lam_mult *= rh
                    mu_mult *= ra
                    adj_note.append(f"ref={row.referee}({rh:.2f}/{ra:.2f})")
                    adjustments.append({"kind": "referee", "detail": str(row.referee),
                                        "home_factor": round(rh, 4), "away_factor": round(ra, 4)})
            home_rest = away_rest = None
            if args.use_rest or args.use_travel:
                home_last = last_match_by_team.get(row.home_team)
                away_last = last_match_by_team.get(row.away_team)
                home_rest = (row.date.normalize() - home_last).days if pd.notna(home_last) else None
                away_rest = (row.date.normalize() - away_last).days if pd.notna(away_last) else None
            if args.use_rest:
                rf_h, rf_a = rest_factor(home_rest, k=args.rest_k), rest_factor(away_rest, k=args.rest_k)
                lam_mult *= rf_h
                mu_mult *= rf_a
                if home_rest is not None or away_rest is not None:
                    adj_note.append(f"rest={home_rest}d/{away_rest}d")
                if (rf_h, rf_a) != (1.0, 1.0):
                    adjustments.append({"kind": "rest",
                                        "detail": f"{home_rest}d home / {away_rest}d away",
                                        "home_factor": round(rf_h, 4), "away_factor": round(rf_a, 4)})
            if args.use_travel:
                dist = trip_distance_km(row.home_team, row.away_team)
                tf = travel_factor(dist, rest_days=away_rest, k=args.travel_k)
                mu_mult *= tf
                if dist is not None:
                    adj_note.append(f"travel={round(dist)}km({tf:.2f})")
                    if tf != 1.0:
                        adjustments.append({"kind": "travel", "detail": f"{round(dist)} km",
                                            "home_factor": 1.0, "away_factor": round(tf, 4)})
            if snap.is_prior(row.home_team):
                adj_note.append(f"home rated by squad value (€{squad_values[row.home_team]/1e6:.0f}m)")
            if snap.is_prior(row.away_team):
                adj_note.append(f"away rated by squad value (€{squad_values[row.away_team]/1e6:.0f}m)")
            if args.use_injuries and div == "E0":
                home_avail = pl_availability.get(row.home_team)
                away_avail = pl_availability.get(row.away_team)
                hf = injury_factor(home_avail["availability"] if home_avail else None, k=args.injury_k)
                af = injury_factor(away_avail["availability"] if away_avail else None, k=args.injury_k)
                lam_mult *= hf
                mu_mult *= af
                if home_avail and home_avail["n_flagged"]:
                    adj_note.append(f"home availability {home_avail['availability']:.0%} "
                                    f"({home_avail['n_flagged']} flagged, {hf:.2f})")
                if away_avail and away_avail["n_flagged"]:
                    adj_note.append(f"away availability {away_avail['availability']:.0%} "
                                    f"({away_avail['n_flagged']} flagged, {af:.2f})")
                if hf != 1.0 or af != 1.0:
                    adjustments.append({
                        "kind": "injuries",
                        "detail": (f"home {home_avail['availability']:.0%} avail" if hf != 1.0 else "")
                                  + (" / " if hf != 1.0 and af != 1.0 else "")
                                  + (f"away {away_avail['availability']:.0%} avail" if af != 1.0 else ""),
                        "home_factor": round(hf, 4), "away_factor": round(af, 4)})

            pred = predict_match(snap, row.home_team, row.away_team, rho=args.rho,
                                 delta=args.delta, lam_mult=lam_mult, mu_mult=mu_mult)
            mkt = market.get((div, row.home_team, row.away_team))
            when = row.date.strftime("%a %d %b %H:%M") if pd.notna(row.date) else "?"
            if pred is None:
                print(f"  {when}  {row.home_team:20s} v {row.away_team:20s}  "
                      f"UNRATED (not enough history yet)")
                results.append({"league": row.league, "date": row.date,
                                "home_team": row.home_team, "away_team": row.away_team,
                                "unrated": True})
                continue

            # The same fixture priced with the section-10.1 adjustments turned
            # off (lam_mult=mu_mult=1) - the "pure ratings + Dixon-Coles" number.
            # render_why shows this next to the adjusted one so a reader can see
            # how much of any model/market gap is the ratings vs the nudges.
            base_pred = predict_match(snap, row.home_team, row.away_team,
                                      rho=args.rho, delta=args.delta)
            base_lam, base_mu = expected_goals(snap, row.home_team, row.away_team)
            line = (f"  {when}  {row.home_team:20s} v {row.away_team:20s}  "
                    f"model: {pred['p_home']*100:4.1f}% / {pred['p_draw']*100:4.1f}% / "
                    f"{pred['p_away']*100:4.1f}%  (likely {pred['likely_score']})")
            if mkt:
                line += (f"   market: {mkt['p_home']*100:4.1f}% / {mkt['p_draw']*100:4.1f}% / "
                         f"{mkt['p_away']*100:4.1f}%")
            if adj_note:
                line += "   [" + ", ".join(adj_note) + "]"
            print(line)
            hg = snap.teams.get(row.home_team, {}).get("matches")
            ag = snap.teams.get(row.away_team, {}).get("matches")
            rec = {"league": row.league, "date": row.date, "home_team": row.home_team,
                   "away_team": row.away_team, "unrated": False,
                   "match_id": schema.make_match_id(div, row.date, row.home_team, row.away_team),
                   "lam_mult": lam_mult, "mu_mult": mu_mult,
                   "adj_note": ", ".join(adj_note) if adj_note else "",
                   "adjustments": adjustments,
                   "stat_used": stat_used,
                   "home_attack": round(snap.attack(row.home_team), 4),
                   "home_defence": round(snap.defence(row.home_team), 4),
                   "away_attack": round(snap.attack(row.away_team), 4),
                   "away_defence": round(snap.defence(row.away_team), 4),
                   "home_matches_used": int(hg) if hg is not None else None,
                   "away_matches_used": int(ag) if ag is not None else None,
                   "lg_home_goals": round(snap.lg_home_goals, 4),
                   "lg_away_goals": round(snap.lg_away_goals, 4),
                   "base_home_pred": round(float(base_lam), 3),
                   "base_away_pred": round(float(base_mu), 3),
                   "base_p_home": base_pred["p_home"], "base_p_draw": base_pred["p_draw"],
                   "base_p_away": base_pred["p_away"],
                   **pred}
            if mkt:
                rec.update({f"market_{k}": v for k, v in mkt.items()})
            results.append(rec)
        print()

    out = pd.DataFrame(results)
    out_path = args.out or (config.ARTEFACTS / "upcoming_predictions.csv")
    out.drop(columns=["adjustments"], errors="ignore").to_csv(out_path, index=False)
    def _jsonable(rec: dict) -> dict:
        d = dict(rec)
        if isinstance(d.get("date"), pd.Timestamp):
            d["date"] = d["date"].strftime("%Y-%m-%dT%H:%M")
        return d

    json_path = Path(out_path).with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump([_jsonable(r) for r in results], fh, default=str, indent=2)
    print(f"-> {out_path}")
    print(f"-> {json_path}")

    if args.log_predictions:
        from evaluate import prediction_log
        n_new = prediction_log.append(results)
        print(f"-> prediction_log.csv (+{n_new} new fixture(s))")

    if adjustments_off:
        print("(plain model - referee/rest/travel adjustments disabled via --no-*; "
              "these are ON by default, see README Milestone 4 section)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
