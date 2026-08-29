# Football Match Predictor

Weighted-counting attack/defence ratings + a Dixon-Coles Poisson grid, validated
walk-forward against the devigged bookmaker closing line and a Club Elo proxy.
No machine learning, no training loop. See the project doc you supplied for the
full design rationale (metric choice, decay constant, roadmap, risks); this file
is the "what actually got built" companion to it.

## Status

**Milestone 4 done**: ingest, normalise, ratings, Dixon-Coles prediction,
RPS/log-loss/calibration evaluation, a walk-forward backtest, a completed
xi-tuning sweep (default updated from the doc's literature value to the
tuned one - see below), the sot/goals input-stat comparison, and the first
two ranked section 10.1 feature candidates (referee identity, rest days)
implemented, tested, and rejected with reasoning. Section 3 criteria 3-6 are
met on the evaluation window. 60 tests pass, including leakage-guard tests
for ratings, referee factors, and rest-day computation.

Also built past the original Milestone-4-only scope: `ingest/fixtures.py` and
`scripts/predict_upcoming.py` predict real not-yet-played fixtures from
football-data.co.uk's free fixtures feed, alongside that same fixture's
market price - useful for eyeballing the model against reality without
waiting for a backtest window to close.

Not built: Milestone 5/6 (containerised scheduling, FastAPI service) and the
Postgres schema from section 8 - this repo runs off Parquet/CSV files under
`data/`, which is enough to prove the model out. Wire in the DB and API once
the numbers earn it, per the doc's own sequencing.

## Quickstart

```bash
python3.13 -m venv .venv && source .venv/bin/activate   # see "A Python 3.14 note" below
pip install -r requirements.txt

# Milestone 1: one command -> a clean match table
python -m ingest.historical              # download football-data.co.uk CSVs
python -m normalise.build_aliases        # seed normalise/teams.yaml
python -m scripts.build_dataset          # -> data/processed/matches.{csv,parquet}

# Milestone 2 + 4: baselines, then the model, on an untouched evaluation window
python -m scripts.run_backtest --eval-start 2025-08-01 --stat sot

# Milestone 4: tune xi and compare xg/sot/goals on an EARLIER, separate window
python -m scripts.run_tune --tune-start 2024-08-01 --tune-end 2025-06-30

# Predict real upcoming fixtures (needs --refresh once, to pull the in-progress
# season and re-seed team aliases for any newly promoted sides)
python -m scripts.predict_upcoming --refresh --days 4

pytest -q
```

### A Python 3.14 note

This machine's Homebrew Python 3.14 has a broken `pyexpat` (a libexpat symbol
mismatch), which breaks `platform.mac_ver()`, which breaks pip's vendored
`truststore` SSL context, which breaks every `pip install` - including inside
a fresh venv. Python 3.13 doesn't have this problem. The venv above was built
against `python3.13` for that reason; there's nothing project-specific about
it, so switch back to `python3` once your system Python is fixed.

## What's covered by data

8 leagues x seasons 2023/24-2025/26 (the most recent 3 complete seasons as of
Aug 2026), pulled from football-data.co.uk: Premier League (E0), Championship
(E1), Scottish Premiership (SC0), La Liga (SP1), Bundesliga (D1), Serie A (I1),
Ligue 1 (F1), Primeira Liga (P1). 8,514 played matches, 100% with closing odds,
99.99% with shots-on-target.

Expected goals is now populated: `python -m ingest.understat` (via `soccerdata`)
pulls real Understat xG for the 5 leagues it covers (E0, SP1, D1, I1, F1 - not
E1/SC0/P1), 5,240 matches joined. Default `--stat` is still `sot` (shots on
target) for the live pipeline, since it's the one with full 8-league coverage -
see Milestone 4 below for why `xg` isn't the default despite scoring better.

## Milestone 4: tuning and feature tests

**xi.** Swept 0 to 0.010 on a tuning window (2024-08-01..2025-06-30) kept
separate from the evaluation window below - the doc's own warning against
tuning and reporting on the same seasons. Clean, shallow U-curve, minimum at
**xi=0.0035** (RPS 0.2017 vs 0.2023 at xi=0, vs 0.2031 at xi=0.01) - see
`artefacts/xi_sweep.png`. That's roughly double the doc's literature-derived
0.0018 (Dixon & Coles 1997, converted from half-weeks), consistent with the
doc's own expectation that today's heavier fixture load calls for faster
decay. Set as the new `config.DEFAULT_XI`.

**Input stat.** First pass (`goals` 0.2028, `sot` 0.2017 on all 8 leagues,
n=2659) couldn't include `xg` on equal terms - Understat only covers 5/8
leagues, so an 8-league `sot` number isn't a fair comparison against a
5-league `xg` number (different, and on average less predictable, leagues).
Redone properly: `xg` vs `sot`, **same 5 leagues** (E0/SP1/D1/I1/F1), same
tuning window, same exact 1,644 fixtures both sides (verified: 0 dropped
either way), paired per match:

| stat | RPS (matched 5-league set) | log loss |
|---|---|---|
| `sot` | 0.20275 | - |
| `xg` | **0.19778** | - |
| mean paired difference | **0.00497** (xg lower/better) | |

Bootstrap 95% CI on the mean difference: **[0.0027, 0.0073]** - excludes
zero. Wilcoxon signed-rank p=5.3e-08, paired t-test p=1.9e-05. `xg` wins,
genuinely, not as an artefact of comparing different leagues.

**Promoted to the live default as `stat="auto"`**: tries `xg` first per
division, falls back to `sot` if that division has no xg data (E1/SC0/P1 -
Understat's gap, not fixed by this). Resolved in
`evaluate.backtest.backtest_league` / `scripts.predict_upcoming`, never
passed into `model.ratings.build_ratings` directly - that function only
understands concrete stats. Confirmed on the real weekend card: Bundesliga,
Premier League, Ligue 1, Serie A, La Liga all report "using stat=xg";
Championship, Primeira Liga, Scottish Premiership report "using stat=sot".
`config.AUTO_STAT_PRIMARY`/`AUTO_STAT_FALLBACK` control which stats;
`--stat sot` / `--stat xg` still force a single stat everywhere if you want
the old fixed behaviour back.

**Section 10.1 feature candidates - all four now implemented and tested:**

| Candidate | Verdict | Evidence |
|---|---|---|
| Referee identity (`model/referee.py`) | Tested worse, **kept ON anyway** | RPS +0.0008 to +0.0017 across every decay/min-matches setting tried |
| Days since last match (`model/rest.py`) | Tested worse, **kept ON anyway** | RPS rises monotonically as the fatigue penalty `k` increases from 0; `k=0` exactly reproduces baseline, confirming the plumbing (not a bug) |
| Travel distance (`model/travel.py`) | Tested worse, **kept ON anyway** | RPS 0.20194 vs 0.20172 baseline, alone |
| Squad market value (`model/squad_value.py`) | Built, tested, **OFF by product decision** | Not RPS-testable at all (see below) - turned off because a team without enough history should stay excluded, not priced from money |

All three RPS-tested candidates raised RPS individually; combined (referee +
rest + travel together) RPS is **0.20378** vs a **0.20172** pure baseline -
worse by design terms, kept on anyway per an explicit decision to prioritise
using more of the real-world signal bookmakers use over this one narrow
metric, with the tradeoff written down rather than hidden. `--no-referee`,
`--no-rest`, `--no-travel` on `scripts.predict_upcoming` (or the matching
`use_*=False` on `BacktestConfig`) get the better-tested plain model back.
The doc's own methodology note still applies: "referees differ in home
advantage produced" (Nevill, Balmer & Williams 2007) is a real, separately-
replicated finding about raw goal/card differentials - it just doesn't
translate into better match-outcome *forecasts* layered on an already-tuned
rating model at this sample size (3 seasons).

All four share the same architecture: a generic `lam_mult`/`mu_mult` hook on
`model.predict.predict_match` that multiplies expected goals after the
attack/defence calculation, so each adjustment composes independently and a
disabled one is a true no-op (multiply by 1.0).

**Squad market value is different in kind, not just verdict** - it's a
fallback for teams with too little match history, not a per-fixture nudge to
an existing prediction, so "does it improve RPS" isn't even the right
question for it (see `ingest/squad_value.py` and `model/squad_value.py`).
Real data, scraped from Transfermarkt's per-league market-value pages (8
requests, one per league, not per-club) via `python -m ingest.squad_value`,
resolved through `ingest/transfermarkt_aliases.yaml` (150/150 clubs matched).
Fits `attack ~ a + b*log(value)` from teams that already have a normal
rating, then applies it to teams that don't. Demonstrated working on the
actual weekend card - coverage went from 61/72 (85%) to 72/72 (100%), every
previously "UNRATED" fixture (Real Madrid v Malaga, Wolves v Stoke, Coventry
v Hull, etc.) got a real prediction instead of being excluded - **then
turned off by explicit product decision**: `config.DEFAULT_USE_SQUAD_VALUE`
is `False`, so a team without enough history is UNRATED again, same as
before this feature existed. The code, tests, and Transfermarkt data stay in
the repo; `--use-squad-value` on `scripts.predict_upcoming` opts back in.
Never wired into `evaluate.backtest.BacktestConfig`'s default either way
(`use_squad_value=False` there always) since the value data is a single
current-day snapshot - scoring a 2023/24 match with 2026 squad values would
be look-ahead bias, not a fair backtest, regardless of the live-use decision.

## Post-Milestone-4: diagonal-inflated bivariate Poisson

Sourced from literature research, not the original doc's own section 10.1
list: **Karlis & Ntzoufras (2003)**'s diagonal-inflated bivariate Poisson,
used again in **Egidi et al., "Bayesian weighted discrete-time dynamic models
for association football prediction," JRSS Series C (2026)**, which reports
RPS 0.189 on Bundesliga/EPL/La Liga. That paper actually bundles two separate
techniques - diagonal inflation, and a full Bayesian adaptive time-weighting
scheme fit via MCMC (spike-and-slab hyperpriors, replacing a fixed xi
entirely). Only the first is built here: the second is a different fitting
paradigm (Bayesian, iterative) from this codebase's weighted-counting-plus-
tunable-parameter approach throughout, and would mean taking on a
probabilistic-programming dependency (Stan/PyMC) - a separate, larger
decision, not folded in silently.

**What it is**: Dixon-Coles' own correction (`model.dixon_coles.dc_tau`) only
touches four cells - 0-0, 1-0, 0-1, 1-1. Plain and DC-corrected Poisson both
still underestimate draws at higher scorelines (2-2, 3-3, ...), because
nothing touches those cells. `diagonal_inflate()` generalises the idea to the
*whole* diagonal: every draw cell, at every scoreline, inflated
proportionally by `(1 + delta)`, then renormalised. `delta=0` is an exact
no-op - same discipline as xi and rho, tuned on the tuning window rather than
taken from the paper (which doesn't fit delta as a plain grid-search
constant the way this does).

**Result - unlike every section 10.1 candidate, this one actually helped:**

| delta | RPS (current default config) |
|---|---|
| 0.00 (off) | 0.20064 |
| 0.10 | 0.20040 |
| 0.15 | 0.20033 |
| **0.20** | **0.20031** (minimum) |
| 0.30 | 0.20036 |
| 0.50 | 0.20081 |
| 1.00 | 0.20542 |

Clean U-shape, minimum at **delta=0.20**, set as `config.DEFAULT_DELTA`. A
modest improvement (-0.00033 RPS) but a real, positive one - referee, rest,
and travel all made things worse and were kept anyway by decision; this is
the first section-10.1-style addition that earned its place on the metric
itself. Re-verified section 3 criteria 3-6 still hold with it on: RPS 0.2044
(< 0.21), beats Elo, within 0.01 of the closing line (gap +0.0067) -
calibration error on P(home win) actually *improved*, 0.0234 -> 0.0155.

## Premier-League-only injury/availability (live-only, untestable, off by default)

`ingest/fpl.py` + `model/injuries.py`. Source is Fantasy Premier League's own
free, official, keyless API (`fantasy.premierleague.com/api/bootstrap-static/`)
- confirmed live, current-season data (Gameweek 2, 2026/27), not stale.
Covers Premier League only; FPL doesn't exist for the other 7 leagues.

**Cannot be RPS-tested at all**, not just "not tested yet" like squad value:
the free historical FPL archive (vaastav/Fantasy-Premier-League on GitHub)
only ever recorded post-match performance, never pre-match availability, for
any season - so there is no historical ground truth to walk-forward test
against. `--use-injuries` on `scripts.predict_upcoming` is a live-only
experiment; `DEFAULT_USE_INJURIES = False`.

**A real bug found and fixed before shipping this**: the first version
weighted each player's importance by this season's `total_points`. That's
wrong for exactly the players the feature exists to catch - a player injured
before playing a single minute this season (Arsenal's Saliba, the case that
caught it) has `total_points=0`, so weighting by points made the model treat
its own best defender as unimportant *because* he'd been out all season.
Fixed by weighting on FPL's own `now_cost` (transfer-market price) instead,
which reputable players keep even while absent. Regression test:
`test_season_long_absentee_with_zero_points_still_weighted_by_cost`.

## Evaluation

`python -m scripts.run_backtest --eval-start 2025-08-01 --stat sot` (walk-forward
across all 8 leagues, ratings built only from strictly-prior matches, tuned
xi=0.0035):

| | RPS | log loss | n |
|---|---|---|---|
| devigged closing line | 0.1978 | 0.979 | 2816 |
| Club Elo (self-fit proxy, see below) | 0.2053 | 1.004 | 2816 |
| **this model** | **0.2039** | **1.004** | 2816 |

- RPS < 0.21: **yes** (0.2039, inside the 0.184-0.213 published band)
- beats the Elo proxy on the same fixtures: **yes**
- within 0.01 RPS of the devigged closing line: **yes** (gap +0.0062)
- coverage: 95.5% (the rest are early-season/promoted teams under the
  `min_matches=8` guard - by design, not a bug; coverage dipped slightly from
  earlier runs because this window now reaches into the very first weeks of
  three new seasons at once)
- calibration error (P(home win), 10 bins): 0.028 - see `artefacts/calibration.png`

**On the Club Elo row**: the design doc's baseline is the real clubelo.com
rating pulled via `soccerdata`. To keep this repo dependency-free and fully
reproducible offline, `evaluate/baselines.py` instead fits a standard
goal-difference-weighted Elo (Hvattum & Arntzen 2010 style) walk-forward on
the same match history. Swap in true Club Elo (`soccerdata.ClubElo`) if you
want the literal number the doc names.

## Repository layout

Matches `football-predictor/` in the design doc section 7, minus `api/` and
`infra/` (not built yet):

```
config.py            league codes, seasons, model defaults incl. section 10.1 feature flags
ingest/
  historical.py       football-data.co.uk CSV download + load
  fixtures.py          upcoming fixtures, football-data.co.uk's free fixtures.csv
  understat.py         xG pull (soccerdata) - 5/8 leagues, run once to populate
  squad_value.py        Transfermarkt scrape, 8 requests (one per league)
  transfermarkt_aliases.yaml   Transfermarkt name -> canonical name (150 clubs)
  fpl.py                Fantasy Premier League API - live injury/availability, PL only
normalise/
  teams.py, teams.yaml canonical team resolution, raises on unknown names
  build_aliases.py     seeds teams.yaml from observed CSV names
  schema.py            raw rows -> unified match record
model/
  ratings.py           attack/defence, stat-agnostic, leakage-guarded, squad-value fallback
  dixon_coles.py        tau correction, diagonal inflation, + scoreline grid
  predict.py            fixture -> probabilities, generic lam_mult/mu_mult hook
  referee.py             section 10.1 rank 1 - tested worse, kept ON (see Milestone 4)
  rest.py                 section 10.1 rank 2 - tested worse, kept ON
  travel.py               section 10.1 rank 3 - tested worse, kept ON
  stadiums.py             team -> (lat, lon), 189 teams, for travel.py
  squad_value.py          section 10.1 rank 4 - value->rating prior fit, not RPS-tested
  injuries.py             PL-only, live-only, cannot be RPS-tested, OFF by default
evaluate/
  metrics.py            RPS, log loss, calibration
  baselines.py          devig, Elo proxy
  backtest.py            walk-forward runner (referee/rest/travel ON by default, squad-value OFF)
  tune.py                 xi sweep + plot, stat comparison
scripts/
  build_dataset.py, run_backtest.py, run_tune.py, predict_upcoming.py    entry points
tests/                 104 tests incl. leakage-guard tests for ratings, referee, rest, travel
data/raw/, data/processed/, artefacts/
```

## Design decisions worth flagging

- **Input stat is a parameter**, not a hardcoded column (`stat="sot"|"goals"|"xg"`
  throughout `ratings.py`/`predict.py`/`backtest.py`), per section 4.5/11.
- **Leakage guard**: `build_ratings` only ever reads rows with `date < as_of`;
  `tests/test_ratings.py::test_leakage_guard_ignores_matches_on_or_after_as_of`
  poisons a future row with an absurd scoreline and asserts ratings don't move.
- **Unknown team names raise**, never fuzzy-match (`normalise/teams.py`) -
  a silent mismatch produces a confident prediction for the wrong fixture.
- **Unrated teams return `None`**, not a league-average guess - newly promoted
  teams and small early-season samples are excluded via `min_matches=8` rather
  than defaulting to attack/defence = 1.0.
- **Closing-odds column choice**: Pinnacle closing (`PSCH/D/A`) preferred as the
  sharpest book, falling back to market-average and Bet365 closing, then to
  pre-close prices as a last resort (`normalise/schema.py::_pick_closing_odds`).

## Not built (by choice, this pass)

- Postgres schema (section 8) - flat files were enough to prove the model.
- `api/` FastAPI service, containerisation, scheduling (Milestones 5-6).
- Historical (as-of-season) squad values - only a current-day Transfermarkt
  snapshot exists, which is why squad value is live-only, not backtested.
- xG for Championship/Scottish Premiership/Primeira Liga - Understat simply
  doesn't cover them; `stat="auto"` falls back to `sot` there, it doesn't
  paper over the gap.
- football-data.org fixtures API - `ingest/fixtures.py` uses
  football-data.co.uk's own free fixtures.csv instead (same division codes,
  no API key), documented there; swap if you need leagues outside this set.
