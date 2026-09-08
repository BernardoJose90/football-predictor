"""Central configuration: league codes, seasons, paths, model defaults.

Kept deliberately small and import-only so every module can read it without
side effects.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Load .env once, here, so any module reading a credential off config (or off
# os.environ) sees it without repeating load_dotenv() everywhere. This is the
# one deliberate side effect in this file; it only populates os.environ from a
# local file and is a no-op when python-dotenv or the file is absent.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ModuleNotFoundError:  # pragma: no cover - python-dotenv is a pinned dep
    pass
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTEFACTS = ROOT / "artefacts"

for _p in (DATA_RAW, DATA_PROCESSED, ARTEFACTS):
    _p.mkdir(parents=True, exist_ok=True)

# football-data.co.uk division codes -> human name.
# https://www.football-data.co.uk/notes.txt
#
# COUPON_LEAGUES are the ones the weekend page and the public track record cover
# (unchanged since the project started). The four after them - Eredivisie,
# Belgian Pro League, Greek Super League, Turkish Super Lig - were added purely
# so their clubs get an attack/defence rating for the Champions League
# cross-league model (see model/league_bridge.py). They are NOT shown on the
# weekend coupon: scripts.predict_upcoming pulls its domestic card from
# COUPON_LEAGUES, and only the CL block reaches past it.
COUPON_LEAGUES: dict[str, str] = {
    "E0": "Premier League",
    "E1": "EFL Championship",
    "SC0": "Scottish Premiership",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    "P1": "Primeira Liga",
}

# Rating-only additions - ingested and rated, never couponed. Keep these in
# their own dict so "is this a couponed league?" stays a one-liner.
RATING_ONLY_LEAGUES: dict[str, str] = {
    "N1": "Eredivisie",
    "B1": "Belgian Pro League",
    "G1": "Greek Super League",
    "T1": "Turkish Super Lig",
}

# Everything ingest.historical downloads and normalise.schema names.
LEAGUES: dict[str, str] = {**COUPON_LEAGUES, **RATING_ONLY_LEAGUES}

# UEFA club competitions (not football-data.co.uk - see ingest/champions_league.py).
UEFA_COMPS: dict[str, str] = {"CL": "Champions League"}

# Seasons in football-data.co.uk's 4-digit form (start year + end year, 2 digits each).
# 2627 is the in-progress season - included so ratings reflect current form and
# newly promoted/relegated teams appear at all. It will be a partial file
# (only played rounds so far) all season; that's expected, not a bug.
SEASONS: list[str] = ["2324", "2425", "2526", "2627"]

# The one season whose CSV still gains rows week to week. Completed seasons are
# immutable once their file is on disk, so ingest.historical never re-fetches
# them even under force - only this one is worth the network round-trip and the
# exposure to an upstream 5xx (see ingest/historical.py download_all).
CURRENT_SEASON: str = SEASONS[-1]

FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281"

# ---- Champions League ----------------------------------------------------
# Fixtures and results only (no odds on the free tier - the comparison price
# is Club Elo's implied 1X2, see model/club_elo_implied.py). Register a free
# key at https://www.football-data.org/client/register and put it in .env.
FOOTBALL_DATA_ORG_BASE = "https://api.football-data.org/v4"
FOOTBALL_DATA_ORG_TOKEN = os.environ.get("FOOTBALL_DATA_ORG_TOKEN", "")

# Seasons to pull CL history for, as the starting calendar year (football-data.org
# keys a season by its start year). 2021 is the first season with the current
# free-tier depth; 2024 onward is the 36-team league phase.
CL_SEASONS: list[int] = [2021, 2022, 2023, 2024, 2025]

# Club Elo -> expected-goals bridge exponent for a cross-league tie. 0.0 = use
# domestic form only (no bridge). Set from evaluate.tune.sweep_cl_gamma on a
# tuning window kept separate from the reporting one - see README's Champions
# League section. Provisional until that sweep has been run on real data.
DEFAULT_CL_GAMMA = 0.0

# ---- model defaults -------------------------------------------------------
# Time-decay. Dixon & Coles (1997) optimum 0.0065 per half-week => /3.5 => 0.00186 per day,
# independently reproduced across ENG/GER/NED/FRA - that's where 0.0018 comes from.
#
# Tuned on this repo's own data: evaluate/tune.py xi sweep over 8 leagues,
# stat=sot, tuning window 2024-08-01..2025-06-30 (kept separate from the
# 2025-08-01+ evaluation window to avoid tuning-on-the-test-set leakage).
# Curve is a shallow, clean U bottoming at xi=0.0035 (RPS 0.2017 vs 0.2023 at
# xi=0, vs 0.2031 at xi=0.01) - see artefacts/xi_sweep.png. Roughly double the
# 1997 constant, consistent with the doc's expectation that heavier modern
# fixture loads call for faster decay. Re-tune if the dataset changes.
DEFAULT_XI = 0.0035

# Dixon-Coles low-score interaction term. Fitted values typically 0.05-0.15.
DEFAULT_RHO = 0.10

# Whole-diagonal inflation on top of Dixon-Coles (Karlis & Ntzoufras 2003;
# Egidi et al. 2026 - see model/dixon_coles.py). Swept 0 to 1.0 on the tuning
# window: a real, if modest, U-shaped improvement (unlike referee/rest/travel,
# which all made RPS worse) - RPS 0.20064 -> 0.20031 in the current production
# config (stat=auto, referee+rest+travel on), minimum around delta=0.20.
# Isolated (stat=sot alone, no other adjustments) the optimum was flatter and
# slightly lower, ~0.15 - the two agree closely enough to use one constant.
DEFAULT_DELTA = 0.20

# A team needs at least this many matches in the lookback window to be rated.
DEFAULT_MIN_MATCHES = 8

# Poisson grid truncation.
MAX_GOALS = 10

# Which raw statistic feeds the ratings. 'xg' requires an external xG join
# (see ingest/understat.py); 'sot' and 'goals' come straight from the CSVs.
# 'auto' walks AUTO_STAT_CHAIN per division, using the first stat that division
# has usable data for: xg (top 5 only), then shots on target, then plain goals
# as a last resort for any division whose CSVs carry no shot columns at all
# (some N1/B1/G1/T1 seasons). Resolved in evaluate.backtest.build_snapshot /
# scripts.predict_upcoming, never passed into model.ratings.build_ratings
# directly (it only understands concrete stats). Promoted to the default after
# a matched, paired comparison on the 5 xg-covered leagues: xg beat sot by a
# mean RPS of 0.00497 per match (bootstrap 95% CI [0.0027, 0.0073], excludes
# zero; Wilcoxon p=5.3e-08, n=1644 identical fixtures) - see README Milestone 4.
STAT_CHOICES = ("xg", "sot", "goals", "auto")
DEFAULT_STAT = "auto"
AUTO_STAT_CHAIN = ("xg", "sot", "goals")
AUTO_STAT_PRIMARY = AUTO_STAT_CHAIN[0]      # back-compat aliases
AUTO_STAT_FALLBACK = AUTO_STAT_CHAIN[1]

# ---- section 10.1 feature candidates ---------------------------------------
# Referee identity, days-since-last-match and travel distance were each tested
# walk-forward against RPS (see README's Milestone 4 section). All three raised
# RPS - made the forecasts less accurate - individually and together, on this
# repo's 3-season, 8-league dataset. They were kept ON for a while as a
# "use more of the signal bookmakers use" call, then turned OFF: the evidence
# says they don't help, the referee factor in particular produces large
# small-sample artefacts (a lone ref can swing a fixture 20 points), and
# "we tested these and rejected them" is a cleaner, more honest story than
# keeping them despite the metric. The code, flags and tests stay; re-enable
# per run with --referee / --rest / --travel on scripts.predict_upcoming (or
# use_*=True on BacktestConfig), and recompute the RPS comparison if the
# dataset grows - the 3-season sample-size noise may not hold on more data.
DEFAULT_USE_REFEREE = False
DEFAULT_USE_REST = False
DEFAULT_USE_TRAVEL = False
REFEREE_MIN_MATCHES = 12
REST_K = 0.02
TRAVEL_K = 0.00006     # per km, see model/travel.py
TRAVEL_REST_SYNERGY = True   # apply the extra penalty only under short rest

# Squad market value (rank 4) is a fallback, not a walk-forward-tested
# adjustment like the three above - it only ever activates for a team with
# too little history to rate normally, rather than excluding it. Built and
# tested (model/squad_value.py, ingest/squad_value.py), but OFF by default
# per an explicit product decision: a team with too little history goes back
# to being excluded (UNRATED) rather than priced from squad value. Still
# available via --use-squad-value on scripts.predict_upcoming.
DEFAULT_USE_SQUAD_VALUE = False
VALUE_PRIOR_MIN_POINTS = 5   # min rated teams w/ known value needed to fit the prior

# Premier-League-only injury/availability data (Fantasy Premier League's free
# API, ingest/fpl.py / model/injuries.py). Unlike every other adjustment here,
# this has NO free historical archive to walk-forward test against RPS at
# all - not "not tested yet" like squad value's backtest gap, but genuinely
# untestable with data that exists. OFF by default: a live-only experiment,
# not a decision backed by evidence either way. --use-injuries opts in.
DEFAULT_USE_INJURIES = False
INJURY_K = 0.3                 # per-team expected-goals penalty scale, see model/injuries.py
INJURY_MIN_IMPORTANCE = 100.0  # min total FPL now_cost across a squad before weighting by it
