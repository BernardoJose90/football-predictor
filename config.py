"""Central configuration: league codes, seasons, paths, model defaults.

Kept deliberately small and import-only so every module can read it without
side effects.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTEFACTS = ROOT / "artefacts"

for _p in (DATA_RAW, DATA_PROCESSED, ARTEFACTS):
    _p.mkdir(parents=True, exist_ok=True)

# football-data.co.uk division codes -> human name.
# https://www.football-data.co.uk/notes.txt
LEAGUES: dict[str, str] = {
    "E0": "Premier League",
    "E1": "EFL Championship",
    "SC0": "Scottish Premiership",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    "P1": "Primeira Liga",
}

# Seasons in football-data.co.uk's 4-digit form (start year + end year, 2 digits each).
# 2627 is the in-progress season - included so ratings reflect current form and
# newly promoted/relegated teams appear at all. It will be a partial file
# (only played rounds so far) all season; that's expected, not a bug.
SEASONS: list[str] = ["2324", "2425", "2526", "2627"]

FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281"

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
# 'auto' tries AUTO_STAT_PRIMARY first per division, falling back to
# AUTO_STAT_FALLBACK if that division has no usable data for it (e.g. xg for
# E1/SC0/P1, which Understat doesn't cover) - resolved in
# evaluate.backtest.backtest_league / scripts.predict_upcoming, never passed
# into model.ratings.build_ratings directly (it only understands concrete
# stats). Promoted to the default after a matched, paired comparison on the
# 5 xg-covered leagues: xg beat sot by a mean RPS of 0.00497 per match
# (bootstrap 95% CI [0.0027, 0.0073], excludes zero; Wilcoxon p=5.3e-08,
# n=1644 identical fixtures both sides) - see README's Milestone 4 section.
STAT_CHOICES = ("xg", "sot", "goals", "auto")
DEFAULT_STAT = "auto"
AUTO_STAT_PRIMARY = "xg"
AUTO_STAT_FALLBACK = "sot"

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
