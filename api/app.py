"""The FastAPI application.

Read-only, versioned (`/v1`), CORS-open (it's public match data), and cached
at the edge via `Cache-Control`. Every endpoint reads a JSON artefact through
``api.data``; if the artefact hasn't been generated yet the endpoint returns
503 with a hint rather than a stack trace.

Interactive docs at `/docs`, OpenAPI schema at `/openapi.json`.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import config
from api import data

app = FastAPI(
    title="Football Match Predictor API",
    version="1.0.0",
    description=(
        "Read-only access to a walk-forward Poisson model's output: this "
        "weekend's predictions next to the market price, the model-vs-market "
        "disagreements with their reasoning, and the running track record. "
        "Serves the same artefacts the pages at "
        "https://bernardojose90.github.io/football-predictor/ are built from."
    ),
    contact={"name": "football-predictor", "url": "https://github.com/BernardoJose90/football-predictor"},
    license_info={"name": "See repository"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

_CACHE_CONTROL = "public, max-age=600, stale-while-revalidate=3600"


@app.middleware("http")
async def _cache_headers(request: Request, call_next):
    resp: Response = await call_next(request)
    if request.method == "GET" and resp.status_code == 200 and "cache-control" not in resp.headers:
        resp.headers["Cache-Control"] = _CACHE_CONTROL
    return resp


# --------------------------------------------------------------------------
# models — a curated view, not the raw internal dicts
# --------------------------------------------------------------------------
class Outcome(BaseModel):
    home: float = Field(..., description="P(home win), %")
    draw: float
    away: float


class Adjustment(BaseModel):
    kind: str = Field(..., description="referee | rest | travel | injuries")
    detail: str
    home_factor: float = Field(..., description="multiplier on home expected goals")
    away_factor: float


class Fixture(BaseModel):
    match_id: str
    league: str
    kickoff: str | None
    home_team: str
    away_team: str
    unrated: bool
    likely_score: str | None = None
    model: Outcome | None = None
    market: Outcome | None = None
    expected_goals: list[float] | None = Field(None, description="[home, away], published")
    over_2_5: float | None = None
    btts: float | None = None
    stat: str | None = Field(None, description="rating stat used for this league (xg|sot)")


class FixtureDetail(Fixture):
    base_model: Outcome | None = Field(None, description="model probabilities before the 10.1 adjustments")
    base_expected_goals: list[float] | None = None
    home_attack: float | None = None
    home_defence: float | None = None
    away_attack: float | None = None
    away_defence: float | None = None
    home_matches_used: int | None = None
    away_matches_used: int | None = None
    adjustments: list[Adjustment] = []
    adj_note: str = ""


class Disagreement(BaseModel):
    match_id: str | None = None
    league: str
    kickoff: str | None = Field(None, alias="date")
    home: str
    away: str
    primary_outcome: str = Field(..., alias="primary", description="home | draw | away")
    direction: str = Field(..., description="model is 'higher' or 'lower' than the market on that outcome")
    gap_points: float = Field(..., alias="gap")
    adjustment_movement_points: float = Field(..., alias="moved")
    attribution: str = Field(..., description="ratings | adjustments | mixed")
    model: Outcome
    market: Outcome
    ratings_only: Outcome

    model_config = {"populate_by_name": True}


class League(BaseModel):
    code: str
    name: str


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _require(payload, name: str):
    if payload is None:
        raise HTTPException(
            status_code=503,
            detail=f"the '{name}' artefact has not been generated yet - "
                   f"run the matching render script (see the project README)",
        )
    return payload


def _outcome(h, d, a, *, scale=100.0) -> Outcome | None:
    if h is None:
        return None
    return Outcome(home=round(h * scale, 1), draw=round(d * scale, 1), away=round(a * scale, 1))


def _to_fixture(r: dict, *, detail: bool) -> dict:
    base = {
        "match_id": r.get("match_id"),
        "league": r.get("league"),
        "kickoff": r.get("date"),
        "home_team": r.get("home_team"),
        "away_team": r.get("away_team"),
        "unrated": bool(r.get("unrated")),
    }
    if not r.get("unrated"):
        base.update({
            "likely_score": r.get("likely_score"),
            "model": _outcome(r.get("p_home"), r.get("p_draw"), r.get("p_away")),
            "market": _outcome(r.get("market_p_home"), r.get("market_p_draw"), r.get("market_p_away")),
            "expected_goals": [r.get("home_pred"), r.get("away_pred")],
            "over_2_5": _pct(r.get("p_over_2_5")),
            "btts": _pct(r.get("p_btts")),
            "stat": r.get("stat_used") or r.get("stat"),
        })
        if detail:
            base.update({
                "base_model": _outcome(r.get("base_p_home"), r.get("base_p_draw"), r.get("base_p_away")),
                "base_expected_goals": [r.get("base_home_pred"), r.get("base_away_pred")],
                "home_attack": r.get("home_attack"), "home_defence": r.get("home_defence"),
                "away_attack": r.get("away_attack"), "away_defence": r.get("away_defence"),
                "home_matches_used": r.get("home_matches_used"),
                "away_matches_used": r.get("away_matches_used"),
                "adjustments": r.get("adjustments") or [],
                "adj_note": r.get("adj_note") or "",
            })
    return base


def _pct(x):
    return round(x * 100, 1) if x is not None else None


def _matches_league(rec: dict, wanted: str) -> bool:
    w = wanted.strip().lower()
    if w in {k.lower() for k in config.LEAGUES}:
        return str(rec.get("match_id", "")).lower().startswith(w + "_")
    return w in str(rec.get("league", "")).lower()


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/docs")


@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    """Liveness + which artefacts are currently available."""
    return {"status": "ok", "artefacts": data.artefact_status()}


@app.get("/v1/leagues", response_model=list[League], tags=["reference"])
def leagues():
    """The eight leagues covered, with their football-data.co.uk division codes."""
    return [League(code=c, name=n) for c, n in config.LEAGUES.items()]


@app.get("/v1/predictions", response_model=list[Fixture], tags=["predictions"])
def predictions(
    league: str | None = Query(None, description="division code (e.g. E0) or a name substring (e.g. 'premier')"),
    rated_only: bool = Query(True, description="drop fixtures where a team has too little history to rate"),
):
    """This weekend's card: every upcoming fixture with the model's forecast
    and the same fixture's pre-match market price."""
    recs = _require(data.predictions(), "predictions")
    if league:
        recs = [r for r in recs if _matches_league(r, league)]
    if rated_only:
        recs = [r for r in recs if not r.get("unrated")]
    return [_to_fixture(r, detail=False) for r in recs]


@app.get("/v1/predictions/{match_id}", response_model=FixtureDetail, tags=["predictions"])
def prediction_detail(match_id: str):
    """One fixture with the full breakdown: attack/defence ratings, the
    pre-adjustment ('ratings only') probabilities, and every referee / rest /
    travel / availability nudge that was applied."""
    recs = _require(data.predictions(), "predictions")
    for r in recs:
        if r.get("match_id") == match_id:
            return _to_fixture(r, detail=True)
    raise HTTPException(status_code=404, detail=f"no fixture with match_id {match_id!r} in the current card")


@app.get("/v1/disagreements", response_model=list[Disagreement], response_model_by_alias=False,
         tags=["predictions"])
def disagreements(
    min_gap: float = Query(5.0, ge=0, le=100, description="minimum model-vs-market gap, in percentage points"),
):
    """Fixtures where the model and the market disagree, with the gap split
    into a ratings part and an adjustments part (see the 'attribution' field)."""
    recs = _require(data.why(), "disagreements")
    out = []
    for r in recs:
        if r.get("gap", 0) < min_gap:
            continue
        out.append({
            **r,
            "model": _outcome(r["pHome"], r["pDraw"], r["pAway"], scale=1.0),
            "market": _outcome(r["mHome"], r["mDraw"], r["mAway"], scale=1.0),
            "ratings_only": _outcome(r["pureHome"], r["pureDraw"], r["pureAway"], scale=1.0),
        })
    return out


@app.get("/v1/track-record", tags=["track record"])
def track_record() -> dict:
    """The full track record: the live log scored against results as they come
    in, plus the walk-forward validation backtest (RPS, calibration, Brier
    decomposition, per-season and per-league breakdowns, significance tests)."""
    return _require(data.track_record(), "track_record")


@app.get("/v1/track-record/fixtures", tags=["track record"])
def track_record_fixtures() -> list[dict]:
    """Just the 'what we said / what happened' list - every published
    prediction that has since been played, most recent first."""
    tr = _require(data.track_record(), "track_record")
    return tr.get("live", {}).get("fixtures", [])
