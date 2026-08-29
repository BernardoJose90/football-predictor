"""One-off diagnostic: does API-Football actually have injury data for our
8 leagues, not just "the API technically covers this competition."

    python -m scripts.check_injury_coverage

API-Football's own docs warn the `coverage` object is per season, per league -
a competition can be "in" the API for fixtures/odds while carrying no injury
data at all, because that depends on there being reporters/club staff feeding
it, not on API-Football's own infrastructure. This checks the real, current
answer rather than assuming from the marketing page.

Costs 8 requests (one per league) against the free tier's 100/day cap.
"""
from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"

# (search name, country) - matches what our project already calls these leagues.
LEAGUES = [
    ("Premier League", "England"),
    ("Championship", "England"),
    ("Premiership", "Scotland"),
    ("La Liga", "Spain"),
    ("Bundesliga", "Germany"),
    ("Serie A", "Italy"),
    ("Ligue 1", "France"),
    ("Primeira Liga", "Portugal"),
]


def check_league(name: str, country: str) -> dict:
    resp = requests.get(
        f"{BASE_URL}/leagues",
        headers={"x-apisports-key": API_KEY},
        params={"name": name, "country": country},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("errors"):
        return {"name": name, "country": country, "error": data["errors"]}
    results = data.get("response", [])
    if not results:
        return {"name": name, "country": country, "error": "no league matched this name/country"}

    league_info = results[0]["league"]
    seasons = results[0]["seasons"]
    latest = max(seasons, key=lambda s: s["year"])

    return {
        "name": name,
        "country": country,
        "league_id": league_info["id"],
        "latest_season": latest["year"],
        "current": latest.get("current", False),
        "injuries": latest.get("coverage", {}).get("injuries"),
        "predictions": latest.get("coverage", {}).get("predictions"),
        "lineups": latest.get("coverage", {}).get("fixtures", {}).get("lineups"),
    }


def main() -> int:
    if not API_KEY:
        print("FAIL: API_FOOTBALL_KEY not set - check your .env file", file=sys.stderr)
        return 1

    print(f"{'League':<20} {'Country':<10} {'Season':<8} {'Injuries':<10} {'Lineups':<9} {'Predictions'}")
    print("-" * 75)
    any_error = False
    for name, country in LEAGUES:
        try:
            r = check_league(name, country)
        except requests.HTTPError as exc:
            print(f"{name:<20} {country:<10} ERROR: {exc}")
            any_error = True
            continue

        if "error" in r:
            print(f"{name:<20} {country:<10} ERROR: {r['error']}")
            any_error = True
            continue

        print(f"{r['name']:<20} {r['country']:<10} {r['latest_season']:<8} "
              f"{str(r['injuries']):<10} {str(r['lineups']):<9} {r['predictions']}")

    return 1 if any_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
