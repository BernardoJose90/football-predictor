# API service (Milestone 6)

A small, read-only HTTP API over the model's output. It serves the same JSON
artefacts the web pages are built from — it never runs the model per request,
so it's fast, stateless, and safe behind a CDN.

## Run it

```bash
pip install -r requirements.txt -r api/requirements.txt

# make sure the artefacts exist (these are what the API serves)
python -m scripts.predict_upcoming --refresh --days 4 --log-predictions
python -m scripts.render_why
python -m scripts.track_record --eval-start 2025-08-01

python -m api --reload          # http://127.0.0.1:8000
# or: uvicorn api.app:app --reload
```

Interactive docs at `/docs`, OpenAPI schema at `/openapi.json`.

## Endpoints

| Method & path | Returns |
|---|---|
| `GET /health` | liveness + which artefacts are currently on disk |
| `GET /v1/leagues` | the eight leagues + their division codes |
| `GET /v1/predictions?league=&rated_only=` | this weekend's card: model forecast vs market price per fixture |
| `GET /v1/predictions/{match_id}` | one fixture, full breakdown: attack/defence ratings, the "ratings only" probabilities, every referee/rest/travel/availability nudge |
| `GET /v1/disagreements?min_gap=5` | fixtures where model and market disagree, with the gap split into a ratings part and an adjustments part |
| `GET /v1/track-record` | the full track record (live log scored vs results + walk-forward validation backtest) |
| `GET /v1/track-record/fixtures` | just the "what we said / what happened" list |

`league` accepts a division code (`E0`) or a name substring (`premier`).
Every `GET` response carries `Cache-Control: public, max-age=600,
stale-while-revalidate=3600`. CORS is open (public match data).

A missing artefact returns `503` with a hint rather than a stack trace, so the
API can be deployed before the first artefact is generated.

## Keeping the data fresh

The API reads each artefact only when its mtime changes, so a running server
picks up a new file without a restart. Regenerate the artefacts on a schedule
the same way the pages are (`.github/workflows/weekly-predictions.yml` already
runs the three render scripts twice a week). Two ways to wire that to a live
API:

1. **Redeploy on commit.** Point a host (Fly.io, Render, Railway) at the repo;
   the weekly CI commit that updates `docs/` also updates the committed
   artefacts, and the host redeploys with them.
2. **Static instead of a server.** If you don't need the query parameters,
   commit `upcoming_predictions.json` / `why.json` / `track_record.json` into
   `docs/` and serve them straight off GitHub Pages — no server at all. The
   API is the right call only once something needs `?league=` /
   `?min_gap=` / the per-fixture detail endpoint.

## Deployment sketch (Fly.io)

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt api/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r api/requirements.txt
COPY . .
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

`fly launch` from that, set no secrets (there are none), scale to 1 shared-cpu
instance. Render/Railway are the same shape with a `Procfile`:
`web: uvicorn api.app:app --host 0.0.0.0 --port $PORT`.
