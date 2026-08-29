"""Milestone 6: a small read-only HTTP API over the model's output.

It serves the same JSON artefacts the web pages are built from
(``artefacts/upcoming_predictions.json``, ``why.json``,
``track_record.json``) - it does NOT run the model per request, so it is
fast, stateless and safe to put behind a cache. Regenerate the artefacts on
a schedule (``scripts.render_coupon`` / ``render_why`` / ``track_record``)
the same way the pages are.

    uvicorn api.app:app --reload        # dev
    python -m api                       # same, convenience wrapper

See ``api/README.md`` for endpoints and deployment notes.
"""
