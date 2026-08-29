"""Artefact loading for the API, with an mtime-keyed in-process cache.

The API never touches the model or the dataset directly - it only reads the
JSON files the render scripts already produce. Each file is re-read only when
its mtime changes, so a running server picks up a fresh artefact (e.g. after
the weekly CI job commits one and the box redeploys) without a restart, and
otherwise every request is a dict lookup.
"""
from __future__ import annotations

import json
from pathlib import Path

import config

PREDICTIONS = config.ARTEFACTS / "upcoming_predictions.json"
WHY = config.ARTEFACTS / "why.json"
TRACK_RECORD = config.ARTEFACTS / "track_record.json"

_cache: dict[str, tuple[float, object]] = {}


def _load(path: Path):
    """Parsed JSON for ``path``, or ``None`` if it doesn't exist yet."""
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return None
    key = str(path)
    hit = _cache.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    data = json.loads(path.read_text(encoding="utf-8"))
    _cache[key] = (mtime, data)
    return data


def predictions() -> list[dict] | None:
    return _load(PREDICTIONS)


def why() -> list[dict] | None:
    return _load(WHY)


def track_record() -> dict | None:
    return _load(TRACK_RECORD)


def artefact_status() -> dict[str, bool]:
    return {
        "predictions": PREDICTIONS.exists(),
        "disagreements": WHY.exists(),
        "track_record": TRACK_RECORD.exists(),
    }


def clear_cache() -> None:
    _cache.clear()
