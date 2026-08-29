"""Canonical team-name resolution.

Rule from the design doc (section 11): three sources will spell the same club
three ways. Maintain an explicit alias file and raise on any unknown name.
Never fuzzy-match silently - a wrong match produces confident predictions for
the wrong fixture.

teams.yaml maps every known spelling (the "alias", including the canonical name
itself) to a canonical name. Resolution is exact-match only. An unseen name
raises UnknownTeamError, which the dataset build turns into a hard failure.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_YAML = Path(__file__).with_name("teams.yaml")


class UnknownTeamError(KeyError):
    """Raised when a team name is not present in teams.yaml."""


@lru_cache(maxsize=1)
def _alias_map() -> dict[str, str]:
    if not _YAML.exists():
        raise FileNotFoundError(
            f"{_YAML} missing - run `python -m normalise.build_aliases` to seed it"
        )
    raw = yaml.safe_load(_YAML.read_text()) or {}
    aliases: dict[str, str] = {}
    for canonical, alt_names in raw.items():
        aliases[canonical.strip()] = canonical.strip()
        for alt in alt_names or []:
            aliases[str(alt).strip()] = canonical.strip()
    return aliases


def canonical_names() -> set[str]:
    return set(_alias_map().values())


def resolve(name: str) -> str:
    """Return the canonical name for ``name`` or raise UnknownTeamError."""
    if name is None:
        raise UnknownTeamError("<None>")
    key = str(name).strip()
    try:
        return _alias_map()[key]
    except KeyError as exc:
        raise UnknownTeamError(
            f"{name!r} is not in {_YAML.name}. Add it under the correct canonical "
            f"club rather than letting the pipeline guess."
        ) from exc


def resolve_series(names) -> list[str]:
    """Resolve an iterable of names, collecting *all* failures before raising."""
    missing = sorted({str(n).strip() for n in names if str(n).strip() not in _alias_map()})
    if missing:
        raise UnknownTeamError(
            f"{len(missing)} unresolved team name(s): {missing}. "
            f"Add them to {_YAML.name}."
        )
    return [resolve(n) for n in names]


def reload_cache() -> None:
    _alias_map.cache_clear()
