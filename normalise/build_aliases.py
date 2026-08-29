"""Seed / refresh normalise/teams.yaml from whatever raw CSVs are on disk.

For a single source (football-data.co.uk) the names are already internally
consistent, so this writes an identity mapping: each observed name becomes its
own canonical entry. When you add a second source (e.g. Understat) you hand-edit
teams.yaml to fold its spellings under the existing canonical names.

This script never deletes existing aliases - it only adds newly observed names
as fresh canonical entries, so hand edits survive a re-run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from ingest import historical

_YAML = Path(__file__).with_name("teams.yaml")


def observed_names() -> set[str]:
    df = historical.load_all()
    names: set[str] = set()
    for col in ("HomeTeam", "AwayTeam"):
        names |= {str(n).strip() for n in df[col].dropna() if str(n).strip()}
    return names


def main() -> None:
    existing: dict[str, list[str]] = {}
    if _YAML.exists():
        existing = yaml.safe_load(_YAML.read_text()) or {}

    known = set(existing)
    for canon, alts in existing.items():
        known |= {str(a).strip() for a in alts or []}

    added = 0
    for name in sorted(observed_names()):
        if name not in known:
            existing[name] = []
            known.add(name)
            added += 1

    ordered = {k: existing[k] for k in sorted(existing)}
    _YAML.write_text(
        "# canonical name -> list of alternative spellings.\n"
        "# Hand-maintained. build_aliases.py only appends new canonical entries.\n"
        + yaml.safe_dump(ordered, sort_keys=True, allow_unicode=True, default_flow_style=False)
    )
    print(f"{_YAML}: {len(ordered)} canonical teams (+{added} new)", file=sys.stderr)


if __name__ == "__main__":
    main()
