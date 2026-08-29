"""Milestone 1: one command -> a clean match table.

    python -m scripts.build_dataset [--download] [--force] [--stat xg]

Output: data/processed/matches.parquet (and .csv), with results, shots on
target, closing odds, and - if --stat xg and data/processed/understat_xg.csv
exists - expected goals. Zero unresolved team names or it exits non-zero.
"""
from __future__ import annotations

import argparse
import sys

import config
from ingest import historical
from ingest import understat
from normalise import schema
from normalise.teams import UnknownTeamError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true", help="fetch CSVs first")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--stat", choices=config.STAT_CHOICES, default=None,
                    help="if 'xg', attempt the Understat join and fail if it is empty")
    args = ap.parse_args(argv)

    if args.download or args.force:
        historical.download_all(force=args.force)

    raw = historical.load_all()
    print(f"loaded {len(raw)} raw rows from {raw['Div'].nunique()} leagues", file=sys.stderr)

    try:
        matches = schema.normalise(raw)
    except UnknownTeamError as exc:
        print(f"\nFAIL: {exc}\n", file=sys.stderr)
        print("Run `python -m normalise.build_aliases` then edit normalise/teams.yaml.",
              file=sys.stderr)
        return 2

    if args.stat in ("xg", "auto"):
        matches = understat.join(matches)
        if args.stat == "xg" and matches["home_xg"].notna().sum() == 0:
            print("FAIL: --stat xg but no expected-goals rows joined.", file=sys.stderr)
            return 3

    pq = config.DATA_PROCESSED / "matches.parquet"
    csv = config.DATA_PROCESSED / "matches.csv"
    try:
        matches.to_parquet(pq, index=False)
    except Exception as exc:  # pyarrow missing
        print(f"note: parquet write skipped ({exc})", file=sys.stderr)
    matches.to_csv(csv, index=False)

    span = f"{matches['date'].min():%Y-%m-%d} .. {matches['date'].max():%Y-%m-%d}"
    print(
        f"\nOK  {len(matches)} matches  |  {matches['div'].nunique()} leagues  |  {span}\n"
        f"    -> {csv}",
        file=sys.stderr,
    )
    with_sot = matches[["home_sot", "away_sot"]].notna().all(axis=1).sum()
    with_odds = matches[["close_home", "close_draw", "close_away"]].notna().all(axis=1).sum()
    print(f"    shots on target: {with_sot}/{len(matches)}   closing odds: {with_odds}/{len(matches)}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
