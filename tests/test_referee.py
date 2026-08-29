import pandas as pd

from model.referee import build_referee_factors


def _ref_matches(n_per_ref=20, home_friendly_ref="Ref B"):
    """A league where 'Ref B' officiates matches with systematically more home
    goals than the rest of the league - the exact pattern Nevill/Balmer/
    Williams found, exaggerated so the test doesn't depend on noise."""
    rows = []
    start = pd.Timestamp("2024-08-01")
    i = 0
    # neutral referees: home goals hover around 1.4, away around 1.1
    for ref in ["Ref A", "Ref C"]:
        for _ in range(n_per_ref):
            i += 1
            rows.append({"date": start + pd.Timedelta(days=i), "referee": ref,
                        "home_goals": 1.4, "away_goals": 1.1})
    # home_friendly_ref: home goals much higher
    for _ in range(n_per_ref):
        i += 1
        rows.append({"date": start + pd.Timedelta(days=i), "referee": home_friendly_ref,
                    "home_goals": 2.6, "away_goals": 0.9})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_home_friendly_referee_gets_a_boosted_home_factor():
    m = _ref_matches()
    as_of = m["date"].max() + pd.Timedelta(days=1)
    rf = build_referee_factors(m, as_of=as_of, xi=0.0, min_matches=10)
    home_ref, away_ref = rf.factor("Ref B")
    neutral_home, neutral_away = rf.factor("Ref A")
    # Ref B's matches average 2.6 home goals vs the neutral refs' 1.4 - that
    # gap must survive as home_ref clearly outscoring neutral_home, in both
    # directions (Ref B pulled above the league average, neutral pulled below
    # it, since Ref B's high-scoring matches drag the league average up too).
    assert home_ref > 1.0
    assert neutral_home < 1.0
    assert home_ref > neutral_home
    assert away_ref < neutral_away  # same pattern, away goals suppressed under Ref B


def test_unknown_referee_gets_neutral_factor():
    m = _ref_matches()
    as_of = m["date"].max() + pd.Timedelta(days=1)
    rf = build_referee_factors(m, as_of=as_of, xi=0.0, min_matches=10)
    assert rf.factor("Someone Nobody Has Heard Of") == (1.0, 1.0)


def test_none_referee_gets_neutral_factor():
    m = _ref_matches()
    rf = build_referee_factors(m, as_of=m["date"].max() + pd.Timedelta(days=1), xi=0.0)
    assert rf.factor(None) == (1.0, 1.0)
    assert rf.factor(float("nan")) == (1.0, 1.0)


def test_min_matches_guard_excludes_sparse_referees():
    m = _ref_matches(n_per_ref=5)  # below default min_matches=12
    rf = build_referee_factors(m, as_of=m["date"].max() + pd.Timedelta(days=1))
    assert rf.factor("Ref B") == (1.0, 1.0)  # too little data - no adjustment applied


def test_leakage_guard_ignores_matches_on_or_after_as_of():
    m = _ref_matches()
    as_of = pd.Timestamp("2024-08-15")  # partway through - before Ref B's block even starts
    rf = build_referee_factors(m, as_of=as_of, xi=0.0, min_matches=5)
    assert "Ref B" not in rf.referees  # none of Ref B's matches happened yet


def test_no_referee_data_returns_empty_factors():
    m = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]"),
                      "referee": pd.Series([], dtype=object),
                      "home_goals": pd.Series([], dtype=float),
                      "away_goals": pd.Series([], dtype=float)})
    rf = build_referee_factors(m, as_of="2025-01-01")
    assert rf.referees == {}
    assert rf.factor("Anyone") == (1.0, 1.0)
