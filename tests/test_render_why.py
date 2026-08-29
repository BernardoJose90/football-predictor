from scripts.render_why import build


def _rec(gap_home=0.0, moved_home=0.0, **overrides):
    base_home = 0.5
    final_home = base_home + moved_home
    market_home = final_home - gap_home
    rec = {
        "league": "Premier League", "date": "2026-08-30T15:00",
        "home_team": "Arsenal", "away_team": "Chelsea", "unrated": False,
        "likely_score": "2-1", "adj_note": "rest=6d/6d", "adjustments": [],
        "p_home": final_home, "p_draw": 0.25, "p_away": round(1 - final_home - 0.25, 4),
        "base_p_home": base_home, "base_p_draw": 0.25, "base_p_away": round(1 - base_home - 0.25, 4),
        "market_p_home": market_home, "market_p_draw": 0.27,
        "market_p_away": round(1 - market_home - 0.27, 4),
    }
    rec.update(overrides)
    return rec


def test_unrated_fixtures_are_excluded():
    out = build([{"unrated": True}], min_gap=0)
    assert out == []


def test_fixtures_without_market_price_are_excluded():
    rec = _rec()
    rec["market_p_home"] = None
    assert build([rec], min_gap=0) == []


def test_gap_below_threshold_is_excluded():
    rec = _rec(gap_home=0.03)  # 3pt gap
    assert build([rec], min_gap=5) == []


def test_gap_above_threshold_is_kept_with_correct_fields():
    rec = _rec(gap_home=0.10, moved_home=0.04)
    out = build([rec], min_gap=5)
    assert len(out) == 1
    row = out[0]
    assert row["home"] == "Arsenal" and row["away"] == "Chelsea"
    assert row["gap"] == 10.0
    assert row["moved"] == 4.0
    assert row["pureHome"] == 50.0
    assert row["pHome"] == 54.0


def test_results_sorted_by_gap_descending():
    small = _rec(gap_home=0.06, home_team="Small", away_team="Gap")
    big = _rec(gap_home=0.20, home_team="Big", away_team="Gap")
    out = build([small, big], min_gap=0)
    assert [r["home"] for r in out] == ["Big", "Small"]
