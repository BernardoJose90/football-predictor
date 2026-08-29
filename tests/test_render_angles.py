import pytest

from scripts.render_angles import TEMPLATE, build, render


def _rec(ph, pd_, pa, mh, md, ma, home="A", away="B", **over):
    r = {
        "unrated": False, "league": "Premier League", "date": "2026-08-30T15:00",
        "home_team": home, "away_team": away, "likely_score": "2-1",
        "p_home": ph, "p_draw": pd_, "p_away": pa,
        "market_p_home": mh, "market_p_draw": md, "market_p_away": ma,
    }
    r.update(over)
    return r


def test_unrated_and_marketless_are_skipped():
    assert build([{"unrated": True}]) == []
    r = _rec(0.6, 0.25, 0.15, None, None, None)
    r["market_p_home"] = None
    assert build([r]) == []


def test_pick_is_the_biggest_positive_edge_outcome():
    # model 60/25/15, market 50/28/22 -> edges +10 / -3 / -7 -> pick home
    out = build([_rec(0.60, 0.25, 0.15, 0.50, 0.28, 0.22, home="Arsenal", away="Chelsea")])
    assert len(out) == 1
    row = out[0]
    assert row["pick"] == "Arsenal" and row["side"] == "home"
    assert row["edge"] == 10.0
    assert row["model"] == 60.0 and row["market"] == 50.0


def test_no_edge_favourite_is_excluded():
    # model and market agree on a strong home side -> no edge anywhere >= 5
    assert build([_rec(0.80, 0.13, 0.07, 0.79, 0.14, 0.07)]) == []


def test_min_prob_filters_a_thin_longshot_edge():
    # away edge is +12 but model only gives it 20% -> below the 35% floor
    assert build([_rec(0.55, 0.25, 0.20, 0.62, 0.30, 0.08)], min_edge=5, min_prob=35) == []


def test_draw_and_underdog_categories():
    draw = build([_rec(0.33, 0.40, 0.27, 0.35, 0.30, 0.35)])[0]
    assert draw["pick"] == "Draw" and draw["category"] == "draw"
    # away pick where the market has the away side as clear outsider
    dog = build([_rec(0.30, 0.25, 0.45, 0.40, 0.28, 0.32, home="Burnley", away="Arsenal")])[0]
    assert dog["pick"] == "Arsenal" and dog["category"] == "underdog"
    # away pick where the market already has the away side favoured -> "favourite"
    fav = build([_rec(0.20, 0.22, 0.58, 0.28, 0.24, 0.48, home="Luton", away="Man City")])[0]
    assert fav["pick"] == "Man City" and fav["category"] == "favourite"


def test_sorted_by_edge_descending():
    small = _rec(0.50, 0.25, 0.25, 0.44, 0.28, 0.28, home="Small")
    big = _rec(0.60, 0.20, 0.20, 0.45, 0.28, 0.27, home="Big")
    assert [r["home"] for r in build([small, big])] == ["Big", "Small"]


def test_render_inlines_base_css_and_data():
    html = render(build([_rec(0.60, 0.25, 0.15, 0.50, 0.28, 0.22, home="Arsenal")]))
    assert "__BASE_CSS__" not in html and "__DATA_JSON__" not in html and "__GENERATED__" not in html
    assert "The Angles" in html and '"pick":"Arsenal"' in html
    assert "Not a betting recommendation" in html


def test_render_fails_without_placeholder(tmp_path):
    bad = tmp_path / "bad.html"
    bad.write_text("<html>no</html>")
    with pytest.raises(RuntimeError):
        render([], template=bad)


def test_template_has_placeholders():
    h = TEMPLATE.read_text(encoding="utf-8")
    assert "__BASE_CSS__" in h and "const DATA = __DATA_JSON__;" in h and "__GENERATED__" in h
    assert h.lstrip().startswith("<!DOCTYPE html>")
