from model.club_elo_implied import implied_1x2


def test_probs_are_a_distribution():
    p = implied_1x2(1800, 1750)
    assert abs(p["p_home"] + p["p_draw"] + p["p_away"] - 1.0) < 1e-9
    assert all(0.0 <= v <= 1.0 for v in p.values())


def test_stronger_team_is_favoured():
    p = implied_1x2(2000, 1600)
    assert p["p_home"] > p["p_away"]
    assert p["p_home"] > 0.5


def test_home_edge_shows_on_equal_ratings():
    p = implied_1x2(1800, 1800)
    assert p["p_home"] > p["p_away"]


def test_draw_share_shrinks_as_the_gap_widens():
    close = implied_1x2(1800, 1790)
    wide = implied_1x2(2100, 1500)
    assert close["p_draw"] > wide["p_draw"]
