import pandas as pd
import pytest

from scripts.render_coupon import build_payload, render, TEMPLATE


def _write_csv(tmp_path, rows):
    path = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_build_payload_rated_fixture_has_full_fields(tmp_path):
    csv = _write_csv(tmp_path, [{
        "league": "Premier League", "date": "2026-08-29 15:00:00",
        "home_team": "Arsenal", "away_team": "Chelsea", "unrated": False,
        "home_pred": 1.8, "away_pred": 1.1, "likely_score": "2-1",
        "p_home": 0.55, "p_draw": 0.25, "p_away": 0.20,
        "p_over_2_5": 0.6, "p_btts": 0.5, "adj_note": "rest=6d/6d",
        "market_p_home": 0.5, "market_p_draw": 0.27, "market_p_away": 0.23,
    }])
    payload = build_payload(csv)
    assert len(payload) == 1
    rec = payload[0]
    assert rec["home"] == "Arsenal" and rec["away"] == "Chelsea"
    assert rec["unrated"] is False
    assert rec["pHome"] == 55.0
    assert rec["mHome"] == 50.0
    assert rec["note"] == "rest=6d/6d"


def test_build_payload_market_source_swaps_headline_and_comparison(tmp_path):
    csv = _write_csv(tmp_path, [{
        "league": "Premier League", "date": "2026-08-29 15:00:00",
        "home_team": "Arsenal", "away_team": "Chelsea", "unrated": False,
        "home_pred": 1.8, "away_pred": 1.1, "likely_score": "2-1",
        "p_home": 0.55, "p_draw": 0.25, "p_away": 0.20,
        "p_over_2_5": 0.6, "p_btts": 0.5, "adj_note": "",
        "market_p_home": 0.50, "market_p_draw": 0.27, "market_p_away": 0.23,
        "market_likely_score": "1-1", "market_home_pred": 1.4, "market_away_pred": 1.2,
        "market_p_over_2_5": 0.55, "market_p_btts": 0.52,
    }])
    rec = build_payload(csv, source="market")[0]
    assert rec["source"] == "market"
    assert rec["pHome"] == 50.0 and rec["score"] == "1-1"   # headline = market
    assert rec["over25"] == 55.0                            # from the market grid
    assert rec["mHome"] == 55.0                             # comparison = model
    # and the default is unchanged
    assert build_payload(csv)[0]["pHome"] == 55.0


def test_build_payload_market_source_falls_back_to_model_when_no_price(tmp_path):
    csv = _write_csv(tmp_path, [{
        "league": "Serie A", "date": "2026-08-30 18:00:00",
        "home_team": "Milan", "away_team": "Roma", "unrated": False,
        "home_pred": 1.5, "away_pred": 1.2, "likely_score": "1-1",
        "p_home": 0.4, "p_draw": 0.3, "p_away": 0.3,
        "p_over_2_5": 0.5, "p_btts": 0.5, "adj_note": "",
        "market_p_home": float("nan"), "market_p_draw": float("nan"), "market_p_away": float("nan"),
    }])
    rec = build_payload(csv, source="market")[0]
    assert rec["pHome"] == 40.0 and "mHome" not in rec


def test_build_payload_unrated_fixture_is_minimal(tmp_path):
    csv = _write_csv(tmp_path, [{
        "league": "La Liga", "date": "2026-08-30 16:00:00",
        "home_team": "Real Madrid", "away_team": "Malaga", "unrated": True,
    }])
    payload = build_payload(csv)
    assert payload[0]["unrated"] is True
    assert "pHome" not in payload[0]


def test_build_payload_missing_market_odds_omits_market_fields(tmp_path):
    csv = _write_csv(tmp_path, [{
        "league": "Serie A", "date": "2026-08-30 18:00:00",
        "home_team": "Milan", "away_team": "Roma", "unrated": False,
        "home_pred": 1.5, "away_pred": 1.2, "likely_score": "1-1",
        "p_home": 0.4, "p_draw": 0.3, "p_away": 0.3,
        "p_over_2_5": 0.5, "p_btts": 0.5, "adj_note": "",
        "market_p_home": float("nan"), "market_p_draw": float("nan"), "market_p_away": float("nan"),
    }])
    payload = build_payload(csv)
    assert "mHome" not in payload[0]


def test_render_injects_payload_and_keeps_placeholder_gone():
    html = render([{"league": "Test", "date": "2026-01-01T12:00", "home": "A", "away": "B", "unrated": True}])
    assert "__DATA_JSON__" not in html
    assert '"home":"A"' in html
    assert "Auto-generated" in html


def test_render_fails_loudly_if_template_placeholder_missing(tmp_path, monkeypatch):
    bad_template = tmp_path / "bad.html"
    bad_template.write_text("<html>no placeholder here</html>")
    import scripts.render_coupon as rc
    monkeypatch.setattr(rc, "TEMPLATE", bad_template)
    with pytest.raises(RuntimeError):
        rc.render([])


def test_template_file_actually_exists_and_has_placeholder():
    assert TEMPLATE.exists()
    assert "__DATA_JSON__" in TEMPLATE.read_text(encoding="utf-8")


# ---- sanity gate ---------------------------------------------------------

def _rated(**over):
    base = {
        "unrated": False, "match_id": "m1", "home_team": "A", "away_team": "B",
        "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2,
        "home_pred": 1.6, "away_pred": 1.1, "base_home_pred": 1.5, "base_away_pred": 1.1,
        "home_attack": 1.2, "home_defence": 0.9, "away_attack": 1.0, "away_defence": 1.1,
        "lam_mult": 1.0, "mu_mult": 1.0,
    }
    base.update(over)
    return base


def test_sanity_check_passes_a_normal_fixture():
    from scripts.predict_upcoming import _sanity_check
    msgs, bad = _sanity_check([_rated(), {"unrated": True}])
    assert msgs == [] and bad == set()


def test_sanity_check_flags_probabilities_that_dont_sum_to_one():
    from scripts.predict_upcoming import _sanity_check
    msgs, bad = _sanity_check([_rated(match_id="x", p_home=0.9, p_draw=0.9, p_away=0.9)])
    assert bad == {"x"} and any("sum to" in m for m in msgs)


def test_sanity_check_flags_absurd_expected_goals_and_ratings():
    from scripts.predict_upcoming import _sanity_check
    _, bad = _sanity_check([_rated(match_id="g", home_pred=42.0)])
    assert bad == {"g"}
    _, bad2 = _sanity_check([_rated(match_id="r", away_defence=0.0)])
    assert bad2 == {"r"}


def test_sanity_check_only_warns_on_aggressive_multipliers():
    from scripts.predict_upcoming import _sanity_check
    msgs, bad = _sanity_check([_rated(mu_mult=0.3)])
    assert bad == set()  # not dropped
    assert any("multipliers" in m and m.startswith("warn") for m in msgs)
