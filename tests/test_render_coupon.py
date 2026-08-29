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
