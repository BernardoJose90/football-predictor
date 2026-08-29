import pandas as pd

import scripts.track_record as tr


def test_monthly_running_rps_includes_late_month_kickoffs():
    # Regression: an earlier version compared against each month's own
    # midnight-31st, so a match kicking off after 00:00 on the last day of
    # the month (i.e. almost every real fixture) was excluded from its own
    # month's running mean.
    df = pd.DataFrame([
        {"date": pd.Timestamp("2025-08-05 15:00"), "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2, "result": "H"},
        {"date": pd.Timestamp("2025-08-31 20:00"), "p_home": 0.4, "p_draw": 0.3, "p_away": 0.3, "result": "D"},
        {"date": pd.Timestamp("2025-09-02 15:00"), "p_home": 0.6, "p_draw": 0.2, "p_away": 0.2, "result": "A"},
    ])
    monthly = tr._monthly_rps(df)
    assert monthly.loc[monthly["month"] == pd.Timestamp("2025-08-01"), "n"].iloc[0] == 2
    aug_rps = monthly.loc[monthly["month"] == pd.Timestamp("2025-08-01"), "rps"].iloc[0]
    aug_running = monthly.loc[monthly["month"] == pd.Timestamp("2025-08-01"), "running_rps"].iloc[0]
    assert aug_running == aug_rps  # nothing precedes August, so running == that month's own mean


def test_monthly_rps_on_empty_frame_does_not_raise():
    empty = pd.DataFrame(columns=["date", "p_home", "p_draw", "p_away", "result"])
    out = tr._monthly_rps(empty)
    assert out.empty
    assert list(out.columns) == ["month", "rps", "n", "running_rps"]


def test_build_runs_end_to_end_on_synthetic_league(synthetic_matches, monkeypatch, tmp_path):
    monkeypatch.setattr(tr, "_load_matches", lambda: synthetic_matches)
    from evaluate import prediction_log
    empty_log = prediction_log.load(tmp_path / "no_log.csv")
    monkeypatch.setattr(prediction_log, "load", lambda *a, **k: empty_log)

    payload = tr.build(eval_start="2024-08-01", eval_end=None)

    assert payload["live"]["n_logged"] == 0  # nothing logged in this test env
    val = payload["validation"]["headline"]
    assert val["model_n"] > 0
    assert val["devig_n"] > 0
    assert 0.0 <= val["model_rps"] <= 1.0
    assert len(payload["validation"]["monthly"]["model"]) > 0
    assert set(payload["validation"]["calibration"].keys()) == {"home", "draw", "away"}
    assert len(payload["validation"]["leagues"]) == 1
    assert payload["validation"]["leagues"][0]["league"] == "Test League"


def test_build_validation_includes_extended_battery(synthetic_matches, monkeypatch, tmp_path):
    monkeypatch.setattr(tr, "_load_matches", lambda: synthetic_matches)
    from evaluate import prediction_log
    empty_log = prediction_log.load(tmp_path / "no_log.csv")
    monkeypatch.setattr(prediction_log, "load", lambda *a, **k: empty_log)

    val = tr.build(eval_start="2024-08-01", eval_end=None)["validation"]
    assert set(val["brier"]) >= {"reliability", "resolution", "uncertainty", "brier"}
    assert set(val["sharpness"]) >= {"entropy", "mean_max_prob"}
    assert isinstance(val["by_season"], list) and val["by_season"]
    # significance vs the market is always computable on the synthetic league
    assert "market" in val["significance"]
    assert set(val["significance"]["market"]) >= {"mean_diff", "p_value", "ci_low", "ci_high"}
    # headline gains the "is it significant" companion flags
    assert "beats_elo_significant" in val["headline"]
    assert "gap_to_market_significant" in val["headline"]


def test_build_validation_config_reflects_production_defaults(synthetic_matches, monkeypatch, tmp_path):
    import config
    monkeypatch.setattr(tr, "_load_matches", lambda: synthetic_matches)
    from evaluate import prediction_log
    empty_log = prediction_log.load(tmp_path / "no_log.csv")
    monkeypatch.setattr(prediction_log, "load", lambda *a, **k: empty_log)

    payload = tr.build(eval_start="2024-08-01", eval_end=None)
    cfg = payload["validation"]["config"]
    assert cfg["stat"] == config.DEFAULT_STAT
    assert cfg["referee"] == config.DEFAULT_USE_REFEREE
    assert cfg["rest"] == config.DEFAULT_USE_REST
    assert cfg["travel"] == config.DEFAULT_USE_TRAVEL
