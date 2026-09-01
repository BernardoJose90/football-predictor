import pandas as pd

import scripts.score_results as sr


def test_main_refreshes_then_scores_without_touching_the_log(monkeypatch, tmp_path):
    """score_results is scoring-only: it rebuilds the match table and re-runs
    track_record, but must never write the prediction log."""
    calls = []

    monkeypatch.setattr(sr, "refresh_results", lambda: calls.append("refresh"))

    def fake_track_record_main(argv):
        calls.append(("track_record", tuple(argv)))
        return 0

    monkeypatch.setattr(sr.track_record, "main", fake_track_record_main)

    from evaluate import prediction_log
    before = prediction_log.LOG_PATH.read_bytes() if prediction_log.LOG_PATH.exists() else None

    def boom(*a, **k):
        raise AssertionError("score_results must not write the prediction log")

    monkeypatch.setattr(prediction_log, "append", boom)

    rc = sr.main(["--eval-start", "2025-08-01"])

    assert rc == 0
    assert calls == ["refresh", ("track_record", ("--eval-start", "2025-08-01"))]
    if before is not None:
        assert prediction_log.LOG_PATH.read_bytes() == before


def test_refresh_results_rebuilds_matches_parquet(monkeypatch, tmp_path, synthetic_matches):
    """The download/normalise chain is stubbed; we only check refresh_results
    writes the processed match table the scorer reads."""
    monkeypatch.setattr(sr.historical, "download_all", lambda *a, **k: [])
    monkeypatch.setattr(sr.historical, "load_all", lambda: pd.DataFrame())
    monkeypatch.setattr(sr.build_aliases, "main", lambda: None)
    monkeypatch.setattr(sr, "reload_cache", lambda: None)
    monkeypatch.setattr(sr.schema, "normalise", lambda raw: synthetic_matches)
    monkeypatch.setattr(sr.understat, "join", lambda m: m)
    monkeypatch.setattr(sr.config, "DATA_PROCESSED", tmp_path)

    sr.refresh_results()

    assert (tmp_path / "matches.csv").exists()
    reloaded = pd.read_csv(tmp_path / "matches.csv")
    assert len(reloaded) == len(synthetic_matches)
