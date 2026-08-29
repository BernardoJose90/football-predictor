import numpy as np
import pandas as pd

from evaluate.backtest import BacktestConfig, backtest, report


def test_report_on_empty_frame_does_not_raise():
    rep = report(pd.DataFrame())
    assert rep["rated"] == 0
    assert rep["matches"] == 0
    assert np.isnan(rep["rps"])


def test_backtest_with_no_data_for_stat_returns_shaped_empty_frame(synthetic_matches):
    # No team ever clears min_matches this early - nothing should be rated,
    # and the result must still have an 'unrated' column for report() to use.
    cfg = BacktestConfig(stat="goals", min_matches=8)
    preds = backtest(synthetic_matches, start="2024-08-01", end="2024-08-05", cfg=cfg)
    assert "unrated" in preds.columns or preds.empty
    rep = report(preds)  # must not raise
    assert rep["rated"] == 0


def test_backtest_missing_stat_column_is_a_clean_empty_result(synthetic_matches):
    # Simulate the real xg-not-populated case: the stat column exists but is
    # entirely NaN, same as home_xg/away_xg before ingest.understat has run.
    poisoned = synthetic_matches.copy()
    poisoned["home_xg"] = float("nan")
    poisoned["away_xg"] = float("nan")
    cfg = BacktestConfig(stat="xg")
    preds = backtest(poisoned, start="2025-01-01", end="2025-06-01", cfg=cfg)
    assert preds.empty
    rep = report(preds)  # must not raise (this is the bug that crashed run_tune.py)
    assert rep["rated"] == 0


def test_auto_stat_falls_back_to_sot_when_xg_unavailable(synthetic_matches):
    # Same "xg not populated" situation as above, but stat="auto" should
    # fall back to sot instead of coming back empty.
    no_xg = synthetic_matches.copy()
    no_xg["home_xg"] = float("nan")
    no_xg["away_xg"] = float("nan")
    cfg = BacktestConfig(stat="auto", use_referee=False, use_rest=False, use_travel=False)
    preds = backtest(no_xg, start="2025-01-01", end="2025-06-01", cfg=cfg)
    rep = report(preds)
    assert rep["rated"] > 0   # fell back to sot rather than giving up


def test_auto_stat_uses_xg_when_available(synthetic_matches):
    # synthetic_matches has home_xg/away_xg fully populated - auto should use it.
    cfg_auto = BacktestConfig(stat="auto", use_referee=False, use_rest=False, use_travel=False)
    cfg_xg = BacktestConfig(stat="xg", use_referee=False, use_rest=False, use_travel=False)
    auto_preds = backtest(synthetic_matches, start="2025-01-01", end="2025-06-01", cfg=cfg_auto)
    xg_preds = backtest(synthetic_matches, start="2025-01-01", end="2025-06-01", cfg=cfg_xg)
    # identical results, since auto picked xg (the same as an explicit xg run)
    pd.testing.assert_series_equal(
        auto_preds.sort_values("match_id")["p_home"].reset_index(drop=True),
        xg_preds.sort_values("match_id")["p_home"].reset_index(drop=True),
    )
