import pandas as pd
import pytest

from ingest import club_elo


def test_anchor_uses_rated_overlap_then_falls_back():
    elo_map = {"Arsenal": 1900, "Chelsea": 1800, "Spurs": 1750, "Everton": 1600,
               "Wolves": 1550, "Brighton": 1650}
    rated = ["Arsenal", "Chelsea", "Spurs", "Everton", "Wolves"]
    assert club_elo.anchor(elo_map, rated) == pytest.approx((1900 + 1800 + 1750 + 1600 + 1550) / 5)

    # thin overlap -> mean of the whole map
    assert club_elo.anchor(elo_map, ["Arsenal", "Chelsea"]) == pytest.approx(
        sum(elo_map.values()) / len(elo_map))


def test_snapshot_caches_and_reads_back(monkeypatch, tmp_path):
    monkeypatch.setattr(club_elo, "CACHE_DIR", tmp_path)
    calls = []

    def fake_fetch(date):
        calls.append(pd.Timestamp(date))
        return pd.DataFrame({"team": ["Arsenal", "Barcelona"], "elo": [1950.0, 2010.0]})

    monkeypatch.setattr(club_elo, "_fetch", fake_fetch)

    a = club_elo.snapshot("2024-09-17")
    b = club_elo.snapshot("2024-09-17")            # second call served from disk
    assert len(calls) == 1
    assert list(a["team"]) == list(b["team"]) == ["Arsenal", "Barcelona"]
    assert (tmp_path / "2024-09-17.csv").exists()


def test_elo_by_date_maps_matchday_to_team_elo(monkeypatch, tmp_path):
    monkeypatch.setattr(club_elo, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(club_elo, "_fetch", lambda d: pd.DataFrame(
        {"team": ["Arsenal", "Barcelona"], "elo": [1950.0, 2010.0]}))

    lookup = club_elo.elo_by_date(["2024-09-17", "2024-09-17"])
    key = pd.Timestamp("2024-09-17")
    assert lookup[key]["Barcelona"] == 2010.0


def test_snapshot_degrades_to_cache_on_fetch_error(monkeypatch, tmp_path):
    monkeypatch.setattr(club_elo, "CACHE_DIR", tmp_path)
    (tmp_path).mkdir(exist_ok=True)
    pd.DataFrame({"team": ["Arsenal"], "elo": [1900.0]}).to_csv(
        tmp_path / "2024-09-17.csv", index=False)

    def boom(date):
        raise RuntimeError("clubelo down")

    monkeypatch.setattr(club_elo, "_fetch", boom)
    out = club_elo.snapshot("2024-09-17", refresh=True)
    assert list(out["team"]) == ["Arsenal"]
