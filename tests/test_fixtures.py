import pytest
import requests

from ingest import fixtures


_GOOD = (b"\xef\xbb\xbfDiv,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\n"
         b"E0,13/09/2026,15:00,Arsenal,Chelsea,1.8,3.6,4.5\n")


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self.content = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}", response=self)


def test_download_fixtures_caches_last_good(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, "_CACHE", tmp_path / "fixtures.csv")
    monkeypatch.setattr(fixtures.SESSION, "get", lambda *a, **k: _Resp(200, _GOOD))

    df = fixtures.download_fixtures()
    assert list(df["HomeTeam"]) == ["Arsenal"]
    assert (tmp_path / "fixtures.csv").read_bytes() == _GOOD


def test_download_fixtures_falls_back_to_cache_on_503(tmp_path, monkeypatch):
    cache = tmp_path / "fixtures.csv"
    cache.write_bytes(_GOOD)
    monkeypatch.setattr(fixtures, "_CACHE", cache)
    monkeypatch.setattr(fixtures.SESSION, "get", lambda *a, **k: _Resp(503, b"<html>nope</html>"))

    df = fixtures.download_fixtures()
    assert list(df["AwayTeam"]) == ["Chelsea"]


def test_download_fixtures_rejects_html_and_falls_back(tmp_path, monkeypatch):
    cache = tmp_path / "fixtures.csv"
    cache.write_bytes(_GOOD)
    monkeypatch.setattr(fixtures, "_CACHE", cache)
    # HTTP 200 but an nginx "temporarily unavailable" body, not the CSV.
    monkeypatch.setattr(fixtures.SESSION, "get",
                        lambda *a, **k: _Resp(200, b"<html><body>unavailable</body></html>"))

    df = fixtures.download_fixtures()
    assert list(df["HomeTeam"]) == ["Arsenal"]
    # the bad body must not have overwritten the good cache
    assert cache.read_bytes() == _GOOD


def test_download_fixtures_raises_when_down_and_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, "_CACHE", tmp_path / "fixtures.csv")
    monkeypatch.setattr(fixtures.SESSION, "get",
                        lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.ConnectionError()))

    with pytest.raises(RuntimeError, match="no cached copy"):
        fixtures.download_fixtures()
