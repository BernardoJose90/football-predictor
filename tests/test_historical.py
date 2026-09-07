import pytest
import requests

from ingest import historical


class _FakeResp:
    def __init__(self, status_code, content_type, body):
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.content = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}", response=self)


@pytest.fixture(autouse=True)
def _no_courtesy_pause(monkeypatch):
    monkeypatch.setattr(historical, "INTER_REQUEST_PAUSE", 0.0)


def _patch_get(monkeypatch, fn):
    """Route historical's session GET through a fake, whatever the session is."""
    monkeypatch.setattr(historical.SESSION, "get", fn)


GOOD_CSV = (b'\xef\xbb\xbfDiv,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n'
            b'E0,15/08/2026,Arsenal,Chelsea,2,1,H\n') * 20


def test_download_rejects_html_error_page(tmp_path, monkeypatch):
    # football-data.co.uk answers a missing file with HTTP 300 + an HTML
    # "Multiple Choices" listing, not a 404 - this reproduces that exact case.
    html = b"<!DOCTYPE HTML><html><body>Multiple Choices...</body></html>" * 20
    monkeypatch.setattr(historical.config, "DATA_RAW", tmp_path)
    _patch_get(monkeypatch, lambda *a, **k: _FakeResp(300, "text/html; charset=iso-8859-1", html))
    with pytest.raises(historical.FileUnavailable):
        historical.download("D1", "2627")
    assert not (tmp_path / "D1_2627.csv").exists()


def test_download_all_skips_unavailable_files_without_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(historical.config, "DATA_RAW", tmp_path)
    monkeypatch.setattr(historical.config, "LEAGUES", {"E0": "Premier League", "D1": "Bundesliga"})
    monkeypatch.setattr(historical.config, "SEASONS", ["2627"])
    monkeypatch.setattr(historical.config, "CURRENT_SEASON", "2627")

    def fake_get(url, *a, **k):
        if "/D1.csv" in url:
            return _FakeResp(300, "text/html", b"<html>not found</html>" * 20)
        return _FakeResp(200, "text/csv", GOOD_CSV)

    _patch_get(monkeypatch, fake_get)
    paths = historical.download_all()
    names = {p.name for p in paths}
    assert "E0_2627.csv" in names
    assert "D1_2627.csv" not in names
    assert not (tmp_path / "D1_2627.csv").exists()


def test_download_accepts_valid_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(historical.config, "DATA_RAW", tmp_path)
    _patch_get(monkeypatch, lambda *a, **k: _FakeResp(200, "text/csv", GOOD_CSV))
    path = historical.download("E0", "2627")
    assert path.exists()
    assert path.read_bytes() == GOOD_CSV


def test_session_retries_transient_status_codes():
    # The retry policy is what absorbed the real-world 503s - assert it is
    # actually wired onto the session for the football-data.co.uk host.
    adapter = historical.SESSION.get_adapter("https://www.football-data.co.uk/")
    retry = adapter.max_retries
    assert retry.total and retry.total >= 3
    for code in (429, 500, 502, 503, 504):
        assert code in retry.status_forcelist
    # A multi-minute Retry-After must not be honoured verbatim - that would
    # blow the CI job budget; the sustained-throttle path is the disk cache.
    assert retry.respect_retry_after_header is False


def test_download_all_falls_back_to_cache_on_transient_error(tmp_path, monkeypatch):
    # The in-progress season IS re-fetched under force, the upstream is down,
    # but last week's copy is on disk - the run must survive on that.
    monkeypatch.setattr(historical.config, "DATA_RAW", tmp_path)
    monkeypatch.setattr(historical.config, "LEAGUES", {"E0": "Premier League"})
    monkeypatch.setattr(historical.config, "SEASONS", ["2627"])
    monkeypatch.setattr(historical.config, "CURRENT_SEASON", "2627")

    cached = tmp_path / "E0_2627.csv"
    cached.write_bytes(GOOD_CSV)

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("reset by peer")

    _patch_get(monkeypatch, boom)
    paths = historical.download_all(force=True, current_only=True)
    assert paths == [cached]
    assert cached.read_bytes() == GOOD_CSV


def test_download_all_reraises_transient_error_when_nothing_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(historical.config, "DATA_RAW", tmp_path)
    monkeypatch.setattr(historical.config, "LEAGUES", {"E0": "Premier League"})
    monkeypatch.setattr(historical.config, "SEASONS", ["2627"])
    monkeypatch.setattr(historical.config, "CURRENT_SEASON", "2627")

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("reset by peer")

    _patch_get(monkeypatch, boom)
    with pytest.raises(requests.exceptions.ConnectionError):
        historical.download_all()


def test_current_only_forces_just_the_in_progress_season(tmp_path, monkeypatch):
    monkeypatch.setattr(historical.config, "DATA_RAW", tmp_path)
    monkeypatch.setattr(historical.config, "LEAGUES", {"E0": "Premier League"})
    monkeypatch.setattr(historical.config, "SEASONS", ["2324", "2627"])
    monkeypatch.setattr(historical.config, "CURRENT_SEASON", "2627")

    for season in ("2324", "2627"):
        (tmp_path / f"E0_{season}.csv").write_bytes(GOOD_CSV)

    fetched = []

    def fake_get(url, *a, **k):
        fetched.append(url)
        return _FakeResp(200, "text/csv", GOOD_CSV)

    _patch_get(monkeypatch, fake_get)
    historical.download_all(force=True, current_only=True)
    assert fetched == [f"{historical.config.FOOTBALL_DATA_BASE}/2627/E0.csv"]

    # ...whereas a plain force=True (the manual entry points) re-fetches all.
    fetched.clear()
    historical.download_all(force=True)
    assert sorted(fetched) == [
        f"{historical.config.FOOTBALL_DATA_BASE}/2324/E0.csv",
        f"{historical.config.FOOTBALL_DATA_BASE}/2627/E0.csv",
    ]
