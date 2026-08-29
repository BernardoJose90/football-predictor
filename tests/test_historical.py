import pytest

from ingest import historical


class _FakeResp:
    def __init__(self, status_code, content_type, body):
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.content = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_download_rejects_html_error_page(tmp_path, monkeypatch):
    # football-data.co.uk answers a missing file with HTTP 300 + an HTML
    # "Multiple Choices" listing, not a 404 - this reproduces that exact case.
    html = b"<!DOCTYPE HTML><html><body>Multiple Choices...</body></html>" * 20
    monkeypatch.setattr(historical.config, "DATA_RAW", tmp_path)
    monkeypatch.setattr(
        historical.requests, "get",
        lambda *a, **k: _FakeResp(300, "text/html; charset=iso-8859-1", html),
    )
    with pytest.raises(historical.FileUnavailable):
        historical.download("D1", "2627")
    assert not (tmp_path / "D1_2627.csv").exists()


def test_download_all_skips_unavailable_files_without_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(historical.config, "DATA_RAW", tmp_path)
    monkeypatch.setattr(historical.config, "LEAGUES", {"E0": "Premier League", "D1": "Bundesliga"})
    monkeypatch.setattr(historical.config, "SEASONS", ["2627"])

    good_csv = (b'\xef\xbb\xbfDiv,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n'
               b'E0,15/08/2026,Arsenal,Chelsea,2,1,H\n') * 20

    def fake_get(url, *a, **k):
        if "/D1.csv" in url:
            return _FakeResp(300, "text/html", b"<html>not found</html>" * 20)
        return _FakeResp(200, "text/csv", good_csv)

    monkeypatch.setattr(historical.requests, "get", fake_get)
    paths = historical.download_all()
    names = {p.name for p in paths}
    assert "E0_2627.csv" in names
    assert "D1_2627.csv" not in names
    assert not (tmp_path / "D1_2627.csv").exists()


def test_download_accepts_valid_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(historical.config, "DATA_RAW", tmp_path)
    good_csv = (b'\xef\xbb\xbfDiv,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n'
               b'E0,15/08/2026,Arsenal,Chelsea,2,1,H\n') * 20
    monkeypatch.setattr(historical.requests, "get",
                        lambda *a, **k: _FakeResp(200, "text/csv", good_csv))
    path = historical.download("E0", "2627")
    assert path.exists()
    assert path.read_bytes() == good_csv
