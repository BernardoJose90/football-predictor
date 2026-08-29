import pytest

from normalise import teams


def test_known_team_resolves_to_itself():
    canon = next(iter(teams.canonical_names()))
    assert teams.resolve(canon) == canon


def test_unknown_team_raises():
    with pytest.raises(teams.UnknownTeamError):
        teams.resolve("Definitely Not A Real Club FC")


def test_resolve_series_collects_all_missing_names():
    names = ["Definitely Not Real FC", "Also Not Real United"]
    with pytest.raises(teams.UnknownTeamError) as exc:
        teams.resolve_series(names)
    msg = str(exc.value)
    assert "Definitely Not Real FC" in msg
    assert "Also Not Real United" in msg


def test_alias_resolves_to_canonical(tmp_path, monkeypatch):
    yaml_path = tmp_path / "teams.yaml"
    yaml_path.write_text("Manchester United:\n  - Man Utd\n  - Man United\n")
    monkeypatch.setattr(teams, "_YAML", yaml_path)
    teams.reload_cache()
    try:
        assert teams.resolve("Man Utd") == "Manchester United"
        assert teams.resolve("Manchester United") == "Manchester United"
        with pytest.raises(teams.UnknownTeamError):
            teams.resolve("Man United FC")  # not listed, must not fuzzy-match
    finally:
        teams.reload_cache()  # restore the real alias map for other tests
