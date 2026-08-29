import pytest

from scripts import render_common


def test_finalize_inlines_base_css_and_data_and_timestamp():
    tmpl = ('<style>__BASE_CSS__</style>'
            '<div>__GENERATED__</div>'
            '<script>const DATA = __DATA_JSON__;</script>')
    out = render_common.finalize(tmpl, [{"a": 1}], "2026-08-29 10:00 UTC")
    assert "__BASE_CSS__" not in out
    assert "__GENERATED__" not in out
    assert "__DATA_JSON__" not in out
    assert "2026-08-29 10:00 UTC" in out
    assert 'const DATA = [{"a":1}];' in out
    # a token that only exists in base.css
    assert "--viz-model" in out and "scrollbar-gutter" in out


@pytest.mark.parametrize("missing", [
    '<div>__GENERATED__</div><script>const DATA = __DATA_JSON__;</script>',      # no __BASE_CSS__
    '<style>__BASE_CSS__</style><script>const DATA = __DATA_JSON__;</script>',    # no __GENERATED__
    '<style>__BASE_CSS__</style><div>__GENERATED__</div>',                        # no data marker
])
def test_finalize_fails_loudly_on_missing_placeholder(missing):
    with pytest.raises(RuntimeError):
        render_common.finalize(missing, [], "t")


def test_base_css_file_is_non_trivial():
    css = render_common.base_css()
    assert len(css) > 2000
    assert ":root" in css and "@media (prefers-color-scheme: dark)" in css


def test_all_three_templates_carry_the_placeholders():
    import scripts.render_coupon as rc
    import scripts.render_why as rw
    import scripts.track_record as tr
    for tmpl in (rc.TEMPLATE, rw.TEMPLATE, tr.TEMPLATE):
        html = tmpl.read_text(encoding="utf-8")
        assert "__BASE_CSS__" in html, tmpl
        assert "const DATA = __DATA_JSON__;" in html, tmpl
        assert "__GENERATED__" in html, tmpl
        assert html.lstrip().startswith("<!DOCTYPE html>"), tmpl
