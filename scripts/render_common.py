"""Shared rendering plumbing for the three public pages.

Each template is a self-contained HTML document with three placeholders:

    __BASE_CSS__                 -> the contents of web/base.css (tokens, layout,
                                   shared components) inlined at render time, so
                                   base.css is the single source of truth and the
                                   page still ships as one file with no extra request
    const DATA = __DATA_JSON__;  -> the page's data payload as a JS literal
    __GENERATED__               -> the build timestamp

`finalize()` performs all three substitutions and fails loudly if a template is
missing one - a silently un-substituted placeholder is a broken page.
"""
from __future__ import annotations

import json
from pathlib import Path

import config

BASE_CSS = config.ROOT / "web" / "base.css"

_DATA_MARKER = "const DATA = __DATA_JSON__;"


def base_css() -> str:
    return BASE_CSS.read_text(encoding="utf-8")


def finalize(template_html: str, payload, generated: str, *, where: str = "template") -> str:
    if "__BASE_CSS__" not in template_html:
        raise RuntimeError(f"{where}: missing __BASE_CSS__ placeholder")
    if _DATA_MARKER not in template_html:
        raise RuntimeError(f"{where}: missing '{_DATA_MARKER}' placeholder")
    if "__GENERATED__" not in template_html:
        raise RuntimeError(f"{where}: missing __GENERATED__ placeholder")

    data_js = json.dumps(payload, separators=(",", ":"))
    return (
        template_html
        .replace("__BASE_CSS__", base_css())
        .replace(_DATA_MARKER, f"const DATA = {data_js};")
        .replace("__GENERATED__", generated)
    )


def utc_now_str() -> str:
    import pandas as pd
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
