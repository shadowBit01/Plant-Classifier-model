"""Lane L7: the project report builds from runs/ with every placeholder filled (no PDF step, no Edge needed)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("build_report", REPO / "pitch" / "report" / "build_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_html_builds_and_is_filled():
    br = _load()
    html = br.build_html().read_text(encoding="utf-8")
    assert "{{" not in html and "}}" not in html
    assert "—" not in html
    for sec in ("Introduction", "How the system works", "Approach and methods", "Dataset", "Experimental setup",
                "Results", "Limitations", "How we plan", "Conclusion", "References"):
        assert sec in html, sec
    assert html.count("<table>") >= 8
    assert "Table a table below" not in html and "a figure" not in html


def test_formatting_helpers():
    br = _load()
    assert br.pct(0.9386) == "93.9%"
    assert br.pct(None) == br.NM
    assert br.pct_ci({"value": 0.5, "ci": [0.4, 0.6]}) == "50.0% (95% CI 40.0 to 60.0)"
    assert br.effect(0.90, 0.92) == "raised top-1 by 2.0 points"
    assert br.dig({"a": [{"b": 3}]}, "a.0.b") == 3
