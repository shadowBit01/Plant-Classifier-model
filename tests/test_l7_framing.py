"""L7-f: final framing pass. Frozen encoders read as a design choice, scope section is concise, self-review exists."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EM = "—"


def _text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_no_admission_phrasing_in_report_deck_or_script() -> None:
    banned = ("no fine-tuning", "never fine-tunes", "ruled out", "obvious approach", "why no fine-tuning",
              "not perfect yet", "before the dino fix", "random attention")
    for rel in ("pitch/report/VanaLens_report.md", "pitch/report/VanaLens_report.html", "scripts/build_deck.py",
                "pitch/script.md"):
        s = _text(rel).lower()
        for b in banned:
            assert b not in s, (rel, b)
        assert EM not in s, rel


def test_design_choice_and_scope_sections() -> None:
    html = _text("pitch/report/VanaLens_report.html")
    assert "Design choice: frozen foundation features with light trained heads" in html
    assert "Contributions at a glance" in html
    assert "Engineering safeguard: encoder weight integrity" in html
    md = _text("pitch/report/VanaLens_report.md")
    sec = md.split("## Limitations: current scope and next steps", 1)[1].split("## How we plan", 1)[0]
    rows = [ln for ln in sec.splitlines() if ln.startswith("| **")]
    assert 1 <= len(rows) <= 6


def test_no_hidden_or_tiny_text() -> None:
    html = _text("pitch/report/VanaLens_report.html").lower()
    assert "display:none" not in html.replace(" ", "")
    assert not re.search(r"font-size:\s*[0-6](\.\d+)?(px|pt)", html)


def test_ai_review_written() -> None:
    s = _text("pitch/report/ai_review.md").lower()
    for t in ("strengths", "novelty", "drawbacks", "pass 2"):
        assert t in s, t
    assert EM not in s
