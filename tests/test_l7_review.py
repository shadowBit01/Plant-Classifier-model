"""L7-d: review, demo script and refreshed report stay consistent (no network, no GPU)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EM = "—"


def test_review_covers_required_topics() -> None:
    s = (ROOT / "pitch/report/review.md").read_text(encoding="utf-8").lower()
    for topic in ("claims without numbers", "confidence intervals", "contamination", "geographic bias",
                  "few test photos per species", "baselines not run", "unfair comparisons", "overclaiming",
                  "unclear explanations", "slide design"):
        assert topic in s, topic
    assert EM not in s


def test_demo_covers_five_scenarios() -> None:
    s = (ROOT / "pitch/demo.md").read_text(encoding="utf-8").lower()
    for topic in ("a known plant", "an unknown plant", "add a species", "three photos", "location", "wi-fi fails"):
        assert topic in s, topic
    assert EM not in s


def test_report_has_no_stale_claims() -> None:
    html = (ROOT / "pitch/report/VanaLens_report.html").read_text(encoding="utf-8")
    assert "from 0.6% to 0.6%" not in html
    assert "rests on a single re-run" not in html
    assert "{{" not in html and EM not in html
    assert "Wilson interval" in html
