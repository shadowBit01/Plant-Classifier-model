"""Human-owned tests: ownership map is complete and non-overlapping; status parser works."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


co = _load("check_owners")
st = _load("status")
OWNERS = co.load_owners(REPO / "docs" / "OWNERS")


def test_lane_permissions():
    assert co.allowed("vanalens/scorers.py", "L3", OWNERS)
    assert co.allowed("worklog/L3.md", "L3", OWNERS)
    assert co.allowed("runs/L3/ablation.json", "L3", OWNERS)
    assert co.allowed("tests/test_l3_extra.py", "L3", OWNERS)
    assert not co.allowed("vanalens/encoders.py", "L3", OWNERS)
    assert not co.allowed("worklog/L2.md", "L3", OWNERS)
    assert not co.allowed("vanalens/contracts.py", "L3", OWNERS)
    assert co.owner_of("vanalens/encoders.py", OWNERS) == "L2"
    assert co.owner_of("vanalens/contracts.py", OWNERS) == "HUMAN"


def test_every_repo_file_has_exactly_one_owner():
    skip_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", ".codegraph"}
    files = [p.relative_to(REPO).as_posix() for p in REPO.rglob("*")
             if p.is_file() and not (set(p.relative_to(REPO).parts) & skip_dirs) and p.name != ".lane"]
    unowned, multi = [], []
    for f in files:
        hits = {lane for lane, pats in OWNERS.items() if any(co.match(f, p) for p in pats)}
        hits |= {lane for lane in co.LANES if any(co.match(f, p) for p in co.implicit(lane))}
        if f.startswith("runs/logs/"):
            continue
        if not hits:
            unowned.append(f)
        elif len(hits) > 1:
            multi.append((f, sorted(hits)))
    assert not unowned, f"add these to docs/OWNERS: {unowned}"
    assert not multi, f"owned by more than one lane: {multi}"


def test_violations_report_owner():
    bad = co.violations(["vanalens/scorers.py", "app/app.py"], "L3", OWNERS)
    assert bad == [("app/app.py", "L5")]


def test_status_parse():
    text = """# Worklog L3
## 2026-11-15 02:10 | L3-a | STARTED
- Plan: x
- Needs: none
## 2026-11-15 03:05 | L3-a | DONE
- Tests: pytest -> 13 passed
- Needs: L2: embeddings for dimpsar
- Next: L3-b fit_core.py
"""
    es = st.parse(text)
    assert [e["status"] for e in es] == ["STARTED", "DONE"]
    assert es[-1]["needs"] == "L2: embeddings for dimpsar" and es[-1]["next"] == "L3-b fit_core.py"


def test_lane_file_encodings(tmp_path):
    f = tmp_path / ".lane"
    f.write_bytes("L3\r\n".encode("utf-16"))          # PowerShell 5.1 `echo L3 > .lane`
    assert co.read_lane_file(f) == "L3"
    f.write_bytes(b"\xef\xbb\xbfl5\n")                 # UTF-8 with BOM, lowercase
    assert co.read_lane_file(f) == "L5"
    f.write_text("L2\n")
    assert co.read_lane_file(f) == "L2"
    f.write_text("nothing")
    assert co.read_lane_file(f) is None
