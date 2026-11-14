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


def _repo_files() -> list[str]:
    """Tracked + untracked-but-not-ignored files (so .lanes/, .venv/ etc. are skipped); plain walk without git."""
    import subprocess
    try:
        r = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=REPO,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0 and r.stdout.strip():
            return [f for f in r.stdout.splitlines() if (REPO / f).is_file()]
    except OSError:
        pass
    skip_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", ".codegraph", ".lanes", ".t3", ".t3-worktrees"}
    return [p.relative_to(REPO).as_posix() for p in REPO.rglob("*")
            if p.is_file() and not (set(p.relative_to(REPO).parts) & skip_dirs) and p.name != ".lane"]


def test_every_repo_file_has_exactly_one_owner():
    files = _repo_files()
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


def test_main_only_takes_human_files():
    assert co.violations(["docs/HANDOFF.md", "worklog/DISPATCH.md", "docs/PLAN.json"], "HUMAN", OWNERS) == []
    assert co.violations(["vanalens/scorers.py"], "HUMAN", OWNERS) == [("vanalens/scorers.py", "L3")]


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


nc = _load("next_cards")
cp = _load("card_prompt")
PLAN = nc.load_plan(REPO / "docs" / "PLAN.json")


def _e(time, card, status, needs="none"):
    return {"time": time, "card": card, "status": status, "needs": needs, "next": ""}


def test_next_cards_first_wave():
    from datetime import datetime
    out = nc.compute(PLAN, {}, {}, datetime(2026, 11, 15, 3, 0))
    cards = [d["card"] for d in out["dispatch"]]
    assert cards == ["L1-a", "L2-a", "L3-a", "L4-a", "L5-a"]   # full first wave, 2 per Antigravity account
    assert not out["all_done"] and out["needs_merge"] == []


def test_next_cards_provider_cap_and_flow():
    from datetime import datetime
    now = datetime(2026, 11, 15, 4, 0)
    allx = {"L1-a": [_e("2026-11-15 03:00", "L1-a", "STARTED")], "L2-a": [_e("2026-11-15 03:00", "L2-a", "STARTED")]}
    out = nc.compute(PLAN, {}, allx, now)
    assert set(out["in_flight"]) == {"L1-a", "L2-a"}
    assert all(d["provider"] != "Antigravity A" for d in out["dispatch"])   # A is at its 2-thread cap
    # L1-a DONE on its branch but not merged -> needs merge, L1-b must wait
    allx["L1-a"].append(_e("2026-11-15 03:40", "L1-a", "DONE"))
    out = nc.compute(PLAN, {}, allx, now)
    assert out["needs_merge"] == ["L1"] and "L1-b" not in [d["card"] for d in out["dispatch"]]
    # merged -> L1-b dispatchable
    out = nc.compute(PLAN, {"L1-a": allx["L1-a"]}, allx, now)
    assert "L1-b" in [d["card"] for d in out["dispatch"]]


def test_next_cards_stale_and_blocked():
    from datetime import datetime
    now = datetime(2026, 11, 15, 9, 0)
    allx = {"L3-a": [_e("2026-11-15 05:00", "L3-a", "STARTED")],
            "L4-a": [_e("2026-11-15 08:30", "L4-a", "BLOCKED", "L2: encode_text")]}
    out = nc.compute(PLAN, {}, allx, now)
    d = {x["card"]: x for x in out["dispatch"]}
    assert d["L3-a"]["resume"] is True and out["stale"] == ["L3-a"]
    assert out["blocked"] == [{"card": "L4-a", "needs": "L2: encode_text"}] and "L4-a" not in d


def test_every_plan_card_has_a_prompt():
    tasks = (REPO / "docs" / "TASKS.md").read_text(encoding="utf-8")
    for c in PLAN["cards"] + [{"id": "L6-m"}]:
        text = cp.extract(tasks, c["id"])
        assert text, f"no prompt for {c['id']}"
        if c["id"] != "L6-m":
            lane = c["id"].split("-")[0]
            assert f"worklog/{lane}.md" in text and PLAN["lanes"][lane]["branch"] in text
            assert f"start {c['id']}" in text, f"{c['id']} must commit its STARTED entry immediately"


def test_card_prompts_use_lane_folders():
    tasks = (REPO / "docs" / "TASKS.md").read_text(encoding="utf-8")
    assert "pull --rebase" not in tasks
    for c in PLAN["cards"]:
        lane = c["id"].split("-")[0]
        text = cp.extract(tasks, c["id"])
        assert f"scripts/lane.py {lane} --start {c['id']}" in text and f".lanes/{lane}" in text
        assert PLAN["lanes"][lane]["project"] == PLAN["project"]          # one T3 project for everything
    fix = cp.fix_prompt("L2", "encode_text must return float32")
    assert "scripts/lane.py L2 --start L2-fix" in fix and "lane/L2-encoders" in fix and "float32" in fix
    import re
    assert re.search(r"--start L2-fix\d{6}`", fix)                                   # unique per round


def test_fix_card_keeps_lane_busy():
    from datetime import datetime
    now = datetime(2026, 11, 15, 4, 0)
    allx = {"L3-fix150350": [_e("2026-11-15 03:50", "L3-fix150350", "STARTED")]}
    out = nc.compute(PLAN, {}, allx, now)
    assert "L3-fix150350" in out["in_flight"]
    assert "L3-a" not in [d["card"] for d in out["dispatch"]]          # lane L3 is busy with the fix


def test_second_fix_round_needs_merge():
    from datetime import datetime
    now = datetime(2026, 11, 15, 6, 0)
    done1 = _e("2026-11-15 04:00", "L3-fix", "DONE")
    done2 = _e("2026-11-15 05:30", "L3-fix", "DONE")
    out = nc.compute(PLAN, {"L3-fix": [done1]}, {"L3-fix": [done1, done2]}, now)
    assert out["needs_merge"] == ["L3"]


def test_status_parse_tolerates_windows_encodings():
    text = "﻿## 2026-11-15 02:10 | L3-a | STARTED\r\n- Needs: none\r\n"
    assert [e["card"] for e in st.parse(text)] == ["L3-a"]
    garbled = "## 2026-11-15 02:10 | L1-a | DONE\n".encode("utf-16-le").decode("utf-8", errors="replace")
    assert [e["status"] for e in st.parse(garbled)] == ["DONE"]


def test_merge_gate_selection(tmp_path):
    ml = _load("merge_lanes")
    (tmp_path / "tests").mkdir()
    for f in ("test_tooling.py", "test_eval.py", "test_core.py", "test_context.py", "test_l3_extra.py"):
        (tmp_path / "tests" / f).write_text("")
    gate = ml.gate_tests(PLAN, "L3", green={"L1"}, root=tmp_path)
    assert gate == ["tests/test_tooling.py", "tests/test_eval.py", "tests/test_core.py", "tests/test_l3_extra.py"]
    assert "tests/test_context.py" not in gate                       # L4 not merged yet: its stubs would fail
    assert ml.gate_tests(PLAN, "L7", green=set(), root=tmp_path) == ["tests/test_tooling.py"]
