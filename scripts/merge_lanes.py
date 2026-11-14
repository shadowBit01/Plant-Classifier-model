#!/usr/bin/env python3
"""Merge finished lanes into main, safely and deterministically (stdlib only).

Run in the MAIN checkout, on branch main, with a clean working tree:
    python scripts/merge_lanes.py

For every lane that has a DONE entry not yet on main:
  1. ownership check on main...origin/<lane branch> (docs/OWNERS)
  2. git merge --no-ff --no-commit origin/<lane branch>   (conflict -> git merge --abort)
  3. gate tests: tests/test_tooling.py + this lane's tests (docs/PLAN.json "tests" + tests/test_<lane>_*.py)
     + the tests of every lane already merged green. Unstarted lanes are skipped: their modules are still stubs.
  4. green -> commit "merge <LANE>"; red -> git merge --abort (main untouched)
Lanes with a fix thread in flight are skipped this run.
Then logs one entry to worklog/MERGE.md, pushes main, and prints JSON {"merged": [...], "failed": [...]}.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import next_cards as nc  # noqa: E402
from check_owners import load_owners, violations  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def run(*cmd: str) -> tuple[int, str]:
    r = subprocess.run(list(cmd), cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout + r.stderr).strip()


def lane_tests(plan: dict, lane: str, root: Path = REPO) -> list[str]:
    files = [f for f in plan["lanes"][lane].get("tests", []) if (root / f).exists()]
    files += sorted(p.relative_to(root).as_posix() for p in (root / "tests").glob(f"test_{lane.lower()}_*.py"))
    return files


def gate_tests(plan: dict, lane: str, green: set[str], root: Path = REPO) -> list[str]:
    """tooling tests + this lane's tests + tests of every lane already merged green (they must stay green).
    Lanes that have not finished a card yet are left out: their modules are still stubs."""
    out = ["tests/test_tooling.py"]
    for ln in plan["lanes"]:
        if ln == lane or ln in green:
            out += [f for f in lane_tests(plan, ln, root) if f not in out]
    return out


def tail(text: str, n: int = 12) -> str:
    return "\n".join(text.splitlines()[-n:])


def main() -> int:
    _, branch = run("git", "rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        print(json.dumps({"error": f"run on main, not {branch}"}))
        return 1
    _, dirty = run("git", "status", "--porcelain", "--untracked-files=no")
    if dirty:
        print(json.dumps({"error": "main checkout has uncommitted changes", "detail": dirty}))
        return 1
    if run("git", "pull", "--ff-only", "origin", "main")[0] != 0:
        run("git", "pull", "--no-rebase", "--no-edit", "origin", "main")

    plan, owners = nc.load_plan(), load_owners()
    st = nc.state(fetch=True)
    plan_ids = {c["id"] for c in plan["cards"]}
    fixing = {c.split("-")[0] for c in st["in_flight"] if c not in plan_ids}   # a fix thread is working on it
    green = {c.split("-")[0] for c in st["done_merged"]}
    merged, failed, skipped = [], [], []
    for lane in st["needs_merge"]:
        if lane in fixing:
            skipped.append(lane)
            continue
        ref = f"origin/{plan['lanes'][lane]['branch']}"
        tip = run("git", "rev-parse", "--short", ref)[1]
        _, diff = run("git", "diff", "--name-only", f"main...{ref}")
        bad = violations([p for p in diff.splitlines() if p], lane, owners)
        if bad:
            failed.append({"lane": lane, "tip": tip, "reason": "ownership", "detail": "; ".join(f"{p} (owned by {o})" for p, o in bad)})
            continue
        code, out = run("git", "merge", "--no-ff", "--no-commit", ref)
        if code != 0:
            run("git", "merge", "--abort")
            failed.append({"lane": lane, "tip": tip, "reason": "conflict", "detail": tail(out)})
            continue
        if run("git", "rev-parse", "-q", "--verify", "MERGE_HEAD")[0] != 0:
            continue  # already up to date
        tests = gate_tests(plan, lane, green)
        code, out = run(sys.executable, "-m", "pytest", "-q", *tests)
        if code != 0:
            run("git", "merge", "--abort")
            failed.append({"lane": lane, "tip": tip, "reason": "tests", "detail": tail(out)})
            continue
        run("git", "commit", "-m", f"merge {lane}")
        merged.append(lane)
        green.add(lane)

    key = ",".join(f"{f['lane']}@{f['tip']}:{f['reason']}" for f in failed)
    log = REPO / "worklog" / "MERGE.md"
    last_key = ""
    if log.exists():
        keys = [ln[len("- Key: "):].strip() for ln in log.read_text(encoding="utf-8").splitlines() if ln.startswith("- Key: ")]
        last_key = keys[-1] if keys else ""
    if merged or (failed and key != last_key):   # do not re-log an identical failure every run
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        needs = "; ".join(f"{f['lane']}: merge failed ({f['reason']}): {f['detail'].splitlines()[-1] if f['detail'] else ''}"
                          for f in failed) or "none"
        entry = (f"\n## {now} | L6-m | DONE\n- Did: merged {', '.join(merged) or 'nothing'}; "
                 f"failed {', '.join(f['lane'] for f in failed) or 'none'}\n- Needs: {needs}\n"
                 f"- Key: {key}\n- Next: dispatcher routes failures to their lanes\n")
        with log.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(entry)
        run("git", "add", "worklog/MERGE.md")
        run("git", "commit", "-m", f"[MERGE] merged {','.join(merged) or '-'} failed {','.join(f['lane'] for f in failed) or '-'}")
    if run("git", "rev-list", "--count", "origin/main..main")[1] not in ("", "0"):   # also retries an earlier failed push
        code, out = run("git", "push", "origin", "main")
        if code != 0:
            failed.append({"lane": "-", "tip": "", "reason": "push", "detail": tail(out)})
    print(json.dumps({"merged": merged, "failed": failed, "skipped_fix_in_flight": skipped}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
