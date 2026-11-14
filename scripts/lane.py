#!/usr/bin/env python3
"""Prepare a lane's working folder inside this one project, and optionally log the card start (stdlib only).

  python scripts/lane.py L1                  # create or refresh .lanes/L1, print where it is
  python scripts/lane.py L1 --start L1-a     # same, then append + commit + push the STARTED entry

.lanes/<LANE> (gitignored) is a git worktree on local branch lanework/<LANE>, tracking origin/lane/<...>.
Every run merges origin/<lane branch> and origin/main into it (merge, never rebase), so pushes stay
fast-forward. All lane work happens inside that folder; the project root stays on main for the dispatcher.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def git(cwd: Path, *args: str, check: bool = False) -> tuple[int, str]:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout + r.stderr).strip()
    if check and r.returncode != 0:
        sys.exit(f"lane.py: `git {' '.join(args)}` failed in {cwd}:\n{out[-1500:]}\n"
                 "If local changes block it, commit them inside the lane folder ([LANE] wip) and run lane.py again.")
    return r.returncode, out


def main_root() -> Path:
    """The project's main checkout, even when called from inside a lane folder or another worktree."""
    _, out = git(REPO, "worktree", "list", "--porcelain")
    for line in out.splitlines():
        if line.startswith("worktree "):
            return Path(line[len("worktree "):].strip())
    return REPO


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lane")
    ap.add_argument("--start", metavar="CARD", help="card id to log as STARTED, e.g. L1-a or L2-fix151030")
    ap.add_argument("--plan", default="see card", help="one-line plan for the STARTED entry")
    a = ap.parse_args(argv)

    lane = a.lane.upper()
    root = main_root()
    plan = json.loads((root / "docs" / "PLAN.json").read_text(encoding="utf-8"))
    if lane not in plan["lanes"]:
        sys.exit(f"lane.py: unknown lane {lane}; expected one of {', '.join(plan['lanes'])}")
    branch = plan["lanes"][lane]["branch"]
    wt = root / ".lanes" / lane

    git(root, "fetch", "origin", "--quiet", check=True)
    has_remote = git(root, "rev-parse", "-q", "--verify", f"refs/remotes/origin/{branch}")[0] == 0
    if not (wt / ".git").exists():
        git(root, "worktree", "prune")
        start = f"origin/{branch}" if has_remote else "origin/main"
        git(root, "worktree", "add", "-B", f"lanework/{lane}", str(wt), start, check=True)
    elif has_remote:
        git(wt, "pull", "--no-rebase", "--no-edit", "origin", branch, check=True)
    git(wt, "pull", "--no-rebase", "--no-edit", "origin", "main", check=True)
    (wt / ".lane").write_text(lane + "\n", encoding="ascii")

    if a.start:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        with (wt / "worklog" / f"{lane}.md").open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"\n## {now} | {a.start} | STARTED\n- Plan: {a.plan}\n- Needs: none\n")
        git(wt, "add", f"worklog/{lane}.md", check=True)
        git(wt, "commit", "-m", f"[{lane}] start {a.start}", check=True)
    if a.start or not has_remote:
        if git(wt, "push", "origin", f"HEAD:{branch}")[0] != 0:
            git(wt, "pull", "--no-rebase", "--no-edit", "origin", branch, check=True)
            git(wt, "push", "origin", f"HEAD:{branch}", check=True)

    _, head = git(wt, "log", "-1", "--format=%h %s")
    _, dirty = git(wt, "status", "--short")
    print(f"lane {lane} ready in {wt}\n  branch on GitHub: {branch} (push with: git push origin HEAD:{branch})\n"
          f"  last commit: {head}\n  uncommitted files: {len(dirty.splitlines())}\n"
          f"  started: {a.start or '-'}\nWork ONLY inside this folder from now on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
