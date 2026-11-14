#!/usr/bin/env python3
"""One-screen status board from every lane's worklog, read across ALL local and remote branches,
so you see progress before anything is merged. Stdlib only.

Usage: python scripts/status.py [--fetch]
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LANES = ("L1", "L2", "L3", "L4", "L5", "L6", "L7")
HEAD_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) \| ([^|]+?) \| (STARTED|DONE|BLOCKED)\s*$")


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    return r.stdout if r.returncode == 0 else ""


def parse(text: str) -> list[dict]:
    entries: list[dict] = []
    cur: dict | None = None
    for line in text.splitlines():
        s = line.strip()
        m = HEAD_RE.match(s)
        if m:
            cur = {"time": m.group(1), "card": m.group(2).strip(), "status": m.group(3), "needs": "", "next": ""}
            entries.append(cur)
        elif cur is not None and s.startswith("- "):
            key, _, val = s[2:].partition(":")
            if key.strip().lower() in ("needs", "next"):
                cur[key.strip().lower()] = val.strip()
    return entries


def versions(lane: str):
    f = REPO / "worklog" / f"{lane}.md"
    if f.exists():
        yield f.read_text()
    for ref in git("for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes").splitlines():
        text = git("show", f"{ref}:worklog/{lane}.md")
        if text:
            yield text


def latest(lane: str) -> dict | None:
    best = None
    for text in versions(lane):
        es = parse(text)
        if es:
            e = sorted(es, key=lambda x: x["time"])[-1]
            if best is None or e["time"] >= best["time"]:
                best = e
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="git fetch --all first")
    if ap.parse_args().fetch:
        git("fetch", "--all", "--quiet")
    rows = [(lane, latest(lane)) for lane in LANES + ("MERGE", "DISPATCH")]
    print(f"{'LANE':5} {'UPDATED':17} {'CARD':8} {'STATUS':8} {'NEXT'}")
    for lane, e in rows:
        if e is None:
            print(f"{lane:5} {'-':17} {'-':8} {'idle':8}")
        else:
            print(f"{lane:5} {e['time']:17} {e['card'][:8]:8} {e['status']:8} {e['next'][:60]}")
    needs = [(lane, e["needs"]) for lane, e in rows if e and e["needs"] and e["needs"].lower() != "none"]
    if needs:
        print("\nOPEN NEEDS")
        for lane, n in needs:
            print(f"  {lane} -> {n}")


if __name__ == "__main__":
    main()
