#!/usr/bin/env python3
"""Reject changes outside the current lane's files (stdlib only).

Lane: --lane, else $VANALENS_LANE, else ./.lane, else branch name lane/<LANE>-*. No lane -> skip (human on main).
Modes:
  (default)          check staged files (used by .githooks/pre-commit)
  --range A...B      check files changed between refs (merge gate for L6), e.g. --lane L2 --range main...lane/L2-encoders
  --files f1 f2 ...  check explicit paths
Exit code 1 on any violation.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LANES = ("L1", "L2", "L3", "L4", "L5", "L6")


def load_owners(path: Path = REPO / "docs" / "OWNERS") -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lane, pat = line.split(None, 1)
            owners.setdefault(lane, []).append(pat.strip())
    return owners


def implicit(lane: str) -> list[str]:
    return [f"worklog/{lane}.md", f"runs/{lane}/", f"tests/test_{lane.lower()}_*"]


def match(path: str, pat: str) -> bool:
    if pat.endswith("/"):
        return path.startswith(pat)
    if pat.endswith("*"):
        return path.startswith(pat[:-1])
    return path == pat


def allowed(path: str, lane: str, owners: dict[str, list[str]]) -> bool:
    return any(match(path, p) for p in owners.get(lane, []) + implicit(lane))


def owner_of(path: str, owners: dict[str, list[str]]) -> str:
    for lane, pats in owners.items():
        if any(match(path, p) for p in pats):
            return lane
    for lane in LANES:
        if any(match(path, p) for p in implicit(lane)):
            return lane
    return "UNOWNED"


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    return r.stdout.strip()


def read_lane_file(path: Path) -> str | None:
    """Tolerates UTF-8, UTF-8 BOM and UTF-16 (PowerShell 5.1 `echo L3 > .lane` writes UTF-16)."""
    raw = path.read_bytes()
    for enc in ("utf-16", "utf-8-sig") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig", "latin-1"):
        try:
            m = re.search(r"L[1-6]", raw.decode(enc), re.I)
            if m:
                return m.group(0).upper()
        except UnicodeDecodeError:
            continue
    return None


def resolve_lane(cli: str | None) -> str | None:
    if cli:
        return cli
    if os.environ.get("VANALENS_LANE"):
        return os.environ["VANALENS_LANE"]
    lane_file = REPO / ".lane"
    if lane_file.exists():
        lane = read_lane_file(lane_file)
        if lane:
            return lane
    m = re.match(r"lane/(L\d)", git("rev-parse", "--abbrev-ref", "HEAD"))
    return m.group(1) if m else None


def violations(paths: list[str], lane: str, owners: dict[str, list[str]]) -> list[tuple[str, str]]:
    return [(p, owner_of(p, owners)) for p in paths if p and not allowed(p, lane, owners)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane")
    ap.add_argument("--range")
    ap.add_argument("--files", nargs="*")
    a = ap.parse_args(argv)

    lane = resolve_lane(a.lane)
    if lane is None:
        print("check_owners: no lane set (.lane / VANALENS_LANE / lane/* branch); skipping")
        return 0
    lane = lane.upper()
    if lane not in LANES:
        print(f"check_owners: unknown lane {lane!r}; expected one of {', '.join(LANES)}")
        return 1

    if a.files is not None:
        paths = a.files
    elif a.range:
        paths = git("diff", "--name-only", a.range).splitlines()
    else:
        paths = git("diff", "--cached", "--name-only").splitlines()

    bad = violations(paths, lane, load_owners())
    if bad:
        print(f"check_owners: lane {lane} may not change:")
        for p, o in bad:
            print(f"  {p}  (owned by {o})")
        print("Unstage them (git restore --staged <file>) and ask under 'Needs:' in your worklog.")
        return 1
    print(f"check_owners: {len(paths)} file(s) OK for lane {lane}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
