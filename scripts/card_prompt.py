#!/usr/bin/env python3
"""Print the exact prompt for a task card from docs/TASKS.md, so dispatchers paste it verbatim.

Usage:
  python scripts/card_prompt.py L3-b              # card prompt
  python scripts/card_prompt.py L3-b --resume     # same, prefixed with resume instructions
  python scripts/card_prompt.py --fix L2 "encode_text must return float32"   # small fix request for an idle lane
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESUME = ("RESUME: an earlier thread started this card and stopped without a DONE entry. Read your worklog and "
          "`git log -5` on this branch, keep what is good, and finish the card.\n")


def extract(tasks: str, card: str) -> str | None:
    lines = tasks.splitlines()
    for i, line in enumerate(lines):
        if re.match(rf"^\*\*{re.escape(card)}(\*\*| )", line.strip()):
            fences = [k for k in range(i + 1, len(lines)) if lines[k].strip() == "```"]
            if len(fences) >= 2:
                return "\n".join(lines[fences[0] + 1:fences[1]])
    return None


def fix_prompt(lane: str, request: str) -> str:
    branch = json.loads((REPO / "docs" / "PLAN.json").read_text(encoding="utf-8"))["lanes"][lane]["branch"]
    card = f"{lane}-fix{datetime.now():%d%H%M}"   # unique per fix round
    return (f"Lane {lane}, card {card} (a fix). Setup, from the project root: "
            f"`& $env:VANALENS_PY scripts/lane.py {lane} --start {card}` "
            f"(bash: `\"$VANALENS_PY\" scripts/lane.py {lane} --start {card}`). It prepares `.lanes/{lane}` and commits + "
            f"pushes your STARTED entry. Work ONLY inside `.lanes/{lane}` (AGENTS.md section 1).\n"
            f"Fix request from the lead: {request}\n"
            f"Change only files your lane owns, keep your lane's tests green.\n"
            f"When done, inside `.lanes/{lane}`: DONE entry for {card} in worklog/{lane}.md, commit with prefix [{lane}], "
            f"`git push origin HEAD:{branch}`, stop.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("card", nargs="?")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fix", nargs=2, metavar=("LANE", "REQUEST"))
    a = ap.parse_args()
    if a.fix:
        print(fix_prompt(a.fix[0].upper(), a.fix[1]))
        return 0
    text = extract((REPO / "docs" / "TASKS.md").read_text(encoding="utf-8"), a.card or "")
    if text is None:
        print(f"card {a.card!r} not found in docs/TASKS.md", file=sys.stderr)
        return 1
    print((RESUME if a.resume else "") + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
