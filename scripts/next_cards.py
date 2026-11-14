#!/usr/bin/env python3
"""Autopilot brain (stdlib only): which task cards can start now?

A dependency counts as satisfied only when its DONE entry is visible in main's worklog (i.e. merged).
In-flight = latest entry STARTED and younger than stale_minutes; older STARTED = stale -> re-dispatch with resume.
One card per lane at a time (each lane has one folder); at most max_threads_per_provider per provider.

Usage: python scripts/next_cards.py [--fetch]      -> JSON on stdout
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from status import git, parse, versions  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RANK = {"STARTED": 0, "BLOCKED": 1, "DONE": 2}


def load_plan(path: Path = REPO / "docs" / "PLAN.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def group(texts: list[str]) -> dict[str, list[dict]]:
    """card -> unique entries across all text versions."""
    seen, out = set(), {}
    for t in texts:
        for e in parse(t):
            key = (e["time"], e["card"], e["status"])
            if key not in seen:
                seen.add(key)
                out.setdefault(e["card"], []).append(e)
    return out


def latest(entries: list[dict]) -> dict:
    return max(entries, key=lambda e: (e["time"], RANK[e["status"]]))


def compute(plan: dict, main_entries: dict, all_entries: dict, now: datetime) -> dict:
    lanes, stale = plan["lanes"], plan.get("stale_minutes", 120)
    cap = plan.get("max_threads_per_provider", 2)
    cards = [c for c in plan["cards"] if c.get("enabled", True)]
    done_main = {c for c, es in main_entries.items() if any(e["status"] == "DONE" for e in es)}
    done_any = {c for c, es in all_entries.items() if any(e["status"] == "DONE" for e in es)}

    in_flight, stale_cards, blocked = [], [], []
    for c in cards:
        cid = c["id"]
        if cid in done_any or cid not in all_entries:
            continue
        e = latest(all_entries[cid])
        if e["status"] == "BLOCKED":
            blocked.append({"card": cid, "needs": e["needs"]})
        elif e["status"] == "STARTED":
            age = (now - datetime.strptime(e["time"], "%Y-%m-%d %H:%M")).total_seconds() / 60
            (in_flight if age <= stale else stale_cards).append(cid)

    plan_ids = {c["id"] for c in plan["cards"]}
    extra = []   # e.g. "L2-fix" threads started by the dispatcher
    for cid, es in all_entries.items():
        lane = cid.split("-")[0]
        if cid in plan_ids or lane not in lanes:
            continue
        e = latest(es)
        if e["status"] == "STARTED":
            age = (now - datetime.strptime(e["time"], "%Y-%m-%d %H:%M")).total_seconds() / 60
            if age <= stale:
                extra.append(cid)

    busy_lanes = {cid.split("-")[0] for cid in in_flight + extra + [b["card"] for b in blocked]}
    load: dict[str, int] = {}
    for cid in in_flight + extra:
        p = lanes[cid.split("-")[0]]["provider"]
        load[p] = load.get(p, 0) + 1

    dispatch = []
    for c in cards:
        cid, lane = c["id"], c["id"].split("-")[0]
        if cid in done_any or cid in in_flight or cid in [b["card"] for b in blocked]:
            continue
        if lane in busy_lanes or not all(d in done_main for d in c["deps"]):
            continue
        info = lanes[lane]
        if load.get(info["provider"], 0) >= cap:
            continue
        dispatch.append({"card": cid, "lane": lane, "resume": cid in stale_cards, **info})
        busy_lanes.add(lane)
        load[info["provider"]] = load.get(info["provider"], 0) + 1

    # compare DONE entries, not card ids, so a second "L2-fix..." round is still noticed
    keys = lambda ents: {(c, e["time"]) for c, es in ents.items() for e in es if e["status"] == "DONE"}
    needs_merge = sorted({c.split("-")[0] for c, _ in keys(all_entries) - keys(main_entries) if c.split("-")[0] in lanes})
    return {
        "now": now.strftime("%Y-%m-%d %H:%M"),
        "dispatch": dispatch,
        "in_flight": in_flight + extra,
        "stale": stale_cards,
        "blocked": blocked,
        "needs_merge": needs_merge,
        "done_merged": sorted(done_main & {c["id"] for c in cards}),
        "all_done": all(c["id"] in done_main for c in cards),
    }


def main_ref() -> str:
    return "refs/remotes/origin/main" if git("rev-parse", "--verify", "-q", "refs/remotes/origin/main").strip() else "refs/heads/main"


def state(fetch: bool = False) -> dict:
    if fetch:
        git("fetch", "--all", "--quiet")
    plan = load_plan()
    ref = main_ref()
    main_texts, all_texts = [], []
    for lane in plan["lanes"]:
        t = git("show", f"{ref}:worklog/{lane}.md")
        if t:
            main_texts.append(t)
        all_texts.extend(versions(lane))
    return compute(plan, group(main_texts), group(all_texts), datetime.now())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    print(json.dumps(state(ap.parse_args().fetch), indent=1))


if __name__ == "__main__":
    main()
