# Manual mode (only if autopilot is not running)

Use this when you want to drive the cards yourself, or when the dispatcher can't start threads.
Setup is the same as `docs/AUTOPILOT.md` steps 1 to 7 (repo, worktrees, T3 projects, permissions, deny lists).

## Providers (T3 Settings > Providers)
- Display names must be exactly `Claude`, `Antigravity A`, `Antigravity B` (PLAN.json uses them).
- The two Antigravity rows must show two different account emails.
- The Claude row must show your account. If it says "Custom API endpoint", remove `ANTHROPIC_BASE_URL`,
  `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_API_KEY` from your environment and from `~/.claude/settings.json`,
  then `claude auth login` and restart T3.

## Running a card by hand (PowerShell, main checkout)
```powershell
& $env:VANALENS_PY scripts\next_cards.py --fetch        # what can start now, with project + provider
& $env:VANALENS_PY scripts\card_prompt.py L3-a          # exact prompt for that card
```
Open a new thread in the project it names (`PCM-L3`), on the provider it names, and paste the prompt.
One card per thread; one thread per lane at a time; at most 2 threads per provider.

## Merging finished cards (main checkout, clean tree)
```powershell
& $env:VANALENS_PY scripts\merge_lanes.py
```
It merges every lane with an unmerged DONE entry behind the ownership check and that lane's gate tests,
aborts anything red, logs `worklog/MERGE.md` and pushes main. A failed lane gets a fix card:
```powershell
& $env:VANALENS_PY scripts\card_prompt.py --fix L3 "merge failed: <last lines from merge_lanes output>"
```

## Progress
```powershell
& $env:VANALENS_PY scripts\status.py --fetch
```
