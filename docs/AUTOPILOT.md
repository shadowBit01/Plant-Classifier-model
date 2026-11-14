# Autopilot: the agents finish the project on their own

A Claude **dispatcher** wakes every 30 minutes in the main checkout. `scripts/next_cards.py` tells it which task
cards are ready (dependencies merged, lane idle, provider under its 2-thread cap). It merges finished lanes,
routes BLOCKED requests, starts new threads through T3's MCP server, logs to `worklog/DISPATCH.md`, and stops.
Lane agents follow AGENTS.md section 0b: never ask, use the card's fallback, or log BLOCKED and stop.
The data is built automatically (iNaturalist), and lane L7 writes the deck. Your remaining jobs are below.

## One-time setup (about 25 minutes, PowerShell, one line at a time)

**1. Put this version in the repo** (from the main checkout, `C:\Users\divya\Plant-Classifier-model`):
```powershell
Expand-Archive "$HOME\Downloads\vanalens-scaffold.zip" -DestinationPath "$env:TEMP\vl2" -Force
Copy-Item "$env:TEMP\vl2\vanalens\*" . -Recurse -Force
& $env:VANALENS_PY -m pip install python-pptx matplotlib
& $env:VANALENS_PY -m pytest -q tests/test_tooling.py
git add -A
git commit -m "autopilot: plan, dispatcher, worktrees, L7 pitch"
git push origin main
```
The pytest line should print 9 passed.

**2. Create the lane folders**: `powershell -ExecutionPolicy Bypass -File scripts\make_worktrees.ps1`
It creates `..\PCM-L1` to `..\PCM-L7`, each on its lane branch, with `.lane` and a CodeGraph index.

**3. Add 8 projects in T3** with exactly these names: `PCM-main` (the main checkout) and `PCM-L1` to `PCM-L7`
(the new folders). Provider display names must be exactly `Claude`, `Antigravity A`, `Antigravity B`.

**4. Permissions for unattended work.** T3 Settings > General > New threads > Permissions: **Full access**.
Then start a throwaway thread on each Antigravity provider with "run git status" and check it runs without
asking. If it asks, raise that provider's permission mode until it doesn't.

**5. Safety deny lists** (the guard that replaces you approving commands):
- Claude: `.claude/settings.json` is in the repo, so every lane folder already has it.
- Antigravity: create `%USERPROFILE%\.gemini\antigravity-cli\settings.json` (merge if it exists):
```json
{
  "permissions": {
    "deny": [
      "command(rm -rf *)", "command(Remove-Item *)", "command(rmdir *)", "command(del *)",
      "command(git push --force *)", "command(git push -f *)", "command(git reset --hard *)",
      "command(git clean *)", "command(git branch -D *)", "command(format *)", "command(shutdown *)"
    ]
  }
}
```

**6. Let Claude start threads in T3** (T3's MCP server). Find T3's local address in its settings
(Devices / remote access; often `http://localhost:3773`), then:
```powershell
claude mcp add --transport http --scope user t3 http://localhost:3773/mcp
claude mcp login t3
```
In the browser sign-in choose **Full access** (threads it starts can't exceed this level). Test in a Claude
thread in `PCM-main`: "Use the t3 MCP tools to list projects and providers." It must list all 8 projects
and the 3 providers.

**7. Keep the laptop working all night:**
```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```
Plug in the charger, set lid close to "Do nothing" if you close it, pause Windows Update for a day
(Settings > Windows Update > Pause), and leave T3 open.

**8. Smoke test (5 min):** open a Claude thread in `PCM-main` and paste the prompt from `docs/DISPATCHER.md`.
It should start 5 threads (L1-a and L2-a on Antigravity A, L3-a on Claude, L4-a and L5-a on Antigravity B),
write an entry in `worklog/DISPATCH.md`, and push.

**9. Schedule it:** T3 > Scheduled tasks > new task: every 30 minutes, project `PCM-main`, provider `Claude`,
prompt = the block in `docs/DISPATCHER.md`. If scheduled tasks can't pick a project and provider, use the
fallback loop at the bottom of `docs/DISPATCHER.md` in one Claude thread instead.

## Optional, before you sleep
- `setx PLANTNET_API_KEY ...` and `setx VANALENS_LLM_KEY ...`. Without them B0 is skipped, the agent writes plant
  traits itself, and the reranker is off. Restart T3 after setx.
- Accept the DINOv3 license on Hugging Face (otherwise the DINOv2 fallback is used automatically).
- DIMPSAR or campus photos: drop them into `%USERPROFILE%\vanalens-data\dimpsar` or `...\campus` within the first
  hour, before L1-b runs, if you want them included.

## Your remaining jobs
- Glance at progress when you're up: `& $env:VANALENS_PY scripts\status.py --fetch`, or `worklog/DISPATCH.md`.
- Anything under `Needs: HUMAN:` in DISPATCH.md is for you.
- Present: `pitch/VanaLens.pptx`, `pitch/script.md`, `pitch/demo.md` (on `main` once L7-a is merged).

## What can still stop it
| Risk | Effect | Mitigation |
|---|---|---|
| Claude 5-hour limit | Dispatcher and Claude lanes pause until reset | Schedule keeps retrying; Antigravity threads finish their cards |
| One Antigravity account out of quota | Its lanes stall | Dispatcher moves the lane to another provider in PLAN.json |
| Laptop sleeps, reboots, or T3 closes | Everything stops | Step 7; restart T3 and the schedule resumes |
| An agent makes a mess | Bad code on its own branch only | Ownership hook, tests before merge, deny lists, every card committed |
| T3 MCP can't start threads | Dispatch step fails | Dispatcher writes `Needs: HUMAN: start <card> in <project>`; you paste `scripts/card_prompt.py <card>` output |
