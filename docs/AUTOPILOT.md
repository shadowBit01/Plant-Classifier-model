# Autopilot: the agents finish the project on their own

Everything runs in **one T3 project**: the main checkout `C:\Users\divya\Plant-Classifier-model`.
A Claude **dispatcher** wakes every 30 minutes in that project (Local mode, branch main). `scripts/next_cards.py`
tells it which task cards are ready (dependencies merged, lane idle, provider under its 2-thread cap). It merges
finished lanes with `scripts/merge_lanes.py` (ownership check + gate tests, red merges are aborted and sent back as
fix cards), routes BLOCKED requests, starts new threads in the same project through T3's MCP server, logs to
`worklog/DISPATCH.md`, and stops.

Each lane thread starts with `scripts/lane.py <LANE> --start <card>`, which creates its own folder `.lanes\LX`
inside the project (a git worktree on that lane's branch, gitignored) and works only there. So 7 lanes can work in
parallel on 7 branches inside one project, and the project root stays on main. Nobody switches branches.
Lane agents follow AGENTS.md section 0b: never ask, use the card's fallback, or log BLOCKED and stop.
The data is built automatically (iNaturalist), and lane L7 writes the deck. Your remaining jobs are at the end.

## One-time setup (about 20 minutes, PowerShell, one line at a time)

**1. Put this version in the repo.** Open a NEW PowerShell window (so `$env:VANALENS_PY` from setup.ps1 exists):
```powershell
cd C:\Users\divya\Plant-Classifier-model
git checkout main
git pull origin main
Expand-Archive "$HOME\Downloads\vanalens-final.zip" -DestinationPath "$env:TEMP\vl-final2" -Force
Copy-Item "$env:TEMP\vl-final2\vanalens\*" . -Recurse -Force
& $env:VANALENS_PY -m pip install -r requirements.txt -q
& $env:VANALENS_PY -m pytest -q tests\test_tooling.py
git checkout -- worklog
git add -A
git commit --no-verify -m "autopilot: single-project scaffold"
git push origin main
git push origin --all
```
- The pytest line must print `15 passed` (the lane tests fail on purpose until the agents implement them).
- `git checkout -- worklog` keeps any worklog entries already in your repo instead of the empty templates.
- `--no-verify` is needed once: the hook now lets main take only human-owned files, and this update touches a few
  lane stubs.
- `git push origin --all` makes sure every `lane/...` branch exists on GitHub.
- If `$env:VANALENS_PY` is empty, run `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1`, open a new
  window, and repeat.

**2. Only if you ran make_worktrees.ps1 before:** remove the old folders and their T3 projects.
```powershell
foreach ($n in 1..7) { git worktree remove "..\PCM-L$n" }
git worktree prune
git worktree list
```
`git worktree list` must show only `C:/Users/divya/Plant-Classifier-model`. If one says it contains untracked or
modified files, add `--force` to that one (nothing there is needed). Delete any `PCM-*` projects in T3.

**3. One T3 project.** Keep (or add) exactly one project for `C:\Users\divya\Plant-Classifier-model`, named
`Plant-Classifier-model` (the default name). In T3 settings, set new threads to **Local** (not "New worktree").
Provider display names must be exactly `Claude`, `Antigravity A`, `Antigravity B`.
Then fully quit T3 (also from the tray) and reopen it, so threads see the variables setup.ps1 saved.

**4. Permissions for unattended work.** T3 Settings > General > New threads > Permissions: **Full access**.
Start a throwaway thread on each Antigravity provider with `run git status` and check it runs without asking.
If it asks, raise that provider's permission mode until it doesn't. Delete the throwaway threads.

**5. Safety deny lists** (the guard that replaces you approving commands):
- Claude: `.claude/settings.json` is in the repo, so every lane folder has it too.
- Antigravity: open the settings file (`notepad` creates it if missing):
```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.gemini\antigravity-cli"
notepad "$env:USERPROFILE\.gemini\antigravity-cli\settings.json"
```
If the file is new, paste this and save. If it already has content, add the `"permissions"` block inside the outer
`{ }` with a comma between entries.
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
Choose **Full access** in the browser sign-in. If `claude mcp login t3` errors: run `claude`, type `/mcp`, pick
`t3`, Authenticate, choose Full access, exit. Test in a Claude thread in the project:
"Use the t3 MCP tools to list projects and providers." It must show the project and the 3 providers.

**7. Keep the laptop working all night:**
```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setactive SCHEME_CURRENT
```
Keep the charger in, pause Windows Update for a day (Settings > Windows Update > Pause), leave T3 open.

**8. Smoke test (5 min):** open a Claude thread in the project and paste the first block from `docs\DISPATCHER.md`.
It should start 5 threads in the same project (L1-a and L2-a on Antigravity A, L3-a on Claude, L4-a and L5-a on
Antigravity B), write an entry in `worklog/DISPATCH.md`, and push. After about 5 minutes:
```powershell
& $env:VANALENS_PY scripts\status.py --fetch
Get-ChildItem .lanes
```
L1 to L5 should show `STARTED`, and `.lanes` should contain `L1` to `L5`.

**9. Schedule it:** T3 > Scheduled tasks > new task: every 30 minutes, project `Plant-Classifier-model`, provider
`Claude`, Local mode, prompt = the same block from `docs\DISPATCHER.md`. If scheduled tasks can't pick a project and
provider, use the fallback loop at the bottom of `docs\DISPATCHER.md` in one Claude thread instead.

## Optional, before you sleep
- `setx PLANTNET_API_KEY ...` and `setx VANALENS_LLM_KEY ...`. Without them B0 is skipped, the agent writes plant
  traits itself, and the reranker is off. Restart T3 after setx.
- Accept the DINOv3 license on Hugging Face (otherwise the DINOv2 fallback is used automatically).
- DIMPSAR or campus photos: drop them into `%USERPROFILE%\vanalens-data\dimpsar` or `...\campus` within the first
  hour, before L1-b runs, if you want them included.

## Your remaining jobs
- Glance at progress when you're up: `& $env:VANALENS_PY scripts\status.py --fetch`, or `worklog/DISPATCH.md`.
- While it runs, don't edit, commit or pull in the project root or in `.lanes\`: the dispatcher needs a clean root
  on main to merge, and each lane folder belongs to its agent. To look, use GitHub or `git log`.
- Anything under `Needs: HUMAN:` in DISPATCH.md is for you.
- Present: `pitch/VanaLens.pptx`, `pitch/script.md`, `pitch/demo.md` (on `main` once L7-a is merged).

## What can still stop it
| Risk | Effect | Mitigation |
|---|---|---|
| Claude 5-hour limit | Dispatcher and Claude lanes pause until reset | Schedule keeps retrying; Antigravity threads finish their cards |
| One Antigravity account out of quota | Its lanes stall | Dispatcher moves the lane to another provider in PLAN.json |
| Laptop sleeps, reboots, or T3 closes | Everything stops | Step 7; restart T3 and the schedule resumes |
| An agent edits the project root by mistake | Root gets dirty, merges pause | Hook refuses lane files on main; dispatcher stashes stray changes and logs it |
| An agent makes a mess | Bad code on its own branch only | Ownership hook, gate tests before merge (red merges are aborted), deny lists |
| A lane can't get its merge green | That lane's later cards wait | Fix cards go back to the lane; after 3 rounds the fix goes to Claude |
| T3 MCP can't start threads | Dispatch step fails | Dispatcher writes `Needs: HUMAN: start <card> ...`; you paste `scripts\card_prompt.py <card>` output into a new thread |
