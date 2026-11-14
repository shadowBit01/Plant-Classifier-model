# Setup: Claude + 2 Antigravity accounts, all in T3 Code (one machine)

Machine: Lenovo LOQ (RTX 3050 6 GB, 16 GB RAM, Windows). Every agent is a T3 Code thread in its own worktree.

| T3 provider (display name) | Lanes | Parallel threads |
|---|---|---|
| Claude | L3 core-ml, L6 integration, L6-m merges | max 2 |
| Antigravity A | L1 data-eval, L2 encoders (GPU) | max 2 |
| Antigravity B | L4 context, L5 app | max 2 |

## 1. Providers (T3 Settings > Providers)

1. **Rename** the two Antigravity rows: Display name `Antigravity A` / `Antigravity B` (pick accent colours too), so
   the model picker makes it obvious which account a thread uses.
2. **Check they are two different accounts.** Open each row and compare the account email (click to reveal if
   blurred). If both show the same email, the second sign-in replaced the first: sign that instance in again with
   the other account, using a separate home/config path for it if the panel offers one.
3. **Claude must say your account, not "Custom API endpoint".** That label means Claude Code is pointed at a
   non-Anthropic endpoint by an environment variable or settings file. Find it:
   ```powershell
   Get-ChildItem Env: | Where-Object Name -like "*ANTHROPIC*"
   Select-String -Path "$HOME\.claude\settings.json" -Pattern anthropic
   ```
   Remove any `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY` (and an `"env"` block in
   `~/.claude/settings.json` that sets them). Also clear the Environment variables section in T3's Claude provider
   panel. Then `claude auth logout && claude auth login`, quit T3 fully, reopen it, and re-check the row.

## 2. Repo and environment (Windows, PowerShell)

PowerShell 5.1 has no `&&`: run these one line at a time.

```powershell
cd $HOME\Plant-Classifier-model            # wherever T3 cloned it
Expand-Archive "$HOME\Downloads\vanalens-scaffold.zip" -DestinationPath "$env:TEMP\vl" -Force
Copy-Item "$env:TEMP\vl\vanalens\*" . -Recurse -Force
Test-Path .githooks\pre-commit              # must print True
git branch -M main
git add -A
git commit -m "scaffold: contracts, stubs, tests, lane cards"
git push -u origin main
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
git push origin --all
```

`setup.ps1` creates the venv, installs the CUDA build of torch (PyPI torch is CPU-only on Windows) and all
dependencies, runs `codegraph init`, installs the ownership hook, creates the 6 lane branches, and saves
`VANALENS_DATA`, `VANALENS_CACHE`, `VANALENS_VENV`, `VANALENS_PY`, `VANALENS_LLM_MODEL` as user environment
variables. Then add your keys and restart T3:

```powershell
setx PLANTNET_API_KEY "your-key"
setx VANALENS_LLM_KEY "your-anthropic-key"    # optional; never name it ANTHROPIC_API_KEY
```

Fully quit T3 (tray icon too) and reopen it so threads see the new variables.
Accept the DINOv3 license on Hugging Face, then `& "$env:VANALENS_VENV\Scripts\hf.exe" auth login`.

## 3. CodeGraph

```powershell
irm https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.ps1 | iex
# open a NEW terminal
codegraph install                # wires up Claude Code and Antigravity
```

Run `setup.ps1` again if CodeGraph was installed after it (it runs `codegraph init`). Worktrees index themselves:
AGENTS.md tells every agent to run `codegraph init` when `.codegraph/` is missing.

## 4. Start the lanes

T3: add project `~/vanalens`. For each card in `docs/TASKS.md`: new thread, pick the provider from the table above,
turn on the worktree option based on that lane's branch (e.g. `lane/L1-data-eval`), paste the card.

First wave (5 threads):

| Provider | Thread 1 | Thread 2 |
|---|---|---|
| Claude | L3-a | (keep free for L6-m merges) |
| Antigravity A | L1-a | L2-a |
| Antigravity B | L4-a | L5-a |

Each card makes the agent: set `.lane`, init CodeGraph, pull its lane branch + `main`, log STARTED, work, log DONE,
commit `[LANE] ...`, `git push origin HEAD:lane/...`, stop. When a thread finishes, open a new thread for the next
card of that lane (new threads keep context and tokens small).

Antigravity runs terminal commands: in its settings, keep command execution on review rather than fully automatic.

## 5. The rhythm

- **Watch**: `python scripts/status.py --fetch` shows every lane's latest entry and open `Needs:` from GitHub.
- **Merge**: every ~2 h, run card **L6-m** in a Claude thread **without** a worktree (main checkout). It checks
  ownership per lane, merges, runs tests, pushes `main`, logs to `worklog/MERGE.md`.
- **Blocked agent**: writes `BLOCKED` + `Needs:` and stops; you route the request.
- **Contract change**: only you edit `contracts.py`, `config.py`, `AGENTS.md`, on `main`, then run L6-m.
- **GPU**: only L2 threads run GPU jobs. With 16 GB RAM, run big embedding extraction on Kaggle
  (card L2-c includes a Kaggle script) if the laptop starts swapping.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Commit rejected by `check_owners` | Working as intended. Unstage the other lane's file; request via `Needs:` |
| Hook says "no lane set" | `echo <LANE> > .lane` in that worktree |
| Agent can't find `VANALENS_*` vars | T3 was started without them: restart T3 from a terminal that has them |
| Two GPU jobs OOM | Only L2 may use the GPU; stop the other job |
| Claude limit hit | Pause Claude threads; Antigravity lanes keep going |
| One Antigravity account hits its quota | Continue that lane's next card on the other Antigravity provider; lanes are tied to files, not accounts |
