# AGENTS.md: rules for every coding agent on VanaLens

Read by Claude Code (via CLAUDE.md) and by Antigravity (natively). Follow it exactly.

VanaLens = evidence-backed, uncertainty-aware plant species ID (approach F).
Frozen encoders (BioCLIP 2 + DINOv3) -> kNN / probe / text-trait scorers -> fusion -> geo prior
-> hierarchical backoff -> conformal set -> gated VLM reranker. No fine-tuning.

## 0. Who does what

All agents run as T3 Code threads on one machine (Lenovo LOQ, RTX 3050 6 GB). Each lane always works in its own
fixed folder `PCM-LX` (a git worktree on `lane/LX-...`, also a T3 project). The dispatcher works in the main checkout.

| T3 provider | Lanes |
|---|---|
| Claude | Dispatcher/lead (main checkout, merges with `scripts/merge_lanes.py`), L3 core-ml, L6 integration, L7 pitch |
| Antigravity A | L1 data-eval, L2 encoders (only lane allowed on the GPU) |
| Antigravity B | L4 context, L5 app |

## 0b. Autopilot: nobody is watching
- Never ask the human a question or wait for a reply. Pick the most reasonable option consistent with
  `docs/HANDOFF.md`, do it, and record the choice under `Did:` in your worklog.
- Missing data, key or access: use the fallback written in your card (e.g. dinov2-large, skip B0, write traits
  yourself). If none exists, write a BLOCKED entry with a precise `Needs:` and stop; the dispatcher routes it.
- Same error twice after two different fixes: BLOCKED with the error summary, stop. Do not loop.
- Do only your card. No refactors of other lanes, no new features beyond the card.

Lane details, owned files and task cards: `docs/TASKS.md`. Ownership rules: `docs/OWNERS`.

## 1. First thing in every new thread
1. Find your lane in the card prompt (e.g. "lane L3"). Write it to `.lane` in the repo root (gitignored; the
   ownership hook reads it): bash `echo L3 > .lane`, PowerShell `Set-Content .lane L3 -Encoding ascii`. If `.codegraph/` is missing here and `codegraph` is installed,
   run `codegraph init` (seconds).
2. Read, in this order: this file, `vanalens/contracts.py`, `vanalens/config.py`, your own `worklog/<LANE>.md`.
   Run `python scripts/status.py` to see what other lanes have finished or need. Do not read other lanes' code
   or worklogs beyond that board.
3. Append a STARTED entry to `worklog/<LANE>.md` (template in section 6), then commit and push it immediately:
   the dispatcher only sees committed entries and would otherwise start a duplicate thread for your card.

## 2. Stay in your lane
- Edit ONLY files your lane owns (`docs/OWNERS`). A pre-commit hook rejects anything else.
- `vanalens/contracts.py`, `vanalens/config.py`, `AGENTS.md`, `docs/*` are human-owned; only the human or the
  dispatcher (lead, on main) changes them, logging the decision in `docs/HANDOFF.md`. Need a change? Write it under
  `Needs:` (BLOCKED if you can't continue) and stop; do not work around the contract.
- Never edit another lane's tests or worklog. New tests go in `tests/test_<lane lowercase>_*.py` (e.g. `tests/test_l3_extra.py`).
- Never weaken a test to make it pass.

## 3. Git
- Work only on your lane branch (`lane/<LANE>-...`, or the worktree branch T3 Code created for you).
- Commit messages start with your lane: `[L3] fit temperature on cal split`.
- Your branch on GitHub is `lane/<LANE>-...` (see `docs/TASKS.md`). Whatever your local branch is called (T3 Code
  may name worktree branches itself), push with `git push origin HEAD:lane/<LANE>-...`.
- At card start: `git pull --no-rebase --no-edit origin lane/<LANE>-...` then
  `git pull --no-rebase --no-edit origin main` (files are disjoint, so this merge is clean).
- Never rebase, amend or squash commits you have pushed: the push would be rejected and force-push is blocked.
  If a push is rejected, `git pull --no-rebase --no-edit origin <your branch>` and push again.
- Never commit to or push `main`; only the dispatcher (`scripts/merge_lanes.py`, main checkout) or the human
  merges. Merges are logged in `worklog/MERGE.md`; a failed merge comes back to your lane as a fix card.

## 4. Spend tokens carefully
- Navigate code with `codegraph_explore` before grep/Read. Never Read a whole large file to find one function.
- Never print datasets, embeddings, model outputs or full logs. Use `head -n 20`, `tail -n 30`, `wc -l`,
  or a one-line `python -c` summary (shapes, counts, means).
- One task card per thread. When the card is done, write the DONE entry and stop; the dispatcher opens a new thread.

## 5. Data, caches, GPU, safety
- Machine is Windows. Claude Code runs commands in Git Bash, Antigravity likely in PowerShell: use the right syntax
  (PowerShell 5.1 has no `&&`; use `;` or separate commands. `tail -n 30` = `Get-Content <f> -Tail 30`).
- In every card, `python` / `pytest` mean the venv interpreter at `$VANALENS_PY`
  (`<repo>\.venv\Scripts\python.exe`): run `"$VANALENS_PY" -m pytest ...` in bash, `& $env:VANALENS_PY -m pytest ...`
  in PowerShell. Never use the system Python. Worktrees share this one venv.
- Data lives OUTSIDE the repo: `$VANALENS_DATA` (default `~/vanalens-data`), caches in `$VANALENS_CACHE`
  (default `~/vanalens-cache`). Use `vanalens.config.DATA_DIR` / `CACHE_DIR`. Never write data into the repo.
- Only lane L2 launches GPU jobs (6 GB VRAM; two jobs at once will OOM). Others use `tests/conftest.py`
  fixtures or cached `.npz` files.
- Long jobs: run in the background with output to `runs/logs/<name>.log` (bash `nohup ... &`, PowerShell
  `Start-Process ... -RedirectStandardOutput`), then check the last 30 lines.
- Never run destructive commands (`rm -rf`, `Remove-Item`, `git reset --hard`, `git clean`, `git push --force`,
  `git branch -D`, deleting caches). Permission rules block them anyway; find a non-destructive way or write BLOCKED.
- Never enter plan mode or wait for approval: nobody is there to approve.

## 6. Worklog (mandatory)
`worklog/<LANE>.md` is append-only and owned by that lane alone. Add an entry when a card starts, when it ends,
and when you get blocked. Keep each entry at most 8 lines. Append with your file-editing tool, never with shell
redirection (`>>` / `echo` in PowerShell 5.1 writes UTF-16). Exact format (the status board parses it):

```
## 2026-11-15 02:10 | L3-a | STARTED
- Plan: implement scorers, fusion, conformal, hierarchy
- Needs: none

## 2026-11-15 03:05 | L3-a | DONE
- Did: KNN/probe/text scorers, log-linear fusion, temperature, split conformal, backoff
- Files: vanalens/scorers.py, vanalens/fusion.py, vanalens/conformal.py, vanalens/hierarchy.py
- Tests: pytest -q tests/test_core.py -> 13 passed
- Results: none
- Needs: L2: embeddings for dimpsar gallery/cal/test
- Next: L3-b fit_core.py
```
Status is one of `STARTED`, `DONE`, `BLOCKED`. Time is local (IST), 24-hour. `Needs:` is `none` or `<LANE>: <what>`.

## 7. Conventions
- Probability matrices are `(N, C)` float arrays, rows sum to 1, column j = class idx j from `labels.json`.
- Embeddings: `(N, EMB_DIM)` float32 = `[bioclip block | dino block]`, each block L2-normalised separately.
- Results: small JSON files in `runs/<LANE>/`. Commit those, never arrays.
- Python 3.10+, numpy-first, type hints, keep docstrings. Dependencies are pre-listed in `requirements.txt`;
  need another one? Put it under `Needs:`.

## 8. Secrets
- Never print or commit keys.
- Do NOT export `ANTHROPIC_API_KEY` in the shell: Claude Code then bills that key instead of the plan.
  LLM calls in this repo read `VANALENS_LLM_KEY`.

## 9. Definition of done
- Your lane's tests pass.
- DONE entry written in your worklog, lane branch committed and pushed.
- Final chat message: at most 10 lines, same content as the DONE entry.
