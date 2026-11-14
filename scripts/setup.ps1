# Windows setup (PowerShell 5.1+). Run from the repo root, AFTER the scaffold is committed and pushed:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Continue"
$repo = (Get-Location).Path

$data  = if ($env:VANALENS_DATA)  { $env:VANALENS_DATA }  else { Join-Path $HOME "vanalens-data" }
$cache = if ($env:VANALENS_CACHE) { $env:VANALENS_CACHE } else { Join-Path $HOME "vanalens-cache" }
New-Item -ItemType Directory -Force -Path $data | Out-Null
foreach ($d in "manifests", "emb", "state", "text", "gbif", "plantnet_api") {
    New-Item -ItemType Directory -Force -Path (Join-Path $cache $d) | Out-Null
}
New-Item -ItemType Directory -Force -Path (Join-Path $repo "runs\logs") | Out-Null

# venv
if (-not (Test-Path ".venv")) {
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3 -m venv .venv } else { python -m venv .venv }
}
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host ">> venv creation failed. Install Python 3.10+ from python.org (tick 'Add to PATH')."; exit 1 }

& $py -m pip install --upgrade pip -q
# PyPI torch is CPU-only on Windows: install the CUDA build first
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
& $py -m pip install -r requirements.txt
& $py -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# CodeGraph index
if (Get-Command codegraph -ErrorAction SilentlyContinue) { codegraph init } else { Write-Host ">> CodeGraph not found: irm https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.ps1 | iex ; then new terminal, 'codegraph install', 'codegraph init'" }

# Git: ownership hook + lane branches (created from the current main)
git config core.hooksPath .githooks
foreach ($b in "lane/L1-data-eval", "lane/L2-encoders", "lane/L3-core-ml", "lane/L4-context", "lane/L5-app", "lane/L6-integration") {
    git show-ref -q --verify "refs/heads/$b"
    if ($LASTEXITCODE -ne 0) { git branch $b }
}

# Persistent user env vars (new processes only: fully quit and reopen T3 afterwards)
setx VANALENS_DATA $data | Out-Null
setx VANALENS_CACHE $cache | Out-Null
setx VANALENS_VENV (Join-Path $repo ".venv") | Out-Null
setx VANALENS_PY $py | Out-Null
setx VANALENS_LLM_MODEL "claude-sonnet-5-5" | Out-Null

& $py -m pytest -q --co | Select-Object -Last 1
Write-Host ">> Done. VANALENS_PY=$py"
Write-Host ">> Next: git push origin --all ; set PLANTNET_API_KEY (and optionally VANALENS_LLM_KEY) with setx ; quit and reopen T3."
