# Run ONCE from the main checkout after setup.ps1:
#   powershell -ExecutionPolicy Bypass -File scripts\make_worktrees.ps1
# Creates one fixed folder per lane next to the repo (..\PCM-L1 ... ..\PCM-L7), each on its lane branch,
# writes .lane, builds the CodeGraph index, and pushes all branches. Add each folder as a T3 project
# with the same name (PCM-L1 ... PCM-L7), plus the main checkout as project PCM-main.
$ErrorActionPreference = "Continue"
$plan = Get-Content docs\PLAN.json -Raw | ConvertFrom-Json
$parent = Split-Path (Get-Location).Path -Parent
git fetch origin
foreach ($p in $plan.lanes.PSObject.Properties) {
    $lane = $p.Name
    $info = $p.Value
    $dir = Join-Path $parent $info.folder
    git show-ref -q --verify "refs/heads/$($info.branch)"
    if ($LASTEXITCODE -ne 0) { git branch $info.branch main }
    if (-not (Test-Path $dir)) { git worktree add $dir $info.branch }
    Set-Content -Path (Join-Path $dir ".lane") -Value $lane -Encoding ascii
    if (Get-Command codegraph -ErrorAction SilentlyContinue) { Push-Location $dir; codegraph init; Pop-Location }
    Write-Host "$lane -> $dir   T3 project: $($info.project)   provider: $($info.provider)"
}
git push origin --all
Write-Host ">> Add these 7 folders + this checkout (as PCM-main) as T3 projects."
