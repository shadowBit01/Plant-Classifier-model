# Obsolete: VanaLens now runs in ONE T3 project. Lane folders are created automatically as .lanes\LX
# inside the repo by scripts/lane.py when a card starts. Nothing to do here.
#
# If you created ..\PCM-L1 ... ..\PCM-L7 with an older version, remove them (from the main checkout):
#   foreach ($n in 1..7) { git worktree remove "..\PCM-L$n" }
#   git worktree prune
# and delete the PCM-L* projects in T3.
Write-Host "Not needed anymore: lanes live in .lanes\LX, created by scripts\lane.py. See the comments in this file to remove old PCM-L* folders."
