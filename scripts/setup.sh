#!/usr/bin/env bash
# One-time setup in the main checkout. Run from the repo root: bash scripts/setup.sh
set -euo pipefail

: "${VANALENS_DATA:=$HOME/vanalens-data}"
: "${VANALENS_CACHE:=$HOME/vanalens-cache}"
mkdir -p "$VANALENS_DATA" "$VANALENS_CACHE"/{manifests,emb,state,text,gbif,plantnet_api} runs/logs

if [ ! -d .venv ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip -q
python -c "import torch" 2>/dev/null || echo ">> Install torch for your CUDA first: https://pytorch.org/get-started/locally/"
pip install -q -r requirements.txt

if command -v codegraph >/dev/null; then
  codegraph init
else
  echo ">> CodeGraph not found. Install it, open a NEW terminal, run 'codegraph install', then 'codegraph init' here."
fi

if [ ! -d .git ]; then
  git init -q -b main && git add -A && git commit -qm "scaffold: contracts, stubs, tests, lane cards"
fi
git config core.hooksPath .githooks   # ownership check on every commit
for b in lane/L1-data-eval lane/L2-encoders lane/L3-core-ml lane/L4-context lane/L5-app lane/L6-integration lane/L7-pitch; do
  git show-ref -q --verify "refs/heads/$b" || git branch "$b"
done

python -m pytest -q --co | tail -n 1
echo ">> Done. Data: $VANALENS_DATA  Cache: $VANALENS_CACHE"
