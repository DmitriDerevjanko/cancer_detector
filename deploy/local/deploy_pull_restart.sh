#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/home/dmitri/cancer_detector}"
BRANCH="${2:-main}"

cd "$PROJECT_DIR"

echo "[deploy] branch=$BRANCH dir=$PROJECT_DIR"

git fetch --all --prune
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

if [[ ! -x ".venv/bin/python" ]]; then
  python3 -m venv .venv
fi

.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"

if sudo -n true 2>/dev/null; then
  sudo systemctl restart oncoview-api
elif systemctl --user status oncoview-api >/dev/null 2>&1; then
  systemctl --user restart oncoview-api
else
  echo "[deploy] cannot restart oncoview-api (no sudo rights and no user service)."
  exit 1
fi

sleep 2
curl -fsS http://127.0.0.1:18005/health >/dev/null

echo "[deploy] done"
