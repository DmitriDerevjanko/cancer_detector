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

echo "[deploy] waiting for health check"
for attempt in {1..30}; do
  if curl -fsS http://127.0.0.1:18005/health >/dev/null; then
    echo "[deploy] done"
    exit 0
  fi
  echo "[deploy] health check not ready yet (${attempt}/30)"
  sleep 2
done

echo "[deploy] health check failed"
if command -v systemctl >/dev/null 2>&1; then
  systemctl status oncoview-api --no-pager -l || true
fi
exit 1
