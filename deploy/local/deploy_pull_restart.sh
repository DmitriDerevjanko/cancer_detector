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

if sudo -n /usr/bin/systemctl restart oncoview-api 2>/dev/null; then
  echo "[deploy] restarted oncoview-api via system sudo"
elif /usr/bin/systemctl --user status oncoview-api >/dev/null 2>&1; then
  /usr/bin/systemctl --user restart oncoview-api
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
if [[ -x /usr/bin/systemctl ]]; then
  /usr/bin/systemctl status oncoview-api --no-pager -l || true
fi
exit 1
