#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 0 ]]; then
  PORTS=("$@")
else
  PORTS=(18005 18080 2223 8005)
fi

echo "Checking ports: ${PORTS[*]}"
echo

for p in "${PORTS[@]}"; do
  echo "== port ${p} =="
  ss -tulpen | awk -v port=":${p}" '$0 ~ port {print}' || true
  echo
done
