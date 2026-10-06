#!/usr/bin/env bash
# Host preflight: abort if any GPU is in use. Never kills or modifies anything.
set -euo pipefail

MAX_USED_MIB=500
fail() { echo "$1"; echo "GPU busy — stop and ask the owner. Do not kill services."; exit 1; }

command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"

apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true)
[ -z "$apps" ] || fail "GPU compute processes present:
$apps"

while IFS=, read -r idx used; do
  used=${used//[^0-9]/}
  [ "$used" -le "$MAX_USED_MIB" ] || fail "GPU $idx uses ${used} MiB (> ${MAX_USED_MIB} MiB)"
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

# Containers named like llama-swap / vllm that request a GPU.
if command -v docker >/dev/null; then
  for c in $(docker ps --format '{{.Names}}' | grep -Ei 'llama-swap|vllm' || true); do
    if [ -n "$(docker inspect -f '{{range .HostConfig.DeviceRequests}}{{.Driver}}{{end}}' "$c")" ]; then
      fail "container $c (llama-swap/vllm) is running with GPU access"
    fi
  done
fi

# Host processes named like llama-swap / vllm: only a problem if they hold a GPU, which the
# compute-apps check above already covers. Mention them for the record.
pgrep -fa 'llama-swap|vllm' | grep -v "pgrep\|preflight_host" || true

echo "preflight OK: all GPUs idle"
