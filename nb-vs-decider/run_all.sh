#!/usr/bin/env bash
# End-to-end entry point. Run on the execution host (orion), from this directory.
# Stages skip when their output file exists (delete the file to force a rerun).
set -euo pipefail
cd "$(dirname "$0")"

docker compose build pipeline
DECIDER_IMAGE=$(docker image inspect nb-vs-decider:dev --format '{{.Id}}')
export DECIDER_IMAGE
# Create host-owned dirs first: containers run as root and would otherwise own them.
mkdir -p results plots data/raw data/processed data/splits

# stage OUTPUT_FILE SERVICE MODULE [ARGS...]: run a stage unless its output already exists.
stage() {
  local out=$1 service=$2 module=$3
  shift 3
  if [ -e "$out" ]; then
    echo "skip ${module} ${*:-} (found ${out})"
  else
    docker compose run --rm "$service" python -m "scripts.${module}" "$@"
  fi
}

# ---- CPU stages (no GPU needed; these may run while the GPU is busy) ----
REPO=data/raw/strands-decider
COMMIT=3e94e9d84c620ed5a95f1a3310c3decb971e261c
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/strands-labs/strands-decider "$REPO"
fi
git -C "$REPO" checkout --quiet "$COMMIT"

stage data/processed/wos46985_l1_clean.parquet cpu dataset prepare
stage results/contamination.csv cpu contamination
stage results/dataset_summary.json cpu dataset split

# ---- GPU stages: the host GPU preflight is mandatory right before them ----
./preflight_host.sh

for module in versions decider_smoke; do
  # later stages (decider_run, nb_curve, energy, report) are appended here
  docker compose run --rm pipeline python -m "scripts.${module}"
done
