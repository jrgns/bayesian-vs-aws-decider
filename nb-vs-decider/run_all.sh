#!/usr/bin/env bash
# End-to-end entry point. Run on the execution host (orion), from this directory.
# Stages skip when their output file exists (delete the file to force a rerun).
set -euo pipefail
cd "$(dirname "$0")"

docker compose build pipeline
DECIDER_IMAGE=$(docker image inspect nb-vs-decider:dev --format '{{.Id}}')
export DECIDER_IMAGE
# Create host-owned dirs first: containers run as root and would otherwise own them.
mkdir -p results/decider results/power_samples plots data/raw data/processed data/splits

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

# gpu_stage OUTPUT_FILE MODULE [ARGS...]: like stage, on the pipeline service, with the host GPU
# preflight run right before (and only when the stage actually runs).
gpu_stage() {
  local out=$1 module=$2
  shift 2
  if [ -e "$out" ]; then
    echo "skip ${module} ${*:-} (found ${out})"
  else
    ./preflight_host.sh
    docker compose run --rm pipeline python -m "scripts.${module}" "$@"
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

# Noise indicator for the energy records (a container cannot see the host's docker daemon).
HOST_CONTAINERS_RUNNING=$(docker ps -q | wc -l)
export HOST_CONTAINERS_RUNNING

LABELS=${LABELS:-v1}
# Dev run (labels v1 by default), then the throughput sweep (dev subset only).
gpu_stage "results/decider/dev_${LABELS}_bs1.parquet" decider_run --split dev --labels "$LABELS" \
  --batch-size 1 --measure-energy --run-id "dev_${LABELS}_bs1"
gpu_stage results/decider_best_bs.json decider_sweep --labels "$LABELS"

# Test split: only when the owner exports ALLOW_TEST=1 (labels frozen, never set by scripts).
if [ "${ALLOW_TEST:-}" = "1" ]; then
  gpu_stage "results/decider/test_${LABELS}_bs1.parquet" decider_run --split test \
    --labels "$LABELS" --batch-size 1 --measure-energy --run-id "test_${LABELS}_bs1"
  BEST=$(python3 -c "import json; print(json.load(open('results/decider_best_bs.json'))['best_batch_size'])")
  if [ "$BEST" != "1" ]; then
    gpu_stage "results/decider/test_${LABELS}_bs${BEST}.parquet" decider_run --split test \
      --labels "$LABELS" --batch-size "$BEST" --measure-energy --run-id "test_${LABELS}_bs${BEST}"
  fi
else
  echo "skip test-split Decider runs (ALLOW_TEST=1 not set)"
fi

# NB learning curve (single-process, pinned to one thread; CPU RAPL energy). Evaluates on the test
# split, so it needs ALLOW_TEST=1 like the Decider test runs.
if [ "${ALLOW_TEST:-}" = "1" ]; then
  if [ -e results/nb_runs.csv ]; then
    echo "skip nb_curve (found results/nb_runs.csv)"
  else
    ./preflight_host.sh
    docker compose run --rm -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
      pipeline python -m scripts.nb_curve --measure-energy
  fi
else
  echo "skip nb_curve (ALLOW_TEST=1 not set)"
fi
