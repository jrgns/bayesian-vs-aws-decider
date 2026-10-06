#!/usr/bin/env bash
# End-to-end entry point: ALLOW_TEST=1 ./run_all.sh  (run on the execution host, from this directory).
# Every stage skips when its output file exists (delete the file to force a rerun).
# DRY_RUN=1 ./run_all.sh only prints each stage and whether it would skip or run.
set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=${DRY_RUN:-0}

if [ "$DRY_RUN" != 1 ]; then
  docker compose build pipeline
  DECIDER_IMAGE=$(docker image inspect nb-vs-decider:dev --format '{{.Id}}')
  export DECIDER_IMAGE
  # Create host-owned dirs first: containers run as root and would otherwise own them.
  mkdir -p results/decider results/power_samples plots data/raw data/processed data/splits
  # Noise indicator for the energy records (a container cannot see the host's docker daemon).
  HOST_CONTAINERS_RUNNING=$(docker ps -q | wc -l)
  export HOST_CONTAINERS_RUNNING
fi

# stage KIND OUTPUT MODULE [ARGS...]
#   KIND   cpu = `cpu` service; gpu = `pipeline` service after the host GPU preflight;
#          nb  = like gpu, with BLAS/OpenMP pinned to one thread (the NB measurements).
#   OUTPUT the file whose existence marks the stage done; "@fn" uses the bash predicate fn.
stage() {
  local kind=$1 out=$2 module=$3
  shift 3
  local label="${module}${*:+ $*}" finished=0
  if [ "${out#@}" != "$out" ]; then
    "${out#@}" && finished=1 || true
    out="check ${out#@}"
  elif [ -e "$out" ]; then
    finished=1
  fi
  if [ "$finished" = 1 ]; then
    echo "skip  ${label} (done: ${out})"
    return
  fi
  if [ "$DRY_RUN" = 1 ]; then
    echo "RUN   ${label} (not done: ${out})"
    return
  fi
  echo "run   ${label}"
  case $kind in
    cpu) docker compose run --rm cpu python -m "scripts.${module}" "$@" ;;
    gpu)
      ./preflight_host.sh
      docker compose run --rm pipeline python -m "scripts.${module}" "$@"
      ;;
    nb)
      ./preflight_host.sh
      docker compose run --rm -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
        -e MKL_NUM_THREADS=1 pipeline python -m "scripts.${module}" "$@"
      ;;
  esac
}

# test_stage ...: a stage that reads or produces test-split results, gated by ALLOW_TEST=1
# (the owner exports it once the labels are frozen; no script ever sets it).
test_stage() {
  if [ "${ALLOW_TEST:-}" = "1" ]; then
    stage "$@"
  else
    echo "skip  ${3} (ALLOW_TEST=1 not set)"
  fi
}

json_field() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))$2)" "$1" 2>/dev/null; }

# ---- 1. dataset, contamination, splits (CPU only; may run while the GPU is busy) ----
REPO=data/raw/strands-decider
COMMIT=3e94e9d84c620ed5a95f1a3310c3decb971e261c
if [ "$DRY_RUN" != 1 ] && [ ! -e results/contamination.csv ]; then
  if [ ! -d "$REPO/.git" ]; then
    git clone https://github.com/strands-labs/strands-decider "$REPO"
  fi
  git -C "$REPO" checkout --quiet "$COMMIT"
fi
stage cpu data/processed/wos46985_l1_clean.parquet dataset prepare
stage cpu results/contamination.csv contamination
stage cpu results/dataset_summary.json dataset split

# ---- 2. pinned versions and the Decider smoke test ----
stage gpu results/versions.json versions
stage gpu results/decider_smoke.json decider_smoke

# ---- 3. dev runs for every label wording version, then the selection rule ----
for v in v1 v2 v3; do
  stage gpu "results/decider/dev_${v}_bs1.parquet" decider_run --split dev --labels "$v" \
    --batch-size 1 --measure-energy --run-id "dev_${v}_bs1"
done
stage cpu results/selected_labels.json select_labels
LABELS=$(json_field results/selected_labels.json "['selected']" || true)
LABELS=${LABELS:-"<selected>"}
echo "label wording: ${LABELS}"

# ---- 4. throughput sweep with the selected labels (dev subset only) ----
sweep_done() { [ "$(json_field results/decider_best_bs.json "['subset']['labels']")" = "$LABELS" ]; }
stage gpu @sweep_done decider_sweep --labels "$LABELS"
BEST=$(json_field results/decider_best_bs.json "['best_batch_size']" || true)
BEST=${BEST:-"<best>"}

# ---- 5. test split: Decider at bs=1 and at the best batch size, then NB ----
test_stage gpu "results/decider/test_${LABELS}_bs1.parquet" decider_run --split test \
  --labels "$LABELS" --batch-size 1 --measure-energy --run-id "test_${LABELS}_bs1"
if [ "$BEST" != "1" ]; then
  test_stage gpu "results/decider/test_${LABELS}_bs${BEST}.parquet" decider_run --split test \
    --labels "$LABELS" --batch-size "$BEST" --measure-energy --run-id "test_${LABELS}_bs${BEST}"
fi
test_stage nb results/nb_runs.csv nb_curve --measure-energy

# ---- 6. analysis: crossover, NB latency, energy summary, plots ----
test_stage cpu results/crossover.json crossover
test_stage nb results/nb_latency.json nb_latency
test_stage cpu results/energy_summary.csv energy_summary
test_stage cpu plots/learning_curve.png plots
