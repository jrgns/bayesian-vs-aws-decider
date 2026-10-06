#!/usr/bin/env bash
# One-command stability test: ALLOW_TEST=1 ./stability/run_stability.sh  (on the execution host).
# Subset -> N1, N2 -> Decider runs (preflight before each) -> metrics. Every run is its own
# `docker compose run --rm pipeline` process. Runs whose output parquet exists are skipped.
# Spec: docs/PLAN_STABILITY.md. If D3 raises in deterministic mode, D3w (warn_only) runs instead.
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose build -q pipeline
DECIDER_IMAGE=$(docker image inspect nb-vs-decider:dev --format '{{.Id}}')
export DECIDER_IMAGE
mkdir -p stability/runs

RUNS=stability/runs
NB_ENV=(-e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1)

[ -e stability/subset_ids.csv ] || docker compose run --rm -T cpu python -m scripts.stability_subset

# NB: N1 is one process; N2 is one process per run (PYTHONHASHSEED left unset on purpose).
[ -e $RUNS/N1/run_9.parquet ] || docker compose run --rm -T "${NB_ENV[@]}" pipeline \
  python -m scripts.stability_nb --cond N1
for k in $(seq 0 9); do
  [ -e $RUNS/N2/run_$k.parquet ] || docker compose run --rm -T "${NB_ENV[@]}" pipeline \
    python -m scripts.stability_nb --cond N2 --run "$k"
done

# decider COND COUNT [docker run extra args...]
decider() {
  local cond=$1 count=$2 rc
  shift 2
  for k in $(seq 0 $((count - 1))); do
    if [ -e $RUNS/$cond/run_$k.parquet ]; then
      echo "skip  $cond run $k"
      continue
    fi
    ./preflight_host.sh
    echo "run   $cond run $k"
    rc=0
    docker compose run --rm -T "$@" pipeline \
      python -m scripts.stability_run --cond "$cond" --run "$k" || rc=$?
    [ "$rc" = 0 ] || return "$rc"
  done
}

DET=(-e CUBLAS_WORKSPACE_CONFIG=:4096:8)
decider D2 10
decider D1 10
rc=0
decider D3 10 "${DET[@]}" || rc=$?
if [ "$rc" = 3 ]; then
  echo "D3 raised in deterministic mode (see $RUNS/D3/error_run_*.json): running D3w"
  decider D3w 10 "${DET[@]}"
elif [ "$rc" != 0 ]; then
  exit "$rc"
fi
decider D5 5
decider D6a 5
decider D6b 5

docker compose run --rm -T cpu python -m scripts.stability_metrics
echo "done: stability/results.csv, flipped_examples.json, versions.json"
