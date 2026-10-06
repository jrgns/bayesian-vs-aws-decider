#!/usr/bin/env bash
# End-to-end entry point. Run on the execution host (orion), from this directory.
set -euo pipefail
cd "$(dirname "$0")"

./preflight_host.sh

docker compose build pipeline
DECIDER_IMAGE=$(docker image inspect nb-vs-decider:dev --format '{{.Id}}')
export DECIDER_IMAGE
mkdir -p results plots

for stage in versions decider_smoke; do
  # later stages (dataset, splits, decider_run, nb_curve, energy, report) are appended here
  docker compose run --rm pipeline python -m "scripts.${stage}"
done
