#!/usr/bin/env bash
# CSL data workflow: fixtures -> (xG + merge) -> HExpG+/AExpG+
# Run from repo root: ./scripts/run_csl_update.sh

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"
csl_bootstrap

echo "Running in: ${CONDA_DEFAULT_ENV:-$CSL_ENV_NAME}"
step=1

run_step() {
  local title="$1"
  shift
  printf "Step %d: %s -- " "$step" "$title"
  if "$@"; then
    echo "Success"
  else
    echo "Failed"
    exit 1
  fi
  step=$((step + 1))
}

run_step "Update Fixtures" "$PYTHON" -m csl.fixtures.chn_fixture_v5

pull_xg_and_merge() {
  # SofaScore's Cloudflare 403s datacenter IPs, so the fetch can't run on CI.
  # Set CSL_SKIP_XG_FETCH=1 there to skip the fetch and merge the committed
  # xg_data.csv (kept fresh by the home Mac; see scripts/LOCAL_XG_SETUP.md).
  if [ -n "${CSL_SKIP_XG_FETCH:-}" ]; then
    echo "CSL_SKIP_XG_FETCH set -- skipping SofaScore fetch, using committed xg_data.csv"
  else
    "$PYTHON" -m csl.xg.xg_pipeline
  fi
  "$PYTHON" -m csl.xg.chn_merge
}
run_step "Pull xG & Update to Fixtures" pull_xg_and_merge

run_step "Calculate Expected Goal+" "$PYTHON" -m csl.xg.compute_expg

echo "All steps completed."
