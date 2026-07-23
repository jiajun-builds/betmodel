#!/usr/bin/env bash
# Liga MX data workflow (single source: SofaScore direct API):
#   fixtures + results + upcoming + xG  ->  HExpG+/AExpG+
# Run from repo root: ./scripts/run_ligamx_update.sh

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"
ligamx_bootstrap

echo "Running in: ${CONDA_DEFAULT_ENV:-$LIGAMX_ENV_NAME}"
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

run_step "Fetch Fixtures/Results/Upcoming + xG (SofaScore)" "$PYTHON" -m ligamx.fixtures.mex_fixture
run_step "Calculate Expected Goal+" "$PYTHON" -m ligamx.xg.compute_expg

echo "All steps completed."
