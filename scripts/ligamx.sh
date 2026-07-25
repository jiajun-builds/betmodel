#!/usr/bin/env bash
#
# Single entry point for every Liga MX workflow. Activates the conda env via
# common.sh, then dispatches to the ligamx.* python modules.
#
#   ./scripts/ligamx.sh help
#
# build_dashboard_site.sh stays a separate file on purpose: the GitHub Pages
# workflow (.github/workflows/deploy-pages.yml) invokes it directly.

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"
ligamx_bootstrap

format_duration() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))

  if [ "$hours" -gt 0 ]; then
    printf '%dh%02dm%02ds' "$hours" "$minutes" "$seconds"
  elif [ "$minutes" -gt 0 ]; then
    printf '%dm%02ds' "$minutes" "$seconds"
  else
    printf '%ds' "$seconds"
  fi
}

# run_step <label> <command...> -- prints a header, times it, reports failures.
run_step() {
  local label="$1"
  shift

  local started_at finished_at elapsed
  printf '\n[%s] %s\n' "$label" "$*"

  started_at="$(date +%s)"
  if "$@"; then
    finished_at="$(date +%s)"
    elapsed=$((finished_at - started_at))
    printf '[%s] Done in %s\n' "$label" "$(format_duration "$elapsed")"
  else
    finished_at="$(date +%s)"
    elapsed=$((finished_at - started_at))
    printf '[%s] Failed after %s\n' "$label" "$(format_duration "$elapsed")" >&2
    printf 'Failed command: %s\n' "$*" >&2
    return 1
  fi
}

# run_pipeline <title> <command-string>...
# Runs each command in order with auto-numbered STEP n/N labels. Each argument
# is one whitespace-separated command line, so adding a phase means adding a
# line -- there are no step counters to renumber.
run_pipeline() {
  local title="$1"
  shift

  local total=$#
  local index=0
  local started_at finished_at elapsed
  started_at="$(date +%s)"

  printf '%s (%d steps)\n' "$title" "$total"

  local cmd
  for cmd in "$@"; do
    index=$((index + 1))
    # shellcheck disable=SC2086 -- intentional word splitting into argv
    run_step "STEP $index/$total" $cmd
  done

  finished_at="$(date +%s)"
  elapsed=$((finished_at - started_at))
  printf '\nAll steps completed in %s\n' "$(format_duration "$elapsed")"
}

# --- commands ---------------------------------------------------------------

# Fetch fixtures/results/upcoming + xG from SofaScore, then blend HExpG+/AExpG+.
# Incremental and forward-only: matches at or before the newest cached date are
# never revisited, so run `verify-xg` periodically to catch what it missed.
cmd_update() {
  run_pipeline "Liga MX data update" \
    "$PYTHON -m ligamx.fixtures.mex_fixture" \
    "$PYTHON -m ligamx.xg.compute_expg"
}

# Recompute HExpG+/AExpG+ and normalize the Date column WITHOUT re-fetching from
# SofaScore. Run this after hand-editing MEX_ligamx.csv (adding matches, fixing
# xG) so the blended columns and Excel-mangled dates are repaired in place.
cmd_recompute() {
  "$PYTHON" -m ligamx.xg.compute_expg
}

# Audit stored xG, scores and Round/Season against the SofaScore API. Read-only;
# pass --fix (and the gated --fix-xg-diffs / --fix-meta) to repair. This is the
# only command that re-checks rows `update` has already written.
cmd_verify_xg() {
  "$PYTHON" -m ligamx.xg.verify_xg "$@"
}

cmd_model() {
  "$PYTHON" -m ligamx.models.dc
}

# The only step needing network + THE_ODDS_API_KEY. The market comparison it
# feeds belongs to `publish`, so odds can be fetched and published separately.
cmd_odds() {
  ligamx_require_env THE_ODDS_API_KEY || return 1
  "$PYTHON" -m ligamx.odds.fetch_pinnacle_h2h
}

# Fully offline. Rebuilds the exports from what is already on disk, which
# includes the MODEL OUTPUT (MEX_team_stats.csv, ..._match_simulations.csv), not
# just MEX_ligamx.csv -- so a CSV edit does not reach the dashboard until `model`
# re-runs. Order matters: export_dashboard reads the market comparison CSV.
cmd_publish() {
  run_pipeline "Rebuild exports and site/" \
    "$PYTHON -m ligamx.odds.export_upcoming_market_comparison" \
    "$PYTHON -m ligamx.dashboard.export_dashboard" \
    "./scripts/build_dashboard_site.sh"
}

cmd_all() {
  ligamx_require_env THE_ODDS_API_KEY || return 1

  run_pipeline "Full Liga MX workflow" \
    "$PYTHON -m ligamx.fixtures.mex_fixture" \
    "$PYTHON -m ligamx.xg.compute_expg" \
    "$PYTHON -m ligamx.models.dc" \
    "$PYTHON -m ligamx.odds.fetch_pinnacle_h2h" \
    "$PYTHON -m ligamx.odds.export_upcoming_market_comparison" \
    "$PYTHON -m ligamx.dashboard.export_dashboard" \
    "./scripts/build_dashboard_site.sh"

  printf 'site/ is ready for GitHub Pages deploy.\n'
}

show_help() {
  cat <<'EOF'
Usage: ./scripts/ligamx.sh <command> [args]

Commands:
  update     Fetch new fixtures/results/xG from SofaScore, recompute ExpG+
  recompute  Recompute ExpG+ and fix dates from the local CSV
  verify-xg  Audit the CSV against SofaScore; add --fix to repair
  model      Run the goals model
  odds       Fetch Pinnacle 1X2 odds (needs THE_ODDS_API_KEY)
  publish    Rebuild market comparison, dashboard and site/
  all        update -> model -> odds -> publish
  help       Show this message

When to use what:
  After a matchday       all
  Data looks wrong       verify-xg   (then --fix to repair)
  Hand-edited the CSV    recompute -> model -> publish
  Only rebuilding site/  publish     (offline, spends no odds credit)

publish reuses the last model run. If MEX_ligamx.csv changed, run model first or
the dashboard will rebuild from stale team strengths.
EOF
}

case "${1:-help}" in
  update) shift; cmd_update ;;
  recompute) shift; cmd_recompute ;;
  verify-xg) shift; cmd_verify_xg "$@" ;;
  model) shift; cmd_model ;;
  odds) shift; cmd_odds ;;
  publish) shift; cmd_publish ;;
  all) shift; cmd_all ;;
  help | -h | --help) show_help ;;
  *)
    echo "Unknown command: $1" >&2
    show_help >&2
    exit 1
    ;;
esac
