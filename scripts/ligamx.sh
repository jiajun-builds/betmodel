#!/usr/bin/env bash

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

print_phase_header() {
  local step_label="$1"
  local title="$2"

  printf '\n[%s] %s\n' "$step_label" "$title"
}

run_timed_phase() {
  local step_label="$1"
  local title="$2"
  local command_label="$3"
  shift 3

  local started_at
  local finished_at
  local elapsed

  print_phase_header "$step_label" "$title"
  printf 'Command: %s\n' "$command_label"

  started_at="$(date +%s)"
  if "$@"; then
    finished_at="$(date +%s)"
    elapsed=$((finished_at - started_at))
    printf '[%s] Done in %s\n' "$step_label" "$(format_duration "$elapsed")"
  else
    finished_at="$(date +%s)"
    elapsed=$((finished_at - started_at))
    printf '[%s] Failed after %s\n' "$step_label" "$(format_duration "$elapsed")" >&2
    printf 'Failed command: %s\n' "$command_label" >&2
    return 1
  fi
}

run_update() {
  ./scripts/run_ligamx_update.sh
}

# Recompute HExpG+/AExpG+ and normalize the Date column WITHOUT re-fetching from
# SofaScore. Run this after hand-editing MEX_ligamx.csv (adding matches, fixing
# xG) so the blended columns and Excel-mangled dates are repaired in place.
run_recompute() {
  "$PYTHON" -m ligamx.xg.compute_expg
}

# Audit stored xG, scores and Round/Season against the SofaScore API. Read-only;
# pass --fix (and the gated --fix-xg-diffs / --fix-meta) to repair.
run_verify_xg() {
  "$PYTHON" -m ligamx.xg.verify_xg "$@"
}

run_model() {
  ./scripts/ligamx-model.sh
}

run_dashboard() {
  "$PYTHON" -m ligamx.dashboard.export_dashboard
}

run_odds_fetch() {
  ligamx_require_env THE_ODDS_API_KEY || return 1
  "$PYTHON" -m ligamx.odds.fetch_pinnacle_h2h
}

run_market_comparison() {
  "$PYTHON" -m ligamx.odds.export_upcoming_market_comparison
}

run_odds() {
  run_odds_fetch
  run_market_comparison
}

run_site_build() {
  ./scripts/build_dashboard_site.sh
}

run_publish() {
  run_dashboard
  run_site_build
}

# Rebuild the market comparison + site WITHOUT re-fetching odds.
run_republish() {
  run_market_comparison
  run_publish
}

run_all() {
  local started_at
  local finished_at
  local elapsed

  ligamx_require_env THE_ODDS_API_KEY || return 1

  started_at="$(date +%s)"

  cat <<'EOF'
Running full Liga MX workflow:
  1. Data update
  2. Model export
  3. Odds fetch
  4. Market comparison export
  5. Dashboard export
  6. Publish site
EOF

  run_timed_phase "STEP 1/6" "Data Update" "./scripts/run_ligamx_update.sh" run_update
  run_timed_phase "STEP 2/6" "Model Export" "./scripts/ligamx-model.sh" run_model
  run_timed_phase "STEP 3/6" "Odds Fetch" "python -m ligamx.odds.fetch_pinnacle_h2h" run_odds_fetch
  run_timed_phase "STEP 4/6" "Market Comparison Export" "python -m ligamx.odds.export_upcoming_market_comparison" run_market_comparison
  run_timed_phase "STEP 5/6" "Dashboard Export" "python -m ligamx.dashboard.export_dashboard" run_dashboard
  run_timed_phase "STEP 6/6" "Publish Site" "./scripts/build_dashboard_site.sh" run_site_build

  finished_at="$(date +%s)"
  elapsed=$((finished_at - started_at))

  printf '\nAll steps completed in %s\n' "$(format_duration "$elapsed")"
  printf 'site/ is ready for GitHub Pages deploy.\n'
}

show_help() {
  cat <<'EOF'
Usage:
  ./scripts/ligamx.sh <command>
  ./scripts/ligamx.sh

Commands:
  update     Run fixtures/xG/expg data update pipeline
  recompute  Recompute HExpG+/AExpG+ and fix dates after hand-editing the CSV
  verify-xg  Audit stored xG/scores/Round/Season vs SofaScore; read-only unless --fix
  model      Run the goals model export
  dashboard  Export dashboard CSV and JSON
  odds       Fetch Pinnacle odds and export market comparison
  publish    Rebuild dashboard exports and site/
  republish  Rebuild market comparison + dashboard + site/ WITHOUT fetching odds
  all        Run the full local workflow, including odds
  help       Show this help message
EOF
}

dispatch_command() {
  local cmd="${1:-}"
  shift || true
  case "$cmd" in
    update) run_update ;;
    recompute) run_recompute ;;
    verify-xg) run_verify_xg "$@" ;;
    model) run_model ;;
    dashboard) run_dashboard ;;
    odds) run_odds ;;
    publish) run_publish ;;
    republish) run_republish ;;
    all) run_all ;;
    help|-h|--help) show_help ;;
    *)
      echo "Unknown command: $cmd" >&2
      show_help >&2
      return 1
      ;;
  esac
}

show_menu() {
  echo "Liga MX workflow menu"
  echo "  1) update"
  echo "  2) model"
  echo "  3) dashboard"
  echo "  4) odds"
  echo "  5) publish"
  echo "  6) all"
  echo "  7) help"
  echo "  q) quit"
  printf "Choose an action: "

  local choice
  read -r choice

  case "$choice" in
    1) dispatch_command update ;;
    2) dispatch_command model ;;
    3) dispatch_command dashboard ;;
    4) dispatch_command odds ;;
    5) dispatch_command publish ;;
    6) dispatch_command all ;;
    7) dispatch_command help ;;
    q|Q) exit 0 ;;
    *)
      echo "Unknown menu choice: $choice" >&2
      return 1
      ;;
  esac
}

if [ "$#" -eq 0 ]; then
  show_menu
else
  dispatch_command "$@"
fi
