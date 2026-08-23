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

# Paths the pipeline owns. Explicit and never `git add -A`: the working tree can
# carry unrelated edits -- a half-finished script, a scratch file, a hand xG
# correction still being checked -- and sweeping those into an unattended commit
# publishes work nobody read. Everything else the run touches (site/,
# data/output_data/, MEX_pinnacle_h2h.csv) is gitignored and regenerable.
PUBLISH_PATHS=(
  data/MEX_ligamx.csv
  data/MEX_upcoming_fixtures.csv
  data/MEX_fixtures_meta.json
  data/dashboard/json
  models/MEX_team_stats.csv
  models/MEX_team_stats_match_simulations.csv
  models/MEX_model_meta.json
)

# Commit the pipeline's output and push it, rebasing onto whatever CI pushed while
# we were running.
#
# NO `[skip ci]` here, deliberately. These paths include data/dashboard/json/**,
# which is exactly what deploy-pages.yml triggers on -- with the suffix the JSON
# lands on main and the published board never rebuilds, and the two silently
# disagree. capture-odds.yml's ticks may skip CI because nothing watches their
# files; this commit is the opposite case.
#
# The retry exists because the capture workflow pushes to main every few minutes,
# so a push racing one of its commits is normal rather than exceptional.
ligamx_publish() {
  local branch attempt=0
  local max_attempts=3

  if ! git rev-parse --git-dir >/dev/null 2>&1; then
    printf 'Not a git repository; skipping publish.\n' >&2
    return 0
  fi

  branch="$(git rev-parse --abbrev-ref HEAD)"
  if [ "$branch" = "HEAD" ]; then
    printf 'Detached HEAD; refusing to publish. Check out a branch and push by hand.\n' >&2
    return 1
  fi

  # `--` so a path that does not exist yet cannot be read as a revision.
  git add -- "${PUBLISH_PATHS[@]}"

  if git diff --cached --quiet; then
    printf '\nNothing to publish: the run reproduced what is already committed.\n'
    return 0
  fi

  printf '\nPublishing:\n'
  git diff --cached --stat | sed 's/^/  /'

  # Every run commits, including one that found nothing new: the freshness stamps
  # have to reach the published board or it reports STALE despite having just been
  # refreshed. So the message distinguishes the two instead, and MEX_ligamx.csv is
  # what it asks -- that file moves only when a match, a score or an xG figure did.
  local subject="chore(data): refresh via ligamx.sh all"
  if git diff --cached --quiet -- data/MEX_ligamx.csv; then
    subject="chore(data): confirm freshness (no new results)"
  fi
  git commit -q -m "$subject"

  while [ "$attempt" -lt "$max_attempts" ]; do
    if git push origin "$branch" 2>&1 | sed 's/^/  /'; then
      printf 'Pushed to origin/%s.\n' "$branch"
      return 0
    fi
    attempt=$((attempt + 1))
    if [ "$attempt" -ge "$max_attempts" ]; then
      break
    fi
    printf 'Push rejected (CI likely pushed first); rebasing and retrying (%d/%d)...\n' \
      "$attempt" "$max_attempts"
    # --autostash so an unrelated dirty file cannot abort the rebase; the commit
    # itself is already made, so nothing of the run is at risk here.
    if ! git pull --rebase --autostash origin "$branch" 2>&1 | sed 's/^/  /'; then
      printf 'Rebase failed. The refresh is committed locally; resolve and push by hand.\n' >&2
      return 1
    fi
  done

  printf 'Push failed after %d attempts. The refresh is committed locally and safe;\n' "$max_attempts" >&2
  printf 'run `git push origin %s` once the network or the conflict is sorted.\n' "$branch" >&2
  return 1
}

# --- commands ---------------------------------------------------------------

# Fetch fixtures/results/upcoming + xG from SofaScore, then blend HExpG+/AExpG+,
# then drain the odds-capture history into the newly-arrived rows.
# Incremental and forward-only: matches at or before the newest cached date are
# never revisited, so run `verify-xg` periodically to catch what it missed.
#
# The reduce step belongs here and nowhere else. The capture loop runs every few
# minutes in CI, but it can only ever append to MEX_odds_capture_history.csv:
# reducing writes into MEX_ligamx.csv, which has no row for a fixture until it has
# been played. So the openers and closes sit in the history until mex_fixture
# creates their row, and this is the moment right after that happens. Offline and
# idempotent -- it fills blanks only, so re-running it costs nothing and can never
# overwrite the hand-entered openers or the repaired Pinnacle closes.
cmd_update() {
  run_pipeline "Liga MX data update" \
    "$PYTHON -m ligamx.fixtures.mex_fixture" \
    "$PYTHON -m ligamx.xg.compute_expg" \
    "$PYTHON -m ligamx.odds.reduce_capture_history"
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

# One capture tick, run by hand. Normally this runs in GitHub Actions every
# 5-15 minutes (see SETUP_CAPTURE.md); locally it is mostly useful with
# --dry-run to see what the next tick would spend.
#
# Both captures self-gate: the opener skips fixtures whose books have already
# opened, and the close spends nothing outside its 20-minute pre-kickoff window.
# So running this repeatedly is safe and usually free. Extra args pass through to
# both (--dry-run being the one worth using).
cmd_capture() {
  local rc=0
  if [ -n "${ODDS_API_IO_KEY:-}" ]; then
    "$PYTHON" -m ligamx.odds.fetch_oddsapiio_opens "$@" || rc=$?
  else
    printf 'ODDS_API_IO_KEY not set; skipping the opening-line capture.\n' >&2
  fi
  if [ -n "${THE_ODDS_API_KEY:-}" ]; then
    "$PYTHON" -m ligamx.odds.capture_close "$@" || rc=$?
  else
    printf 'THE_ODDS_API_KEY not set; skipping the closing-line capture.\n' >&2
  fi
  return "$rc"
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

# Runs the whole pipeline and publishes the result. `--no-push` stops at the
# working tree, for when you want to inspect a run before it reaches the board.
cmd_all() {
  local push=1
  if [ "${1:-}" = "--no-push" ]; then
    push=0
    shift
  fi

  ligamx_require_env THE_ODDS_API_KEY || return 1

  run_pipeline "Full Liga MX workflow" \
    "$PYTHON -m ligamx.fixtures.mex_fixture" \
    "$PYTHON -m ligamx.xg.compute_expg" \
    "$PYTHON -m ligamx.odds.reduce_capture_history" \
    "$PYTHON -m ligamx.models.dc" \
    "$PYTHON -m ligamx.odds.fetch_pinnacle_h2h" \
    "$PYTHON -m ligamx.odds.export_upcoming_market_comparison" \
    "$PYTHON -m ligamx.dashboard.export_dashboard" \
    "./scripts/build_dashboard_site.sh"

  printf 'site/ is ready for GitHub Pages deploy.\n'

  if [ "$push" -eq 1 ]; then
    ligamx_publish
  else
    printf '\n--no-push: leaving the refresh uncommitted in the working tree.\n'
  fi
}

show_help() {
  cat <<'EOF'
Usage: ./scripts/ligamx.sh <command> [args]

Commands:
  update     Fetch new fixtures/results/xG from SofaScore, recompute ExpG+,
             fold captured openers/closes into the newly-played rows
  recompute  Recompute ExpG+ and fix dates from the local CSV
  verify-xg  Audit the CSV against SofaScore; add --fix to repair
  model      Run the goals model
  odds       Fetch Pinnacle 1X2 odds (needs THE_ODDS_API_KEY)
  capture    One odds-capture tick: openers + pre-kickoff closes
  publish    Rebuild market comparison, dashboard and site/
  all        update -> model -> odds -> publish -> commit + push
             (add --no-push to stop at the working tree)
  help       Show this message

When to use what:
  After a matchday       all         (commits and pushes; Pages redeploys)
  Data looks wrong       verify-xg   (then --fix to repair)
  Hand-edited the CSV    recompute -> model -> publish
  Only rebuilding site/  publish     (offline, spends no odds credit)
  Checking capture cost  capture --dry-run   (spends nothing)

capture normally runs unattended in GitHub Actions; see SETUP_CAPTURE.md. Both of
its captures self-gate, so running it by hand is safe and usually free.

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
  capture) shift; cmd_capture "$@" ;;
  publish) shift; cmd_publish ;;
  all) shift; cmd_all "$@" ;;
  help | -h | --help) show_help ;;
  *)
    echo "Unknown command: $1" >&2
    show_help >&2
    exit 1
    ;;
esac
