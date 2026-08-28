#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Local xG refresh — run on a residential-IP machine (e.g. an old Mac at home).
#
# SofaScore's Cloudflare 403s GitHub Actions' datacenter IPs, so xG can't be
# fetched from CI. This script fetches it from *this* machine's residential IP,
# then commits & pushes ONLY data/raw_data/xg_data.csv. GitHub CI's scheduled
# run then reads that fresh file and does everything else (merge, model, odds,
# dashboard, site). Because the merge is no-erase, a CI run that 403s just
# retains this file unchanged — no conflict.
#
# Safe to run anytime: it is a no-op if xG hasn't changed. Designed for launchd
# (see scripts/install_local_xg.sh) but also fine to run by hand.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BRANCH="${CSL_BRANCH:-main}"
XG_FILE="data/raw_data/xg_data.csv"
# Full-season is robust (schedule-independent) and safe via the no-erase merge.
# Override with XG_MODE="" for the faster two-round incremental refresh.
XG_MODE="${XG_MODE:---full-season}"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# Single-instance lock so a slow network run can't overlap the next trigger.
LOCK="${TMPDIR:-/tmp}/cslmonitor-fetch-xg.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  log "Another fetch is already running ($LOCK). Exiting."
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

# Activate the conda env, load .env.local, set PYTHONPATH (shared bootstrap).
source "$REPO/scripts/common.sh"
csl_bootstrap

log "Repo: $REPO  |  env: ${CONDA_DEFAULT_ENV:-?}  |  branch: $BRANCH  |  mode: ${XG_MODE:-incremental}"

# Anything dirty besides the generated xG file means a human is using this clone.
foreign_changes() {
  git status --porcelain --untracked-files=no -- ":(exclude)$XG_FILE"
}

# Point the clone at the published tip, discarding whatever is local. Deliberately a
# reset and not a rebase: xg_data.csv is machine-generated, and the pipeline's own
# no-erase merge (run right after) is the correct way to reconcile it with anything
# that landed remotely. A line-level rebase can conflict even when both sides computed
# identical numbers — that is precisely how this fetcher died for 10 days in July 2026.
sync_to_remote() {
  git switch --quiet "$BRANCH"
  git fetch --quiet origin "$BRANCH"
  git reset --quiet --hard "origin/$BRANCH"
}

regenerate_xg() {
  log "Fetching xG ${XG_MODE:+($XG_MODE)} ..."
  "$PYTHON" -m csl.xg.xg_pipeline ${XG_MODE:+$XG_MODE}
}

commit_xg() {
  git add "$XG_FILE"
  git commit --quiet -m "chore(xg): refresh xG from SofaScore (local $(date +%Y-%m-%d))"
  log "Committed xG update."
}

# 0) Self-heal a wedged clone. A conflicting edit to xg_data.csv can stop an earlier run
#    mid-rebase; from then on EVERY run dies at `git switch` with "cannot switch branch
#    while rebasing", so the feed stops without a single error reaching anyone. Unwinding
#    is always right here — the next steps rebuild the file from scratch anyway.
GIT_DIR_PATH="$(git rev-parse --git-dir)"
if [ -d "$GIT_DIR_PATH/rebase-merge" ] || [ -d "$GIT_DIR_PATH/rebase-apply" ]; then
  log "WARNING: clone was left mid-rebase by an earlier run — unwinding it."
  git rebase --abort 2>/dev/null || git rebase --quit 2>/dev/null || true
fi

# Everything below assumes this clone is disposable. That is only safe while xg_data.csv
# is the sole thing that ever changes in it, so bail out rather than reset over real work.
if [ -n "$(foreign_changes)" ]; then
  log "ERROR: uncommitted changes outside $XG_FILE — refusing to reset. Resolve by hand:"
  foreign_changes >&2
  exit 1
fi

# 1) + 2) Sync to the published tip, then fetch fresh xG (writes xg_data.csv).
sync_to_remote
regenerate_xg

# 3) Commit only xg_data.csv, and only if it actually changed.
if git diff --quiet -- "$XG_FILE"; then
  log "No xG changes — nothing to commit. Done."
  exit 0
fi
commit_xg

# 4) Push. If the remote moved under us, rebuild on its new tip instead of rebasing onto
#    it: reset, re-run the pipeline (which merges no-erase over whatever is now
#    published), re-commit. Idempotent, and it can never leave a conflict behind.
for attempt in 1 2 3; do
  if git push --quiet origin "$BRANCH"; then
    log "Pushed to origin/$BRANCH. Done."
    exit 0
  fi
  log "Push attempt $attempt rejected (remote moved); rebuilding on the new tip ..."
  sync_to_remote
  regenerate_xg
  if git diff --quiet -- "$XG_FILE"; then
    log "Remote already carries this xG — nothing left to push. Done."
    exit 0
  fi
  commit_xg
done

log "ERROR: push failed after 3 attempts." >&2
exit 1
