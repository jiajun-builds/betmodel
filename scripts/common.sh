#!/usr/bin/env bash

LIGAMX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIGAMX_CONDA_SH="${LIGAMX_CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
LIGAMX_ENV_NAME="${LIGAMX_ENV_NAME:-ligamx-workflows}"
LIGAMX_ENV_FILE="${LIGAMX_ENV_FILE:-$LIGAMX_ROOT/.env.local}"

ligamx_repo_root() {
  printf '%s\n' "$LIGAMX_ROOT"
}

ligamx_activate_environment() {
  local conda_sh="$LIGAMX_CONDA_SH"

  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [ -n "$conda_base" ] && [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
      conda_sh="$conda_base/etc/profile.d/conda.sh"
    fi
  fi

  if [ ! -f "$conda_sh" ]; then
    echo "Conda init script not found: $conda_sh" >&2
    return 1
  fi

  # shellcheck disable=SC1090
  source "$conda_sh"
  conda activate "$LIGAMX_ENV_NAME"
}

ligamx_load_local_env() {
  local f
  for f in "$LIGAMX_ENV_FILE" "$LIGAMX_ROOT/.env"; do
    if [ -f "$f" ]; then
      set -a
      # shellcheck disable=SC1090
      source "$f"
      set +a
    fi
  done
}

ligamx_bootstrap() {
  ligamx_activate_environment || return 1
  cd "$LIGAMX_ROOT" || return 1
  ligamx_load_local_env
  export PYTHONPATH="${LIGAMX_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  export PYTHON="${PYTHON:-python3}"
}

ligamx_require_env() {
  local missing=0
  local var_name
  for var_name in "$@"; do
    if [ -z "${!var_name:-}" ]; then
      echo "Missing required environment variable: $var_name" >&2
      missing=1
    fi
  done
  return "$missing"
}
