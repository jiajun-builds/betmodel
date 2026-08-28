#!/usr/bin/env bash

CSL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Explicit override only; when empty the conda base is discovered automatically.
CSL_CONDA_SH="${CSL_CONDA_SH:-}"
CSL_ENV_NAME="${CSL_ENV_NAME:-csl-workflows}"
CSL_ENV_FILE="${CSL_ENV_FILE:-$CSL_ROOT/.env.local}"

csl_repo_root() {
  printf '%s\n' "$CSL_ROOT"
}

csl_activate_environment() {
  local conda_sh="$CSL_CONDA_SH"

  if [ -z "$conda_sh" ]; then
    local base
    # `conda info --base` is authoritative when conda is on PATH. The literal
    # paths cover shells where it is not (cron, non-interactive, stripped PATH),
    # which is where a hardcoded default used to fail with a misleading error.
    for base in \
      "$(command -v conda >/dev/null 2>&1 && conda info --base 2>/dev/null)" \
      /opt/homebrew/Caskroom/miniconda/base \
      "$HOME/miniconda3" \
      "$HOME/anaconda3" \
      /opt/miniconda3
    do
      if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then
        conda_sh="$base/etc/profile.d/conda.sh"
        break
      fi
    done
  fi

  if [ ! -f "$conda_sh" ]; then
    echo "Could not locate conda.sh in any known conda base." >&2
    echo "Set CSL_CONDA_SH=<conda-base>/etc/profile.d/conda.sh to point at yours." >&2
    return 1
  fi

  # shellcheck disable=SC1090
  source "$conda_sh"
  conda activate "$CSL_ENV_NAME"
}

csl_load_local_env() {
  if [ -f "$CSL_ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$CSL_ENV_FILE"
    set +a
  fi
}

csl_bootstrap() {
  csl_activate_environment || return 1
  cd "$CSL_ROOT" || return 1
  csl_load_local_env
  export PYTHONPATH="${CSL_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  export PYTHON="${PYTHON:-python3}"
}

csl_require_env() {
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
