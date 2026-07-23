#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"
ligamx_bootstrap

"$PYTHON" DC_MEX.py
