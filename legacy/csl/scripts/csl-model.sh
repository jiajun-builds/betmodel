#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"
csl_bootstrap

"$PYTHON" DC_CHN.py
