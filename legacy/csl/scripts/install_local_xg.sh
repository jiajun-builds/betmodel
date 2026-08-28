#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Install the daily local xG refresh as a launchd job (macOS).
#
# Run this ONCE, on the home Mac, from an interactive shell where `conda` works:
#     ./scripts/install_local_xg.sh [HOUR]
# HOUR is the 24h local hour to run at (default 8 = 08:00). Pick a time BEFORE
# GitHub's daily full run (09:17 London) so CI picks up the fresh xG.
#
# It bakes the detected conda path into the job's PATH so launchd (which has a
# minimal environment) can find and activate the env. Re-run it to update.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOUR="${1:-8}"
LABEL="com.cslmonitor.fetch-xg"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/cslmonitor-fetch-xg.log"
SCRIPT="$REPO/scripts/fetch_xg_local.sh"

# Detect the conda base so the launchd job can find `conda` (its PATH is bare).
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -z "$CONDA_BASE" ]; then
  echo "ERROR: 'conda' not found. Run this from a shell where conda is active" >&2
  echo "       (so its path can be baked into the launchd job)." >&2
  exit 1
fi
JOB_PATH="$CONDA_BASE/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

chmod +x "$SCRIPT"
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$SCRIPT</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>$JOB_PATH</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>$HOUR</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
  <key>RunAtLoad</key><false/>
</dict>
</plist>
EOF

# (Re)load the job.
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load -w "$PLIST"

echo "✅ Installed launchd job '$LABEL' — runs daily at $(printf '%02d' "$HOUR"):00."
echo "   Script : $SCRIPT"
echo "   Log    : $LOG"
echo "   conda  : $CONDA_BASE"
echo
echo "Test it now:           launchctl start $LABEL   &&   tail -f \"$LOG\""
echo "Let the Mac auto-wake (so it can sleep otherwise), e.g. 5 min before:"
echo "   sudo pmset repeat wakeorpoweron MTWRFSU $(printf '%02d' $((HOUR>0?HOUR-1:23))):55:00"
echo "Uninstall:             launchctl unload \"$PLIST\" && rm \"$PLIST\""
