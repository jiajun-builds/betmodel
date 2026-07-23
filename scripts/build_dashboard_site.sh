#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_DASHBOARD="$ROOT/dashboard"
SOURCE_JSON="$ROOT/data/dashboard/json"
SITE_DIR="$ROOT/site"
SITE_DATA_DIR="$SITE_DIR/data"

if [ ! -d "$SOURCE_DASHBOARD" ]; then
  echo "Dashboard source directory not found: $SOURCE_DASHBOARD" >&2
  exit 1
fi

if [ ! -d "$SOURCE_JSON" ]; then
  echo "Dashboard JSON directory not found: $SOURCE_JSON" >&2
  exit 1
fi

for required in \
  dashboard_meta.json \
  upcoming_fixtures.json \
  match_predictions.json \
  team_strength_rankings.json \
  upcoming_market_comparison.json
do
  if [ ! -f "$SOURCE_JSON/$required" ]; then
    echo "Missing required dashboard JSON file: $SOURCE_JSON/$required" >&2
    exit 1
  fi
done

rm -rf "$SITE_DIR"
mkdir -p "$SITE_DATA_DIR"

cp -R "$SOURCE_DASHBOARD/." "$SITE_DIR/"
cp "$SOURCE_JSON"/*.json "$SITE_DATA_DIR/"

echo "Built site bundle:"
echo "  Site root : $SITE_DIR"
echo "  Data files: $SITE_DATA_DIR"
