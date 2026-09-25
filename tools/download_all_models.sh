#!/bin/bash
# Download all models for Studio Effects.
# This is a wrapper around src/download_models.sh — kept separate so tools/
# has all operational scripts in one place.
#
set -euo pipefail

SCRIPT_DIR="tools"
if [ ! -f "$SCRIPT_DIR/../src/download_models.sh" ]; then
    SCRIPT_DIR="$HOME/.local/share/studio-effects"
fi

exec "$SCRIPT_DIR/src/download_models.sh" "$@"
