#!/usr/bin/env bash
# Thin wrapper around deploy.py for convenience.
# All RunPod management is in deploy.py.
#
# Usage:
#   ./deploy.sh                    # sync code to pod
#   ./deploy.sh --setup            # first-time pod setup
#   ./deploy.sh --train            # sync + start training with wandb
#   ./deploy.sh --run "cmd"        # sync + run arbitrary command
#   ./deploy.sh --status           # check pod status
#   ./deploy.sh --stop             # stop pod (preserves data)
#   ./deploy.sh --terminate        # destroy pod
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/deploy.py" "$@"
