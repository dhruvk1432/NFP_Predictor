#!/usr/bin/env bash
# Tail the live log of the most recent training run on the instance.
# Exits when you Ctrl-C (training keeps running in tmux).

set -euo pipefail
source "$(dirname "$0")/lib.sh"
preflight
require_running

exec ssh "${SSH_OPTS[@]}" "$(ssh_target)" \
  "tail -F \$(ls -1t ${REMOTE_PROJECT_DIR}/_output/runs/run_*.log 2>/dev/null | head -1)"
