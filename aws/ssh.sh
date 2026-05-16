#!/usr/bin/env bash
# SSH to the training instance. Forwards any extra args to ssh (e.g. a command).

set -euo pipefail
source "$(dirname "$0")/lib.sh"
preflight
require_running

exec ssh "${SSH_OPTS[@]}" "$(ssh_target)" "$@"
