#!/usr/bin/env bash
# Securely copy your local .env to the instance.
# .env never goes through S3 — it has API keys (FRED_API_KEY, Unifier token).

set -euo pipefail
source "$(dirname "$0")/lib.sh"
preflight
require_running

LOCAL_ENV="${REPO_ROOT}/.env"
[[ -f "${LOCAL_ENV}" ]] || die "No local .env at ${LOCAL_ENV}"

TARGET="$(ssh_target)"
ssh "${SSH_OPTS[@]}" "${TARGET}" "mkdir -p ${REMOTE_PROJECT_DIR}"
log "Copying .env -> ${TARGET}:${REMOTE_PROJECT_DIR}/.env"
scp "${SSH_OPTS[@]}" "${LOCAL_ENV}" "${TARGET}:${REMOTE_PROJECT_DIR}/.env"
ssh "${SSH_OPTS[@]}" "${TARGET}" "chmod 600 ${REMOTE_PROJECT_DIR}/.env"
log "Done."
