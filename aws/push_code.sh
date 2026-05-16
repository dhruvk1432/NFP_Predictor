#!/usr/bin/env bash
# rsync your code dirs to the instance. Excludes data/, _output/, _temp/,
# git, caches, notebooks, etc. — anything large or generated.
#
# Run this every time you edit code locally and want to retrain on AWS.

set -euo pipefail
source "$(dirname "$0")/lib.sh"
preflight
require_running

TARGET="$(ssh_target)"

ssh "${SSH_OPTS[@]}" "${TARGET}" "mkdir -p ${REMOTE_PROJECT_DIR}"

WRAPPER="$(ensure_ssh_wrapper)"
log "Rsync ${REPO_ROOT}/ -> ${TARGET}:${REMOTE_PROJECT_DIR}/"
rsync -avz --delete \
  -e "${WRAPPER}" \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '.ipynb_checkpoints/' \
  --exclude '*.ipynb' \
  --exclude '.DS_Store' \
  --exclude 'data/' \
  --exclude '_output/' \
  --exclude '_temp/' \
  --exclude 'continuous_futures/' \
  --exclude 'economist_panel/' \
  --exclude 'catboost_info/' \
  --exclude 'aws/.keys/' \
  --exclude 'aws/.state/' \
  --exclude '.env' \
  "${REPO_ROOT}/" "${TARGET}:${REMOTE_PROJECT_DIR}/"
log "Done. (.env was NOT copied — use aws/push_env.sh for that.)"
