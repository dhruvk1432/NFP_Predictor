#!/usr/bin/env bash
# Runs ON the EC2 instance. Idempotent.
#
# Installs:
#   - apt: build essentials, libomp (for LightGBM), python3.12, git, awscli, jq,
#          rsync, tmux, htop, libopenblas, graphviz (for plots).
#   - pip: project deps into ~/NFP_Predictor/.venv
#
# Run with:
#   bash ~/NFP_Predictor/aws/on_instance/bootstrap.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

log() { printf '[bootstrap %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "apt-get update / install"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential git curl unzip jq rsync tmux htop \
  python3.12 python3.12-venv python3.12-dev \
  libomp-dev libopenblas-dev \
  graphviz

# AWS CLI v2 (the apt awscli is v1 and outdated).
if ! command -v aws >/dev/null 2>&1; then
  log "Installing AWS CLI v2"
  TMP="$(mktemp -d)"
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "${TMP}/awscliv2.zip"
  (cd "${TMP}" && unzip -q awscliv2.zip && sudo ./aws/install)
  rm -rf "${TMP}"
fi

# Faster S3 sync defaults.
aws configure set default.s3.max_concurrent_requests 32
aws configure set default.s3.multipart_chunksize 64MB

# --- Python venv -----------------------------------------------------------
if [[ ! -d "${PROJECT_DIR}/.venv" ]]; then
  log "Creating venv at ${PROJECT_DIR}/.venv"
  python3.12 -m venv "${PROJECT_DIR}/.venv"
fi
# shellcheck disable=SC1091
source "${PROJECT_DIR}/.venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools

log "Installing Python deps"
pip install \
  "numpy" "pandas" "pyarrow" "scipy" "scikit-learn" \
  "lightgbm" "shap" "optuna" \
  "matplotlib" \
  "joblib" "tqdm" "psutil" "python-dateutil" "python-dotenv" \
  "fredapi" "yfinance" "beautifulsoup4" "requests" \
  "statsmodels" "catboost"

# Optional: Unifier private package. If you don't use the Unifier source,
# this can be skipped.
if ! python -c "import unifier" >/dev/null 2>&1; then
  log "NOTE: 'unifier' is not installed. If your runs use the Unifier source,"
  log "  install it manually, e.g.:  pip install unifier==0.1.13"
fi

log "Activating venv on login (adding to ~/.bashrc)"
if ! grep -q "NFP_Predictor/.venv" "${HOME}/.bashrc" 2>/dev/null; then
  cat >> "${HOME}/.bashrc" <<EOF

# NFP Predictor venv auto-activate
if [[ -f "${PROJECT_DIR}/.venv/bin/activate" ]]; then
  source "${PROJECT_DIR}/.venv/bin/activate"
  cd "${PROJECT_DIR}"
fi
EOF
fi

log "Done. Next: bash aws/on_instance/pull_data.sh"
