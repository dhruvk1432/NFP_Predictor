#!/usr/bin/env bash
# Shared helpers. Source this from every aws/*.sh script.

set -euo pipefail

AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${AWS_DIR}/.." && pwd)"
KEYS_DIR="${AWS_DIR}/.keys"
STATE_DIR="${AWS_DIR}/.state"
mkdir -p "${KEYS_DIR}" "${STATE_DIR}"

# Load committed defaults, then local overrides.
# shellcheck disable=SC1091
source "${AWS_DIR}/config.env"
if [[ -f "${AWS_DIR}/config.local.env" ]]; then
  # shellcheck disable=SC1091
  source "${AWS_DIR}/config.local.env"
fi

KEY_PATH="${KEYS_DIR}/${KEY_NAME}.pem"
INSTANCE_ID_FILE="${STATE_DIR}/instance_id"
S3_BUCKET_FILE="${STATE_DIR}/s3_bucket"

# --- logging --------------------------------------------------------------
log()   { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()   { log "ERROR: $*"; exit 1; }
need()  { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }

# --- aws helpers ----------------------------------------------------------
aws_() { aws --profile "${AWS_PROFILE}" --region "${AWS_REGION}" "$@"; }

account_id() {
  aws_ sts get-caller-identity --query 'Account' --output text
}

default_bucket_name() {
  echo "${PROJECT_NAME}-$(account_id)"
}

ensure_bucket_name() {
  if [[ -z "${S3_BUCKET}" ]]; then
    if [[ -f "${S3_BUCKET_FILE}" ]]; then
      S3_BUCKET="$(cat "${S3_BUCKET_FILE}")"
    else
      S3_BUCKET="$(default_bucket_name)"
    fi
  fi
  export S3_BUCKET
}

instance_id() {
  [[ -f "${INSTANCE_ID_FILE}" ]] || die "No instance id on file. Run aws/provision.sh first."
  cat "${INSTANCE_ID_FILE}"
}

instance_state() {
  aws_ ec2 describe-instances --instance-ids "$(instance_id)" \
    --query 'Reservations[0].Instances[0].State.Name' --output text
}

instance_public_ip() {
  aws_ ec2 describe-instances --instance-ids "$(instance_id)" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}

ssh_target() {
  echo "${REMOTE_USER}@$(instance_public_ip)"
}

# Array form so paths containing spaces (e.g. "Github Repos") work correctly.
# UserKnownHostsFile is /dev/null because SSH splits its value on whitespace,
# which breaks paths with spaces. Host identity here is the provisioned IP,
# so skipping known_hosts is acceptable for this toolkit.
SSH_OPTS=(
  -i "${KEY_PATH}"
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o ServerAliveInterval=60
)

# For tools like rsync that want a single `-e` string, we materialize a
# wrapper script in $HOME (no spaces) that invokes ssh with our options.
# This avoids quoting headaches with shell-tokenized -e values.
SSH_WRAPPER="${HOME}/.cache/nfp-predictor/ssh-wrapper.sh"
ensure_ssh_wrapper() {
  mkdir -p "$(dirname "${SSH_WRAPPER}")"
  cat > "${SSH_WRAPPER}" <<EOF
#!/usr/bin/env bash
exec ssh \\
  -i "${KEY_PATH}" \\
  -o StrictHostKeyChecking=no \\
  -o UserKnownHostsFile=/dev/null \\
  -o LogLevel=ERROR \\
  -o ServerAliveInterval=60 \\
  "\$@"
EOF
  chmod +x "${SSH_WRAPPER}"
  echo "${SSH_WRAPPER}"
}

require_running() {
  local state
  state="$(instance_state)"
  [[ "${state}" == "running" ]] || die "Instance is '${state}', not 'running'. Run aws/start.sh first."
}

# --- preflight ------------------------------------------------------------
preflight() {
  need aws
  need jq
  aws_ sts get-caller-identity >/dev/null 2>&1 \
    || die "AWS CLI profile '${AWS_PROFILE}' is not configured. See aws/README.md."
}
