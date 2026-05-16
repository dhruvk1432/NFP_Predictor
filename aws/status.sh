#!/usr/bin/env bash
# Show current instance state, public IP, and approximate hourly cost.

set -euo pipefail
source "$(dirname "$0")/lib.sh"
preflight
ensure_bucket_name

if [[ ! -f "${INSTANCE_ID_FILE}" ]]; then
  echo "No instance provisioned. Run aws/provision.sh."
  exit 0
fi

IID="$(cat "${INSTANCE_ID_FILE}")"
STATE="$(aws_ ec2 describe-instances --instance-ids "${IID}" \
  --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo unknown)"
TYPE="$(aws_ ec2 describe-instances --instance-ids "${IID}" \
  --query 'Reservations[0].Instances[0].InstanceType' --output text 2>/dev/null || echo unknown)"
IP="$(aws_ ec2 describe-instances --instance-ids "${IID}" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null || echo none)"

cat <<EOF
Instance id   : ${IID}
Type          : ${TYPE}
State         : ${STATE}
Public IP     : ${IP}
Region        : ${AWS_REGION}
S3 bucket     : s3://${S3_BUCKET}
Key path      : ${KEY_PATH}
EOF
