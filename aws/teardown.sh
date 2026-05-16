#!/usr/bin/env bash
# Tears down everything provision.sh created. Prompts before deleting.
#
# WARNING: with --purge-bucket, ALL data in s3://<bucket> is deleted, including
# every versioned object. Without that flag, the bucket is kept (you can delete
# manually in the console).

set -euo pipefail
source "$(dirname "$0")/lib.sh"

PURGE_BUCKET=0
for a in "$@"; do
  case "$a" in
    --purge-bucket) PURGE_BUCKET=1 ;;
    *) die "Unknown flag: $a" ;;
  esac
done

preflight
ensure_bucket_name

read -r -p "Tear down ${INSTANCE_NAME_TAG} in ${AWS_REGION}? Type 'yes' to confirm: " ANSWER
[[ "${ANSWER}" == "yes" ]] || die "Aborted."

# Instance
if [[ -f "${INSTANCE_ID_FILE}" ]]; then
  IID="$(cat "${INSTANCE_ID_FILE}")"
  if aws_ ec2 describe-instances --instance-ids "${IID}" >/dev/null 2>&1; then
    log "Terminating instance ${IID} ..."
    aws_ ec2 terminate-instances --instance-ids "${IID}" >/dev/null
    aws_ ec2 wait instance-terminated --instance-ids "${IID}"
  fi
  rm -f "${INSTANCE_ID_FILE}"
fi

# Security group
SG_ID="$(aws_ ec2 describe-security-groups --filters "Name=group-name,Values=${SG_NAME}" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo None)"
if [[ "${SG_ID}" != "None" && -n "${SG_ID}" ]]; then
  log "Deleting security group ${SG_ID} ..."
  aws_ ec2 delete-security-group --group-id "${SG_ID}" || log "  (could not delete; check dependencies)"
fi

# Key pair (keeps local copy unless --purge-bucket given, in case you want to redeploy)
if aws_ ec2 describe-key-pairs --key-names "${KEY_NAME}" >/dev/null 2>&1; then
  log "Deleting AWS key pair ${KEY_NAME} (local ${KEY_PATH} kept) ..."
  aws_ ec2 delete-key-pair --key-name "${KEY_NAME}"
fi

# Instance profile + role
if aws_ iam get-instance-profile --instance-profile-name "${INSTANCE_PROFILE_NAME}" >/dev/null 2>&1; then
  aws_ iam remove-role-from-instance-profile \
    --instance-profile-name "${INSTANCE_PROFILE_NAME}" --role-name "${INSTANCE_ROLE_NAME}" 2>/dev/null || true
  aws_ iam delete-instance-profile --instance-profile-name "${INSTANCE_PROFILE_NAME}"
fi
if aws_ iam get-role --role-name "${INSTANCE_ROLE_NAME}" >/dev/null 2>&1; then
  aws_ iam delete-role-policy --role-name "${INSTANCE_ROLE_NAME}" \
    --policy-name "${PROJECT_NAME}-s3-access" 2>/dev/null || true
  aws_ iam delete-role --role-name "${INSTANCE_ROLE_NAME}"
fi

# Bucket
if [[ "${PURGE_BUCKET}" == "1" ]]; then
  log "Purging bucket s3://${S3_BUCKET} (all versions) ..."
  aws_ s3api delete-objects --bucket "${S3_BUCKET}" \
    --delete "$(aws_ s3api list-object-versions --bucket "${S3_BUCKET}" \
      --output json --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' \
      | jq -c '. + {Quiet:true}')" 2>/dev/null || true
  aws_ s3api delete-objects --bucket "${S3_BUCKET}" \
    --delete "$(aws_ s3api list-object-versions --bucket "${S3_BUCKET}" \
      --output json --query '{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}' \
      | jq -c '. + {Quiet:true}')" 2>/dev/null || true
  aws_ s3 rb "s3://${S3_BUCKET}" --force
else
  log "Bucket s3://${S3_BUCKET} kept (re-run with --purge-bucket to delete it)."
fi

log "Teardown complete."
