#!/usr/bin/env bash
# One-time provisioning. Idempotent: re-runs are safe.
#
# Creates:
#   - S3 bucket  (nfp-predictor-<account-id>)
#   - IAM role + instance profile for the EC2 box
#   - SSH key pair (private saved to aws/.keys/<KEY_NAME>.pem)
#   - Security group allowing SSH from your current IP
#   - EC2 instance (m7i.4xlarge by default)
#
# After this completes, run:
#   aws/push_data.sh           # uploads ./data to S3   (first time only, slow)
#   aws/push_env.sh            # copies .env to instance
#   aws/ssh.sh                 # log in
#   on instance: ~/NFP_Predictor/aws/on_instance/bootstrap.sh
#   on instance: ~/NFP_Predictor/aws/on_instance/pull_data.sh

set -euo pipefail
source "$(dirname "$0")/lib.sh"

preflight

ACCOUNT_ID="$(account_id)"
S3_BUCKET="${S3_BUCKET:-${PROJECT_NAME}-${ACCOUNT_ID}}"
export S3_BUCKET
echo "${S3_BUCKET}" > "${S3_BUCKET_FILE}"
log "Using account=${ACCOUNT_ID} region=${AWS_REGION} bucket=${S3_BUCKET}"

# ---------------------------------------------------------------- S3 bucket
# AES256 default encryption is applied automatically by AWS to all new buckets
# since January 2023, so no explicit PutBucketEncryption call is needed.
if aws_ s3api head-bucket --bucket "${S3_BUCKET}" 2>/dev/null; then
  log "Bucket s3://${S3_BUCKET} already exists."
else
  log "Creating bucket s3://${S3_BUCKET} ..."
  if [[ "${AWS_REGION}" == "us-east-1" ]]; then
    aws_ s3api create-bucket --bucket "${S3_BUCKET}" >/dev/null
  else
    aws_ s3api create-bucket --bucket "${S3_BUCKET}" \
      --create-bucket-configuration "LocationConstraint=${AWS_REGION}" >/dev/null
  fi
fi
# Idempotent: re-applying these is a no-op if already set.
aws_ s3api put-public-access-block --bucket "${S3_BUCKET}" \
  --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
aws_ s3api put-bucket-versioning --bucket "${S3_BUCKET}" \
  --versioning-configuration Status=Enabled

# ---------------------------------------------------------------- IAM role
if aws_ iam get-role --role-name "${INSTANCE_ROLE_NAME}" >/dev/null 2>&1; then
  log "IAM role ${INSTANCE_ROLE_NAME} exists, skipping create."
else
  log "Creating IAM role ${INSTANCE_ROLE_NAME} ..."
  aws_ iam create-role --role-name "${INSTANCE_ROLE_NAME}" \
    --assume-role-policy-document "file://${AWS_DIR}/iam/instance-trust-policy.json" >/dev/null
fi

INSTANCE_POLICY_RENDERED="$(mktemp)"
sed "s|__BUCKET__|${S3_BUCKET}|g" "${AWS_DIR}/iam/instance-policy.json" > "${INSTANCE_POLICY_RENDERED}"
aws_ iam put-role-policy --role-name "${INSTANCE_ROLE_NAME}" \
  --policy-name "${PROJECT_NAME}-s3-access" \
  --policy-document "file://${INSTANCE_POLICY_RENDERED}"
rm -f "${INSTANCE_POLICY_RENDERED}"

if aws_ iam get-instance-profile --instance-profile-name "${INSTANCE_PROFILE_NAME}" >/dev/null 2>&1; then
  log "Instance profile ${INSTANCE_PROFILE_NAME} exists, skipping create."
else
  log "Creating instance profile ${INSTANCE_PROFILE_NAME} ..."
  aws_ iam create-instance-profile --instance-profile-name "${INSTANCE_PROFILE_NAME}" >/dev/null
  aws_ iam add-role-to-instance-profile \
    --instance-profile-name "${INSTANCE_PROFILE_NAME}" \
    --role-name "${INSTANCE_ROLE_NAME}"
  log "Waiting 10s for IAM propagation ..."
  sleep 10
fi

# ---------------------------------------------------------------- Key pair
if [[ -f "${KEY_PATH}" ]]; then
  log "Local key file ${KEY_PATH} exists, skipping key pair create."
elif aws_ ec2 describe-key-pairs --key-names "${KEY_NAME}" >/dev/null 2>&1; then
  die "AWS already has key pair '${KEY_NAME}' but local ${KEY_PATH} is missing. \
Either restore ${KEY_PATH} from your password manager, or delete the AWS key pair \
(aws_ ec2 delete-key-pair --key-name ${KEY_NAME}) and rerun this script."
else
  log "Creating key pair ${KEY_NAME} ..."
  aws_ ec2 create-key-pair --key-name "${KEY_NAME}" \
    --query 'KeyMaterial' --output text > "${KEY_PATH}"
  chmod 600 "${KEY_PATH}"
  log "Private key saved to ${KEY_PATH} (mode 600). Back this up to your password manager."
fi

# ---------------------------------------------------------------- VPC + SG
VPC_ID="$(aws_ ec2 describe-vpcs --filters Name=is-default,Values=true \
  --query 'Vpcs[0].VpcId' --output text)"
[[ "${VPC_ID}" == "None" ]] && die "No default VPC in ${AWS_REGION}."

SG_ID="$(aws_ ec2 describe-security-groups \
  --filters "Name=group-name,Values=${SG_NAME}" "Name=vpc-id,Values=${VPC_ID}" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo None)"
if [[ "${SG_ID}" == "None" || -z "${SG_ID}" ]]; then
  log "Creating security group ${SG_NAME} ..."
  SG_ID="$(aws_ ec2 create-security-group --group-name "${SG_NAME}" \
    --description "NFP Predictor training box" --vpc-id "${VPC_ID}" \
    --query 'GroupId' --output text)"
fi

if [[ "${SSH_INGRESS_CIDR}" == "auto" ]]; then
  MY_IP="$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')"
  SSH_INGRESS_CIDR="${MY_IP}/32"
fi
log "Authorizing SSH from ${SSH_INGRESS_CIDR} on sg ${SG_ID} ..."
aws_ ec2 authorize-security-group-ingress --group-id "${SG_ID}" \
  --protocol tcp --port 22 --cidr "${SSH_INGRESS_CIDR}" 2>/dev/null \
  || log "  (rule already present, skipping)"

# ---------------------------------------------------------------- AMI
AMI_ID="$(aws_ ssm get-parameter --name "${UBUNTU_AMI_SSM_PARAM}" \
  --query 'Parameter.Value' --output text)"
log "Using Ubuntu 24.04 AMI ${AMI_ID}"

# ---------------------------------------------------------------- Instance
EXISTING_ID="$(aws_ ec2 describe-instances \
  --filters "Name=tag:Name,Values=${INSTANCE_NAME_TAG}" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || echo None)"

if [[ "${EXISTING_ID}" != "None" && -n "${EXISTING_ID}" ]]; then
  log "Instance ${EXISTING_ID} already tagged ${INSTANCE_NAME_TAG}, reusing."
  INSTANCE_ID="${EXISTING_ID}"
else
  log "Launching ${INSTANCE_TYPE} (${ROOT_VOLUME_GB}GB ${ROOT_VOLUME_TYPE}) ..."
  BDM_JSON="$(jq -n --argjson sz "${ROOT_VOLUME_GB}" --arg tp "${ROOT_VOLUME_TYPE}" \
    '[{DeviceName:"/dev/sda1",Ebs:{VolumeSize:$sz,VolumeType:$tp,DeleteOnTermination:true}}]')"
  INSTANCE_ID="$(aws_ ec2 run-instances \
    --image-id "${AMI_ID}" \
    --instance-type "${INSTANCE_TYPE}" \
    --key-name "${KEY_NAME}" \
    --security-group-ids "${SG_ID}" \
    --iam-instance-profile "Name=${INSTANCE_PROFILE_NAME}" \
    --block-device-mappings "${BDM_JSON}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME_TAG}},{Key=Project,Value=${PROJECT_NAME}}]" \
    --query 'Instances[0].InstanceId' --output text)"
fi
echo "${INSTANCE_ID}" > "${INSTANCE_ID_FILE}"

log "Waiting for instance ${INSTANCE_ID} to reach 'running' ..."
aws_ ec2 wait instance-running --instance-ids "${INSTANCE_ID}"
PUBLIC_IP="$(instance_public_ip)"
log "Instance running at ${PUBLIC_IP}"

cat <<EOF

PROVISION COMPLETE.
  Instance id      : ${INSTANCE_ID}
  Public IP        : ${PUBLIC_IP}
  Bucket           : s3://${S3_BUCKET}
  SSH              : ssh -i ${KEY_PATH} ${REMOTE_USER}@${PUBLIC_IP}
                     (or: aws/ssh.sh)

NEXT STEPS:
  1. aws/push_data.sh         # upload ./data to S3 (slow, ~69GB)
  2. aws/push_env.sh          # copy .env to instance
  3. aws/push_code.sh         # rsync Train/, Data_ETA_Pipeline/, etc.
  4. aws/ssh.sh
  5. on instance:
       cd ~/NFP_Predictor
       bash aws/on_instance/bootstrap.sh
       bash aws/on_instance/pull_data.sh
       python Train/train_lightgbm_nfp.py --train-all
       bash aws/on_instance/push_outputs.sh
  6. aws/pull_outputs.sh      # pull _output/ to laptop
  7. aws/stop.sh              # stop instance to save \$\$
EOF
