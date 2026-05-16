#!/usr/bin/env bash
# Start the (stopped) training instance and print connection info.

set -euo pipefail
source "$(dirname "$0")/lib.sh"
preflight

IID="$(instance_id)"
STATE="$(instance_state)"
case "${STATE}" in
  running)  log "Instance ${IID} is already running." ;;
  stopped)
    log "Starting instance ${IID} ..."
    aws_ ec2 start-instances --instance-ids "${IID}" >/dev/null
    aws_ ec2 wait instance-running --instance-ids "${IID}"
    ;;
  stopping|pending)
    log "Instance is '${STATE}', waiting for stable state ..."
    sleep 10
    exec "$0"
    ;;
  *) die "Cannot start instance in state '${STATE}'." ;;
esac

PUBLIC_IP="$(instance_public_ip)"
log "Instance ${IID} running at ${PUBLIC_IP}"
echo "  SSH: aws/ssh.sh   (or: ssh -i ${KEY_PATH} ${REMOTE_USER}@${PUBLIC_IP})"
