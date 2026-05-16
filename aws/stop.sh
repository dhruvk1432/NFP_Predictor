#!/usr/bin/env bash
# Stop the instance to save compute $$. EBS keeps costing (~$0.08/GB/mo for gp3).

set -euo pipefail
source "$(dirname "$0")/lib.sh"
preflight

IID="$(instance_id)"
STATE="$(instance_state)"
case "${STATE}" in
  stopped) log "Instance ${IID} is already stopped." ;;
  running|pending)
    log "Stopping instance ${IID} ..."
    aws_ ec2 stop-instances --instance-ids "${IID}" >/dev/null
    aws_ ec2 wait instance-stopped --instance-ids "${IID}"
    log "Stopped."
    ;;
  *) log "Instance is '${STATE}'; nothing to do." ;;
esac
