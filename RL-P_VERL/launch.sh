#!/usr/bin/env bash

set -e -o pipefail

# Verify envvars
set +x
if [ -z "${GPUS_PER_NODE}" ]; then
  echo "[FATAL] Launcher script could not find 'GPUS_PER_NODE'!"
  exit 1
fi

if [ -z "${FIRST_NODE_HOSTNAME}" ]; then
  echo "[FATAL] Launcher script could not find 'FIRST_NODE_HOSTNAME'!"
  exit 1
fi

if [ -z "${FIRST_NODE_IP}" ]; then
  echo "[FATAL] Launcher script could not find 'FIRST_NODE_IP'!"
  exit 1
fi

if [ -z "${COMMS_PORT}" ]; then
  echo "[FATAL] Launcher script could not find 'COMMS_PORT'!"
  exit 1
fi

if [ -z "${SUBMIT_PORT}" ]; then
  echo "[FATAL] Launcher script could not find 'SUBMIT_PORT'!"
  exit 1
fi

# Print information about GPUs
echo "I am running on $(hostname) and here is my GPU information:"
set -x +e
nvidia-smi -L
set +x -e

# Determine which to launch (match on the first node hostname)
PROCEDURE="UNKNOWN"
if [ "$(hostname)" == "${FIRST_NODE_HOSTNAME}" ]; then
  PROCEDURE="HEAD"
elif [ "$(hostname)" != "${FIRST_NODE_HOSTNAME}" ]; then
  PROCEDURE="WORKER"
fi

if [ "${PROCEDURE}" == "HEAD" ]; then
  echo "[INFO] Launcher on $(hostname) will launch a HEAD instance."
  ray start --head --num-gpus="${GPUS_PER_NODE}" --node-ip-address="${FIRST_NODE_IP}" --port="${COMMS_PORT}" --dashboard-host=0.0.0.0 --dashboard-port="${SUBMIT_PORT}" --block
elif [ "${PROCEDURE}" == "WORKER" ]; then
  echo "[INFO] Launcher on $(hostname) will launch a WORKER instance, and is currently waiting for 60 seconds..."
  sleep 60
  echo "[INFO] Launcher on $(hostname) is launching a WORKER instance now."
  ray start --num-gpus="${GPUS_PER_NODE}" --address="${FIRST_NODE_IP}:${COMMS_PORT}" --block
fi
