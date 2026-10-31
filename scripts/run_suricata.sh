#!/usr/bin/env bash
# Runs Suricata against a pcap file, forcing logs into botnet-detection/logs
# regardless of the current working directory.
#
# default-log-dir in suricata.yaml is NOT reliably honored by Suricata when
# the path is relative, so we always pass an absolute path via -l.
#
# Usage: run_suricata.sh [PCAP_FILE] [extra suricata args...]
#   PCAP_FILE defaults to the bundled sample pcap if omitted.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${REPO_ROOT}/botnet-detection/config/suricata.yaml"
DEFAULT_PCAP="${REPO_ROOT}/botnet-detection/datasets/opt/Malware-Project/BigDataset/IoTScenarios/CTU-IoT-Malware-Capture-1-1/2018-05-09-192.168.100.103.pcap"
LOG_DIR="${REPO_ROOT}/botnet-detection/logs"

PCAP="${1:-${DEFAULT_PCAP}}"
if [ "$#" -gt 0 ]; then
  shift
fi

mkdir -p "${LOG_DIR}"

exec suricata -c "${CONFIG}" -r "${PCAP}" -l "${LOG_DIR}" "$@"
