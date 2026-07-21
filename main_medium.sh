#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PROJECT_ROOT}/scripts/launch_interactive_training.sh" \
    medium "${CONFIG:-configs/train_medium.yaml}"
