#!/usr/bin/env bash
# Discover idle GPUs so a run never evicts another user's process (RU-90 kill
# condition (3)). A device counts as idle when it holds less than
# ICF_FREE_GPU_MEM_MIB (default 512) MiB and reports utilisation below
# ICF_FREE_GPU_UTIL_PCT (default 20) %.
#
# Usage:  mapfile -t GPUS < <(icf_free_gpus)      # after sourcing this file
icf_free_gpus() {
  local mem_cap="${ICF_FREE_GPU_MEM_MIB:-512}"
  local util_cap="${ICF_FREE_GPU_UTIL_PCT:-20}"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
             --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', *' -v m="$mem_cap" -v u="$util_cap" '$3 < m && $2 < u { print $1 }'
}
