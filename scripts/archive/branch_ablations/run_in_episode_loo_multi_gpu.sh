#!/usr/bin/env bash
set -euo pipefail

. scripts/node_env.sh

LOG_DIR="logs/in_episode_loo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "========================================================================"
echo "Launching 4-GPU Parallel 50-Fold In-Episode Context LOO Evaluation"
echo "Log dir: $LOG_DIR"
echo "========================================================================"

# Task assignments across 4 GPUs
# GPU 0: task 0 (ARID1A), task 1 (Grade)
# GPU 1: task 2 (KEAP1),  task 3 (KRAS)
# GPU 2: task 4 (SMAD4),  task 5 (Progression)
# GPU 3: task 6 (PBRM1)

$PYTHON scripts/eval_in_episode_loo.py --device cuda:0 --task-idx 0 > "$LOG_DIR/task_0_ARID1A.log" 2>&1 &
PID0=$!

$PYTHON scripts/eval_in_episode_loo.py --device cuda:0 --task-idx 1 > "$LOG_DIR/task_1_Grade.log" 2>&1 &
PID1=$!

$PYTHON scripts/eval_in_episode_loo.py --device cuda:1 --task-idx 2 > "$LOG_DIR/task_2_KEAP1.log" 2>&1 &
PID2=$!

$PYTHON scripts/eval_in_episode_loo.py --device cuda:1 --task-idx 3 > "$LOG_DIR/task_3_KRAS.log" 2>&1 &
PID3=$!

$PYTHON scripts/eval_in_episode_loo.py --device cuda:2 --task-idx 4 > "$LOG_DIR/task_4_SMAD4.log" 2>&1 &
PID4=$!

$PYTHON scripts/eval_in_episode_loo.py --device cuda:2 --task-idx 5 > "$LOG_DIR/task_5_Progression.log" 2>&1 &
PID5=$!

$PYTHON scripts/eval_in_episode_loo.py --device cuda:3 --task-idx 6 > "$LOG_DIR/task_6_PBRM1.log" 2>&1 &
PID6=$!

echo "Processes launched with PIDs: $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6"
wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6
echo "All 7 tasks finished successfully!"

$PYTHON -c "
import re
from pathlib import Path

log_dir = Path('$LOG_DIR')
results = {}
for p in sorted(log_dir.glob('task_*.log')):
    txt = p.read_text()
    m = re.search(r'\[([A-Za-z0-9_]+)\s*\] 50-Fold In-Episode LOO AUROC:\s*([0-9\.]+)', txt)
    if m:
        results[m.group(1).strip()] = float(m.group(2))

baseline = {
    'ARID1A': 0.5471, 'Grade': 0.6823, 'KEAP1': 0.6129,
    'KRAS': 0.7295, 'SMAD4': 0.4465, 'Progression': 0.7986, 'PBRM1': 0.5685,
    'Macro': 0.6265,
}

print('\n' + '=' * 85)
print(f'{\"Task\":<15} | {\"v120 Baseline\":<15} | {\"In-Episode LOO\":<15} | {\"Diff\":<10}')
print('-' * 85)
scores = []
for name in ['ARID1A', 'Grade', 'KEAP1', 'KRAS', 'SMAD4', 'Progression', 'PBRM1']:
    val = results.get(name, 0.0)
    scores.append(val)
    diff = val - baseline[name]
    print(f'{name:<15} | {baseline[name]:<15.4f} | {val:<15.4f} | {diff:+10.4f}')

macro = sum(scores) / len(scores) if scores else 0.0
diff_m = macro - baseline['Macro']
print('=' * 85)
print(f'{\"Macro AUROC\":<15} | {baseline[\"Macro\"]:<15.4f} | {macro:<15.4f} | {diff_m:+10.4f}')
print('=' * 85)
"
