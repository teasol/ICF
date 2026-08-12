"""Train/evaluate v78 UNWEIGHTED (dd_projection_gradient_weight 1.0) on GPUs 0-3.

The balanced arm (weight 0.02) was a clean null vs v77 -- fold-paired delta -0.0004
[-0.0021, +0.0013] (docs SS102-5). That leaves two readings: DD has nothing to add,
or 0.02 throttled it to irrelevance. This removes the throttle, so DD's contribution
to P's gradient runs at ~52x the CV ridge's and nearly orthogonal to it. Expect a
loss; the value is bracketing the weight axis, not winning.

The rank-1 direction is still not differentiated -- see `_dd_direction`.

Judgment is fold-paired delta + bootstrap CI against the v77 control (docs SS99):

    python scripts/compare_arms_paired.py \
      --baseline v76_classsep_hard_best --arm v78_dd_projection_unweighted_best
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/aibio_3/miniconda3/envs/BagPFN/bin/python"
CONFIG = "configs/train_v78_dd_projection_unweighted_1536.yaml"
RUN_ROOT = ROOT / "checkpoints/20260812_v78_dd_projection_unweighted"
LOG_ROOT = ROOT / "logs/20260812_v78_dd_projection_unweighted"
TAG = "v78_dd_projection_unweighted_best"
GPUS = (0, 1, 2, 3)
TASKS = (
    "bc_therapy/er_status", "bc_therapy/grade", "bc_therapy/her2_status",
    "cptac_brca/PIK3CA_mutation", "cptac_brca/TP53_mutation",
    "cptac_ccrcc/BAP1_mutation", "cptac_ccrcc/VHL_mutation",
    "cptac_luad/EGFR_mutation", "cptac_luad/STK11_mutation",
    "cptac_luad/TP53_mutation",
)


def validation_best(directory: Path) -> Path:
    candidates: list[tuple[float, Path]] = []
    for path in directory.glob("epoch=*-val_ce_loss=*.ckpt"):
        match = re.search(r"val_ce_loss=([0-9.]+)\.ckpt$", path.name)
        if match:
            candidates.append((float(match.group(1)), path))
    if not candidates:
        raise RuntimeError(f"No validation checkpoint found in {directory}")
    return min(candidates, key=lambda item: item[0])[1]


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPUS))
    print(f"=== TRAIN START v78_dd_projection_unweighted GPUs={GPUS}", flush=True)
    with (LOG_ROOT / "train.out").open("w") as output:
        subprocess.run(
            [PYTHON, "scripts/train.py", "--config", CONFIG,
             "--checkpoint-dir", str(RUN_ROOT)], cwd=ROOT, env=env,
            stdout=output, stderr=subprocess.STDOUT, check=True,
        )
    checkpoint = validation_best(RUN_ROOT)
    print(f"=== TRAIN END best={checkpoint.name}", flush=True)
    groups = [TASKS[index::4] for index in range(4)]
    workers = [
        subprocess.Popen(
            ["bash", "scripts/eval_seal_tasks.sh", str(gpu), str(checkpoint),
             CONFIG, TAG, *tasks], cwd=ROOT,
        )
        for gpu, tasks in zip(GPUS, groups)
    ]
    codes = [worker.wait() for worker in workers]
    if any(codes):
        raise RuntimeError(f"SEAL evaluation failed: {codes}")
    print("=== EVAL END v78_dd_projection_unweighted", flush=True)


if __name__ == "__main__":
    main()
