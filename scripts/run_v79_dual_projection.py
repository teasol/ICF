"""Train/evaluate v79 (dual projection: learnable CV + fixed CV + fixed DD + CT).

v77 shares one learnable P between CV and DD. v78 tried to arbitrate that with a
gradient weight and the dose-response ran the wrong way -- 0.6873 / 0.6869 / 0.6826
for weight 0 / 0.02 / 1.0, the last with a CI excluding 0. DD moves P in a direction
that hurts. This arm stops sharing instead: CV keeps a learnable P, while a second
fixed-P CV block and DD both read the original deterministic basis. Four branches,
16 features, 16 -> 32 -> 1.

architecture_version 56; does not strict-load a v77 checkpoint.

Judgment is fold-paired delta + bootstrap CI against the v77 control (docs SS99):

    python scripts/compare_arms_paired.py \
      --baseline v76_classsep_hard_best --arm v79_dual_projection_best
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/aibio_3/miniconda3/envs/BagPFN/bin/python"
CONFIG = "configs/train_v79_dual_projection_1536.yaml"
RUN_ROOT = ROOT / "checkpoints/20260812_v79_dual_projection"
LOG_ROOT = ROOT / "logs/20260812_v79_dual_projection"
TAG = "v79_dual_projection_best"
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
    print(f"=== TRAIN START v79_dual_projection GPUs={GPUS}", flush=True)
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
    print("=== EVAL END v79_dual_projection", flush=True)


if __name__ == "__main__":
    main()
