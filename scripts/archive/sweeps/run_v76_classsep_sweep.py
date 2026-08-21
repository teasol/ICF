"""Train/evaluate new class-separation levels sequentially on four GPUs."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/aibio_3/miniconda3/envs/BagPFN/bin/python"
RUN_ROOT = ROOT / "checkpoints/20260812_v76_classsep_sweep"
LOG_ROOT = ROOT / "logs/20260812_v76_classsep_sweep"
ARMS = (
    ("mild", "configs/archive/v69_v76_relation/train_v76_classsep_mild_1536.yaml"),
    ("hard", "configs/archive/v69_v76_relation/train_v76_classsep_hard_1536.yaml"),
    ("veryhard", "configs/archive/v69_v76_relation/train_v76_classsep_veryhard_1536.yaml"),
)
TASK_GROUPS = (
    ("0", "bc_therapy/er_status", "bc_therapy/grade", "bc_therapy/her2_status"),
    (
        "1",
        "cptac_brca/PIK3CA_mutation",
        "cptac_brca/TP53_mutation",
        "cptac_ccrcc/BAP1_mutation",
    ),
    ("2", "cptac_ccrcc/VHL_mutation", "cptac_luad/EGFR_mutation"),
    ("3", "cptac_luad/STK11_mutation", "cptac_luad/TP53_mutation"),
)


def validation_best(checkpoint_dir: Path) -> Path:
    candidates: list[tuple[float, Path]] = []
    for path in checkpoint_dir.glob("epoch=*-val_ce_loss=*.ckpt"):
        match = re.search(r"val_ce_loss=([0-9.]+)\.ckpt$", path.name)
        if match:
            candidates.append((float(match.group(1)), path))
    if not candidates:
        raise RuntimeError(f"No validation checkpoint found in {checkpoint_dir}")
    return min(candidates, key=lambda item: item[0])[1]


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    for arm, config in ARMS:
        checkpoint_dir = RUN_ROOT / arm
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== TRAIN START {arm}", flush=True)
        with (LOG_ROOT / f"{arm}_train.out").open("w") as output:
            subprocess.run(
                [
                    PYTHON,
                    "scripts/train.py",
                    "--config",
                    config,
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                ],
                cwd=ROOT,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )

        checkpoint = validation_best(checkpoint_dir)
        tag = f"v76_classsep_{arm}_best"
        print(f"=== TRAIN END {arm} best={checkpoint.name}", flush=True)
        print(f"=== EVAL START {arm} tag={tag}", flush=True)
        workers = [
            subprocess.Popen(
                [
                    "bash",
                    "scripts/eval_seal_tasks.sh",
                    gpu,
                    str(checkpoint),
                    config,
                    tag,
                    *tasks,
                ],
                cwd=ROOT,
            )
            for gpu, *tasks in TASK_GROUPS
        ]
        return_codes = [worker.wait() for worker in workers]
        if any(return_codes):
            raise RuntimeError(f"SEAL evaluation failed for {arm}: {return_codes}")
        print(f"=== EVAL END {arm}", flush=True)


if __name__ == "__main__":
    main()
