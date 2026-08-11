"""Sequentially train and SEAL-evaluate the v76 difficulty-axis ablations."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/aibio_3/miniconda3/envs/BagPFN/bin/python")
RUN_ROOT = ROOT / "checkpoints" / "20260812_v76_axis_sweep"
LOG_ROOT = ROOT / "logs" / "20260812_v76_axis_sweep"

ARMS = (
    ("classsep", "configs/train_v76_axis_classsep_medium_1536.yaml"),
    ("response", "configs/train_v76_axis_response_medium_1536.yaml"),
    ("rare", "configs/train_v76_axis_rare_medium_1536.yaml"),
    ("noise", "configs/train_v76_axis_noise_medium_1536.yaml"),
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
        train_log = LOG_ROOT / f"{arm}_train.out"
        print(f"=== TRAIN START {arm}", flush=True)
        with train_log.open("w") as output:
            subprocess.run(
                [
                    str(PYTHON),
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
        print(f"=== TRAIN END {arm} best={checkpoint.name}", flush=True)

        tag = f"v76_axis_{arm}_best"
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
