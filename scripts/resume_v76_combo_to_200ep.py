"""Resume the combo arm through epoch 199, then evaluate its global val best."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/aibio_3/miniconda3/envs/BagPFN/bin/python"
CONFIG = "configs/train_v76_combo_classsep_rare_noise_8gpu_200ep_1536.yaml"
CHECKPOINT_DIR = ROOT / "checkpoints/20260812_v76_combo/classsep_rare_noise_8gpu"
RESUME_CHECKPOINT = CHECKPOINT_DIR / "last.ckpt"
LOG_DIR = ROOT / "logs/20260812_v76_combo"
TAG = "v76_combo_classsep_rare_noise_8gpu_200ep_best"
TASK_GROUPS = (
    ("0", "bc_therapy/er_status", "cptac_luad/TP53_mutation"),
    ("1", "bc_therapy/grade", "cptac_luad/STK11_mutation"),
    ("2", "bc_therapy/her2_status"),
    ("3", "cptac_brca/PIK3CA_mutation"),
    ("4", "cptac_brca/TP53_mutation"),
    ("5", "cptac_ccrcc/BAP1_mutation"),
    ("6", "cptac_ccrcc/VHL_mutation"),
    ("7", "cptac_luad/EGFR_mutation"),
)


def validation_best() -> Path:
    candidates: list[tuple[float, Path]] = []
    for path in CHECKPOINT_DIR.glob("epoch=*-val_ce_loss=*.ckpt"):
        match = re.search(r"val_ce_loss=([0-9.]+)\.ckpt$", path.name)
        if match:
            candidates.append((float(match.group(1)), path))
    if not candidates:
        raise RuntimeError(f"No validation checkpoint found in {CHECKPOINT_DIR}")
    return min(candidates, key=lambda item: item[0])[1]


def main() -> None:
    if not RESUME_CHECKPOINT.is_file():
        raise FileNotFoundError(RESUME_CHECKPOINT)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    print(f"=== RESUME START checkpoint={RESUME_CHECKPOINT}", flush=True)
    with (LOG_DIR / "resume_to_200ep.out").open("w") as output:
        subprocess.run(
            [
                PYTHON,
                "scripts/train.py",
                "--config",
                CONFIG,
                "--checkpoint-dir",
                str(CHECKPOINT_DIR),
                "--ckpt-path",
                str(RESUME_CHECKPOINT),
            ],
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )

    checkpoint = validation_best()
    print(f"=== TRAIN END best={checkpoint.name}", flush=True)
    print(f"=== EVAL START tag={TAG}", flush=True)
    workers = [
        subprocess.Popen(
            [
                "bash",
                "scripts/eval_seal_tasks.sh",
                gpu,
                str(checkpoint),
                CONFIG,
                TAG,
                *tasks,
            ],
            cwd=ROOT,
        )
        for gpu, *tasks in TASK_GROUPS
    ]
    return_codes = [worker.wait() for worker in workers]
    if any(return_codes):
        raise RuntimeError(f"SEAL evaluation failed: {return_codes}")
    print("=== EVAL END 200 epochs", flush=True)


if __name__ == "__main__":
    main()
