"""Train and evaluate the Hard finite-MLP-bank sweep on GPUs 0-3."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/aibio_3/miniconda3/envs/BagPFN/bin/python"
RUN_ROOT = ROOT / "checkpoints/20260812_v76_hard_mlpbank_sweep"
LOG_ROOT = ROOT / "logs/20260812_v76_hard_mlpbank_sweep"
GPUS = (0, 1, 2, 3)
BANK_SIZES = (128, 512, 1024, 2048, 4096)
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


def run_arm(bank_size: int) -> None:
    arm = f"mlpbank{bank_size}"
    config = f"configs/train_v76_hard_{arm}_1536.yaml"
    checkpoint_dir = RUN_ROOT / arm
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPUS))
    print(f"=== TRAIN START {arm} GPUs={GPUS}", flush=True)
    with (LOG_ROOT / f"{arm}_train.out").open("w") as output:
        subprocess.run(
            [PYTHON, "scripts/train.py", "--config", config,
             "--checkpoint-dir", str(checkpoint_dir)],
            cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, check=True,
        )
    checkpoint = validation_best(checkpoint_dir)
    print(f"=== TRAIN END {arm} best={checkpoint.name}", flush=True)
    tag = f"v76_hard_{arm}_best"
    groups = [TASKS[index::4] for index in range(4)]
    workers = [
        subprocess.Popen(
            ["bash", "scripts/eval_seal_tasks.sh", str(gpu), str(checkpoint),
             config, tag, *tasks], cwd=ROOT,
        )
        for gpu, tasks in zip(GPUS, groups)
    ]
    codes = [worker.wait() for worker in workers]
    if any(codes):
        raise RuntimeError(f"SEAL evaluation failed for {arm}: {codes}")
    print(f"=== EVAL END {arm}", flush=True)


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for bank_size in BANK_SIZES:
        run_arm(bank_size)


if __name__ == "__main__":
    main()
