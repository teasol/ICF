from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "v18" / "learnability_ladder_runs.csv"
EPOCH_PATTERN = re.compile(r"epoch[=](\d+)")

REPORT_METRICS = (
    "train_accuracy",
    "train_ce_loss",
    "val_accuracy",
    "val_ce_loss",
    "val_balanced_accuracy",
    "val_auroc",
    "val_majority_accuracy",
    "val_empirical_prior_ce",
    "val_positive_recall",
    "val_negative_recall",
    "val_mean_logit_std",
    "val_population_logit_std",
    "val_tail_logit_std",
    "val_oracle_abundance_auroc",
    "val_oracle_model_auroc_gap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record one completed learnability training run."
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--wandb-file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--status", default="completed")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def history_by_epoch(path: Path) -> dict[int, dict[str, Any]]:
    datastore = DataStore()
    datastore.open_for_scan(str(path.resolve()))
    rows: dict[int, dict[str, Any]] = {}
    while True:
        data = datastore.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue
        history: dict[str, Any] = {}
        for item in record.history.item:
            key = "/".join(item.nested_key) if item.nested_key else item.key
            history[key] = json.loads(item.value_json)
        if "epoch" not in history:
            continue
        epoch = int(history["epoch"])
        rows.setdefault(epoch, {}).update(history)
    return rows


def checkpoint_epoch(path: Path) -> int:
    match = EPOCH_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Checkpoint filename has no epoch: {path.name}")
    return int(match.group(1))


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    wandb_file = args.wandb_file.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not wandb_file.is_file():
        raise FileNotFoundError(wandb_file)

    selected_epoch = checkpoint_epoch(checkpoint)
    histories = history_by_epoch(wandb_file)
    if selected_epoch not in histories:
        raise RuntimeError(
            f"Offline history has no metrics for selected epoch {selected_epoch}."
        )
    selected = histories[selected_epoch]
    final_epoch = max(histories)
    record: dict[str, Any] = {
        "architecture_version": 18,
        "git_commit": git_commit(),
        "stage": args.stage.upper().replace("_", "-"),
        "seed": args.seed,
        "status": args.status,
        "selected_epoch": selected_epoch,
        "final_epoch": final_epoch,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "wandb_file": str(wandb_file.relative_to(PROJECT_ROOT)),
    }
    record.update({metric: selected.get(metric, "") for metric in REPORT_METRICS})

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if output.is_file():
        with output.open(newline="", encoding="utf-8") as stream:
            existing = list(csv.DictReader(stream))
    existing = [
        row
        for row in existing
        if not (row["stage"] == record["stage"] and int(row["seed"]) == args.seed)
    ]
    existing.append(record)
    existing.sort(key=lambda row: (row["stage"], int(row["seed"])))
    fieldnames = list(record)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
    print(json.dumps(record, indent=2, sort_keys=False))
    print(f"Updated {output}")


if __name__ == "__main__":
    main()
