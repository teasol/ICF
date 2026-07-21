"""Report the effective global data and optimizer-update budget."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import merge_train_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/train_medium.yaml"
    )
    args = parser.parse_args()
    config = merge_train_config(args.config.expanduser().resolve())
    data = config["data"]["dataset_kwargs"]
    trainer = config["trainer"]
    world_size = int(trainer["devices"])
    epochs = int(trainer["max_epochs"])
    global_episodes_per_epoch = int(data["episodes_per_epoch"])
    per_rank_episodes = math.ceil(global_episodes_per_epoch / world_size)
    optimizer_updates = per_rank_episodes * epochs
    total_episodes = global_episodes_per_epoch * epochs
    target_range = config["model_kwargs"]["training_targets_per_episode"]
    min_queries = total_episodes * int(target_range[0])
    max_queries = total_episodes * int(target_range[1])
    bag_range = data["num_bags"]
    cell_range = data["num_cells"]
    min_cells = total_episodes * int(bag_range[0]) * int(cell_range[0])
    max_cells = total_episodes * int(bag_range[1]) * int(cell_range[1])
    rank_seed_stride = 1_000_003
    seed_stream_safe = per_rank_episodes * epochs < rank_seed_stride
    print(f"world_size={world_size}")
    print(f"epochs={epochs}")
    print(f"global_episodes_per_epoch={global_episodes_per_epoch}")
    print(f"per_rank_episodes_per_epoch={per_rank_episodes}")
    print(f"total_unique_episodes={total_episodes}")
    print(f"optimizer_updates={optimizer_updates}")
    print(f"query_supervision_range=[{min_queries}, {max_queries}]")
    print(f"generated_cell_range=[{min_cells}, {max_cells}]")
    print(f"rank_seed_streams_non_overlapping={seed_stream_safe}")
    if not seed_stream_safe:
        raise RuntimeError("Per-rank sample streams can overlap at this epoch budget.")


if __name__ == "__main__":
    main()
