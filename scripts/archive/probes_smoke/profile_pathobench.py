"""Profile the PathoBench all-context inference path — where does the time go?

Times each phase of scripts/test_pathobench.py's zero-shot evaluation:
(1) preprocessed cache load, (2) model build + checkpoint load, and (3) the
per-episode loop split into context-bag prep vs model forward vs post. Uses
torch.cuda.synchronize() around the forward so GPU execution time is included
(wall-clock, not just Python launch overhead). Reports per-episode forward
distribution and a total-time breakdown.

Usage:
    python scripts/profile_pathobench.py \
        --checkpoint checkpoints/20260804_132334/v30_cardinality_poolz_l2/epoch=048-val_ce_loss=0.4442.ckpt \
        --csv /NHNHOME/BASE/kimds/Data/PathoBench/csv/cptac_lscc_keap1.csv \
        [--episodes 20]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import lightning as L
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import build_model, merge_train_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/train_v33_phase0_armC_ddp8_batch2.yaml",
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "pathobench",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Cap the number of query episodes profiled (default: all test slides).",
    )
    parser.add_argument(
        "--profiler",
        action="store_true",
        help="Run torch.profiler over --profiler-episodes episodes and print the "
        "op-level bottleneck table instead of the wall-clock phase summary.",
    )
    parser.add_argument("--profiler-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    # ---- Phase 1: cache load + move to GPU ----
    table = pd.read_csv(args.csv)
    table = table[table["split"].isin(("train", "test"))]
    table["slide_id"] = table["slide_id"].astype(str)
    train_table = table[table["split"] == "train"]
    test_table = table[table["split"] == "test"]

    t0 = time.perf_counter()
    data_dir = args.data_dir.expanduser().resolve()
    train_state = torch.load(
        data_dir / f"{args.csv.stem}_train.pt", map_location="cpu", weights_only=False
    )
    test_state = torch.load(
        data_dir / f"{args.csv.stem}_test.pt", map_location="cpu", weights_only=False
    )
    t_cache = time.perf_counter() - t0

    t0 = time.perf_counter()
    projected: dict[str, torch.Tensor] = {}
    for sid, bag in zip(train_state["slide_id"], train_state["bag"]):
        projected[sid] = bag.to(device)
    for sid, bag in zip(test_state["slide_id"], test_state["bag"]):
        projected[sid] = bag.to(device)
    t_move = time.perf_counter() - t0

    train_ids = list(train_state["slide_id"])
    test_ids = list(test_state["slide_id"])
    train_y = {
        sid: int(train_table.loc[train_table["slide_id"] == sid, "label"].iloc[0])
        for sid in train_ids
    }
    test_y = {
        sid: int(test_table.loc[test_table["slide_id"] == sid, "label"].iloc[0])
        for sid in test_ids
    }

    n_train_tiles = sum(b.shape[0] for b in train_state["bag"])
    n_test_tiles = sum(b.shape[0] for b in test_state["bag"])
    print(
        f"task={args.csv.stem} train={len(train_ids)} slides/{n_train_tiles:,} tiles "
        f"test={len(test_ids)} slides/{n_test_tiles:,} tiles"
    )

    # ---- Phase 2: model build + checkpoint load ----
    t0 = time.perf_counter()
    config = merge_train_config(args.config.expanduser().resolve())
    config["seed"] = args.seed
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu")
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)
    t_model = time.perf_counter() - t0
    print(f"model: arch v{model.model.architecture_version} ckpt={args.checkpoint.name}")

    # ---- Phase 3: per-episode loop with phase timing ----
    if args.episodes is not None:
        test_ids = test_ids[: args.episodes]
    n_ep = len(test_ids)
    n_ctx = len(train_ids)
    ctx_cells = sum(projected[s].shape[0] for s in train_ids)
    print(f"all-context: {n_ctx} context slides, {ctx_cells:,} context tiles per episode")

    if args.profiler:
        from torch.profiler import ProfilerActivity, profile

        prof_ids = test_ids[: args.profiler_episodes]
        with torch.no_grad():
            with profile(
                activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                record_shapes=False,
            ) as prof:
                for qid in prof_ids:
                    context_bags = [projected[s] for s in train_ids]
                    episode_bags = [*context_bags, projected[qid]]
                    episode_y = torch.tensor(
                        [train_y[s] for s in train_ids] + [test_y[qid]],
                        dtype=torch.long,
                        device=device,
                    )
                    mask_index = torch.tensor([n_ctx], device=device)
                    model.model.forward(episode_bags, episode_y, mask_index)
                    torch.cuda.synchronize()
        print("\n=== torch.profiler — by self CUDA time (top 25) ===")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
        print("\n=== by self CPU time (top 12) ===")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))
        return

    ctx_prep_s: list[float] = []
    fwd_s: list[float] = []
    post_s: list[float] = []
    total_s: list[float] = []
    warmup = 0.0

    with torch.no_grad():
        for q_idx, qid in enumerate(test_ids):
            t_ep0 = time.perf_counter()

            t0 = time.perf_counter()
            context_bags = [projected[s] for s in train_ids]
            t_ctx = time.perf_counter() - t0

            query_bag = projected[qid]
            episode_bags = [*context_bags, query_bag]
            episode_y = torch.tensor(
                [train_y[s] for s in train_ids] + [test_y[qid]],
                dtype=torch.long,
                device=device,
            )
            mask_index = torch.tensor([n_ctx], device=device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model.model.forward(episode_bags, episode_y, mask_index)
            torch.cuda.synchronize()
            t_fwd = time.perf_counter() - t0

            t0 = time.perf_counter()
            float(torch.softmax(logits.float(), dim=-1)[0, 1].item())
            t_post = time.perf_counter() - t0

            t_ep = time.perf_counter() - t_ep0
            if q_idx == 0:
                warmup = t_fwd
                continue  # skip the warmup episode (kernel/allocator)
            ctx_prep_s.append(t_ctx)
            fwd_s.append(t_fwd)
            post_s.append(t_post)
            total_s.append(t_ep)
            if (q_idx + 1) % 10 == 0:
                print(f"  ... {q_idx + 1}/{n_ep} episodes", flush=True)

    def stats(x: list[float]) -> str:
        xs = sorted(x)
        p90 = xs[int(0.9 * len(xs)) - 1]
        return (
            f"mean={statistics.mean(x) * 1000:.1f}ms "
            f"p90={p90 * 1000:.1f}ms max={max(x) * 1000:.1f}ms"
        )

    total = sum(total_s)
    n_measured = len(total_s)
    print(f"\n=== profile: {n_measured} episodes (warmup skipped), all-context ===")
    print(f"  cache load       : {t_cache:.2f}s")
    print(f"  move to GPU      : {t_move:.2f}s")
    print(f"  model build+ckpt : {t_model:.2f}s")
    print(f"  warmup forward   : {warmup * 1000:.1f}ms")
    print(
        f"  context-prep     : {stats(ctx_prep_s)}  "
        f"({100 * sum(ctx_prep_s) / total:.1f}% of loop)"
    )
    print(
        f"  forward          : {stats(fwd_s)}  "
        f"({100 * sum(fwd_s) / total:.1f}% of loop)"
    )
    print(
        f"  post             : {stats(post_s)}  "
        f"({100 * sum(post_s) / total:.1f}% of loop)"
    )
    print(f"  episode total    : {stats(total_s)}")
    print(f"  loop total       : {total:.2f}s over {n_measured} episodes -> {total / n_measured:.3f} s/ep")
    slow = max(range(len(fwd_s)), key=lambda i: fwd_s[i])
    print(
        f"  slowest episode  : #{slow} {test_ids[slow]} "
        f"forward {fwd_s[slow] * 1000:.1f}ms ({ctx_cells + projected[test_ids[slow]].shape[0]:,} cells)"
    )


if __name__ == "__main__":
    main()
