"""VRAM benchmark script for Large Context (N=96, K=24) 2-Pass Signal-Aware Pre-training."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import torch
from src.models.baseline import BaseModel

def benchmark_vram():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    device = torch.device("cuda:0")
    print(f"=== GPU Memory Benchmark (NVIDIA B200) ===")
    print(f"Device Name : {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM  : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n")

    model = BaseModel(
        input_dim=512,
        meta_hidden_dim=256,
        num_classes=2,
    ).to(device=device)

    model.train()

    num_bags = 96
    num_cells = 1000
    input_dim = 512
    retrieval_k = 24

    print(f"Configuration: Candidate Pool N={num_bags} bags | Cell Instances=1000 | Feature Dim=512 | Context Retrieval K={retrieval_k}", flush=True)
    print("-" * 80, flush=True)

    # Extreme Episode batch scaling test: E = 256, 512, 1024
    for outer_episodes in [256, 512, 1024]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

        try:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            total_loss = torch.tensor(0.0, device=device)
            start_time.record()

            for e in range(outer_episodes):
                x = torch.randn(num_bags, num_cells, input_dim, device=device, dtype=torch.float32)
                y = torch.randint(0, 2, (num_bags,), device=device)
                mask_index = num_bags - 1  # query bag index is the last bag

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model.forward(
                        x=x,
                        y=y,
                        mask_index=mask_index,
                        retrieval_k=retrieval_k,
                    )
                    total_loss = total_loss + logits.sum()

            total_loss.backward()
            end_time.record()
            torch.cuda.synchronize()

            elapsed_ms = start_time.elapsed_time(end_time)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            peak_allocated = torch.cuda.max_memory_allocated(0) / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved(0) / (1024**3)
            episodes_per_sec = (outer_episodes / (elapsed_ms / 1000.0))

            print(f"Outer Episodes E={outer_episodes:3d} (Total Candidate Bags: {outer_episodes * num_bags:5d})", flush=True)
            print(f"  ├─ Step Time               : {elapsed_ms:7.2f} ms ({episodes_per_sec:6.2f} ep/s)", flush=True)
            print(f"  ├─ Current Allocated VRAM  : {allocated:6.4f} GB", flush=True)
            print(f"  ├─ Peak Allocated VRAM     : {peak_allocated:6.4f} GB", flush=True)
            print(f"  └─ Peak Reserved VRAM      : {peak_reserved:6.4f} GB", flush=True)
            print("-" * 80, flush=True)

        except Exception as err:
            print(f"Outer Episodes E={outer_episodes} OOM/Error: {err}", flush=True)

if __name__ == "__main__":
    benchmark_vram()
