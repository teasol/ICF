import sys
import os
sys.path.insert(0, os.path.abspath("."))
import torch
from src.models.baseline import BaseModel

def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if not torch.cuda.is_available():
        return

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

    for e_count in [1, 4, 8, 16]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        
        x = torch.randn(num_bags, num_cells, input_dim, device=device)
        y = torch.randint(0, 2, (num_bags,), device=device)
        mask_index = num_bags - 1

        total_loss = torch.tensor(0.0, device=device)
        for _ in range(e_count):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model.forward(x=x, y=y, mask_index=mask_index, retrieval_k=retrieval_k)
                total_loss = total_loss + logits.sum()

        total_loss.backward()

        peak_allocated = torch.cuda.max_memory_allocated(0) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(0) / (1024**3)
        print(f"E={e_count:2d} Episodes | Peak Allocated: {peak_allocated:.4f} GB | Peak Reserved: {peak_reserved:.4f} GB", flush=True)

if __name__ == "__main__":
    run()
