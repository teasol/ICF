import torch
import numpy as np
from src.utils.metrics import auroc

def compute_in_context_fisher_basis(
    context_bags: list[torch.Tensor],
    context_labels: torch.Tensor,
    sketch_dim: int = 256,
    pca_dim: int = 32,
    shrinkage: float = 0.1,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute In-Context Fisher Discriminant direction and Fisher-augmented basis.
    
    Returns:
        fisher_basis: (1536, sketch_dim) orthogonal basis where col 0 is Fisher direction.
        w_fisher: (1536,) raw Fisher discriminant vector.
    """
    device = torch.device(device)
    labels = context_labels.long().to(device)
    n_ctx = len(context_bags)
    
    # 1. Compute per-bag means and within-bag scatter
    bag_means = []
    total_cells = 0
    dim = context_bags[0].shape[-1]
    S_within = torch.zeros((dim, dim), device=device, dtype=torch.float32)
    
    for bag in context_bags:
        b = bag.float().to(device)
        n = b.shape[0]
        total_cells += n
        m = b.mean(dim=0)
        bag_means.append(m)
        dev = b - m
        S_within += dev.T @ dev
        
    S_within = S_within / max(1, total_cells)
    bag_means = torch.stack(bag_means) # (N_ctx, dim)
    
    # 2. Class centroids
    mu0 = bag_means[labels == 0].mean(dim=0)
    mu1 = bag_means[labels == 1].mean(dim=0)
    delta = mu1 - mu0 # (dim,)
    
    # 3. Slide-level within-class scatter
    dev0 = bag_means[labels == 0] - mu0
    dev1 = bag_means[labels == 1] - mu1
    S_slide = (dev0.T @ dev0 + dev1.T @ dev1) / max(1, n_ctx)
    
    # 4. Total within-class pooled covariance with shrinkage
    Sigma_W = S_within + shrinkage * S_slide
    trace_scale = Sigma_W.diagonal().mean().clamp_min(1e-6)
    identity = torch.eye(dim, device=device, dtype=torch.float32)
    Sigma_W_reg = Sigma_W + (1e-4 * trace_scale) * identity
    
    # 5. Fisher linear discriminant direction: w = Sigma_W^{-1} (mu1 - mu0)
    try:
        w_fisher = torch.linalg.solve(Sigma_W_reg, delta)
    except RuntimeError:
        w_fisher = torch.linalg.lstsq(Sigma_W_reg, delta).solution
        
    v1 = w_fisher / w_fisher.norm().clamp_min(1e-12)
    
    # 6. Top eigenvectors of S_within
    eigvals, eigvecs = torch.linalg.eigh(S_within)
    # top eigenvectors in descending order
    pca_vecs = eigvecs[:, -sketch_dim:].flip(-1) # (dim, sketch_dim)
    
    # 7. Gram-Schmidt orthogonalization: v1 as 1st vector, then project pca_vecs
    basis_cols = [v1]
    for k in range(sketch_dim - 1):
        vk = pca_vecs[:, k]
        # Remove projection onto existing basis columns
        for b in basis_cols:
            vk = vk - (vk @ b) * b
        norm = vk.norm()
        if norm > 1e-6:
            basis_cols.append(vk / norm)
        else:
            # Random orthogonal vector if degenerate
            rand_v = torch.randn(dim, device=device)
            for b in basis_cols:
                rand_v = rand_v - (rand_v @ b) * b
            basis_cols.append(rand_v / rand_v.norm())
            
    fisher_basis = torch.stack(basis_cols, dim=-1) # (dim, sketch_dim)
    return fisher_basis, w_fisher

def test_fisher_antisymmetry():
    torch.manual_seed(42)
    n_ctx = 40
    dim = 64
    ctx_bags = [torch.randn(torch.randint(50, 150, (1,)).item(), dim) for _ in range(n_ctx)]
    ctx_labels = torch.randint(0, 2, (n_ctx,), dtype=torch.long)
    ctx_labels[0], ctx_labels[1] = 0, 1
    
    b_orig, w_orig = compute_in_context_fisher_basis(ctx_bags, ctx_labels, sketch_dim=32, pca_dim=16)
    b_flip, w_flip = compute_in_context_fisher_basis(ctx_bags, 1 - ctx_labels, sketch_dim=32, pca_dim=16)
    
    # w_fisher must flip sign exactly: w_orig = -w_flip
    diff_w = (w_orig + w_flip).abs().max().item()
    print(f"w_fisher Max Antisymmetry Error: {diff_w:.8e}")
    assert diff_w < 1e-5, f"w_fisher antisymmetry error: {diff_w}"
    
    # v1 must flip sign exactly: b_orig[:, 0] = -b_flip[:, 0]
    diff_v1 = (b_orig[:, 0] + b_flip[:, 0]).abs().max().item()
    print(f"v1 (col 0) Max Antisymmetry Error: {diff_v1:.8e}")
    assert diff_v1 < 1e-5, f"v1 antisymmetry error: {diff_v1}"
    print("In-Context Fisher Basis Unit Test Passed Successfully!")

if __name__ == "__main__":
    test_fisher_antisymmetry()
