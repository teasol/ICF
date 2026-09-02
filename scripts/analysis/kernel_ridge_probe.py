import torch
import torch.nn.functional as F

def _kernel_matrix(left: torch.Tensor, right: torch.Tensor, kernel: str = "rbf", gamma: float | None = None, degree: int = 2, coef0: float = 1.0) -> torch.Tensor:
    if kernel == "linear":
        return left @ right.T
    left_f = left.float()
    right_f = right.float()
    squared = left_f @ right_f.T
    dims = left.shape[-1]
    if gamma is None:
        gamma = 1.0 / dims
    if kernel == "rbf":
        sq_left = (left_f * left_f).sum(dim=1, keepdim=True)
        sq_right = (right_f * right_f).sum(dim=1, keepdim=True)
        distances = (sq_left - 2.0 * squared + sq_right.T).clamp_min(0.0)
        return torch.exp(-gamma * distances)
    elif kernel == "poly":
        return (gamma * squared + coef0).pow(degree)
    elif kernel == "cosine":
        l_norm = torch.nn.functional.normalize(left_f, dim=-1)
        r_norm = torch.nn.functional.normalize(right_f, dim=-1)
        return l_norm @ r_norm.T
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")


def _solve_kernel_ridge(
    ctx_feats: torch.Tensor,
    ctx_labels: torch.Tensor,
    qry_feats: torch.Tensor,
    kernel: str = "rbf",
    gamma: float | None = None,
    degree: int = 2,
    coef0: float = 1.0,
    reg_lambda: float = 1.0,
) -> torch.Tensor:
    """Class-balanced centered Kernel Ridge Regression (n x n dual solve)."""
    labels = ctx_labels.long()
    device = ctx_feats.device

    # 1. Per-feature standardisation using context statistics
    centre = ctx_feats.mean(dim=0, keepdim=True)
    scale = (ctx_feats - centre).square().mean(dim=0).sqrt().clamp_min(1e-6)
    ctx_std = (ctx_feats - centre) / scale
    qry_std = (qry_feats - centre) / scale

    # 2. Kernel Gram matrices
    k_ctx = _kernel_matrix(ctx_std, ctx_std, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
    k_qry = _kernel_matrix(qry_std, ctx_std, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)

    # 3. Class-balanced centered targets and centered Gram matrix
    targets = torch.nn.functional.one_hot(labels, 2).float()
    counts = torch.bincount(labels, minlength=2)
    if bool((counts == 0).any()):
        raise ValueError("Every class must occur in the context set.")
    weight = counts.float().reciprocal()[labels]
    total = weight.sum().clamp_min(1e-12)
    root = weight.sqrt()

    m_ctx = (weight[None, :] @ k_ctx).squeeze(0) / total
    mu2 = (weight[None, :] @ k_ctx @ weight[:, None]).squeeze() / (total * total)
    target_mean = (weight[:, None] * targets).sum(0, keepdim=True) / total
    centred_targets = (targets - target_mean) * root[:, None]

    gram = root[:, None] * (k_ctx - m_ctx[:, None] - m_ctx[None, :] + mu2) * root[None, :]
    dimension = gram.shape[0]
    identity = torch.eye(dimension, device=device, dtype=torch.float32)

    # 4. Robust solve
    jitter = 0.0
    dual = None
    for _ in range(6):
        try:
            factor = torch.linalg.cholesky(gram + (reg_lambda + jitter) * identity)
            dual = torch.cholesky_solve(centred_targets, factor)
            break
        except RuntimeError:
            jitter = max(jitter * 10.0, 1e-6 * float(gram.diagonal().abs().mean()) + 1e-12)
    if dual is None:
        dual = torch.linalg.lstsq(gram + (reg_lambda + jitter) * identity, centred_targets).solution

    alpha = root[:, None] * dual
    intercept = target_mean - (m_ctx @ alpha)[None, :]

    m_qry = (weight[None, :] @ k_qry.T).squeeze(0) / total
    logits = k_qry @ alpha - m_qry[:, None] * alpha.sum(0, keepdim=True) + intercept
    return logits[:, 1] - logits[:, 0]


def test_kernel_ridge_exact():
    print("Testing _solve_kernel_ridge exact formulation across kernels...")
    torch.manual_seed(42)
    n_ctx = 40
    n_qry = 20
    dim = 64

    ctx_feats = torch.randn(n_ctx, dim)
    ctx_labels = torch.randint(0, 2, (n_ctx,), dtype=torch.long)
    qry_feats = torch.randn(n_qry, dim)

    for kern in ["linear", "rbf", "poly", "cosine"]:
        m_orig = _solve_kernel_ridge(ctx_feats, ctx_labels, qry_feats, kernel=kern)
        m_flip = _solve_kernel_ridge(ctx_feats, 1 - ctx_labels, qry_feats, kernel=kern)

        diff = (m_orig + m_flip).abs().max().item()
        print(f"Kernel '{kern}' -> max antisymmetry error: {diff:.8e}")
        assert diff < 1e-5, f"Antisymmetry violated for kernel {kern}! diff={diff}"

    print("ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_kernel_ridge_exact()
