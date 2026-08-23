import torch
import torch.nn.functional as F

def compute_patch_likelihood_features(
    context_bags: list[torch.Tensor],
    context_labels: torch.Tensor,
    query_bags: list[torch.Tensor],
    basis: torch.Tensor | None = None,
    pca_dim: int = 32,
    tau: float = 5.0,
    topk_fraction: float = 0.05,
    topk_min: int = 4,
    topk_max: int = 64,
    patches_per_ctx_bag: int = 64,
    reg_lambda: float = 1.0,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Direct In-Context Patch Likelihood Ratio + Top-K MIL features and Ridge Margin."""
    device = context_bags[0].device
    labels = context_labels.long().to(device)

    # 1. Subspace projection if basis provided
    if basis is not None:
        dim = min(pca_dim, basis.shape[1])
        proj = basis[:, :dim].float().to(device)
        ctx_p = [b.float().to(device) @ proj for b in context_bags]
        qry_p = [b.float().to(device) @ proj for b in query_bags]
    else:
        ctx_p = [b.float().to(device) for b in context_bags]
        qry_p = [b.float().to(device) for b in query_bags]

    # 2. Build Class 0 and Class 1 context patch banks
    bank_0, bank_1 = [], []
    for p, y in zip(ctx_p, labels):
        n_c = p.shape[0]
        if n_c == 0:
            continue
        n_sample = min(n_c, patches_per_ctx_bag)
        idx = torch.linspace(0, n_c - 1, n_sample, device=device).long()
        sampled = p[idx]
        if y == 1:
            bank_1.append(sampled)
        else:
            bank_0.append(sampled)

    if not bank_0 or not bank_1:
        raise ValueError("Both classes must have context patches.")

    P0 = torch.cat(bank_0, dim=0)
    P1 = torch.cat(bank_1, dim=0)

    P0_norm = F.normalize(P0, dim=-1)
    P1_norm = F.normalize(P1, dim=-1)

    # 3. Compute Patch-level log-odds likelihood ratio
    def get_slide_lr_feature(proj_bags):
        feats = []
        for bag in proj_bags:
            n_c = bag.shape[0]
            if n_c == 0:
                feats.append(torch.zeros(bag.shape[-1] + 1, device=device))
                continue
            bag_norm = F.normalize(bag, dim=-1)
            # Soft similarity to bank 1 and bank 0
            sim1 = bag_norm @ P1_norm.T  # (N_i, |P1|)
            sim0 = bag_norm @ P0_norm.T  # (N_i, |P0|)

            # Log-sum-exp for numerical stability
            score1 = torch.logsumexp(sim1 * tau, dim=-1) - torch.log(torch.tensor(P1_norm.shape[0], dtype=torch.float32, device=device))
            score0 = torch.logsumexp(sim0 * tau, dim=-1) - torch.log(torch.tensor(P0_norm.shape[0], dtype=torch.float32, device=device))

            lr = score1 - score0  # (N_i,)

            # Top-K positive and Bottom-K negative selection
            k = max(topk_min, min(topk_max, int(n_c * topk_fraction)))
            k = min(k, n_c)

            topk_vals, topk_idx = torch.topk(lr, k=k, largest=True)
            botk_vals, botk_idx = torch.topk(lr, k=k, largest=False)

            z_plus = bag[topk_idx].mean(dim=0)
            z_minus = bag[botk_idx].mean(dim=0)

            delta_z = z_plus - z_minus
            e_scalar = 0.5 * (topk_vals.mean() + botk_vals.mean())

            v_i = torch.cat([delta_z, e_scalar.unsqueeze(0)], dim=-1)
            feats.append(v_i)
        return torch.stack(feats)

    ctx_feats = get_slide_lr_feature(ctx_p)
    qry_feats = get_slide_lr_feature(qry_p)

    # 4. Class-balanced centered linear dual ridge solve
    targets = F.one_hot(labels, 2).float()
    counts = torch.bincount(labels, minlength=2)
    weight = counts.float().reciprocal()[labels]
    total = weight.sum().clamp_min(1e-12)

    feature_mean = (weight[:, None] * ctx_feats).sum(0, keepdim=True) / total
    target_mean = (weight[:, None] * targets).sum(0, keepdim=True) / total
    root = weight.sqrt()[:, None]

    design = (ctx_feats - feature_mean) * root
    centred_targets = (targets - target_mean) * root

    gram = design @ design.T
    identity = torch.eye(gram.shape[0], device=device, dtype=torch.float32)

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

    coefficients = design.T @ dual
    intercept = target_mean - feature_mean @ coefficients
    qry_logits = qry_feats @ coefficients + intercept
    qry_margin = qry_logits[:, 1] - qry_logits[:, 0]

    ctx_logits = ctx_feats @ coefficients + intercept
    ctx_margin = ctx_logits[:, 1] - ctx_logits[:, 0]

    return ctx_margin, qry_margin


def test_lr_antisymmetry():
    torch.manual_seed(42)
    n_ctx = 40
    n_qry = 20
    ctx_bags = [torch.randn(torch.randint(50, 150, (1,)).item(), 64) for _ in range(n_ctx)]
    qry_bags = [torch.randn(torch.randint(50, 150, (1,)).item(), 64) for _ in range(n_qry)]
    ctx_labels = torch.randint(0, 2, (n_ctx,), dtype=torch.long)

    # Ensure both classes exist
    ctx_labels[0] = 0
    ctx_labels[1] = 1

    _, m_orig = compute_patch_likelihood_features(ctx_bags, ctx_labels, qry_bags)
    _, m_flip = compute_patch_likelihood_features(ctx_bags, 1 - ctx_labels, qry_bags)

    diff = (m_orig + m_flip).abs().max().item()
    print(f"LR Branch Max Antisymmetry Error: {diff:.8e}")
    assert diff < 1e-5, f"Antisymmetry violated! diff={diff}"
    print("LR Branch Unit Test Passed Successfully!")

if __name__ == "__main__":
    test_lr_antisymmetry()
