import torch
import torch.nn.functional as F

def test_ds_mathematics():
    print("Testing DS (In-Context Salience Denoising) Branch mathematical properties...")
    
    # 1. Mock context and query bags (in PCA 32 space)
    torch.manual_seed(42)
    n_ctx = 20
    n_qry = 10
    pca_dim = 32
    K = 16
    
    ctx_bags = [torch.randn(100 + i*10, pca_dim) for i in range(n_ctx)]
    ctx_labels = torch.tensor([i % 2 for i in range(n_ctx)], dtype=torch.long)
    qry_bags = [torch.randn(150, pca_dim) for i in range(n_qry)]
    
    # Mock cluster centroids
    centroids = torch.randn(K, pca_dim)
    centroids = F.normalize(centroids, dim=-1)
    
    def compute_ds_margins(ctx_bags, ctx_labels, qry_bags, centroids, beta=1.0, eps=1e-5):
        # 1. Compute soft abundance for context
        def get_abundance_and_patches(bags):
            abundances = []
            patch_assignments = []
            for b in bags:
                b_norm = F.normalize(b, dim=-1)
                sim = b_norm @ centroids.T  # (N_i, K)
                p = F.softmax(sim * 5.0, dim=-1)
                a = p.mean(dim=0)
                abundances.append(a)
                patch_assignments.append(p)
            return torch.stack(abundances), patch_assignments
        
        ctx_a, ctx_p = get_abundance_and_patches(ctx_bags)
        qry_a, qry_p = get_abundance_and_patches(qry_bags)
        
        # 2. Cluster Salience
        a1 = ctx_a[ctx_labels == 1].mean(dim=0)
        a0 = ctx_a[ctx_labels == 0].mean(dim=0)
        s = torch.log((a1 + eps) / (a0 + eps))  # (K,)
        s_abs = s.abs()
        
        # 3. Denoised Bag Representation
        def extract_denoised_mean(bags, assignments):
            features = []
            for b, p in zip(bags, assignments):
                # patch salience
                u = p @ s_abs  # (N_i,)
                # softmax weighting
                w = F.softmax(beta * (u - u.mean()) / (u.std().clamp_min(1e-5)), dim=0)  # (N_i,)
                z_denoised = (w.unsqueeze(-1) * b).sum(dim=0)  # (32,)
                features.append(z_denoised)
            return torch.stack(features)
        
        ctx_feat = extract_denoised_mean(ctx_bags, ctx_p)
        qry_feat = extract_denoised_mean(qry_bags, qry_p)
        
        # 4. Standardise
        mean = ctx_feat.mean(dim=0, keepdim=True)
        ctx_norm = ctx_feat - mean
        qry_norm = qry_feat - mean
        rms = ctx_norm.square().mean().sqrt().clamp_min(1e-6)
        ctx_norm = ctx_norm / rms
        qry_norm = qry_norm / rms
        
        # 5. Dual Ridge
        K_ctx = ctx_norm @ ctx_norm.T
        K_qry = qry_norm @ ctx_norm.T
        
        # Targets
        n1 = (ctx_labels == 1).sum().float()
        n0 = (ctx_labels == 0).sum().float()
        targets = torch.where(ctx_labels == 1, 1.0 / n1, -1.0 / n0)
        
        alpha = torch.linalg.solve(K_ctx + 1.0 * torch.eye(n_ctx), targets)
        m = K_qry @ alpha
        return m

    m_orig = compute_ds_margins(ctx_bags, ctx_labels, qry_bags, centroids)
    m_flipped = compute_ds_margins(ctx_bags, 1 - ctx_labels, qry_bags, centroids)
    
    diff = (m_orig + m_flipped).abs().max().item()
    print(f"Antisymmetry check (m_orig + m_flipped max abs): {diff:.8f}")
    assert diff < 1e-5, f"Antisymmetry violated! diff={diff}"
    print("DS Branch mathematics and antisymmetry PASSED successfully!")

if __name__ == "__main__":
    test_ds_mathematics()
