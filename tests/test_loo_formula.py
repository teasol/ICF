import torch

def solve_ridge_direct(X_tr, y_tr, X_te, reg=1.0):
    I = torch.eye(X_tr.shape[1])
    w = torch.linalg.solve(X_tr.T @ X_tr + reg * I, X_tr.T @ y_tr)
    return X_te @ w

def main():
    torch.manual_seed(42)
    N, D = 40, 16
    X = torch.randn(N, D)
    y = torch.randn(N, 1)
    reg = 1.0

    # 1. Exact loop (N separate fits)
    loo_exact = []
    for i in range(N):
        mask = torch.ones(N, dtype=torch.bool)
        mask[i] = False
        X_tr, y_tr = X[mask], y[mask]
        X_te = X[i:i+1]
        pred = solve_ridge_direct(X_tr, y_tr, X_te, reg=reg)
        loo_exact.append(pred.item())
    loo_exact = torch.tensor(loo_exact)

    # 2. Closed-form Hat matrix formula (0ms)
    # H = X (X^T X + reg I)^(-1) X^T
    I_D = torch.eye(D)
    inv_cov = torch.linalg.inv(X.T @ X + reg * I_D)
    w_all = inv_cov @ (X.T @ y)
    y_hat = (X @ w_all).squeeze(-1)

    # H_ii = row_i(X) @ inv_cov @ col_i(X^T)
    H_diag = (X @ inv_cov * X).sum(dim=-1)

    # LOO formula: (y_hat - H_ii * y) / (1 - H_ii)
    y_flat = y.squeeze(-1)
    loo_closed = (y_hat - H_diag * y_flat) / (1.0 - H_diag)

    diff = (loo_exact - loo_closed).abs().max().item()
    print(f"Max Difference between Exact Loop and Closed-Form LOO: {diff:.8e}")
    assert diff < 1e-5, f"LOO formula mismatch: {diff}"
    print("LOO Formula exact machine precision contract VERIFIED!")

if __name__ == "__main__":
    main()
