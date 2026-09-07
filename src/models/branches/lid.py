"""LID branch: local neighbourhood scaling profile of the whitened cloud.

BD reads the eigenvalue spectrum of the projected covariance and SHJ reads the
radius distribution after whitening; both are identically insensitive to the
local dimension of the support. LID reads that: on a curved manifold the local
intrinsic dimension can differ from the global spectrum entirely. It reaches the
"cluster structure" information CT tried to get from k-means, without the k-means
bottleneck that removed CT from the comparison basis in section 214-V.

Every feature is a ratio of distances, so it is scale-invariant, and whitening
makes it affine-invariant. Two risks are known and handled explicitly rather than
assumed away: token-count proxying (each feature's correlation with log N is
reported so size-driven features can be dropped) and subsample robustness (the
m = 1024 variant exists to be correlated against m = 4096).

Determinism: the subsample is a fixed stride, ties in topk break by ascending
index (torch.topk with sorted=True is stable for equal values on a fixed layout),
and r1 = 0 from duplicate tokens is clamped, with the duplicate rate reported.

Known limitation, measured on synthetic clouds before any benchmark run: the
estimator is only stable when the neighbourhood is not degenerate. On clustered
clouds it reads 14-17 against 22-25 for an isotropic cloud, consistently across
seeds, token counts and subsample sizes. On a noiseless 1-D curve the k-NN
distance ratios collapse towards 1 and ID = 1 / mean(log ratio) diverges - it
exceeded the ambient dimension 32 by an order of magnitude. That is why the
proposal's subsample-robustness falsifier (m = 4096 against m = 1024) is a kill
condition rather than a footnote.
"""

from __future__ import annotations

import torch

LID_FEATURE_DIM = 8
_NEIGHBOURS = 20
_BLOCK = 512
_EPS = 1e-12


def _whiten(bag: torch.Tensor, basis: torch.Tensor, dim: int) -> torch.Tensor:
    """Same affine-invariant coordinates as AKS; see aks.py for why autocast is off."""
    with torch.autocast(device_type=bag.device.type, enabled=False):
        values = bag.to(basis.device, dtype=torch.float32)
        proj = values @ basis[:, :dim].to(dtype=torch.float32)
        centred = proj - proj.mean(dim=0, keepdim=True)
        cov = (centred.T @ centred) / float(centred.shape[0])
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = torch.linalg.eigh(cov.double())
        return ((centred.double() @ eigvecs) * eigvals.clamp_min(1e-8).rsqrt()).float()


def _neighbour_distances(points: torch.Tensor) -> torch.Tensor:
    """Exact k-NN distances r1..r20 per point, computed in row blocks.

    Blocked so the N x N distance matrix is never materialised: at m = 4096 the
    full matrix would be 64 MB per slide, and the pipeline holds several slides.
    """
    n = points.shape[0]
    k = min(_NEIGHBOURS + 1, n)
    out = []
    sq = points.square().sum(dim=1)
    for start in range(0, n, _BLOCK):
        block = points[start:start + _BLOCK]
        d2 = (block.square().sum(dim=1, keepdim=True) - 2.0 * block @ points.T + sq).clamp_min(0.0)
        near = torch.topk(d2, k, dim=1, largest=False, sorted=True).values
        out.append(near[:, 1:k].sqrt())          # drop self-distance
    return torch.cat(out)


def _mle_id(radii: torch.Tensor, scale: int) -> torch.Tensor:
    """Levina-Bickel maximum-likelihood intrinsic dimension at neighbour scale k.

    ID_k = [ mean_j log(r_k / r_j) ]^-1 over j < k, using only r1..rk.
    """
    k = min(scale, radii.shape[1])
    ref = radii[:, k - 1:k].clamp_min(_EPS)
    ratios = (ref / radii[:, :k - 1].clamp_min(_EPS)).log().clamp_min(0.0)
    per_point = ratios.mean(dim=1).clamp_min(_EPS).reciprocal()
    return per_point.median()


def lid_slide_features(bag: torch.Tensor, basis: torch.Tensor, dim: int, subsample: int = 4096
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Eight LID descriptors plus the diagnostics the kill conditions need.

    Returns (features[8], diagnostics[3]) where diagnostics are
    [TwoNN ID (median-robust), |ID_20 - ID_5|, duplicate-token rate].
    """
    z = _whiten(bag, basis, dim)
    n = z.shape[0]
    if n > subsample:
        stride = torch.arange(subsample, device=z.device, dtype=torch.float64)
        z = z[(stride * (n / subsample)).long().clamp(0, n - 1)]

    radii = _neighbour_distances(z)
    duplicate_rate = (radii[:, 0] <= _EPS).float().mean()
    r1 = radii[:, 0].clamp_min(_EPS)
    r2 = radii[:, 1].clamp_min(_EPS) if radii.shape[1] > 1 else r1

    mu = (r2 / r1).log().clamp_min(_EPS)
    id_twonn_mle = 1.0 / mu.mean().clamp_min(_EPS)
    id_twonn_med = torch.log(torch.tensor(2.0, device=z.device)) / mu.median().clamp_min(_EPS)
    id_5, id_20 = _mle_id(radii, 5), _mle_id(radii, min(20, radii.shape[1]))

    mid = radii[:, min(9, radii.shape[1] - 1)].clamp_min(_EPS).log()
    centred = mid - mid.median()
    iqr = centred.quantile(0.75) - centred.quantile(0.25)
    spread = centred.quantile(0.90) - centred.quantile(0.10)
    # Local-to-global scale: the whitened cloud has E||z||^2 = dim, so a random
    # pair sits at sqrt(2 dim); the ratio is dimensionless.
    global_scale = torch.tensor(2.0 * float(dim), device=z.device).sqrt()
    local_global = radii[:, min(9, radii.shape[1] - 1)].median() / global_scale

    features = torch.stack([
        id_twonn_med, id_twonn_mle, id_5, id_20, (id_20 - id_5).abs(),
        iqr, spread, local_global,
    ]).float()
    diagnostics = torch.stack([id_twonn_med, (id_20 - id_5).abs(), duplicate_rate]).float()
    return features, diagnostics
