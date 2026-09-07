"""AKS branch: angular scatter and fourth-moment spectra of the whitened cloud.

Every existing branch reads either where a slide's tokens sit (CV, BM, QA, DS),
how the eigenvalue spectrum spreads (BD), or the shape of the whitened *radius*
distribution (SH, SHJ). AKS reads what whitening and the radius collapse throw
away: the directional scatter of the unit-normalised tokens, and the fourth-moment
tensor's anisotropy with its trace divided out.

Whitening makes the second moment the identity, so the component BD reads is
removed by construction. On any spherically symmetric distribution M = I/32 and
B is proportional to I, which makes all eight features identically constant while
SHJ still varies - so AKS is not a function of SHJ. That is a statement about the
features, not about the ridge margins they produce; the |r| screen is what decides.

The condition number log(q_max/q_min) that the proposal listed is NOT used: it has
no upper bound, which contradicts the proposal's own design goal of bounded
dimensionless features. Four bounded descriptors per spectrum give the eight dims.
"""

from __future__ import annotations

import torch

AKS_FEATURE_DIM = 8
_ANGULAR_DIM = 4      # m_akd: the angular half
_FOURTH_DIM = 4       # m_akf: the fourth-moment half
_EPS = 1e-8


def _whiten(bag: torch.Tensor, basis: torch.Tensor, dim: int) -> torch.Tensor:
    """Affine-invariant coordinates: Z with Z^T Z / N = I.

    Autocast is forced off for the same reason as in shj.py - the projection in
    bf16 carries ~1e-3 relative error and whitening amplifies it by two orders of
    magnitude, which would make the feature depend on the surrounding autocast
    state. The eigendecomposition runs in float64.
    """
    with torch.autocast(device_type=bag.device.type, enabled=False):
        values = bag.to(basis.device, dtype=torch.float32)
        proj = values @ basis[:, :dim].to(dtype=torch.float32)
        centred = proj - proj.mean(dim=0, keepdim=True)
        cov = (centred.T @ centred) / float(centred.shape[0])
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = torch.linalg.eigh(cov.double())
        return (centred.double() @ eigvecs) * eigvals.clamp_min(1e-8).rsqrt()


def _spectrum_features(matrix: torch.Tensor) -> torch.Tensor:
    """Four bounded, rotation-invariant descriptors of a PSD matrix's spectrum.

    The spectrum is normalised to sum 1, so every descriptor is dimensionless and
    lies in [0, 1]. Eigenvalues only: the eigenvector sign ambiguity of eigh
    cannot reach these features (M -> D M D leaves the spectrum unchanged).
    """
    d = matrix.shape[0]
    eig = torch.linalg.eigvalsh(0.5 * (matrix + matrix.T)).clamp_min(0.0)
    total = eig.sum().clamp_min(_EPS)
    q = eig / total

    entropy = -(q * (q + _EPS).log()).sum() / torch.log(torch.tensor(float(d), dtype=q.dtype))
    participation = 1.0 / (q.square().sum().clamp_min(_EPS) * float(d))
    # ||q - uniform|| normalised by its maximum, reached when all mass is on one mode.
    anisotropy = ((float(d) * q.square().sum() - 1.0).clamp_min(0.0) / float(d - 1)).sqrt()
    return torch.stack([entropy, q.max(), participation, anisotropy])


def aks_slide_features(bag: torch.Tensor, basis: torch.Tensor, dim: int
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Eight AKS descriptors plus the diagnostics the kill conditions need.

    Returns (features[8], diagnostics[3]) where diagnostics are
    [angular spectrum entropy H_q, fourth-moment spectrum entropy H_p, log N].
    """
    z = _whiten(bag, basis, dim)
    n = float(z.shape[0])

    radius_sq = z.square().sum(dim=1, keepdim=True)
    unit = z / radius_sq.clamp_min(_EPS).sqrt()
    angular = (unit.T @ unit) / n                       # M: radius fully discarded
    fourth = ((z * radius_sq).T @ z) / n                # B: trace is E||z||^4

    q_feats = _spectrum_features(angular)
    p_feats = _spectrum_features(fourth)
    features = torch.cat([q_feats, p_feats]).float()
    diagnostics = torch.stack([
        q_feats[0], p_feats[0],
        torch.tensor(n, dtype=q_feats.dtype, device=q_feats.device).log(),
    ]).float()
    return features, diagnostics
