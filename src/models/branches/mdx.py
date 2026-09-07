"""MDX branch: multimodality depth profile of the top principal directions.

SH reads per-dimension skew and kurtosis; SHJ reads the whitened radius
distribution. Both are moment summaries, and excess kurtosis is a many-to-one map
that cannot separate a bimodal mixture from a low-kurtosis unimodal cloud (a
uniform distribution and a 3.7-sigma separated two-component mixture both sit near
-1.2). MDX reads the mode structure those summaries collapse.

Every knob is a constant - grid size, range, bandwidth ladder, peak rule - so no
data-dependent choice enters and the branch stays deterministic. A sign flip of an
eigenvector mirrors the density, which leaves peak counts and valley depths
unchanged; near-degenerate eigenvalues can still rotate the top directions, so
that frequency is logged as a diagnostic rather than assumed away.
"""

from __future__ import annotations

import torch

MDX_FEATURE_DIM = 8
_DIRECTIONS = 8
_BANDWIDTHS = (0.15, 0.25, 0.40, 0.60)
_GRID_RANGE = 4.0
#: A local maximum counts as a peak only above this fraction of the density's
#: global maximum. The proposal's peak rule had no prominence floor, and on a
#: plain unimodal Gaussian the tail noise peaks sit at 0.4-0.6% of the maximum -
#: they were being selected as the second peak, which made the valley depth
#: D = 1 - v / p2 large for a single-mode cloud. The floor is a fixed constant,
#: so no data-dependent choice enters.
_PEAK_FLOOR = 0.05
_EPS = 1e-8


def _top_directions(bag: torch.Tensor, basis: torch.Tensor, dim: int
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardised projections onto the slide's own top-8 principal directions.

    Autocast off and float64 eigendecomposition, for the reason given in shj.py.
    Returns (T[N, 8] standardised, adjacent eigenvalue ratios[7]).
    """
    with torch.autocast(device_type=bag.device.type, enabled=False):
        values = bag.to(basis.device, dtype=torch.float32)
        proj = values @ basis[:, :dim].to(dtype=torch.float32)
        centred = proj - proj.mean(dim=0, keepdim=True)
        cov = (centred.T @ centred) / float(centred.shape[0])
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = torch.linalg.eigh(cov.double())
        order = torch.argsort(eigvals, descending=True)[:_DIRECTIONS]
        top = centred.double() @ eigvecs[:, order]
        standard = (top - top.mean(dim=0, keepdim=True)) / top.std(dim=0, keepdim=True).clamp_min(_EPS)
        lead = eigvals[order].clamp_min(_EPS)
        ratios = lead[:-1] / lead[1:]
    return standard, ratios.float()


def _smoothed_histograms(standard: torch.Tensor, grid: int) -> torch.Tensor:
    """Fixed-grid histogram per direction, smoothed at each fixed bandwidth.

    Returns densities[len(_BANDWIDTHS), 8, grid].
    """
    step = 2.0 * _GRID_RANGE / float(grid - 1)
    idx = ((standard + _GRID_RANGE) / step).round().long().clamp(0, grid - 1)
    hist = torch.zeros(standard.shape[1], grid, dtype=torch.float64, device=standard.device)
    hist.scatter_add_(1, idx.T, torch.ones_like(idx.T, dtype=torch.float64))

    out = []
    for bandwidth in _BANDWIDTHS:
        half = max(1, int(round(3.0 * bandwidth / step)))
        offsets = torch.arange(-half, half + 1, dtype=torch.float64, device=standard.device) * step
        kernel = (-0.5 * (offsets / bandwidth).square()).exp()
        kernel = kernel / kernel.sum()
        smoothed = torch.nn.functional.conv1d(
            hist.unsqueeze(1), kernel.view(1, 1, -1), padding=half).squeeze(1)
        out.append(smoothed)
    return torch.stack(out)


def _peaks_and_depth(density: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Peak count and top-two valley depth D = 1 - v / p2 for one 1-D density."""
    interior = density[1:-1]
    floor = _PEAK_FLOOR * density.max()
    is_peak = (interior > density[:-2]) & (interior >= density[2:]) & (interior >= floor)
    peak_pos = torch.nonzero(is_peak, as_tuple=False).flatten() + 1
    count = torch.tensor(float(peak_pos.numel()), dtype=density.dtype, device=density.device)
    if peak_pos.numel() < 2:
        return count, torch.zeros((), dtype=density.dtype, device=density.device)

    heights = density[peak_pos]
    order = torch.argsort(heights, descending=True)[:2]
    a, b = sorted(peak_pos[order].tolist())
    second = heights[order][1].clamp_min(_EPS)
    valley = density[a:b + 1].min()
    return count, (1.0 - valley / second).clamp(0.0, 1.0)


def mdx_slide_features(bag: torch.Tensor, basis: torch.Tensor, dim: int, grid: int = 257
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Eight MDX descriptors plus the diagnostics the kill conditions need.

    Returns (features[8], diagnostics[3]) where diagnostics are
    [unimodal fraction at b = 0.25, max valley depth, degenerate eigenvalue-ratio
    fraction (adjacent ratio < 1.01)].
    """
    standard, ratios = _top_directions(bag, basis, dim)
    densities = _smoothed_histograms(standard, grid)

    counts = torch.zeros(len(_BANDWIDTHS), _DIRECTIONS, dtype=torch.float64, device=standard.device)
    depths = torch.zeros_like(counts)
    for bi in range(len(_BANDWIDTHS)):
        for dj in range(_DIRECTIONS):
            counts[bi, dj], depths[bi, dj] = _peaks_and_depth(densities[bi, dj])

    reference = _BANDWIDTHS.index(0.25)
    multimodal = (counts > 1.5).double()
    # Critical bandwidth index: the coarsest ladder rung that still resolves a
    # second mode, normalised to [0, 1]. Zero when no rung does.
    resolved = multimodal.max(dim=1).values
    critical = (resolved * torch.arange(1, len(_BANDWIDTHS) + 1, dtype=torch.float64,
                                        device=standard.device)).max() / float(len(_BANDWIDTHS))

    features = torch.cat([
        depths.max(dim=1).values,                    # 4: max valley depth per bandwidth
        depths.mean().reshape(1),                    # mean depth
        counts.mean().reshape(1),                    # mean mode count
        critical.reshape(1),                         # critical bandwidth index
        multimodal[reference].mean().reshape(1),     # multimodal direction fraction
    ]).float()

    diagnostics = torch.stack([
        (counts[reference] < 1.5).double().mean(),
        depths.max(),
        (ratios.double() < 1.01).double().mean() if ratios.numel() else torch.zeros((), dtype=torch.float64),
    ]).float()
    return features, diagnostics
