"""Is the synthetic cell distribution statistically consistent with real UNI2 tiles?

Answer, as of docs SS123: no -- the dominant subspace matches well but the
spectral tail, the within-bag coherence and the cell-norm spread do not.
Re-run this before and after any change to the generator's cell-level knobs
(`latent_dim`, `manifold_mode`, `shared_component_fraction`, `donor_shift_scale`,
`normalize_output`) to see whether the change actually moves the statistic it
was supposed to move -- cheaper than a 4-seed arm by three orders of magnitude.

Diagnostic only -- nothing here fits anything to real data.

What is compared, and why at this stage of the pipeline: the model standardizes
every cell by the per-feature mean/std of the episode's CONTEXT cells
(`BaseAggregator._context_pool_stats`), and real features are otherwise fed in
raw from h5 while synthetic cells are L2-normalized (`normalize_output: true`).
That standardization absorbs global scale and per-dimension shift, so comparing
raw magnitudes would mostly measure something the model never sees. Everything
below is therefore reported BOTH before and after that same standardization, and
the post-standardization numbers are the ones that matter.

Real slides are drawn only from cohorts that carry NO SEAL evaluation task
(BRACS, CPTAC-LSCC, CPTAC-PDA, MBC, UCLA_Lung), so this diagnostic never touches
an evaluation cohort.
"""

import argparse
import glob
import os
import sys

import h5py
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.synthetic_data import SyntheticEpisodeDataset  # noqa: E402

FEAT = "/NHNHOME/BASE/kimds/Data/PathoBench/features"
# Cohorts with no in_seal=yes task -- see docs SS123.
LEAK_FREE = ["BRACS", "CPTAC-LSCC_v10", "CPTAC-PDA_v8", "MBC", "UCLA_Lung"]


def real_bags(num_bags, max_cells, seed):
    generator = torch.Generator().manual_seed(seed)
    paths = []
    for cohort in LEAK_FREE:
        paths += sorted(glob.glob(os.path.join(FEAT, cohort, "*.h5")))
    order = torch.randperm(len(paths), generator=generator)[:num_bags].tolist()
    bags = []
    for index in order:
        with h5py.File(paths[index], "r") as handle:
            features = torch.as_tensor(handle["features"][:], dtype=torch.float32)
        if features.shape[0] > max_cells:
            keep = torch.randperm(features.shape[0], generator=generator)[:max_cells]
            features = features.index_select(0, keep)
        bags.append(features)
    return bags


def synthetic_bags(config_path, num_bags, seed):
    config = yaml.safe_load(open(config_path))
    kwargs = dict(config["data"]["dataset_kwargs"])
    kwargs["generation_device"] = "cpu"
    kwargs["seed"] = seed
    dataset = SyntheticEpisodeDataset(**kwargs)
    bags = []
    episode_seed = seed
    while len(bags) < num_bags:
        generator = torch.Generator().manual_seed(episode_seed)
        episode = dataset.episode_generator.sample_episode(generator=generator)
        x = episode.x
        bags += [x[i] for i in range(x.shape[0])] if isinstance(x, torch.Tensor) else list(x)
        episode_seed += 1
    return bags[:num_bags]


def spectrum_stats(cells):
    """Effective rank of the cell cloud.

    `participation` is the participation ratio (sum lambda)^2 / sum lambda^2 --
    the number of directions that actually carry variance, insensitive to the
    long tail of near-zero eigenvalues that a hard threshold would count
    arbitrarily. r90/r99 are how many eigenvalues it takes to reach 90%/99% of
    total variance, reported alongside because they are easier to reason about.
    """
    centered = cells - cells.mean(dim=0, keepdim=True)
    covariance = (centered.T @ centered) / max(1, centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).flip(0).clamp_min(0)
    total = eigenvalues.sum()
    cumulative = eigenvalues.cumsum(0) / total
    participation = float(total**2 / (eigenvalues**2).sum())
    r90 = int((cumulative < 0.90).sum()) + 1
    r99 = int((cumulative < 0.99).sum()) + 1
    top1 = float(eigenvalues[0] / total)
    return participation, r90, r99, top1


def describe(name, bags, dim):
    flat = torch.cat([b for b in bags], dim=0)
    print(f"\n{'=' * 74}\n{name}   bags={len(bags)}  cells={flat.shape[0]:,}  dim={dim}\n{'=' * 74}")

    norms = flat.norm(dim=-1)
    print("[raw]  cell L2 norm      "
          f"mean={norms.mean():.4f}  sd={norms.std():.4f}  "
          f"min={norms.min():.4f}  max={norms.max():.4f}")
    per_dim_std = flat.std(dim=0)
    print("[raw]  per-dim std       "
          f"mean={per_dim_std.mean():.4f}  min={per_dim_std.min():.4f}  max={per_dim_std.max():.4f}  "
          f"max/min={per_dim_std.max() / per_dim_std.min().clamp_min(1e-12):.1f}")

    # Same standardization the model applies: per-feature mean/std over all cells.
    mean = flat.mean(dim=0, keepdim=True)
    std = flat.std(dim=0, keepdim=True).clamp_min(1e-6)
    z = (flat - mean) / std

    participation, r90, r99, top1 = spectrum_stats(z)
    print(f"[std]  effective rank    participation={participation:.1f}  "
          f"r90={r90}  r99={r99}  top-1 eigenvalue share={top1 * 100:.1f}%   (of {dim})")

    kurtosis = (((z - z.mean(0)) / z.std(0).clamp_min(1e-6)) ** 4).mean(0)
    print(f"[std]  per-dim kurtosis  median={kurtosis.median():.2f}  "
          f"p95={kurtosis.quantile(0.95):.2f}  max={kurtosis.max():.2f}   (gaussian=3)")

    # Variance decomposition: how much of the total sits BETWEEN bags vs WITHIN.
    zb, index = [], 0
    for bag in bags:
        zb.append(z[index:index + bag.shape[0]])
        index += bag.shape[0]
    # Both terms are per-cell PER-DIMENSION variances, so both divide by dim.
    # (Dividing only `within` by dim inflates the between-bag share by 1536x.)
    bag_means = torch.stack([b.mean(dim=0) for b in zb])
    counts = torch.tensor([float(b.shape[0]) for b in zb])
    dim_count = z.shape[1]
    grand = (bag_means * counts[:, None]).sum(0) / counts.sum()
    between = ((bag_means - grand) ** 2 * counts[:, None]).sum() / (counts.sum() * dim_count)
    within = sum(float(((b - b.mean(0, keepdim=True)) ** 2).sum()) for b in zb) / (counts.sum() * dim_count)
    print(f"[std]  variance split    between-bag={between / (between + within) * 100:.1f}%  "
          f"within-bag={within / (between + within) * 100:.1f}%   "
          f"(ICC = fraction explained by which bag a cell came from)")

    cosines = []
    for b in zb[: min(24, len(zb))]:
        take = b[torch.randperm(b.shape[0])[:256]]
        unit = torch.nn.functional.normalize(take, dim=-1)
        similarity = unit @ unit.T
        cosines.append(similarity[torch.triu(torch.ones_like(similarity), diagonal=1) > 0])
    cosines = torch.cat(cosines)
    print(f"[std]  within-bag cosine mean={cosines.mean():+.4f}  sd={cosines.std():.4f}")

    cell_counts = torch.tensor([b.shape[0] for b in bags], dtype=torch.float32)
    print(f"       cells/bag         mean={cell_counts.mean():.0f}  "
          f"min={int(cell_counts.min())}  max={int(cell_counts.max())}")


def main():
    parser = argparse.ArgumentParser()
    # Repo-root-relative, like the sibling diagnostics: the absolute /NHNHOME/BASE
    # path broke when the home mount moved. v83 is the generator this diagnostic
    # describes, so it follows the config into configs/archive/ (docs SS110).
    parser.add_argument(
        "--config",
        default="configs/archive/v83_linear_head/train_v83_linear_head_1536_1gpu.yaml",
    )
    parser.add_argument("--bags", type=int, default=80)
    parser.add_argument("--max-cells", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    describe("SYNTHETIC (v83 generator)", synthetic_bags(args.config, args.bags, args.seed), 1536)
    describe("REAL UNI2 (leak-free cohorts)", real_bags(args.bags, args.max_cells, args.seed), 1536)


if __name__ == "__main__":
    main()
