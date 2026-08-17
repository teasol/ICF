"""Does k-means give CT balanced, informative tokens? (docs SS157)

SS148-5 left three suspects upstream of CT's readout; this is the first —
farthest-point token selection. FPS maximises spread, so it places tokens in
LOW-density regions (outlier cells). Every ordinary cell is then far from every
token, the soft assignment flattens, and the per-token discriminative statistic
sits at a median of 1.31 (SS148-4). k-means places centroids at density modes,
which is what "a cell token is a cell population" was supposed to mean.

Tokens are initialised FROM the FPS points and refined with Lloyd iterations, so
0 iterations is today's behaviour exactly and the count is a single knob
interpolating coverage (FPS) -> density (k-means). Everything stays deterministic.

⚠️ The two methods fail in OPPOSITE directions, so the balance diagnostic matters
as much as the AUROC. FPS over-represents rare cells; k-means can spend every
centroid on the dominant population and lose the rare-but-informative ones.
`effective tokens` = exp(entropy of the cluster-size shares): 16 means perfectly
balanced clusters, 1 means one cluster swallowed everything.

⚠️ Diagnostic loader (bags capped once at load, SS138-2), so absolute AUROC is not
comparable to SEAL macro; the between-iteration gaps are.

Usage: python scripts/diagnose_ct_kmeans.py [--heldout]
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.diagnose_full_basis import ALL_TASKS, auroc, index_h5, load_task  # noqa: E402
from src.models.ct_readout import (  # noqa: E402
    CTReadoutConfig,
    ct_abundance,
    discriminative_score,
    farthest_point_tokens,
    lloyd_refine,
    prepare_cells,
    readout_extreme,
    readout_ridge,
)
from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig  # noqa: E402

FEATURES = "/NHNHOME/BASE/kimds/Data/PathoBench/features"
HELDOUT = [
    "cptac_lscc/ARID1A_mutation", "cptac_lscc/Histologic_Grade",
    "cptac_lscc/KEAP1_mutation", "cptac_luad/KRAS_mutation",
    "cptac_pda/SMAD4_mutation", "ucla_lung/progression_regression",
    "cptac_ccrcc/PBRM1_mutation",
]
ITERATIONS = [0, 1, 3, 10, 30]
PCA_DIM = 32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    tasks = HELDOUT if args.heldout else ALL_TASKS
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    reference = TrainingFreeClassifier(TrainingFreeConfig(sketch_dim=256))
    generator = torch.Generator().manual_seed(0)
    h5_index = index_h5(FEATURES)
    stats = {
        count: {"balance": [], "entropy": [], "tmedian": [], "extreme": [], "ridge": []}
        for count in ITERATIONS
    }
    per_task = {count: {} for count in ITERATIONS}

    for task in tasks:
        bags, labels, membership, folds = load_task(task, h5_index, device, 8192, generator)
        task_scores = {count: [] for count in ITERATIONS}
        for fold in folds:
            train, test = membership[fold]["train"], membership[fold]["test"]
            if not train or not test:
                continue
            y = torch.tensor([labels[s] for s in train], device=device)
            if len(set(y.tolist())) < 2:
                continue
            query_y = torch.tensor([labels[s] for s in test], device=device)
            context_bags = [bags[s] for s in train]
            query_bags = [bags[s] for s in test]
            basis = reference.within_slide_basis(context_bags)
            for count in ITERATIONS:
                config = CTReadoutConfig(pca_dim=PCA_DIM, kmeans_iterations=count)
                abundance = ct_abundance(context_bags, query_bags, config, basis)
                # Cluster sizes at this iteration count. `lloyd_refine` with max(count,1)
                # reports the assignment counts even at 0 iterations, where they
                # describe the FPS tokens themselves.
                context, _ = prepare_cells(context_bags, query_bags, config, basis)
                pooled = torch.cat(context, dim=0)
                _, counts = lloyd_refine(
                    pooled, farthest_point_tokens(pooled, config), count
                )
                share = (counts / counts.sum().clamp_min(1.0)).clamp_min(1e-12)
                stats[count]["balance"].append(
                    float(torch.exp(-(share * share.log()).sum()))
                )
                rows = abundance.context.clamp_min(1e-12)
                stats[count]["entropy"].append(float(-(rows * rows.log()).sum(-1).mean()))
                stats[count]["tmedian"].append(
                    float(discriminative_score(abundance, y, config).abs().median())
                )
                for key, readout in (("extreme", readout_extreme), ("ridge", readout_ridge)):
                    value = auroc(
                        query_y.cpu(), readout(abundance, y, config).query.float().cpu()
                    )
                    if value is not None:
                        stats[count][key].append(value)
                        if key == "ridge":
                            task_scores[count].append(value)
        for count in ITERATIONS:
            if task_scores[count]:
                per_task[count][task] = statistics.mean(task_scores[count])
        print(f"{task:34s} " + "  ".join(
            f"{c}={per_task[c].get(task, float('nan')):.4f}" for c in ITERATIONS
        ), flush=True)
        del bags
        torch.cuda.empty_cache()

    print(f"\n{len(tasks)} tasks, {PCA_DIM}-d PCA subspace (v108 CT). "
          f"ln(16) = {math.log(16):.3f}\n")
    print(f"{'iters':>6}{'effective tokens':>18}{'abundance entropy':>19}"
          f"{'|t| median':>12}{'CT-only extreme':>17}{'CT-only ridge':>15}")
    for count in ITERATIONS:
        row = stats[count]
        print(f"{count:>6}{statistics.mean(row['balance']):>18.2f}"
              f"{statistics.mean(row['entropy']):>19.4f}"
              f"{statistics.mean(row['tmedian']):>12.2f}"
              f"{statistics.mean(row['extreme']):>17.4f}"
              f"{statistics.mean(row['ridge']):>15.4f}")
    print("\n=== CT-only (ridge readout) per task ===")
    print(f"{'task':34s}" + "".join(f"{c:>9}" for c in ITERATIONS))
    for task in per_task[0]:
        print(f"{task:34s}" + "".join(f"{per_task[c][task]:>9.4f}" for c in ITERATIONS))
    print("-" * (34 + 9 * len(ITERATIONS)))
    macro = {c: statistics.mean(per_task[c].values()) for c in ITERATIONS}
    print(f"{'MACRO':34s}" + "".join(f"{macro[c]:>9.4f}" for c in ITERATIONS))
    print(f"{'Δ vs 0 (FPS)':34s}" + "".join(f"{macro[c] - macro[0]:>+9.4f}" for c in ITERATIONS))
    wins = {c: sum(per_task[c][t] > per_task[0][t] for t in per_task[0]) for c in ITERATIONS}
    print("  beats FPS on: " + "  ".join(
        f"{c}: {wins[c]}/{len(per_task[0])}" for c in ITERATIONS if c))


if __name__ == "__main__":
    main()
