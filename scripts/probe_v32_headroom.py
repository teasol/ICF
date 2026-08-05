"""Stage-0 DR-CCER probes (P0-P3) on the existing CCER-v2 epoch-18 checkpoint.

These are the mandatory pre-implementation diagnostics from
``docs/architecture_v32b_dr_ccer_proposal.md`` §2. No training happens here.

Probes
------
P0  Decompose the observed v30 -> CCER-v2 delta into the new branch and the
    backbone drift. Runs three prediction sets on identical episodes:
    (1) full CCER-v2, (2) CCER-v2 with the ``ccer_v2`` residual contribution
    removed (``logits - scale * ccer_v2_logits`` from the auxiliary dict),
    (3) original v30. Reports overall, per-task, and the four cardinality
    bands (with n>34 and 11-34 explicitly, since 11-34 regressed in CCER-v2).

P1  Standalone evidence: per-query ``ccer_v2_logits`` standalone AUROC,
    correlation with the v30 margin, AUROC conditional on v30 being wrong,
    route-specific AUROC by task/cardinality, effective contribution SD.

P2  Fusion headroom: episode-grouped cross-validated logistic combiner on
    (v30 margin, CCER-v2 margin). < +0.005 rejects the current representation.

P3  Donor-agreement headroom (tests the v32 §4.1 premise): donor-resolved
    statistics (median, upper-quartile, agreement, MAD) over the per-donor
    class evidence computed from the model's own slot tokens, combined with
    v30 margin in the same episode-grouped CV. < +0.005 falsifies donor-resolved
    pooling regardless of P2.

Usage
-----
    python scripts/probe_v32_headroom.py \
        --checkpoint-ccerv2 checkpoints/20260805_123630/v31_ccer_v2/epoch=018-val_ce_loss=0.4438.ckpt \
        --checkpoint-v30 checkpoints/20260804_132334/v30_cardinality_poolz_l2/epoch=048-val_ce_loss=0.4442.ckpt \
        --config configs/train_v31_ccer_v2.yaml \
        --output logs/probe_v32_headroom_20260805.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import lightning as L
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.synthetic_data import RESPONSE_TASK_NAMES  # noqa: E402
from src.utils.metrics import auroc, log_loss  # noqa: E402
from src.utils.utils import build_datamodule, build_model, merge_train_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-ccerv2", type=Path, required=True)
    parser.add_argument("--checkpoint-v30", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="Config used to build the CCER-v2 model and the "
                             "shared evaluation stream.")
    parser.add_argument("--config-v30", type=Path,
                        default=Path("configs/train_v30_cardinality_poolz_l2.yaml"),
                        help="Config used to build the v30 model (no CCER-v2).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-episodes", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Episode-grouped CV folds for P2/P3 combiners.")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--precision", default="bf16-mixed")
    return parser.parse_args()


# Six-task evaluation distribution (v32b §5): legacy weights renormalized to
# 0.80 of their v30 values, plus any_positive_sparse at 0.20.
SIX_TASK_PROBABILITIES = [0.32, 0.24, 0.04, 0.04, 0.16, 0.20]

_BANDS = ((0, 4), (5, 10), (11, 34), (35, None))
_BAND_NAMES = ("n<=4", "5..10", "11..34", "n>34")


def _band_name(n: int) -> str:
    if n <= 4:
        return "n<=4"
    if n <= 10:
        return "5..10"
    if n <= 34:
        return "11..34"
    return "n>34"


def _load_model(config: dict, checkpoint: Path) -> "torch.nn.Module":
    model = build_model(config)
    ckpt = torch.load(checkpoint.expanduser().resolve(), map_location="cpu",
                      weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, _ = model.load_state_dict(state_dict, strict=False)
    # v30 checkpoints legitimately lack the CCER-v2 branch keys.
    allowed_missing = {
        key
        for key in missing
        if key == "model._architecture_version"
        or key.startswith("model.meta_classifier.ccer_v2_")
    }
    disallowed = sorted(set(missing) - allowed_missing)
    if disallowed:
        raise RuntimeError(f"Unexpected missing keys: {disallowed}")
    model.eval()
    return model


def _build_datamodule(config: dict, val_episodes: int):
    config = dict(config)
    data = dict(config["data"])
    val_kwargs = dict(data.get("val_dataset_kwargs") or {})
    val_kwargs["episodes_per_epoch"] = val_episodes
    val_kwargs["response_task_probabilities"] = list(SIX_TASK_PROBABILITIES)
    data["val_dataset_kwargs"] = val_kwargs
    config["data"] = data
    dm = build_datamodule(config)
    dm.setup("fit")
    return dm


@torch.no_grad()
def _collect(config: dict, checkpoint: Path, dm, device, bootstrap_keep=None):
    """Collect per-query predictions from one checkpoint on the fixed stream.

    Returns dict of CPU tensors with one row per query: episode, margin,
    label, task, cardinality. For the CCER-v2 checkpoint also the branch
    logits, residual scale, route scores/weights and slot tokens.
    """
    model = _load_model(config, checkpoint)
    model.to(device)
    records: dict[str, list] = {
        "episode": [], "margin": [], "label": [], "task": [], "n": [],
    }
    loader = dm.val_dataloader()
    autocast = torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    )
    with autocast:
        for episode_index, batch in enumerate(loader):
            if bootstrap_keep is not None and episode_index not in bootstrap_keep:
                continue
            x, y, mask_index, _, task_index = model._unpack_evaluation_batch(
                batch, "val"
            )
            x = x.to(device)
            y = y.to(device)
            mask_index = mask_index.to(device)
            logits, aux = model.model(x, y, mask_index, return_auxiliary=True)
            margin = (logits[:, 1] - logits[:, 0]).float().cpu()
            n = int(x.shape[-2])
            # task_index is a scalar (one response task per episode).
            task_value = int(task_index.item())
            for q in range(margin.numel()):
                records["episode"].append(episode_index)
                records["margin"].append(margin[q].item())
                records["label"].append(int(y[mask_index[q]].item()))
                records["task"].append(task_value)
                records["n"].append(n)
            if "ccer_v2_logits" in aux:
                c2 = aux["ccer_v2_logits"].float().cpu()
                scale = aux["ccer_v2_residual_scale"].float().cpu()
                routes = aux["ccer_v2_route_scores"].float().cpu()
                weights = aux["ccer_v2_route_weights"].float().cpu()
                records["ccer_margin"] = records.get("ccer_margin", [])
                records["ccer_margin"].extend(
                    (c2[:, 1] - c2[:, 0]).tolist()
                )
                records["ccer_scale"] = records.get("ccer_scale", [])
                records["ccer_scale"].extend(
                    [float(scale)] * margin.numel()
                )
                # routes: [Q, C, R] -> per-query route margins [Q, R].
                route_margin = routes[:, 1, :] - routes[:, 0, :]
                records["ccer_route_margin"] = records.get(
                    "ccer_route_margin", []
                )
                records["ccer_route_margin"].extend(
                    route_margin.tolist()
                )
                records["ccer_route_weights"] = records.get(
                    "ccer_route_weights", []
                )
                records["ccer_route_weights"].extend(weights.tolist())
                # P3 donor features from the model's own slot tokens.
                if "slot_tokens" in aux:
                    donor_feats = _donor_resolved_features(
                        aux, y, mask_index
                    )
                    for key, values in donor_feats.items():
                        out = records.setdefault(key, [])
                        out.extend(values)
    return records


def _donor_resolved_features(aux, y, mask_index):
    """Donor-resolved class-evidence statistics from slot-center tokens.

    For each class, per-donor evidence = cosine of the query bag's slot-pooled
    center token against every support donor's class prototype (slot-pooled
    center of that donor's bag). Robust statistics over the donor axis:
    median, upper-quartile, agreement (fraction > 0), MAD. Returns per-query
    features that are label-equivariant and query-label-free.
    """
    centers = aux["slot_tokens"][:, :, 0, :].float()          # [B, S, D]
    context_mask = aux["context_mask"].bool().cpu()
    query_index = mask_index.long().cpu()
    y = y.cpu()
    centers_n = torch.nn.functional.normalize(centers, dim=-1)
    # Pool over slots within each bag via logsumexp -> per-bag prototype.
    bag_proto = torch.logsumexp(centers_n, dim=1)              # [B, D]
    query_proto = torch.nn.functional.normalize(
        bag_proto[query_index], dim=-1
    )
    sim = query_proto @ bag_proto.T                            # [Q, B]
    # Restrict to context bags only (query bags are never donors).
    labels = y[context_mask]                                   # [B_ctx]
    sim_ctx = sim[:, context_mask]                             # [Q, B_ctx]
    features = {}
    for class_index in range(2):
        class_donors = labels == class_index                   # [B_ctx]
        class_sim = sim_ctx[:, class_donors].to(torch.float64)  # [Q, D_c]
        if class_sim.shape[1] == 0:
            class_sim = torch.zeros_like(sim_ctx[:, :1]).to(torch.float64)
        median = class_sim.median(dim=1).values
        uq = class_sim.quantile(0.75, dim=1)
        agreement = (class_sim > 0.0).to(torch.float64).mean(dim=1)
        mad = (class_sim - median.unsqueeze(1)).abs().mean(dim=1)
        features[f"dr_med_{class_index}"] = median.tolist()
        features[f"dr_uq_{class_index}"] = uq.tolist()
        features[f"dr_agree_{class_index}"] = agreement.tolist()
        features[f"dr_mad_{class_index}"] = mad.tolist()
    return features


def _torch_logistic_cv(X, y, groups, folds=5, steps=600, lr=1e-1):
    """Episode-grouped CV logistic regression fit with torch (no sklearn dep).

    Returns (cv_auroc_list, cv_auc_deltas, fold_sizes). Rows are queries;
    groups are episodes and stay intact within a fold.
    """
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(unique_groups)
    fold_splits = np.array_split(unique_groups, folds)
    aurocs = []
    auc_deltas = []
    fold_sizes = []
    for fold in range(folds):
        test_groups = fold_splits[fold]
        train_mask = np.isin(groups, test_groups, invert=True)
        test_mask = ~train_mask
        Xtr, ytr = X[train_mask], y[train_mask]
        Xte, yte = X[test_mask], y[test_mask]
        fold_sizes.append(len(yte))
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        # Standardize features on train only.
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0) + 1e-8
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd
        Xtr_t = torch.from_numpy(Xtr_s).float()
        ytr_t = torch.from_numpy(ytr).float()
        Xte_t = torch.from_numpy(Xte_s).float()
        w = torch.zeros(Xtr_s.shape[1], dtype=torch.float32, requires_grad=True)
        b = torch.zeros((), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            logit = Xtr_t @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, ytr_t
            ) + 1e-4 * w.square().sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            logit_te = Xte_t @ w + b
            prob = torch.sigmoid(logit_te).numpy()
        auc = auroc(
            torch.from_numpy(prob),
            torch.from_numpy(yte.astype(int)),
        )
        aurocs.append(auc)
        # Baseline AUROC of the first feature (v30 margin) alone on same fold.
        baseline_auc = auroc(
            torch.from_numpy(Xte_s[:, 0]),
            torch.from_numpy(yte.astype(int)),
        )
        auc_deltas.append(auc - baseline_auc)
    return aurocs, auc_deltas, fold_sizes


def _np_auroc(scores, labels) -> float:
    """AUROC over numpy arrays (the metrics module expects torch tensors)."""
    return float(
        auroc(torch.from_numpy(scores), torch.from_numpy(labels))
    )


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        "cuda" if (args.accelerator != "cpu" and torch.cuda.is_available()) else "cpu"
    )
    config = merge_train_config(args.config.expanduser().resolve())
    config["seed"] = args.seed
    dm = _build_datamodule(config, args.val_episodes)
    config_v30 = merge_train_config(args.config_v30.expanduser().resolve())
    config_v30["seed"] = args.seed

    print(f"[probe] collecting CCER-v2 (val episodes={args.val_episodes}) ...")
    records = _collect(config, args.checkpoint_ccerv2, dm, device)
    print(f"[probe] collecting v30 ...")
    records_v30 = _collect(config_v30, args.checkpoint_v30, dm, device)

    import numpy as np
    labels = np.asarray(records["label"])
    n_arr = np.asarray(records["n"])
    tasks = np.asarray(records["task"])
    episodes = np.asarray(records["episode"])

    margin_full = np.asarray(records["margin"])
    ccer_margin = np.asarray(records["ccer_margin"])
    scale = np.asarray(records["ccer_scale"])
    margin_zeroed = margin_full - scale * ccer_margin
    margin_v30 = np.asarray(records_v30["margin"])

    print("\n=== P0: branch vs backbone decomposition ===")
    for key, name in (
        (margin_full, "CCER-v2 full"),
        (margin_zeroed, "CCER-v2 branch-zeroed"),
        (margin_v30, "v30"),
    ):
        auc_all = _np_auroc(key, labels)
        print(f"  {name:24s} overall AUROC = {auc_all:.5f}")
        for t, tname in enumerate(RESPONSE_TASK_NAMES):
            mask = tasks == t
            if mask.sum() < 2:
                continue
            print(f"      {tname:24s} {_np_auroc(key[mask], labels[mask]):.5f} (n={mask.sum()})")
        for bname in _BAND_NAMES:
            mask = np.asarray([_band_name(n) for n in n_arr]) == bname
            if mask.sum() < 2:
                continue
            print(f"      {bname:8s} {_np_auroc(key[mask], labels[mask]):.5f} (n={mask.sum()})")

    print("\n=== P1: standalone evidence ===")
    corr_full_ccer = float(np.corrcoef(margin_full, ccer_margin)[0, 1])
    corr_v30_ccer = float(np.corrcoef(margin_v30, ccer_margin)[0, 1])
    print(f"  corr(CCER-v2 margin, v30 margin) = {corr_v30_ccer:.5f}")
    print(f"  corr(CCER-v2 margin, full margin) = {corr_full_ccer:.5f}")
    standalone_auc = _np_auroc(ccer_margin, labels)
    print(f"  standalone branch AUROC = {standalone_auc:.5f}")
    eff_sd = float(np.std(scale * ccer_margin))
    print(f"  effective contribution SD = {eff_sd:.5f}")

    # AUROC conditional on v30 being wrong (v30 margin wrong sign at 0.5).
    v30_wrong = (margin_v30 > 0) != (labels == 1)
    if v30_wrong.sum() >= 2:
        cond_auc = _np_auroc(ccer_margin[v30_wrong], labels[v30_wrong])
        print(f"  CCER-v2 AUROC | v30 wrong = {cond_auc:.5f} (n={v30_wrong.sum()})")

    route_margins = np.asarray(records["ccer_route_margin"])  # [Q, R]
    route_weights = np.asarray(records["ccer_route_weights"])
    route_count = route_margins.shape[1]
    print(f"  route count = {route_count}, mean weights = "
          f"{np.asarray(route_weights).mean(axis=0).round(3)}")
    for r in range(route_count):
        rm = route_margins[:, r]
        rauc = _np_auroc(rm, labels)
        print(f"      route {r}: standalone AUROC = {rauc:.5f}")

    # boundary crossing fraction
    cross_full = np.mean((margin_full > 0) != (margin_v30 > 0))
    cross_ccer = np.mean((ccer_margin > 0) != (margin_v30 > 0))
    print(f"  0.5-boundary crossing vs v30: full {cross_full:.4f}, branch {cross_ccer:.4f}")

    print("\n=== P2: fusion headroom (v30 margin, CCER-v2 margin) ===")
    X_p2 = np.stack((margin_v30, ccer_margin), axis=1)
    aurocs_p2, deltas_p2, sizes_p2 = _torch_logistic_cv(
        X_p2, labels, episodes, folds=args.cv_folds
    )
    print(f"  combiner CV AUROC = {np.mean(aurocs_p2):.5f} "
          f"(folds {len(aurocs_p2)}, mean size {np.mean(sizes_p2):.0f})")
    print(f"  paired delta vs v30 margin alone = {np.mean(deltas_p2):+.5f}")

    print("\n=== P3: donor-agreement headroom (v30 margin + donor features) ===")
    donor_keys = [k for k in records if k.startswith("dr_")]
    if donor_keys:
        X_p3 = np.stack([margin_v30] + [np.asarray(records[k]) for k in donor_keys], axis=1)
        aurocs_p3, deltas_p3, _ = _torch_logistic_cv(
            X_p3, labels, episodes, folds=args.cv_folds
        )
        print(f"  donor-feature combiner CV AUROC = {np.mean(aurocs_p3):.5f}")
        print(f"  paired delta vs v30 margin alone = {np.mean(deltas_p3):+.5f}")
        # Single-feature deltas (leave-one-out of donor features vs v30 alone).
        for i, k in enumerate(donor_keys, start=1):
            X1 = np.stack((margin_v30, np.asarray(records[k])), axis=1)
            _, d1, _ = _torch_logistic_cv(X1, labels, episodes, folds=args.cv_folds)
            print(f"    + {k:14s} delta {np.mean(d1):+.5f}")
    else:
        print("  (no donor features collected)")

    print("\n=== Gates (v32b §2) ===")
    p2_delta = float(np.mean(deltas_p2)) if deltas_p2 else float("nan")
    p3_delta = float(np.mean(deltas_p3)) if donor_keys and deltas_p3 else float("nan")
    print(f"  P2 gate (< +0.005 -> retire CCER representation): {p2_delta:+.5f}")
    print(f"  P3 gate (< +0.005 -> falsify donor-resolved):     {p3_delta:+.5f}")
    print(f"  Stage-0 gate (P2 or P3 >= +0.005 -> proceed): "
          f"{'PASS' if max(p2_delta, p3_delta) >= 0.005 else 'FAIL'}")

    # Persist compact summary.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["v30_auroc", f"{_np_auroc(margin_v30, labels):.6f}"])
        writer.writerow(["ccer_v2_full_auroc", f"{_np_auroc(margin_full, labels):.6f}"])
        writer.writerow(["ccer_v2_zeroed_auroc", f"{_np_auroc(margin_zeroed, labels):.6f}"])
        writer.writerow(["corr_v30_ccer", f"{corr_v30_ccer:.6f}"])
        writer.writerow(["standalone_ccer_auroc", f"{standalone_auc:.6f}"])
        writer.writerow(["eff_contribution_sd", f"{eff_sd:.6f}"])
        writer.writerow(["p2_combiner_auroc", f"{np.mean(aurocs_p2):.6f}"])
        writer.writerow(["p2_paired_delta", f"{p2_delta:.6f}"])
        if donor_keys:
            writer.writerow(["p3_combiner_auroc", f"{np.mean(aurocs_p3):.6f}"])
            writer.writerow(["p3_paired_delta", f"{p3_delta:.6f}"])
        writer.writerow(["stage0_gate", "PASS" if max(p2_delta, p3_delta) >= 0.005 else "FAIL"])
    print(f"[probe] summary written to {args.output}")


if __name__ == "__main__":
    main()
