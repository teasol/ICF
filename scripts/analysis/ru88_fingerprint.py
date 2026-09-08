"""RU-88: do fold-level, label-free fingerprints predict task-specific effects?

Card: docs/ru/RU-88.json. The pre-registered panel (3 fingerprints x 2 targets =
6 combinations), the cluster structure (7 tasks, df = 6) and the follow-up entry
criterion (95% interval excludes 0 AND |rho| >= 0.3) are fixed THERE, before any
result was seen. This script only applies them mechanically. It does not add a
fingerprint, drop a combination, or re-tune a threshold.

GPU budget for this RU is 0. Everything below is a CPU re-aggregation of stored
tile features and stored predictions.

--------------------------------------------------------------------------------
Definitions fixed by this script (the proposal is under-specified on kernel,
bandwidth, dimension and the per-fold scalar summary; docs/ru/RU-88.json
delegates that choice here and requires it to be stated, not made silently).
--------------------------------------------------------------------------------

Episode reconstruction. For official fold k of a task, the context set is every
slide with fold_k != 'test' and the query set is every slide with fold_k ==
'test', read from data/repro_labels_folds/official/<task>/k=all.tsv -- the same
rule scripts/test_pathobench.py:evaluate_official_folds() uses, so the split is
reproduced exactly rather than approximated. Slides with no .h5 feature file are
dropped first, as the pipeline does.

Slide embedding (label-free). C_within(k) = pooled WITHIN-slide tile covariance
of the CONTEXT bags only (src/models/common/basis.py). B(k) = its top-32
eigenvectors, descending. z_s = mean_tiles(X_s) @ B(k) in R^32. This is exactly
the representation BM/DS consume (src/models/branches/bm.py, ICF_BM_DIM=32,
ICF_SKETCH_DIM=256), so the fingerprints live in the space the intervention
acts on. No label -- context or query -- enters any fingerprint.

F1  outlier proximity (proposal candidate 1). Whiten the 32-d embeddings with
    the CONTEXT mean and covariance (ridge 1e-6 * tr(Sigma)/32, numerical only).
    For each query slide, the mean whitened (Mahalanobis) distance to its k = 5
    nearest CONTEXT slides. Fold scalar = MEDIAN over query slides. k = 5 is the
    proposal's "nearest k context slides"; the median (not the mean) because a
    fold carries 40-75 query slides and a handful of far outliers would
    otherwise set the fold value.

F2  context<->query distribution shift (proposal candidate 4). Gaussian RBF
    kernel on the SAME whitened 32-d embeddings; bandwidth by the median
    heuristic, sigma^2 = median pairwise squared distance over the pooled
    context+query set of that fold; unbiased MMD^2 (Gretton et al.); fold
    scalar = sqrt(max(MMD^2_u, 0)). Whitening first makes the bandwidth
    comparable across tasks whose embedding scales differ by orders of
    magnitude.

F3  context covariance spectrum -- CONTROL (proposals.md improvement 2
    "Task-Geometry"). Eigenvalues lambda_1..lambda_256 of C_within(k) (256 =
    ICF_SKETCH_DIM, the basis the pipeline actually builds).
      F3a decay exponent alpha: -OLS slope of log p_k on log k, k = 1..256,
          p_k = lambda_k / sum lambda.  (p_k ~ k^-alpha, proposals.md)
      F3b participation ratio PR = (sum lambda)^2 / sum lambda^2.
    The card names F3 with BOTH statistics but fixes the table at 3 x 2 = 6
    combinations. F3a (the decay exponent, which is what proposals.md
    improvement 2 actually proposed) is therefore the panel scalar that enters
    the 6-combination table; F3b is computed and reported in full alongside it
    as the second component of the same panel entry. Nothing is dropped and
    nothing outside the panel is added.

Targets (fold level).
T1  paired salience-subsampling delta (SS225). Within ONE file and ONE fold --
    predictions/pathobench_<task>_v121_ds_sweep_official50_bf16.pt, produced by
    a single run (scripts/run_ds_subsample_sweep.sh) that emitted every arm --
    T1 = AUROC(m_ds_f000) - AUROC(m_ds_full). m_ds_full is the full bag,
    m_ds_f000 is background-keep 0.00 = anchors only (15% of patches), i.e. the
    "full -> anchor-only" column of the SS225 table. Same fold, same context,
    same query set, subsampling toggled: this is a controlled paired
    comparison, not two runs laid side by side.
T2  MDX-alone AUROC - 0.5, from predictions/ru86_gate2/summary.json
    raw_fold_auroc (RU-86; fold order = official fold 0..49).

Correlation and interval.
rho = Pearson correlation over the 350 (task, fold) pairs -- PRIMARY, and the
only estimator the card's entry criterion is applied to. 95% interval = cluster
-robust (7 task clusters, small-cluster factor G/(G-1)) influence-function SE
times t_{0.975, df=6}, matching PROJECT.md SS4.1 ("interval from the task-
cluster t, df = 6; the 50 folds inside a task are not independent").
Spearman rho is computed for the SAME 6 combinations with the same interval and
reported as descriptive only; the entry criterion is NOT applied to it. Both
estimators are declared here, before execution, so neither can be picked after
the fact.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.metrics import auroc  # noqa: E402

TASKS = [
    # (official task dir, display name, predictions file stem)
    ("cptac_lscc/ARID1A_mutation", "ARID1A", "cptac_lscc_ARID1A_mutation"),
    ("cptac_lscc/Histologic_Grade", "Grade", "cptac_lscc_Histologic_Grade"),
    ("cptac_lscc/KEAP1_mutation", "KEAP1", "cptac_lscc_KEAP1_mutation"),
    ("cptac_luad/KRAS_mutation", "KRAS", "cptac_luad_KRAS_mutation"),
    ("cptac_pda/SMAD4_mutation", "SMAD4", "cptac_pda_SMAD4_mutation"),
    ("ucla_lung/progression_regression", "Progression",
     "ucla_lung_progression_regression"),
    ("cptac_ccrcc/PBRM1_mutation", "PBRM1", "cptac_ccrcc_PBRM1_mutation"),
]

DATA = ROOT / "data" / "repro_labels_folds"
OFFICIAL = DATA / "official"
FEATURES = DATA / "features"
PRED = ROOT / "predictions"
OUT_DIR = PRED / "ru88_fingerprint"

SKETCH_DIM = 256   # ICF_SKETCH_DIM (scripts/lib/arms.sh)
EMBED_DIM = 32     # ICF_BM_DIM / ICF_DS_DIM
KNN = 5            # F1 neighbourhood
RIDGE = 1e-6       # whitening ridge, relative to tr(Sigma)/d
N_FOLDS = 50
SWEEP_TAG = "v121_ds_sweep"
STAGE_TAG = "ru85_stageB"
FEATURE_DIM = 1536


# ---------------------------------------------------------------- fold layout


def index_h5() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for dataset_dir in sorted(FEATURES.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for path in dataset_dir.glob("*.h5"):
            index.setdefault(path.stem, path)
    return index


def read_task(task_dir: str, h5_index: dict[str, Path]) -> dict:
    """Slide ids, labels and per-fold context/query split, as the pipeline does."""
    base = OFFICIAL / task_dir
    tsv = base / "k=all.tsv"
    task_col = yaml.safe_load((base / "config.yaml").read_text())["task_col"]
    with tsv.open() as fh:
        records = list(csv.DictReader(fh, delimiter="\t"))
    fold_cols = [c for c in records[0] if c.startswith("fold_")]
    slide_ids = [str(r["slide_id"]).strip() for r in records]
    labels = {sid: int(float(r[task_col])) for sid, r in zip(slide_ids, records)}
    if len(set(labels.values())) > 2:
        labels = {s: int(v != 0) for s, v in labels.items()}
    by_id = {sid: r for sid, r in zip(slide_ids, records)}
    kept = [s for s in slide_ids if s in h5_index]
    folds = []
    for fc in fold_cols:
        query = [s for s in kept if by_id[s][fc].strip() == "test"]
        context = [s for s in kept if by_id[s][fc].strip() != "test"]
        folds.append({"query": query, "context": context})
    return {"slide_ids": kept, "labels": labels, "folds": folds,
            "n_dropped": len(slide_ids) - len(kept)}


# ------------------------------------------------------- per-slide statistics


def slide_stats(path: Path) -> tuple[int, torch.Tensor, torch.Tensor]:
    """(n_tiles, mean[D] float64, centred scatter[D, D] float64)."""
    with h5py.File(path, "r") as handle:
        features = torch.from_numpy(handle["features"][:]).double()
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise SystemExit(f"unexpected feature shape at {path}: {tuple(features.shape)}")
    n = features.shape[0]
    mean = features.mean(dim=0)
    centred = features - mean
    return n, mean, centred.T @ centred


# ------------------------------------------------------------- fingerprints


def whiten(context: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = context.mean(dim=0, keepdim=True)
    centred = context - mu
    cov = centred.T @ centred / (context.shape[0] - 1)
    cov = cov + RIDGE * (torch.diagonal(cov).sum() / cov.shape[0]) * torch.eye(
        cov.shape[0], dtype=cov.dtype
    )
    values, vectors = torch.linalg.eigh(cov)
    values = values.clamp_min(1e-12)
    w = vectors @ torch.diag(values.rsqrt()) @ vectors.T
    return (context - mu) @ w, (query - mu) @ w


def f1_outlier_proximity(zc: torch.Tensor, zq: torch.Tensor) -> float:
    distances = torch.cdist(zq, zc)
    k = min(KNN, zc.shape[0])
    nearest = distances.topk(k, dim=1, largest=False).values
    return float(nearest.mean(dim=1).median())


def f2_mmd(zc: torch.Tensor, zq: torch.Tensor) -> float:
    pooled = torch.cat([zc, zq], dim=0)
    sq = torch.cdist(pooled, pooled).pow(2)
    iu = torch.triu_indices(sq.shape[0], sq.shape[0], offset=1)
    sigma2 = float(sq[iu[0], iu[1]].median())
    if sigma2 <= 0:
        return float("nan")

    def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.cdist(a, b).pow(2) / (2.0 * sigma2))

    m, n = zc.shape[0], zq.shape[0]
    kcc, kqq, kcq = kernel(zc, zc), kernel(zq, zq), kernel(zc, zq)
    term_c = (kcc.sum() - torch.diagonal(kcc).sum()) / (m * (m - 1))
    term_q = (kqq.sum() - torch.diagonal(kqq).sum()) / (n * (n - 1))
    mmd2 = float(term_c + term_q - 2.0 * kcq.mean())
    return math.sqrt(max(mmd2, 0.0))


def f3_spectrum(eigenvalues: torch.Tensor) -> tuple[float, float]:
    lam = eigenvalues[:SKETCH_DIM].clamp_min(1e-30).double()
    p = lam / lam.sum()
    k = torch.arange(1, lam.numel() + 1, dtype=torch.float64)
    x, y = torch.log(k), torch.log(p)
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    alpha = -slope
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    return alpha, pr


# ------------------------------------------------------------------ targets


def paired_t1(stem: str) -> tuple[list[int], list[float], list[list[str]], list[int]]:
    """Per-fold AUROC(anchors-only) - AUROC(full bag), same file, same fold."""
    path = PRED / f"pathobench_{stem}_{SWEEP_TAG}_official50_bf16.pt"
    if not path.exists():
        raise SystemExit(f"missing SS225 sweep file: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    deltas, slides, n_context = [], [], []
    for fold in blob["per_fold"]:
        label = fold["label"]
        full = auroc(torch.sigmoid(fold["m_ds_full"].float()), label)
        anchor = auroc(torch.sigmoid(fold["m_ds_f000"].float()), label)
        deltas.append(float(anchor) - float(full))
        slides.append([str(s) for s in fold["slide_id"]])
        n_context.append(int(fold["context_label"].numel()))
    return list(blob["fold_indices"]), deltas, slides, n_context


def stage_b_slides(stem: str) -> tuple[list[int], list[list[str]]]:
    path = PRED / f"pathobench_{stem}_{STAGE_TAG}_official50_bf16.pt"
    if not path.exists():
        raise SystemExit(f"missing RU-85 stage-B file: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return (list(blob["fold_indices"]),
            [[str(s) for s in f["slide_id"]] for f in blob["per_fold"]])


# -------------------------------------------------------------- correlation


def rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    # average ties
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = np.zeros(unique.size)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def cluster_correlation(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Pearson rho with a cluster-robust (influence function) t interval, df = G-1."""
    n = x.size
    xs = (x - x.mean()) / x.std(ddof=0)
    ys = (y - y.mean()) / y.std(ddof=0)
    rho = float((xs * ys).mean())
    influence = xs * ys - 0.5 * rho * (xs ** 2 + ys ** 2)
    labels = np.unique(groups)
    g = labels.size
    per_cluster = np.array([influence[groups == lab].sum() for lab in labels])
    variance = (per_cluster ** 2).sum() / (n ** 2) * (g / (g - 1))
    se = float(math.sqrt(max(variance, 0.0)))
    crit = float(student_t.ppf(0.975, g - 1))
    return {"rho": rho, "se_cluster": se, "df": g - 1,
            "ci95": [rho - crit * se, rho + crit * se]}


def spearman_cluster(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    return cluster_correlation(rank(x), rank(y), groups)


def variance_split(values: np.ndarray, groups: np.ndarray) -> dict:
    labels = np.unique(groups)
    means = np.array([values[groups == lab].mean() for lab in labels])
    within = float(np.mean([values[groups == lab].var(ddof=1) for lab in labels]))
    between = float(means.var(ddof=1))
    return {"within_task_variance": within, "between_task_variance": between,
            "between_over_within": (between / within) if within > 0 else float("inf")}


# ------------------------------------------------------------------- driver


def main() -> None:
    started = time.time()
    torch.set_grad_enabled(False)
    h5_index = index_h5()

    ru86 = json.loads((PRED / "ru86_gate2" / "summary.json").read_text())
    raw_fold_auroc = ru86["raw_fold_auroc"]

    alignment: dict[str, dict] = {}
    rows: list[dict] = []
    stats_cache: dict[str, tuple[int, torch.Tensor, torch.Tensor]] = {}

    for task_dir, name, stem in TASKS:
        layout = read_task(task_dir, h5_index)
        sweep_idx, t1, sweep_slides, sweep_nctx = paired_t1(stem)
        stage_idx, stage_slides = stage_b_slides(stem)

        # ---- kill (2): fold index alignment. Abort, never re-sort. ---------
        problems: list[str] = []
        if sweep_idx != list(range(N_FOLDS)):
            problems.append(f"sweep fold_indices != 0..{N_FOLDS - 1}: {sweep_idx[:5]}...")
        if stage_idx != list(range(N_FOLDS)):
            problems.append(f"stageB fold_indices != 0..{N_FOLDS - 1}: {stage_idx[:5]}...")
        if len(raw_fold_auroc[name]) != N_FOLDS:
            problems.append(f"ru86 raw_fold_auroc[{name}] has {len(raw_fold_auroc[name])}")
        for k in range(N_FOLDS):
            expected = set(layout["folds"][k]["query"])
            if set(sweep_slides[k]) != expected:
                problems.append(f"fold {k}: sweep query set != official fold_{k} test set")
            if set(stage_slides[k]) != expected:
                problems.append(f"fold {k}: stageB query set != official fold_{k} test set")
            if sweep_nctx[k] != len(layout["folds"][k]["context"]):
                problems.append(
                    f"fold {k}: sweep context size {sweep_nctx[k]} != "
                    f"{len(layout['folds'][k]['context'])}"
                )
        alignment[name] = {
            "n_slides": len(layout["slide_ids"]),
            "n_slides_dropped_no_h5": layout["n_dropped"],
            "query_sets_match_official": not problems,
            "problems": problems[:10],
        }
        if problems:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "summary.json").write_text(json.dumps(
                {"ru": "RU-88", "verdict": "실행 무효",
                 "reason": "fold index/query-set misalignment (card kill 2)",
                 "alignment": alignment}, indent=2, ensure_ascii=False))
            raise SystemExit(f"ABORT (kill 2): {name}: {problems[0]}")

        # ---- per-slide sufficient statistics, computed once -----------------
        for sid in layout["slide_ids"]:
            if sid not in stats_cache:
                stats_cache[sid] = slide_stats(h5_index[sid])
        total_n = sum(stats_cache[s][0] for s in layout["slide_ids"])
        total_scatter = torch.zeros(FEATURE_DIM, FEATURE_DIM, dtype=torch.float64)
        for sid in layout["slide_ids"]:
            total_scatter += stats_cache[sid][2]

        for k in range(N_FOLDS):
            context_ids = layout["folds"][k]["context"]
            query_ids = layout["folds"][k]["query"]
            scatter = total_scatter.clone()
            n_cells = total_n
            for sid in query_ids:
                scatter -= stats_cache[sid][2]
                n_cells -= stats_cache[sid][0]
            eigenvalues, eigenvectors = torch.linalg.eigh(scatter / n_cells)
            eigenvalues = eigenvalues.flip(0)
            basis = eigenvectors[:, -SKETCH_DIM:].flip(-1)[:, :EMBED_DIM]

            ctx = torch.stack([stats_cache[s][1] for s in context_ids]) @ basis
            qry = torch.stack([stats_cache[s][1] for s in query_ids]) @ basis
            zc, zq = whiten(ctx, qry)

            alpha, pr = f3_spectrum(eigenvalues)
            rows.append({
                "task": name, "fold": k,
                "n_context": len(context_ids), "n_query": len(query_ids),
                "F1_outlier_proximity": f1_outlier_proximity(zc, zq),
                "F2_mmd": f2_mmd(zc, zq),
                "F3a_decay_exponent": alpha,
                "F3b_participation_ratio": pr,
                "T1_paired_subsample_delta": t1[k],
                "T2_mdx_auroc_minus_half": float(raw_fold_auroc[name][k]) - 0.5,
            })
        print(f"[{name}] {N_FOLDS} folds done  ({time.time() - started:.0f}s)", flush=True)

    # -------------------------------------------------------- correlations
    groups = np.array([r["task"] for r in rows])
    fingerprints = ["F1_outlier_proximity", "F2_mmd", "F3a_decay_exponent"]
    targets = ["T1_paired_subsample_delta", "T2_mdx_auroc_minus_half"]

    combinations: dict[str, dict] = {}
    for f in fingerprints + ["F3b_participation_ratio"]:
        x = np.array([r[f] for r in rows], dtype=np.float64)
        for t in targets:
            y = np.array([r[t] for r in rows], dtype=np.float64)
            pearson = cluster_correlation(x, y, groups)
            spearman = spearman_cluster(x, y, groups)
            lo, hi = pearson["ci95"]
            excludes_zero = (lo > 0) or (hi < 0)
            magnitude = abs(pearson["rho"]) >= 0.3
            combinations[f"{f} x {t}"] = {
                "in_preregistered_6": f in fingerprints,
                "pearson": pearson,
                "spearman_descriptive_only": spearman,
                "ci95_excludes_zero": bool(excludes_zero),
                "abs_rho_ge_0.3": bool(magnitude),
                "meets_followup_entry_criterion": bool(excludes_zero and magnitude),
            }

    diagnostics = {
        name: variance_split(np.array([r[name] for r in rows]), groups)
        for name in fingerprints + ["F3b_participation_ratio"] + targets
    }
    per_task = {}
    for name in sorted(set(groups)):
        mask = groups == name
        per_task[name] = {
            key: float(np.mean([r[key] for r, m in zip(rows, mask) if m]))
            for key in fingerprints + ["F3b_participation_ratio"] + targets
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ru": "RU-88",
        "gpu_hours": 0.0,
        "wall_seconds": round(time.time() - started, 1),
        "n_samples": len(rows),
        "n_clusters": int(len(set(groups))),
        "definitions": {
            "embedding": "mean tile feature @ top-32 eigenvectors of the CONTEXT "
                         "pooled within-slide covariance (ICF_SKETCH_DIM=256, "
                         "ICF_BM_DIM=32); no label used",
            "F1": f"median over query slides of the mean whitened distance to its "
                  f"{KNN} nearest context slides",
            "F2": "sqrt of the unbiased RBF MMD^2 between context and query "
                  "embeddings; median-heuristic bandwidth on the whitened pooled set",
            "F3a": "-OLS slope of log(lambda_k/sum lambda) on log k, k=1..256 "
                   "(panel scalar for the 6-combination table)",
            "F3b": "participation ratio (sum lambda)^2 / sum lambda^2 over the same "
                   "256 eigenvalues (second component of the same panel entry)",
            "T1": "AUROC(m_ds_f000) - AUROC(m_ds_full), same fold, same file "
                  "predictions/pathobench_<task>_v121_ds_sweep_official50_bf16.pt",
            "T2": "predictions/ru86_gate2/summary.json raw_fold_auroc - 0.5",
            "interval": "cluster-robust influence-function SE over 7 task clusters "
                        "(factor G/(G-1)) times t_{0.975, df=6}",
            "entry_criterion": "95% interval excludes 0 AND |rho| >= 0.3 "
                               "(Pearson only; fixed in docs/ru/RU-88.json)",
        },
        "alignment_check": alignment,
        "combinations": combinations,
        "variance_split": diagnostics,
        "per_task_means": per_task,
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n{'combination':<52}{'rho':>9}{'ci_lo':>9}{'ci_hi':>9}{'entry':>8}")
    for key, value in combinations.items():
        p = value["pearson"]
        mark = "YES" if value["meets_followup_entry_criterion"] else "no"
        tag = "" if value["in_preregistered_6"] else " (F3 2nd component)"
        print(f"{key + tag:<52}{p['rho']:>9.3f}{p['ci95'][0]:>9.3f}"
              f"{p['ci95'][1]:>9.3f}{mark:>8}")
    print(f"\nwall {time.time() - started:.0f}s · GPU-h 0 · "
          f"wrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
