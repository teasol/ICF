"""Primary 7 판정 설계의 표준오차와 최소검출효과(MDE) 산출 (RU-80).

GPU를 쓰지 않는다. 저장된 50-fold 예측(`predictions/pathobench_<task>_<tag>_official50_bf16.pt`)만
재집계한다.

무엇을 재는가 (docs/PROJECT.md SS4.1 재정의를 위한 계측):
  추정 대상 = "Primary 7이 대표하는 과제·fold 분포에서의 기대 macro AUROC".
  표본 단위 = fold. 7 과제 x 50 fold = 350.

산출 1 (사용자 지정 설계) -- 흔들림의 규모
  공식 5-branch 각각에 대해 350개 fold AUROC의 표준편차 sigma_b 를 구하고,
  5개를 평균해 sigma 로 삼는다. SE = sigma / sqrt(350).

산출 2 (보조) -- 대응 비교의 실제 정밀도
  승격 판정은 후보와 기준선의 *차이*에 대한 것이므로, 같은 fold에서 짝지은
  per-fold Δ 의 표준편차 s_d 가 판정을 지배한다. 짝짓기 때문에 s_d 는 보통
  sigma 보다 훨씬 작다. 실제 후보 태그들로 s_d 를 산출한다.

산출 3 (보조) -- 의존 구조
  한 과제의 50 fold 는 같은 데이터를 공유하므로 독립 350 이 아니다.
  과제를 군집으로 본 cluster SE 를 함께 낸다. 둘 사이가 벌어지면 sqrt(350) 은
  낙관적이다.

MDE 는 양측 alpha=0.05, power=0.80 정규 근사 `(z_.975 + z_.80) * SE` 로 계획값을
낸다. 이는 근사의 예시이며 관측 후 성공 컷오프가 아니다 (research_protocol SS4.2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# `scripts/analysis/` is two levels below the repo root, so the repo root has
# to go on sys.path before `src` imports resolve (matches drop2_furthest_detail.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import auroc  # noqa: E402

PRIMARY7 = [
    "cptac_lscc_ARID1A_mutation",
    "cptac_lscc_Histologic_Grade",
    "cptac_lscc_KEAP1_mutation",
    "cptac_luad_KRAS_mutation",
    "cptac_pda_SMAD4_mutation",
    "ucla_lung_progression_regression",
    "cptac_ccrcc_PBRM1_mutation",
]
BRANCHES = ["m_cv", "m_bm", "m_bd", "m_qa", "m_ds"]  # CT 제외: 공식 5-branch 기준
Z_ALPHA, Z_POWER = 1.959964, 0.841621  # 양측 alpha=0.05, power=0.80


def trimmed_mean(probs: torch.Tensor) -> torch.Tensor:
    if probs.shape[0] < 3:
        return probs.mean(dim=0)
    sorted_p, _ = torch.sort(probs, dim=0)
    return sorted_p[1:-1].mean(dim=0)


def load(tag: str) -> dict:
    return {
        t: torch.load(
            f"predictions/pathobench_{t}_{tag}_official50_bf16.pt",
            map_location="cpu", weights_only=False,
        )["per_fold"]
        for t in PRIMARY7
    }


def per_fold_branch_auroc(data: dict, branch: str) -> tuple[np.ndarray, np.ndarray]:
    """350개 fold AUROC와 각 값의 과제 인덱스."""
    vals, task_ix = [], []
    for i, t in enumerate(PRIMARY7):
        for f in data[t]:
            vals.append(auroc(torch.sigmoid(f[branch]), f["label"]))
            task_ix.append(i)
    return np.asarray(vals), np.asarray(task_ix)


def per_fold_ensemble_auroc(data: dict) -> np.ndarray:
    out = []
    for t in PRIMARY7:
        for f in data[t]:
            p = torch.stack([torch.sigmoid(f[b]) for b in BRANCHES], dim=0)
            out.append(auroc(trimmed_mean(p), f["label"]))
    return np.asarray(out)


def cluster_se(values: np.ndarray, task_ix: np.ndarray) -> float:
    """과제를 군집으로 본 평균의 표준오차 (군집 평균의 SD / sqrt(군집 수))."""
    means = np.array([values[task_ix == i].mean() for i in range(len(PRIMARY7))])
    return float(means.std(ddof=1) / np.sqrt(len(means)))


def pct(x: float) -> str:
    return f"{x * 100:.4f}%p"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v121_baseline")
    ap.add_argument("--compare", nargs="*", default=[
        "v121_shape_screen", "v121_sh_variants", "v121_rm_screen",
        "v121_ds_sweep", "v121_loo_capacity",
    ])
    a = ap.parse_args()

    base = load(a.tag)
    n_folds = [len(base[t]) for t in PRIMARY7]
    n = int(sum(n_folds))
    print(f"기준 태그: {a.tag}")
    print(f"과제별 fold 수: {dict(zip((t.split('_')[-2] for t in PRIMARY7), n_folds))}")
    print(f"총 표본 N = {n}\n")
    if n != 350:
        print(f"[중단 조건] 총 fold 수가 350이 아니다 (N={n}). 구조를 먼저 확인한다.")

    # ---- 산출 1: branch별 흔들림 -> sigma -------------------------------
    print("=" * 72)
    print("산출 1 · 공식 5-branch 각각의 350-fold AUROC 산포 (사용자 지정 설계)")
    print("=" * 72)
    print(f"{'branch':<8}{'mean':>10}{'sigma_b':>12}{'min':>10}{'max':>10}")
    sigmas, task_ix = [], None
    for b in BRANCHES:
        v, task_ix = per_fold_branch_auroc(base, b)
        s = float(v.std(ddof=1))
        sigmas.append(s)
        print(f"{b:<8}{v.mean():>10.4f}{s:>12.4f}{v.min():>10.4f}{v.max():>10.4f}")
    sigma = float(np.mean(sigmas))
    se_naive = sigma / np.sqrt(n)
    mde_naive = (Z_ALPHA + Z_POWER) * se_naive
    print(f"\nsigma = mean(sigma_b) = {sigma:.4f}")
    print(f"SE  = sigma / sqrt({n}) = {se_naive:.5f}  ({pct(se_naive)})")
    print(f"MDE = (z.975 + z.80) x SE = {mde_naive:.5f}  ({pct(mde_naive)})")
    print("  ! 이 sigma 는 fold 간 '과제 난이도 차이'를 대부분 포함한다.")
    print("    승격은 후보-기준선의 *차이*에 대한 판정이므로 산출 2가 지배적이다.")

    # ---- 산출 3: 의존 구조 ----------------------------------------------
    print("\n" + "=" * 72)
    print("산출 3 · 의존 구조 — 과제 내 50 fold 는 독립이 아니다")
    print("=" * 72)
    ens = per_fold_ensemble_auroc(base)
    se_c = cluster_se(ens, task_ix)
    se_i = float(ens.std(ddof=1) / np.sqrt(n))
    print(f"5-branch 앙상블 macro = {ens.mean():.4f}")
    print(f"  독립 가정 SE (sqrt(350))      = {se_i:.5f}  ({pct(se_i)})")
    print(f"  과제 군집 SE (7 clusters)     = {se_c:.5f}  ({pct(se_c)})")
    print(f"  설계 효과 (군집/독립)         = {se_c / se_i:.2f}x")

    # ---- 산출 2: 대응 비교 ----------------------------------------------
    print("\n" + "=" * 72)
    print("산출 2 · 대응 비교의 실제 정밀도 — per-fold Δ (승격 판정을 지배한다)")
    print("=" * 72)
    print(f"{'비교 태그':<26}{'Δ평균':>11}{'s_d':>10}{'SE_ind':>10}{'SE_clu':>10}{'MDE_ind':>10}{'MDE_clu':>10}")
    rows = []
    for tag in a.compare:
        try:
            cand = load(tag)
        except FileNotFoundError:
            print(f"{tag:<26}  (파일 없음 — 건너뜀)")
            continue
        d = per_fold_ensemble_auroc(cand) - ens
        s_d = float(d.std(ddof=1))
        se_ind = s_d / np.sqrt(n)
        se_clu = cluster_se(d, task_ix)
        rows.append((tag, s_d, se_ind, se_clu))
        print(f"{tag:<26}{d.mean():>11.5f}{s_d:>10.4f}{se_ind:>10.5f}"
              f"{se_clu:>10.5f}{(Z_ALPHA+Z_POWER)*se_ind:>10.5f}{(Z_ALPHA+Z_POWER)*se_clu:>10.5f}")
    if rows:
        s_d_m = float(np.mean([r[1] for r in rows]))
        se_ind_m = float(np.mean([r[2] for r in rows]))
        se_clu_m = float(np.mean([r[3] for r in rows]))
        print(f"\n대응 s_d 평균  = {s_d_m:.4f}   (산출 1의 sigma {sigma:.4f} 대비 {s_d_m/sigma:.2f}x)")
        print(f"대응 SE  독립  = {se_ind_m:.5f}  ({pct(se_ind_m)})")
        print(f"대응 SE  군집  = {se_clu_m:.5f}  ({pct(se_clu_m)})")
        print(f"대응 MDE 독립  = {(Z_ALPHA+Z_POWER)*se_ind_m:.5f}  ({pct((Z_ALPHA+Z_POWER)*se_ind_m)})")
        print(f"대응 MDE 군집  = {(Z_ALPHA+Z_POWER)*se_clu_m:.5f}  ({pct((Z_ALPHA+Z_POWER)*se_clu_m)})")

    print("\n" + "=" * 72)
    print("현행 승격 컷오프 +1.0%p 와의 관계")
    print("=" * 72)
    for name, mde in (("산출1 sigma 기반", mde_naive),
                      *([("대응 비교 독립", (Z_ALPHA+Z_POWER)*se_ind_m),
                         ("대응 비교 군집", (Z_ALPHA+Z_POWER)*se_clu_m)] if rows else [])):
        rel = "컷오프보다 작다 → 컷오프가 판별력보다 높다" if mde < 0.01 else "컷오프보다 크다 → 1.0%p 를 안정 검출할 수 없다"
        print(f"  {name:<16} MDE = {pct(mde):>12}   {rel}")
    print("\n[주의] MDE 는 설계 계획값이지 관측 후 성공 컷오프가 아니다.")
    print("       실용적 최소 효과 δ_min 은 목적과 비용으로 따로 정한다.")


if __name__ == "__main__":
    main()
