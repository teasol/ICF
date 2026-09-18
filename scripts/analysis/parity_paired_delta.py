"""Fold-level paired delta between the 5-branch and 7-branch arms.

The task-level comparison already put the 95% cluster interval across zero.
This goes one level down: the same fold, the same slides, both arms, so the
difference is not confounded by which folds happen to be hard.

PROJECT.md SS4.1 measured that folds are not independent within a task, so the
fold-level interval is reported as the precision of the pairing and the task
cluster interval stays the one that decides. Both are printed; neither is
allowed to stand in for the other.
"""

from __future__ import annotations

import math
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.metrics import auroc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["cptac_lscc_ARID1A_mutation", "cptac_lscc_Histologic_Grade",
         "cptac_lscc_KEAP1_mutation", "cptac_luad_KRAS_mutation",
         "cptac_pda_SMAD4_mutation", "ucla_lung_progression_regression",
         "cptac_ccrcc_PBRM1_mutation"]


def fold_aurocs(path: Path) -> dict[int, float]:
    d = torch.load(path, weights_only=False)
    out = {}
    for rec in d["per_fold"]:
        out[int(rec["fold"])] = float(auroc(rec["probability"], rec["label"]))
    return out


def main() -> int:
    all_deltas: list[float] = []
    task_means: list[float] = []
    print(f"{'과제':<34}{'fold':>5}{'5br':>9}{'7br':>9}{'Δ':>10}")
    for task in TASKS:
        a = ROOT / f"predictions/parity_{task}_v121_active.pt"
        b = ROOT / f"predictions/parity_{task}_v121_7branch_active.pt"
        if not (a.exists() and b.exists()):
            print(f"{task:<34}  (예측 파일 없음)")
            continue
        fa, fb = fold_aurocs(a), fold_aurocs(b)
        shared = sorted(set(fa) & set(fb))
        # A fold present in one arm but not the other cannot be paired; say so
        # rather than dropping it quietly.
        missing = (set(fa) ^ set(fb))
        if missing:
            print(f"  [경고] {task}: 짝지어지지 않은 fold {len(missing)}개 제외")
        d = [fb[k] - fa[k] for k in shared]
        all_deltas.extend(d)
        task_means.append(st.mean(d))
        print(f"{task:<34}{len(shared):>5}{st.mean(fa[k] for k in shared):>9.4f}"
              f"{st.mean(fb[k] for k in shared):>9.4f}{st.mean(d):>+10.4f}")

    if not all_deltas:
        print("\n비교할 예측 파일이 없다.")
        return 1

    n = len(all_deltas)
    m = st.mean(all_deltas)
    se_fold = st.stdev(all_deltas) / math.sqrt(n)
    print(f"\nfold 수준 짝지은 Δ  n={n}  평균 {m:+.4f}  sd {st.stdev(all_deltas):.4f}")
    print(f"  95% 구간 (fold를 독립으로 가정 — 과대신뢰): "
          f"[{m - 1.96 * se_fold:+.4f}, {m + 1.96 * se_fold:+.4f}]")
    print("  ^ 이 구간은 판정에 쓰지 않는다. fold는 과제 안에서 독립이 아니다 (PROJECT.md 4.1).")

    k = len(task_means)
    mt = st.mean(task_means)
    se_task = st.stdev(task_means) / math.sqrt(k)
    t = {6: 2.447, 5: 2.571, 4: 2.776}.get(k - 1, 2.447)
    lo, hi = mt - t * se_task, mt + t * se_task
    print(f"\n과제 군집 Δ  k={k}  평균 {mt:+.4f}  sd {st.stdev(task_means):.4f}")
    print(f"  95% 구간 (df={k - 1}): [{lo:+.4f}, {hi:+.4f}]  "
          f"0 포함: {'예' if lo * hi < 0 else '아니오'}")
    print(f"  악화 과제: {sum(1 for x in task_means if x < 0)}/{k}")
    print(f"  승격 임계 +0.0030 대비 하한: {lo / 0.0030:+.1f}배")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
