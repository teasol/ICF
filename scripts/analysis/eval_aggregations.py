"""Score the pre-registered aggregation candidates from saved margins. No GPU.

Reads the margins dumped by dump_branch_margins.py and applies each aggregation
from src/models/aggregations/voting.py -- the production functions, not
reimplementations, because this repository has already been bitten by a second
copy of an aggregation drifting from the first.

Candidates and rule are fixed in talks/reports/2026-09-18_prereg_aggregation.md
and are not read from the command line, so a candidate cannot be added after
seeing a result.
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.aggregations.voting import (  # noqa: E402
    _adaptive_trimmed_from_probs, _hard_gated_from_probs, _trimmed_mean_from_probs,
)
from src.utils.metrics import auroc  # noqa: E402

#: Fixed by the pre-registration. Not configurable.
CANDIDATES = ("trimmed_mean", "soft_voting", "hard_gated", "adaptive_trimmed")
TAU = 0.05            # hard_gated, code default
ADAPTIVE_TAU = 0.08   # adaptive_trimmed, code defaults -- no grid search
ADAPTIVE_RATIO = 1.5
THRESHOLD = 0.003


def aggregate(name: str, margins: dict[str, torch.Tensor]) -> torch.Tensor:
    probs = [torch.sigmoid(m.float()) for _, m in sorted(margins.items())]
    if name == "trimmed_mean":
        return _trimmed_mean_from_probs(probs)
    if name == "soft_voting":
        return torch.stack(probs, dim=-1).mean(dim=-1)
    if name == "hard_gated":
        return _hard_gated_from_probs(probs, TAU)
    if name == "adaptive_trimmed":
        # Production helper, not a reimplementation: a second copy of an
        # aggregation drifting from the first is exactly what the duplication
        # survey found in this repository.
        return _adaptive_trimmed_from_probs(probs, ADAPTIVE_TAU, ADAPTIVE_RATIO)
    raise ValueError(name)


def main() -> int:
    margin_dir = ROOT / "predictions/margins_7b"
    files = sorted(margin_dir.glob("*.pt"))
    if not files:
        print(f"마진 파일이 없다: {margin_dir}")
        return 1
    prov = json.loads((margin_dir / "provenance.json").read_text(encoding="utf-8"))
    print(f"provenance · config {prov['config_sha256']} · code {prov['code_sha256']} "
          f"· git {prov['git']}\n")

    per_task: dict[str, dict[str, float]] = {}
    for f in files:
        blob = torch.load(f, weights_only=False)
        rows: dict[str, list[float]] = {c: [] for c in CANDIDATES}
        for rec in blob["per_fold"]:
            for c in CANDIDATES:
                p = aggregate(c, rec["branch_margins"])
                if torch.isnan(p).any():
                    rows[c].append(float("nan"))
                else:
                    rows[c].append(float(auroc(p, rec["label"])))
        per_task[blob["task"]] = {c: st.mean(v) for c, v in rows.items()}

    base = "trimmed_mean"
    print(f"{'과제':<34}" + "".join(f"{c[:12]:>13}" for c in CANDIDATES))
    for t, r in per_task.items():
        print(f"{t:<34}" + "".join(f"{r[c]:>13.4f}" for c in CANDIDATES))
    print(f"{'macro':<34}" + "".join(
        f"{st.mean(r[c] for r in per_task.values()):>13.4f}" for c in CANDIDATES))

    print(f"\n기준선 = {base} · 승격 임계 +{THRESHOLD}")
    passed = []
    for c in CANDIDATES:
        if c == base:
            continue
        d = [per_task[t][c] - per_task[t][base] for t in per_task]
        m, sd = st.mean(d), st.stdev(d)
        se = sd / math.sqrt(len(d))
        lo, hi = m - 2.447 * se, m + 2.447 * se
        worse = [t for t in per_task if per_task[t][c] - per_task[t][base] < 0]
        ok = m >= THRESHOLD
        if ok:
            passed.append(c)
        print(f"\n{c}: Δ {m:+.4f}  sd {sd:.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"  sign {len(per_task) - len(worse)}/{len(per_task)}  "
              f"{'임계 충족' if ok else '미달'}")
        if worse:
            print("  악화 과제: " + ", ".join(
                f"{t.split('/')[-1]} {per_task[t][c] - per_task[t][base]:+.4f}"
                for t in worse))

    print()
    if len(passed) > 1:
        print(f"판정: 판별 불가 — {len(passed)}개가 임계를 넘었다({', '.join(passed)}). "
              "사전 등록대로 가장 큰 것을 고르지 않는다. 다중 비교 보정 규칙이 필요하다.")
    elif passed:
        print(f"판정: {passed[0]} 승격 후보. 다중 비교 보정은 수행하지 않았다.")
    else:
        print("판정: 넷 다 미달. 집계 축은 이 넷의 범위에서 개선을 주지 않는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
