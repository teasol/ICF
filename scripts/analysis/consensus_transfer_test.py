"""RU-104: label-free context-consensus signal transfer test.

Pre-registered rule R2: per fold, compare each shape branch (sh, sj) on the
CONTEXT set against the base-5 trimmed-mean consensus (cv,bm,bd,qa,ds). If their
correlation is < 0, drop that shape branch (weight 0); else keep it. Re-aggregate
the query with that per-fold weighting and compare to the fixed 7-branch. Labels
are never used in the signal.

  python scripts/analysis/consensus_transfer_test.py \
      --ctxq predictions/ctxq_7b --config configs/baseline/v121_7branch_active.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import statistics as st
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.config import TrainingFreeConfig  # noqa: E402
from src.models.aggregations.voting import trimmed_mean  # noqa: E402
from src.utils.metrics import auroc  # noqa: E402

BASE = ("cv", "bm", "bd", "qa", "ds")
SHAPE = ("sh", "sj")


def _agg(cfg, m, sh: bool, sj: bool):
    return trimmed_mean(
        cfg, cv=m["cv"], m_cv=m["cv"], m_dd=None, m_ct=None, m_bm=m["bm"],
        m_bd=m["bd"], m_qa=m["qa"], m_ds=m["ds"], m_lr=None, m_de=None,
        m_sw=None, m_sj=m["sj"] if sj else None, m_sh=m["sh"] if sh else None,
    )


def _base_consensus(cfg5, cm):
    return torch.sigmoid(_agg(cfg5, cm, False, False).float())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctxq", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--baseline-macro", type=float, default=0.6226)
    ap.add_argument("--threshold", type=float, default=0.003)
    args = ap.parse_args()

    cfg7 = TrainingFreeConfig.from_yaml(args.config)
    cfg5 = dataclasses.replace(cfg7, weight_sh=0.0, weight_sj=0.0, weight_shj=0.0)
    rows = {}
    for f in sorted(glob.glob(str(args.ctxq / "*.pt"))):
        d = torch.load(f, weights_only=False, map_location="cpu")
        a7, a_r2, dropped = [], [], {k: 0 for k in SHAPE}
        corrs = {k: [] for k in SHAPE}
        n = 0
        for pf in d["per_fold"]:
            n += 1
            cm = {k: v.float() for k, v in pf["context_margins"].items()}
            qm = {k: v.float() for k, v in pf["query_margins"].items()}
            ql = pf["query_label"]
            base = _base_consensus(cfg5, cm)
            use = {}
            for k in SHAPE:
                sp = torch.sigmoid(cm[k])
                if sp.std() == 0 or base.std() == 0:
                    c = 0.0
                else:
                    c = float(torch.corrcoef(torch.stack([sp, base]))[0, 1])
                corrs[k].append(c)
                use[k] = c >= 0  # R2: corr < 0 -> drop
                if not use[k]:
                    dropped[k] += 1
            a7.append(float(auroc(_agg(cfg7, qm, True, True).float(), ql)))
            cfg2 = dataclasses.replace(cfg7, weight_sh=1.0 if use["sh"] else 0.0,
                                       weight_sj=1.0 if use["sj"] else 0.0,
                                       weight_shj=0.0)
            a_r2.append(float(auroc(_agg(cfg2, qm, use["sh"], use["sj"]).float(), ql)))
        rows[d["task"]] = (st.mean(a7), st.mean(a_r2), n,
                           {k: dropped[k] / n for k in SHAPE},
                           {k: st.mean(corrs[k]) for k in SHAPE},)

    print(f"{'task':<34}{'F7':>8}{'R2':>8}{'Δ':>9}  drop(sh/sj)  ctxcorr(sh/sj)")
    for t, (m7, m2, n, dr, co) in rows.items():
        print(f"{t:<34}{m7:>8.4f}{m2:>8.4f}{m2 - m7:>+9.4f}  "
              f"{dr['sh']:.2f}/{dr['sj']:.2f}      {co['sh']:+.2f}/{co['sj']:+.2f}")
    f7 = st.mean(v[0] for v in rows.values())
    r2 = st.mean(v[1] for v in rows.values())
    total_drop = sum(v[3]["sh"] + v[3]["sj"] for v in rows.values())
    print(f"\nF7 macro 재구성 {f7:.4f} (공식 {args.baseline_macro})  R2 macro {r2:.4f}  "
          f"Δ = {r2 - f7:+.4f}")
    if total_drop == 0:
        print("판정: 판별 불가 — R2가 한 번도 발동하지 않았다(퇴화).")
    else:
        print(f"판정: {'전이 확인' if r2 - f7 >= args.threshold else '전이 실패'}"
              f" (임계 +{args.threshold:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
