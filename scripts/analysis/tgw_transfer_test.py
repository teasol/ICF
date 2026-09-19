"""RU-103: does the context-only shape-branch signal transfer to query?

Pre-registered rule R1: per fold, a shape branch (sh, sj) keeps weight 1 if its
AUROC on the CONTEXT set itself (context labels, no query labels) is >= 0.5,
else weight 0. Re-aggregate the query margins with that per-fold weighting and
compare to the fixed 7-branch. If the paired macro gain is below +0.003, TGW is
rejected for lack of context->query transfer.

  python scripts/analysis/tgw_transfer_test.py \
      --ctxq predictions/ctxq_7b \
      --config configs/baseline/v121_7branch_active.yaml
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

SHAPE = ("sh", "sj")


def _agg(cfg, m, sh: bool, sj: bool):
    return trimmed_mean(
        cfg, cv=m["cv"], m_cv=m["cv"], m_dd=None, m_ct=None, m_bm=m["bm"],
        m_bd=m["bd"], m_qa=m["qa"], m_ds=m["ds"], m_lr=None, m_de=None,
        m_sw=None, m_sj=m["sj"] if sj else None, m_sh=m["sh"] if sh else None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctxq", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--baseline-macro", type=float, default=0.6226)
    ap.add_argument("--threshold", type=float, default=0.003)
    args = ap.parse_args()

    cfg7 = TrainingFreeConfig.from_yaml(args.config)
    rows = {}
    f7_all = []
    for f in sorted(glob.glob(str(args.ctxq / "*.pt"))):
        d = torch.load(f, weights_only=False, map_location="cpu")
        a_f7, a_r1, dropped = [], [], {k: 0 for k in SHAPE}
        n = 0
        for pf in d["per_fold"]:
            n += 1
            cl, cm = pf["context_label"], {k: v.float() for k, v in pf["context_margins"].items()}
            ql, qm = pf["query_label"], {k: v.float() for k, v in pf["query_margins"].items()}
            f7 = _agg(cfg7, qm, True, True)
            a_f7.append(float(auroc(f7.float(), ql)))
            use = {}
            for k in SHAPE:
                try:
                    a = float(auroc(cm[k], cl))
                except ValueError:
                    a = 0.5
                use[k] = a >= 0.5
                if not use[k]:
                    dropped[k] += 1
            cfg1 = dataclasses.replace(cfg7, weight_sh=1.0 if use["sh"] else 0.0,
                                       weight_sj=1.0 if use["sj"] else 0.0,
                                       weight_shj=0.0)
            r1 = _agg(cfg1, qm, use["sh"], use["sj"])
            a_r1.append(float(auroc(r1.float(), ql)))
        rows[d["task"]] = (st.mean(a_f7), st.mean(a_r1), n,
                           {k: dropped[k] / n for k in SHAPE})
        f7_all.append(st.mean(a_f7))

    print(f"{'task':<34}{'F7':>8}{'R1':>8}{'Δ':>9}  drop(sh/sj)")
    for t, (m7, m1, n, dr) in rows.items():
        print(f"{t:<34}{m7:>8.4f}{m1:>8.4f}{m1 - m7:>+9.4f}  "
              f"{dr['sh']:.2f}/{dr['sj']:.2f}")
    f7 = st.mean(v[0] for v in rows.values())
    r1 = st.mean(v[1] for v in rows.values())
    print(f"\nF7 macro 재구성 {f7:.4f} (공식 {args.baseline_macro})")
    print(f"R1 macro {r1:.4f}  Δ = {r1 - f7:+.4f}")
    print(f"판정: {'전이 확인' if r1 - f7 >= args.threshold else '전이 실패 → TGW 기각'} "
          f"(임계 +{args.threshold:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
