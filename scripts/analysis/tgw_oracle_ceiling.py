"""TGW go/no-go: the oracle ceiling of per-task branch-family selection (RU-102).

Uses stored per-branch margins (RU-101) to ask how much a *task-adaptive* choice
of branch family could buy, even with query labels -- i.e. an upper bound, not a
method. If that bound is below the promotion threshold (+0.003), no realistic
context-only signal can reach it and TGW should not be started.

  python scripts/analysis/tgw_oracle_ceiling.py \
      --margins predictions/margins_7b \
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

BRANCHES = ("cv", "bm", "bd", "qa", "ds", "sh", "sj")
FAMILIES = ("F7", "F6-SJ", "F6-SH", "F5")


def _agg(cfg, b, use):
    return trimmed_mean(
        cfg, cv=b["cv"], m_cv=b["cv"], m_dd=None, m_ct=None, m_bm=b["bm"],
        m_bd=b["bd"], m_qa=b["qa"], m_ds=b["ds"], m_lr=None, m_de=None,
        m_sw=None,
        m_sj=b["sj"] if use.get("sj") else None,
        m_sh=b["sh"] if use.get("sh") else None,
    )


def _cfg(config_path: Path, sj: bool, sh: bool) -> TrainingFreeConfig:
    return dataclasses.replace(
        TrainingFreeConfig.from_yaml(config_path),
        weight_sj=1.0 if sj else 0.0,
        weight_sh=1.0 if sh else 0.0,
        weight_shj=0.0,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margins", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--baseline-macro", type=float, default=0.6226)
    ap.add_argument("--threshold", type=float, default=0.003)
    args = ap.parse_args()

    configs = {
        "F7": _cfg(args.config, True, True),
        "F6-SJ": _cfg(args.config, False, True),
        "F6-SH": _cfg(args.config, True, False),
        "F5": _cfg(args.config, False, False),
    }
    use = {"F7": {"sh": True, "sj": True}, "F6-SJ": {"sh": True, "sj": False},
           "F6-SH": {"sh": False, "sj": True}, "F5": {"sh": False, "sj": False}}

    per_task = {}
    for f in sorted(glob.glob(str(args.margins / "*.pt"))):
        d = torch.load(f, weights_only=False, map_location="cpu")
        task = d["task"]
        fam = {}
        single = {k: [] for k in BRANCHES}
        for name in FAMILIES:
            acc = []
            for pf in d["per_fold"]:
                b = {k: v.float() for k, v in pf["branch_margins"].items()}
                acc.append(float(auroc(_agg(configs[name], b, use[name]).float(),
                                       pf["label"])))
                if name == "F7":
                    for k in BRANCHES:
                        single[k].append(float(auroc(b[k], pf["label"])))
            fam[name] = st.mean(acc)
        fam["best_single"] = max(st.mean(v) for v in single.values())
        per_task[task] = fam

    # rebuild F7 macro and check it matches the official value
    f7 = st.mean(v["F7"] for v in per_task.values())
    print(f"F7 macro 재구성: {f7:.4f} (공식 {args.baseline_macro})")
    print(f"{'task':<34}" + "".join(f"{k:>9}" for k in FAMILIES) + f"{'best_1':>9}")
    for t, v in per_task.items():
        print(f"{t:<34}" + "".join(f"{v[k]:>9.4f}" for k in FAMILIES)
              + f"{v['best_single']:>9.4f}")

    oracle = st.mean(max(v[k] for k in FAMILIES) for v in per_task.values())
    oracle_single = st.mean(v["best_single"] for v in per_task.values())
    d_fam = oracle - f7
    d_single = oracle_single - f7
    print(f"\noracle family macro = {oracle:.4f}  Δ = {d_fam:+.4f}")
    print(f"oracle best-single macro = {oracle_single:.4f}  Δ = {d_single:+.4f}")
    print(f"승격 임계 +{args.threshold:.3f} | oracle family 기준 "
          f"{'미달 → TGW 착수 보류' if d_fam < args.threshold else '초과 → 설계 진행'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
