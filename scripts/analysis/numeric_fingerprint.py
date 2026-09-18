"""A fingerprint of what the model computes, for checking a refactor changed nothing.

A refactor is only safe here if the numbers come out identical. Tests check
properties; this checks bytes. Synthetic inputs with a fixed seed, so it runs in
seconds on CPU and needs no dataset -- which means it can run before and after a
change without holding the experiment GPU.

  python scripts/analysis/numeric_fingerprint.py --write baseline.json
  python scripts/analysis/numeric_fingerprint.py --check baseline.json

--check exits non-zero on any difference and prints which branch moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.config import TrainingFreeConfig  # noqa: E402
from src.models.training_free import TrainingFreeClassifier  # noqa: E402

CONFIGS = {
    "5branch": dict(weight_cv=1.0, weight_bm=1.0, weight_bd=1.0, weight_qa=1.0,
                    weight_ds=1.0, weight_ct=0.0, weight_dd=0.0),
    "7branch": dict(weight_cv=1.0, weight_bm=1.0, weight_bd=1.0, weight_qa=1.0,
                    weight_ds=1.0, weight_sh=1.0, weight_sj=1.0, weight_ct=0.0,
                    weight_dd=0.0),
    "ct_on": dict(weight_cv=1.0, weight_bm=1.0, weight_bd=1.0, weight_qa=1.0,
                  weight_ds=1.0, weight_ct=1.0, weight_dd=0.0),
}


def fingerprint() -> dict:
    out: dict[str, dict] = {}
    for name, weights in CONFIGS.items():
        torch.manual_seed(20260918)
        ctx = [torch.randn(48, 64) for _ in range(16)]
        lab = torch.tensor([i % 2 for i in range(16)], dtype=torch.long)
        qry = [torch.randn(48, 64) for _ in range(6)]
        cfg = TrainingFreeConfig(sketch_dim=32, aggregation="trimmed_mean", **weights)
        clf = TrainingFreeClassifier(cfg)
        with torch.no_grad():
            m = clf.margins(ctx, lab, qry).double()
            branches = {k: v.double() for k, v in
                        clf.branch_margins(ctx, lab, qry).items()}
        entry = {
            "margins": [f"{float(x):.12e}" for x in m],
            "branches": {k: [f"{float(x):.12e}" for x in v]
                         for k, v in sorted(branches.items())},
        }
        entry["sha256"] = hashlib.sha256(
            json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]
        out[name] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write")
    ap.add_argument("--check")
    args = ap.parse_args()
    got = fingerprint()

    if args.write:
        Path(args.write).write_text(json.dumps(got, indent=2), encoding="utf-8")
        for k, v in got.items():
            print(f"{k:<10} {v['sha256']}")
        print(f"기록: {args.write}")
        return 0

    if args.check:
        want = json.loads(Path(args.check).read_text(encoding="utf-8"))
        bad = False
        for name in sorted(set(want) | set(got)):
            a, b = want.get(name), got.get(name)
            if a is None or b is None:
                print(f"{name:<10} 구성이 한쪽에만 있다"); bad = True; continue
            if a["sha256"] == b["sha256"]:
                print(f"{name:<10} {b['sha256']}  동일")
                continue
            bad = True
            print(f"{name:<10} 달라졌다  {a['sha256']} -> {b['sha256']}")
            for br in sorted(set(a["branches"]) | set(b["branches"])):
                x, y = a["branches"].get(br), b["branches"].get(br)
                if x != y:
                    print(f"   branch {br}: {'사라짐' if y is None else '값 변경'}")
            if a["margins"] != b["margins"]:
                print("   집계 margin 변경")
        return 1 if bad else 0

    for k, v in got.items():
        print(f"{k:<10} {v['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
