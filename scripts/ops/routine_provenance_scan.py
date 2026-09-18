#!/usr/bin/env python3
"""Count prediction artefacts that carry no provenance, as a standing tally.

D-050 found that stored predictions record only task_dir, fold_indices,
fold_aurocs, fold_auroc_mean and per_fold -- no branch list, weights, code hash
or data version -- so no stored result can be attributed to a configuration.
Until the provenance schema lands, this chore keeps the size of that debt
visible instead of letting it be rediscovered.

Cheap by construction: it reads each file's top-level keys via a partial load
and never touches tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED = PROJECT_ROOT / "predictions"
PROVENANCE_KEYS = ("branches", "branch_list", "weights", "code_hash", "git_commit",
                   "data_version", "manifest_hash", "env")


def main() -> None:
    files = sorted(PRED.glob("*.pt"))
    with_prov, without_prov, unreadable = [], [], []
    for f in files:
        try:
            blob = torch.load(f, weights_only=False, map_location="cpu")
        except Exception:  # noqa: BLE001 - an unreadable artefact is itself the finding
            unreadable.append(f.name)
            continue
        keys = set(blob) if isinstance(blob, dict) else set()
        (with_prov if keys & set(PROVENANCE_KEYS) else without_prov).append(f.name)
        del blob

    print(f"predictions/*.pt 총 {len(files)}건")
    print(f"  provenance 있음 : {len(with_prov)}")
    print(f"  provenance 없음 : {len(without_prov)}")
    print(f"  읽기 실패      : {len(unreadable)}")
    if unreadable:
        print("  실패 예:", ", ".join(unreadable[:5]))
    out = PROJECT_ROOT / "talks/ops/provenance_debt.json"
    out.write_text(json.dumps(
        {"total": len(files), "with_provenance": len(with_prov),
         "without_provenance": len(without_prov), "unreadable": unreadable[:50]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"기록: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
