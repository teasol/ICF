"""Build slide-level CSV files with OFFICIAL Patho-Bench labels from k=all.tsv.

Reads the official fold-based splits downloaded by ``fetch_pathobench_official.py``
(/NHNHOME/kimds/Data/PathoBench/official/{source}/{task}/k=all.tsv + config.yaml)
and writes, for each task, a slide-level CSV with the official label column
(``config.yaml.task_col``) and a chosen official fold's split:

    {out_dir}/{source}_{task}.csv   # slide_id,label,split  (split = train/val/test)

Also verifies each official task's labels against the corresponding legacy local
CSV (where one exists) to confirm consistency, and reports tasks with no local
counterpart (e.g. real cptac_ccrcc tasks that were never evaluated).

Usage:
    python scripts/build_pathobench_official_csvs.py [--fold 0] [--out-dir ...]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_pathobench_official import SOURCES_TASKS  # noqa: E402

# Legacy local csv stem -> official source/task (for cross-check only).
LEGACY_TO_OFFICIAL: dict[str, tuple[str, str]] = {
    "bc_therapy_er": ("bc_therapy", "er_status"),
    "bc_therapy_grade": ("bc_therapy", "grade"),
    "bc_therapy_her2": ("bc_therapy", "her2_status"),
    "bc_therapy_residual": ("bc_therapy", "residual_cancer_burden"),
    "bracs_coarse": ("bracs", "slidelevel_coarse"),
    "bracs_fine": ("bracs", "slidelevel_fine"),
    "cptac_brca_pik3ca": ("cptac_brca", "PIK3CA_mutation"),
    "cptac_brca_tp53": ("cptac_brca", "TP53_mutation"),
    "cptac_brca_immune": ("cptac_brca", "Immune_class"),
    "cptac_lscc_arid1a": ("cptac_lscc", "ARID1A_mutation"),
    "cptac_lscc_histologic": ("cptac_lscc", "Histologic_Grade"),
    "cptac_lscc_keap1": ("cptac_lscc", "KEAP1_mutation"),
    "cptac_lscc_immune": ("cptac_lscc", "Immune_class"),
    "cptac_luad_egfr": ("cptac_luad", "EGFR_mutation"),
    "cptac_luad_kras": ("cptac_luad", "KRAS_mutation"),
    "cptac_luad_stk11": ("cptac_luad", "STK11_mutation"),
    "cptac_luad_tp53": ("cptac_luad", "TP53_mutation"),
    "cptac_luad_immune": ("cptac_luad", "Immune_class"),
    "cptac_luad_os": ("cptac_luad", "OS"),
    "cptac_pda_smad4": ("cptac_pda", "SMAD4_mutation"),
    "cptac_pda_immune": ("cptac_pda", "Immune_class"),
    "cptac_pda_os": ("cptac_pda", "OS"),
    "mbc_recist": ("mbc_", "Recist"),
    "mbc_os": ("mbc_", "OS"),
    "ucla_lung_progression_regression": ("ucla_lung", "progression_regression"),
    "herroi_response": ("herroi", "response"),
}


def norm(v: str) -> str:
    """Normalize a label string (e.g. '2.0' -> '2') for comparison."""
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return str(v)


def build_task_csv(tsv_path: Path, cfg_path: Path, fold: int) -> list[dict]:
    """Extract slide_id/label/split for the chosen fold from an official k=all.tsv."""
    cfg = yaml.safe_load(cfg_path.read_text())
    task_col = cfg["task_col"]
    header = tsv_path.read_text().split("\n")[0].split("\t")
    if f"fold_{fold}" not in header:
        raise ValueError(f"fold_{fold} not in {tsv_path}")
    rows = []
    with tsv_path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            rows.append(
                {
                    "slide_id": str(r["slide_id"]).strip(),
                    "label": norm(r[task_col]),
                    "split": r[f"fold_{fold}"].strip(),
                }
            )
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["slide_id", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)


def load_local_csv(path: Path) -> dict[str, str]:
    out = {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            out[str(r["slide_id"]).strip()] = norm(r["label"].strip())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-dir",
        type=Path,
        default=Path("/NHNHOME/kimds/Data/PathoBench/official"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/NHNHOME/kimds/Data/PathoBench/csv_official"),
    )
    parser.add_argument("--legacy-dir", type=Path, default=Path("/NHNHOME/kimds/Data/PathoBench/csv"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--source", default=None)
    args = parser.parse_args()

    official = args.official_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    legacy_dir = args.legacy_dir.expanduser().resolve()

    n_built = 0
    for source, tasks in sorted(SOURCES_TASKS.items()):
        if args.source and source != args.source:
            continue
        for task in tasks:
            tsv = official / source / task / "k=all.tsv"
            cfg = official / source / task / "config.yaml"
            if not (tsv.exists() and cfg.exists()):
                print(f"SKIP missing {source}/{task}")
                continue
            rows = build_task_csv(tsv, cfg, args.fold)
            out_csv = out_dir / f"{source}_{task}.csv"
            write_csv(rows, out_csv)
            n_built += 1

            # Cross-check against legacy local CSV (if any).
            legacy_stem = next(
                (s for s, (src, tsk) in LEGACY_TO_OFFICIAL.items()
                 if (src, tsk) == (source, task)),
                None,
            )
            if legacy_stem:
                leg_path = legacy_dir / f"{legacy_stem}.csv"
                if leg_path.exists():
                    leg = load_local_csv(leg_path)
                    off = {r["slide_id"]: r["label"] for r in rows}
                    inter = set(leg) & set(off)
                    lab_ok = sum(1 for s in inter if leg[s] == off[s])
                    print(
                        f"{source}_{task:32s} n={len(rows):4d} fold{args.fold:2d} "
                        f"| legacy {legacy_stem}: overlap={len(inter):4d} label_match={lab_ok}/{len(inter)}"
                    )
                else:
                    print(f"{source}_{task:32s} n={len(rows):4d} fold{args.fold:2d} | no legacy csv")
            else:
                print(f"{source}_{task:32s} n={len(rows):4d} fold{args.fold:2d} | (no legacy counterpart)")

    print(f"\nBuilt {n_built} official-label CSVs -> {out_dir}")


if __name__ == "__main__":
    main()
