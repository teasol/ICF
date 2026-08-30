# Experiment report: baseline

## Question

Reproduce the v120 baseline (6-branch CV+CT+BM+BD+QA+DS trimmed-mean
voting, training-free, 0 learned parameters) under the harness on the
Primary 7 tasks, and confirm the measured Primary 7 macro fold-mean AUROC
reproduces the recorded value of 0.6265 within +/-0.005.

## Verdict

**NOT READY** — merging is the researcher's call; the harness only reports.

- Branch: `exp/baseline`
- Merge commit: `c776443ce44dc289a39d4b8388ca45a6eadbf5d2`
- Integration: PASSED
- Tasks: 2/2 done
- Determinism: NOT REPRODUCIBLE

### Why not ready

- the experiment is not reproducible

## Modules

| Task | Status | Acceptance | Worker |
| --- | --- | --- | --- |
| `primary7-runner` | done | passed | worker |
| `metrics-extractor` | done | passed | worker |

## Requested metrics

| Metric | Value | Source |
| --- | --- | --- |
| primary7_macro_fold_mean_auroc | 0.621629 | `${HARNESS_RESULTS_DIR}/metrics.json`: `primary7.macro_fold_mean_auroc` |
| primary7_abs_delta_vs_reference | 0.004871 | `${HARNESS_RESULTS_DIR}/metrics.json`: `primary7.abs_delta_vs_reference` |
| primary7_within_tolerance | 1 | `${HARNESS_RESULTS_DIR}/metrics.json`: `primary7.within_tolerance` |
| arid1a_fold_mean_auroc | 0.5512 | `${HARNESS_RESULTS_DIR}/metrics.json`: `tasks.cptac_lscc/ARID1A_mutation.fold_mean_auroc` |
| histologic_grade_fold_mean_auroc | 0.6789 | `${HARNESS_RESULTS_DIR}/metrics.json`: `tasks.cptac_lscc/Histologic_Grade.fold_mean_auroc` |
| keap1_fold_mean_auroc | 0.6121 | `${HARNESS_RESULTS_DIR}/metrics.json`: `tasks.cptac_lscc/KEAP1_mutation.fold_mean_auroc` |
| kras_fold_mean_auroc | 0.7214 | `${HARNESS_RESULTS_DIR}/metrics.json`: `tasks.cptac_luad/KRAS_mutation.fold_mean_auroc` |
| smad4_fold_mean_auroc | 0.4384 | `${HARNESS_RESULTS_DIR}/metrics.json`: `tasks.cptac_pda/SMAD4_mutation.fold_mean_auroc` |
| progression_fold_mean_auroc | 0.7877 | `${HARNESS_RESULTS_DIR}/metrics.json`: `tasks.ucla_lung/progression_regression.fold_mean_auroc` |
| pbrm1_fold_mean_auroc | 0.5617 | `${HARNESS_RESULTS_DIR}/metrics.json`: `tasks.cptac_ccrcc/PBRM1_mutation.fold_mean_auroc` |

## Artifacts

- `results/baseline/metrics.json`
- `results/baseline/manifest.json`

## Not verified

- nondeterministic: metrics.json: run 1 81b5c66251d0… != run 2 06f9da2d3809…

## Tiers

- Planner: first-planner (GLM-5.3)
- Workers: antigravity (gemini-3.7-flash · high)

## Provenance

- Python: 3.12.13 (`/home/kimds/miniconda3/envs/BagPFN/bin/python3.12`)
- Platform: Linux-5.15.0-186-generic-x86_64-with-glibc2.35
- Harness: 0.3.3
