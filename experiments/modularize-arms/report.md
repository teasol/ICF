# Experiment report: modularize-arms

## Question

Modularize the arm definitions currently forked across scripts/eval_v114.sh..eval_v120.sh
(24-38 ICF_* exports each) into a single sourced library (scripts/lib/arms.sh) with thin
per-arm wrappers, so that a new arm is a function, not a copied file. Preservation contract:
each rewritten wrapper must export an environment byte-identical to the original script's,
and the rewritten eval_v120.sh must reproduce the v120 Primary-7 macro fold-mean AUROC
within the established 0.6265 +/- 0.005 gate.

## Verdict

**READY TO MERGE** — merging is the researcher's call; the harness only reports.

- Branch: `exp/modularize-arms`
- Merge commit: `0b3b13efdc663ed4e07d7b6296c231c9f6501ff1`
- Integration: PASSED
- Tasks: 1/1 done
- Determinism: not run

## Modules

| Task | Status | Acceptance | Worker |
| --- | --- | --- | --- |
| `arms-lib` | done | passed | worker |

## Requested metrics

| Metric | Value | Source |
| --- | --- | --- |
| env_parity_all_arms | 1 | `${HARNESS_RESULTS_DIR}/parity.json`: `parity.all_identical` |
| env_parity_v120 | 1 | `${HARNESS_RESULTS_DIR}/parity.json`: `parity.arms.v120.identical` |
| env_parity_usage_ok | 1 | `${HARNESS_RESULTS_DIR}/parity.json`: `parity.usage_ok` |
| primary7_macro_fold_mean_auroc | 0.621614 | `${HARNESS_RESULTS_DIR}/metrics.json`: `primary7.macro_fold_mean_auroc` |
| primary7_abs_delta_vs_reference | 0.004886 | `${HARNESS_RESULTS_DIR}/metrics.json`: `primary7.abs_delta_vs_reference` |
| primary7_within_tolerance | 1 | `${HARNESS_RESULTS_DIR}/metrics.json`: `primary7.within_tolerance` |
| arid1a_fold_mean_auroc | 0.5512 | `${HARNESS_RESULTS_DIR}/metrics.json`: `tasks.cptac_lscc/ARID1A_mutation.fold_mean_auroc` |
| histologic_grade_fold_mean_auroc | 0.6789 | `${HARNESS_RESULTS_DIR}/metrics.json`: `tasks.cptac_lscc/Histologic_Grade.fold_mean_auroc` |
| keap1_fold_mean_auroc | 0.6121 | `${HARNESS_RESULTS_DIR}/metrics.json`: `tasks.cptac_lscc/KEAP1_mutation.fold_mean_auroc` |
| kras_fold_mean_auroc | 0.7214 | `${HARNESS_RESULTS_DIR}/metrics.json`: `tasks.cptac_luad/KRAS_mutation.fold_mean_auroc` |
| smad4_fold_mean_auroc | 0.4385 | `${HARNESS_RESULTS_DIR}/metrics.json`: `tasks.cptac_pda/SMAD4_mutation.fold_mean_auroc` |
| progression_fold_mean_auroc | 0.7875 | `${HARNESS_RESULTS_DIR}/metrics.json`: `tasks.ucla_lung/progression_regression.fold_mean_auroc` |
| pbrm1_fold_mean_auroc | 0.5617 | `${HARNESS_RESULTS_DIR}/metrics.json`: `tasks.cptac_ccrcc/PBRM1_mutation.fold_mean_auroc` |

## Artifacts

- `results/modularize-arms/parity.json`
- `results/modularize-arms/metrics.json`
- `results/modularize-arms/manifest.json`

## Not verified

- task file(s) not part of this plan, ignored for this report: ['metrics-extractor', 'primary7-runner']
- determinism not checked (pass --determinism to run the gate)

## Tiers

- Planner: first-planner (GLM-5.3 · high)
- Workers: opencode (deepseek/deepseek-v4-flash · high)

## Provenance

- Python: 3.12.13 (`/home/kimds/miniconda3/envs/BagPFN/bin/python3.12`)
- Platform: Linux-5.15.0-186-generic-x86_64-with-glibc2.35
- Harness: 0.3.3
