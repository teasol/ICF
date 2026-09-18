# scripts/ 미참조 파일 목록 (deadcode)

이 문서는 `scripts/` 아래 `*.py`·`*.sh` 125개를 **참조 관계만으로** 분류한 목록이다.
파일을 삭제·이동·수정하지 않았으며, `고아`는 "삭제 대상"이 아니라 "현재 어디에서도 이름이 참조되지 않음"을 뜻한다.
과거 실행 재현용 스크립트가 많으므로 `고아`도 보존 대상일 수 있다.

## 방법

- `scripts/archive/` 디렉터리는 존재하지 않아 `보관됨` 분류는 0건이다.
- 각 파일의 stem(확장자 제외 이름)과 파일명(확장자 포함)을 저장소 전체 파일에서 검색했다.
- 자기 자신의 파일은 참조에서 제외했다.
- `사용중`: 다른 `.py`/`.sh`가 import·호출하거나, `talks/ops/recurring/*.json`의 `cmd`가 실행한다.
- `문서만`: 코드 호출은 없고 `docs/`, `talks/`, README, 설정 주석 등 문서에서만 이름이 언급된다.
- `고아`: 자기 자신 외에는 어떤 파일에서도 이름이 검색되지 않는다.
- `참조 수`는 해당 이름을 언급한 **서로 다른 파일의 수**(자기 자신 제외)다.

합계 125개 — 사용중 50, 문서만 65, 고아 10, 보관됨 0

| 파일 | 참조 수 | 참조하는 곳 | 분류 |
|---|---:|---|---|
| `scripts/__init__.py` | 31 | 패키지 마커 — `from scripts.…`/`import scripts.…` 임포트에 필요. 31개 `.py`가 `scripts` 패키지를 임포트한다(예: `scripts/evaluate_pure.py`, `tests/test_gf_background_gmm.py`). stem `__init__` 문자열 검색은 메서드 정의를 오탐하므로 이 행은 임포트 구문 기준으로 세었다. | 사용중 |
| `scripts/analysis/analyze_voting.py` | 2 | 코드: `scripts/analysis/compare_all_voting.py`<br>문서: `docs/history/research_units_all_manifest.json` | 사용중 |
| `scripts/analysis/branch_diagnostics.py` | 14 | 코드: `scripts/analysis/branch_redundancy.py`, `scripts/analysis/branch_screen.py`, `scripts/analysis/branch_significance.py`, `scripts/analysis/loo_capacity.py`, `scripts/analysis/ru89_shape_joint.py`, `scripts/analysis/ru90_wiring_equivalence.py`, `scripts/analysis/subsample_selector.py`, `scripts/analysis/subsample_sweep.py`<br>문서: `docs/agent_handoff.md`, `docs/history/archive.md`, `docs/history/research_units_all.json` 등 6곳 | 사용중 |
| `scripts/analysis/branch_redundancy.py` | 2 | 코드: `scripts/ops/gpu_busy.py`, `scripts/ops/turn_end.sh` | 사용중 |
| `scripts/analysis/branch_screen.py` | 45 | 코드: `scripts/analysis/ru89_shape_joint.py`, `scripts/run_and_wait.sh`<br>문서: `docs/PROJECT.md`, `docs/agent_handoff.md`, `docs/current_architecture.md` 등 43곳 | 사용중 |
| `scripts/analysis/branch_significance.py` | 2 | 문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/analysis/compare_all_voting.py` | 1 | 문서: `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/analysis/compare_arms_paired.py` | 5 | 문서: `configs/archive/v77_hard_orthogonal/train_v77_hard_orthogonal_1536.yaml`, `configs/archive/v80_v82_seed_batch/train_v82_medium_classsep_1536_1gpu.yaml`, `docs/history/history.md` 등 5곳 | 문서만 |
| `scripts/analysis/compare_predictions.py` | 2 | 코드: `src/utils/metrics.py`<br>문서: `docs/history/research_units_all_manifest.json` | 사용중 |
| `scripts/analysis/context_weighting.py` | 4 | 문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json`, `requirements.txt` 등 4곳 | 문서만 |
| `scripts/analysis/decision_precision.py` | 3 | 문서: `docs/history/archive.md`, `docs/history/research_units_all.json`, `docs/history/research_units_all.md` | 문서만 |
| `scripts/analysis/drop2_furthest_detail.py` | 9 | 코드: `scripts/analysis/analyze_voting.py`, `scripts/analysis/branch_diagnostics.py`, `scripts/analysis/decision_precision.py`, `scripts/analysis/eval_gated_voting.py`, `scripts/analysis/parse_ds_aug_results.py`, `scripts/analysis/parse_v121_results.py`<br>문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json`, `scripts/analysis/README.md` | 사용중 |
| `scripts/analysis/ds_branch_probe.py` | 2 | 문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/analysis/eval_combinations.py` | 1 | 문서: `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/analysis/eval_gated_voting.py` | 2 | 문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/analysis/eval_in_episode_loo.py` | 2 | 문서: `docs/history/research_units_all_manifest.json`, `talks/rfc/RFC_2026-09-12_pure_runner_migration_and_legacy_removal.md` | 문서만 |
| `scripts/analysis/eval_pure_in_episode_weighting.py` | 1 | 문서: `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/analysis/fisher_basis_probe.py` | 2 | 문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/analysis/furthest_trimming.py` | 3 | 문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json`, `scripts/analysis/README.md` | 문서만 |
| `scripts/analysis/gf_collate.py` | 9 | 문서: `talks/council/C-20260917-2/blackboard.json`, `talks/council/C-20260917-2/report.md`, `talks/council/C-20260917-4/blackboard.json` 등 9곳 | 문서만 |
| `scripts/analysis/gf_paired_delta.py` | 0 | — | 고아 |
| `scripts/analysis/gf_variance_decomp.py` | 0 | — | 고아 |
| `scripts/analysis/kernel_ridge_probe.py` | 5 | 문서: `docs/history/archive.md`, `docs/history/research_units_all.json`, `docs/history/research_units_all.md` 등 5곳 | 문서만 |
| `scripts/analysis/loo_capacity.py` | 4 | 문서: `docs/history/archive.md`, `docs/history/research_units_all.json`, `docs/history/research_units_all.md` 등 4곳 | 문서만 |
| `scripts/analysis/loo_formula.py` | 2 | 문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/analysis/parity_paired_delta.py` | 1 | 문서: `talks/reports/2026-09-18_arm_parity.md` | 문서만 |
| `scripts/analysis/parse_de_sw_results.py` | 1 | 문서: `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/analysis/parse_ds_aug_results.py` | 3 | 코드: `scripts/run_ds_aug_experiments.sh`, `scripts/run_ds_auto_loo_experiments.sh`<br>문서: `docs/history/research_units_all_manifest.json` | 사용중 |
| `scripts/analysis/parse_v121_results.py` | 1 | 문서: `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/analysis/patch_likelihood.py` | 2 | 문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/analysis/probe_slot_headroom.py` | 5 | 코드: `scripts/analysis/summarize_slot_headroom.py`, `tests/test_precision_contract.py`<br>문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json`, `talks/rfc/RFC_2026-09-12_pure_runner_migration_and_legacy_removal.md` | 사용중 |
| `scripts/analysis/profile_loo_fold.py` | 2 | 문서: `docs/history/research_units_all_manifest.json`, `talks/rfc/RFC_2026-09-12_pure_runner_migration_and_legacy_removal.md` | 문서만 |
| `scripts/analysis/ru81_audit.py` | 3 | 문서: `docs/history/research_units_all.json`, `docs/history/research_units_all.md`, `docs/reports/RU-81_regularization_diagnosis.md` | 문서만 |
| `scripts/analysis/ru81_launch.py` | 5 | 코드: `scripts/analysis/ru81_report.py`, `scripts/analysis/ru82_lambda_ensemble.py`, `scripts/analysis/ru83_rank_size.py`, `scripts/analysis/ru84_gram_diagonal.py`, `scripts/run_ru81.sh` | 사용중 |
| `scripts/analysis/ru81_probe.py` | 5 | 코드: `scripts/analysis/ru81_report.py`, `scripts/analysis/ru81_worker.py`, `scripts/analysis/ru82_lambda_ensemble.py`, `scripts/analysis/ru83_rank_size.py`<br>문서: `talks/rfc/RFC_2026-09-12_dataset_and_training_code_removal.md` | 사용중 |
| `scripts/analysis/ru81_report.py` | 0 | — | 고아 |
| `scripts/analysis/ru81_worker.py` | 2 | 코드: `scripts/analysis/ru81_launch.py`<br>문서: `talks/rfc/RFC_2026-09-12_pure_runner_migration_and_legacy_removal.md` | 사용중 |
| `scripts/analysis/ru82_lambda_ensemble.py` | 3 | 문서: `docs/history/research_units_all.json`, `docs/history/research_units_all.md`, `docs/reports/RU-82_lambda_ensemble.md` | 문서만 |
| `scripts/analysis/ru83_rank_size.py` | 3 | 문서: `docs/history/research_units_all.json`, `docs/history/research_units_all.md`, `docs/reports/RU-83_rank_size_separation.md` | 문서만 |
| `scripts/analysis/ru84_gram_diagonal.py` | 3 | 문서: `docs/history/research_units_all.json`, `docs/history/research_units_all.md`, `docs/reports/RU-84_gram_diagonal_dominance.md` | 문서만 |
| `scripts/analysis/ru85_gate1.py` | 4 | 문서: `docs/history/research_units_all.json`, `docs/history/research_units_all.md`, `docs/reports/RU-85_tier1_gate1.md` 등 4곳 | 문서만 |
| `scripts/analysis/ru86_gate2.py` | 4 | 코드: `scripts/analysis/ru88_fingerprint.py`<br>문서: `docs/history/research_units_all.json`, `docs/history/research_units_all.md`, `docs/reports/RU-86_mdx_gate2.md` | 사용중 |
| `scripts/analysis/ru87_precision.py` | 4 | 문서: `docs/history/research_units_all.json`, `docs/history/research_units_all.md`, `docs/reports/RU-87_precision_bf16_fp32.md` 등 4곳 | 문서만 |
| `scripts/analysis/ru88_fingerprint.py` | 4 | 문서: `docs/history/research_units_all.json`, `docs/history/research_units_all.md`, `docs/reports/RU-88_car1_fingerprint_panel.md` 등 4곳 | 문서만 |
| `scripts/analysis/ru89_shape_joint.py` | 3 | 코드: `scripts/analysis/ru90_wiring_equivalence.py`<br>문서: `docs/history/research_units_all.json`, `docs/history/research_units_all.md` | 사용중 |
| `scripts/analysis/ru90_wiring_equivalence.py` | 3 | 문서: `docs/history/archive.md`, `docs/history/research_units_all.json`, `docs/history/research_units_all.md` | 문서만 |
| `scripts/analysis/search_optimal_combinations.py` | 1 | 문서: `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/analysis/subsample_selector.py` | 1 | 문서: `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/analysis/subsample_sweep.py` | 2 | 문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/analysis/summarize_slot_headroom.py` | 2 | 문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/analysis/test_saved_logits_weighting.py` | 2 | 문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/baseline/run_primary7_v120.sh` | 0 | — | 고아 |
| `scripts/call_agent.sh` | 4 | 문서: `CLAUDE.md`, `docs/agent_handoff.md`, `docs/history/archive.md` 등 4곳 | 문서만 |
| `scripts/council.py` | 27 | 코드: `scripts/ops/check_supervisor.sh`, `scripts/ops/gpu_busy.py`, `scripts/ops/routine_open_questions.py`, `scripts/ops/routine_opencode.sh`, `scripts/ops/run_round.sh`, `scripts/ops/seat_thinking.py`, `scripts/ops/supervisor.py`, `src/council/__init__.py` 등 10개<br>문서: `AGENTS.md`, `docs/SESSION_HANDOFF.md`, `docs/current_status.md` 등 17곳 | 사용중 |
| `scripts/data/build_gf_background_gmm.py` | 6 | 코드: `scripts/run_gf_standalone.py`, `src/models/branches/gf.py`, `tests/test_gf_background_gmm.py`<br>문서: `docs/current_architecture.md`, `talks/council/C-20260917-1/blackboard.json`, `talks/council/C-20260917-1/report.md` | 사용중 |
| `scripts/data/build_gf_whitened_gmm.py` | 1 | 코드: `scripts/run_gf_standalone.py` | 사용중 |
| `scripts/data/build_pathobench_official_csvs.py` | 1 | 문서: `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/data/fetch_pathobench_official.py` | 2 | 코드: `scripts/data/build_pathobench_official_csvs.py`<br>문서: `docs/history/research_units_all_manifest.json` | 사용중 |
| `scripts/data/prepare_pathobench.py` | 3 | 문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json`, `talks/rfc/RFC_2026-09-12_pure_runner_migration_and_legacy_removal.md` | 문서만 |
| `scripts/diagnostics/diagnose_branch_contributions.py` | 3 | 문서: `docs/history/history.md`, `docs/history/research_units_all_manifest.json`, `talks/rfc/RFC_2026-09-12_dataset_and_training_code_removal.md` | 문서만 |
| `scripts/diagnostics/diagnose_covariance_sketch.py` | 4 | 코드: `scripts/diagnostics/diagnose_sketch_basis_variance.py`, `scripts/diagnostics/diagnose_sketch_qmc.py`<br>문서: `docs/history/research_units_all_manifest.json`, `talks/rfc/RFC_2026-09-12_pure_runner_migration_and_legacy_removal.md` | 사용중 |
| `scripts/diagnostics/diagnose_ct_kmeans.py` | 1 | 문서: `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/diagnostics/diagnose_ct_pca_distance.py` | 2 | 문서: `docs/history/history.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/diagnostics/diagnose_ct_readout.py` | 1 | 문서: `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/diagnostics/diagnose_cv_basis.py` | 3 | 문서: `docs/history/history.md`, `docs/history/research_units_all_manifest.json`, `talks/rfc/RFC_2026-09-12_dataset_and_training_code_removal.md` | 문서만 |
| `scripts/diagnostics/diagnose_cv_correlation_scale.py` | 2 | 문서: `docs/history/history.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/diagnostics/diagnose_dd_rank_tstat.py` | 2 | 문서: `docs/history/history.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/diagnostics/diagnose_dd_relative.py` | 2 | 문서: `docs/history/history.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/diagnostics/diagnose_full_basis.py` | 9 | 코드: `scripts/diagnostics/diagnose_ct_kmeans.py`, `scripts/diagnostics/diagnose_ct_pca_distance.py`, `scripts/diagnostics/diagnose_ct_readout.py`, `scripts/diagnostics/diagnose_cv_correlation_scale.py`, `scripts/diagnostics/diagnose_dd_rank_tstat.py`, `scripts/diagnostics/diagnose_dd_relative.py`, `scripts/diagnostics/diagnose_sketch_dim_scale.py`<br>문서: `docs/history/history.md`, `docs/history/research_units_all_manifest.json` | 사용중 |
| `scripts/diagnostics/diagnose_population_routing.py` | 3 | 문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json`, `talks/rfc/RFC_2026-09-12_pure_runner_migration_and_legacy_removal.md` | 문서만 |
| `scripts/diagnostics/diagnose_sketch_basis_variance.py` | 1 | 문서: `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/diagnostics/diagnose_sketch_dim_scale.py` | 2 | 문서: `docs/history/history.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/diagnostics/diagnose_sketch_qmc.py` | 1 | 문서: `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/diagnostics/diagnose_synthetic_vs_real.py` | 9 | 문서: `configs/archive/v94_v102_cell_value/train_v102_tail_bagshared_1536_1gpu.yaml`, `configs/archive/v94_v102_cell_value/train_v94_p1_bagcoherence_1536_1gpu.yaml`, `configs/archive/v94_v102_cell_value/train_v95_p3_spectral_1536_1gpu.yaml` 등 9곳 | 문서만 |
| `scripts/docs/ru.py` | 22 | 코드: `scripts/analysis/ru81_audit.py`, `scripts/analysis/ru81_launch.py`, `scripts/analysis/ru82_lambda_ensemble.py`, `scripts/analysis/ru83_rank_size.py`, `scripts/analysis/ru84_gram_diagonal.py`, `scripts/analysis/ru86_gate2.py`, `scripts/analysis/ru87_precision.py`, `scripts/analysis/ru88_fingerprint.py` 등 10개<br>문서: `docs/README.md`, `docs/agent_handoff.md`, `docs/history/archive.md` 등 12곳 | 사용중 |
| `scripts/eval_ct_alone.sh` | 6 | 코드: `scripts/eval_seal_tasks.sh`<br>문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json`, `talks/council/C-20260918-20/blackboard.json` 등 5곳 | 사용중 |
| `scripts/eval_ds_aug.sh` | 9 | 코드: `scripts/eval_seal_tasks.sh`, `scripts/run_ds_aug_experiments.sh`, `scripts/run_ds_auto_loo_experiments.sh`, `scripts/run_ds_salience_anchor.sh`<br>문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json`, `talks/council/C-20260918-20/blackboard.json` 등 5곳 | 사용중 |
| `scripts/eval_dual16_top3.py` | 3 | 문서: `docs/history/research_units_all_manifest.json`, `talks/rfc/RFC_2026-09-12_dataset_and_training_code_removal.md`, `talks/rfc/RFC_2026-09-12_pure_runner_migration_and_legacy_removal.md` | 문서만 |
| `scripts/eval_seal_tasks.sh` | 36 | 코드: `scripts/diagnostics/diagnose_ct_pca_distance.py`, `scripts/diagnostics/diagnose_ct_readout.py`, `scripts/diagnostics/diagnose_cv_basis.py`, `scripts/diagnostics/diagnose_full_basis.py`, `scripts/eval_ct_alone.sh`, `scripts/eval_ds_aug.sh`, `scripts/eval_v120.sh`, `scripts/eval_v121.sh` 등 12개<br>문서: `configs/archive/v86_v93_episode_shape/train_v90_class_prior_1536_1gpu.yaml`, `docs/PROJECT.md`, `docs/current_status.md` 등 24곳 | 사용중 |
| `scripts/eval_v120.sh` | 8 | 코드: `scripts/baseline/run_primary7_v120.sh`, `scripts/eval_seal_tasks.sh`, `scripts/run_v120_seal_multi_gpu.sh`<br>문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json`, `talks/council/C-20260918-20/blackboard.json` 등 5곳 | 사용중 |
| `scripts/eval_v121.sh` | 35 | 코드: `scripts/analysis/ru86_gate2.py`, `scripts/eval_seal_tasks.sh`, `scripts/run_ds_subsample_compare.sh`, `scripts/run_ds_subsample_sweep.sh`, `scripts/run_ru85_tier1_screen.sh`, `scripts/run_ru87_precision.sh`, `scripts/run_ru90_live_arms.sh`, `scripts/run_ru90_shape_triple.sh` 등 13개<br>문서: `docs/PROJECT.md`, `docs/agent_handoff.md`, `docs/current_status.md` 등 22곳 | 사용중 |
| `scripts/evaluate_pure.py` | 39 | 코드: `scripts/analysis/branch_redundancy.py`, `scripts/analysis/eval_in_episode_loo.py`, `scripts/analysis/probe_slot_headroom.py`, `scripts/analysis/profile_loo_fold.py`, `scripts/data/prepare_pathobench.py`, `scripts/diagnostics/diagnose_covariance_sketch.py`, `scripts/diagnostics/diagnose_population_routing.py`, `scripts/eval_dual16_top3.py` 등 13개<br>문서: `docs/PROJECT.md`, `docs/agent_handoff.md`, `docs/current_architecture.md` 등 26곳 | 사용중 |
| `scripts/inbox.py` | 10 | 코드: `tests/test_inbox.py`<br>문서: `docs/history/archive.md`, `docs/proposals/2026-09-11_tiranos_mail_rfc_import.md`, `talks/archive/inbox/lime/2026-09-12_1300_from_platform_phase_a_0_및_a_1_실행_7_branch_yaml_작성_및_순수.md` 등 9곳 | 사용중 |
| `scripts/lib/arms.sh` | 54 | 코드: `scripts/analysis/compare_arms_paired.py`, `scripts/analysis/parity_paired_delta.py`, `scripts/analysis/parse_ds_aug_results.py`, `scripts/analysis/parse_v121_results.py`, `scripts/analysis/ru83_rank_size.py`, `scripts/analysis/ru87_precision.py`, `scripts/analysis/ru88_fingerprint.py`, `scripts/analysis/subsample_sweep.py` 등 23개<br>문서: `configs/archive/v103_v105_head_proj/train_v105_mlpproj_h128_1536_1gpu.yaml`, `configs/archive/v35_v39_pre_cvonly/train_v36_q1_baseline_1536.yaml`, `configs/archive/v35_v39_pre_cvonly/train_v36_q1_structured_1536.yaml` 등 31곳 | 사용중 |
| `scripts/lib/arms_env_check.py` | 0 | — | 고아 |
| `scripts/lib/free_gpus.sh` | 6 | 코드: `scripts/run_ru90_live_arms.sh`, `scripts/run_ru90_shape_triple.sh`, `scripts/run_seal10_evaluation.py`<br>문서: `docs/history/archive.md`, `docs/history/research_units_all.json`, `docs/history/research_units_all.md` | 사용중 |
| `scripts/node_env.sh` | 30 | 코드: `scripts/analysis/branch_redundancy.py`, `scripts/analysis/probe_slot_headroom.py`, `scripts/analysis/summarize_slot_headroom.py`, `scripts/baseline/run_primary7_v120.sh`, `scripts/diagnostics/diagnose_population_routing.py`, `scripts/eval_ct_alone.sh`, `scripts/eval_ds_aug.sh`, `scripts/eval_seal_tasks.sh` 등 15개<br>문서: `.gitignore`, `docs/README.md`, `docs/agent_handoff.md` 등 15곳 | 사용중 |
| `scripts/ops/__init__.py` | 1 | 패키지 마커 — `tests/test_ops_supervisor.py`의 `import scripts.ops.supervisor`에 필요. stem `__init__` 문자열 검색은 메서드 정의를 오탐하므로 임포트 구문 기준으로 세었다. | 사용중 |
| `scripts/ops/check_supervisor.sh` | 2 | 코드: `scripts/ops/run_round.sh`, `scripts/ops/turn_end.sh` | 사용중 |
| `scripts/ops/gpu_busy.py` | 2 | 코드: `scripts/ops/supervisor.py`, `scripts/ops/turn_end.sh` | 사용중 |
| `scripts/ops/lit_digest.py` | 0 | — | 고아 |
| `scripts/ops/lit_fetch.sh` | 0 | — | 고아 |
| `scripts/ops/lit_index.py` | 2 | 실행설정: `talks/ops/recurring/30_lit_index.json`<br>문서: `talks/lit/INDEX.md` | 사용중 |
| `scripts/ops/llm_env.py` | 4 | 코드: `scripts/ops/lit_digest.py`, `scripts/ops/routine_open_questions.py`, `scripts/ops/seat_thinking.py`, `scripts/ops/supervisor.py` | 사용중 |
| `scripts/ops/routine_open_questions.py` | 1 | 실행설정: `talks/ops/recurring/20_open_questions.json` | 사용중 |
| `scripts/ops/routine_opencode.sh` | 0 | — | 고아 |
| `scripts/ops/routine_provenance_scan.py` | 1 | 실행설정: `talks/ops/recurring/22_provenance_scan.json` | 사용중 |
| `scripts/ops/run_round.sh` | 2 | 문서: `docs/SESSION_HANDOFF.md`, `docs/current_status.md` | 문서만 |
| `scripts/ops/seat_thinking.py` | 1 | 실행설정: `talks/ops/recurring/10_seat_thinking.json` | 사용중 |
| `scripts/ops/supervisor.py` | 6 | 코드: `scripts/ops/check_supervisor.sh`, `scripts/ops/gpu_busy.py`, `scripts/ops/run_round.sh`, `scripts/ops/turn_end.sh`, `tests/test_ops_supervisor.py`<br>문서: `.gitignore` | 사용중 |
| `scripts/ops/turn_end.sh` | 2 | 문서: `docs/SESSION_HANDOFF.md`, `docs/current_status.md` | 문서만 |
| `scripts/run_and_wait.sh` | 3 | 문서: `docs/agent_handoff.md`, `docs/history/archive.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/run_ds_aug_experiments.sh` | 1 | 문서: `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/run_ds_auto_loo_experiments.sh` | 1 | 문서: `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/run_ds_salience_anchor.sh` | 1 | 문서: `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/run_ds_subsample_compare.sh` | 1 | 문서: `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/run_ds_subsample_sweep.sh` | 4 | 코드: `scripts/analysis/ru88_fingerprint.py`<br>문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json`, `docs/reports/RU-88_car1_fingerprint_panel.md` | 사용중 |
| `scripts/run_gf_standalone.py` | 4 | 코드: `scripts/analysis/gf_variance_decomp.py`<br>문서: `docs/current_architecture.md`, `docs/history/research_units_all.json`, `docs/history/research_units_all.md` | 사용중 |
| `scripts/run_primary7_parity.py` | 1 | 문서: `docs/reports/RU-91_pure_runner_primary7_parity.md` | 문서만 |
| `scripts/run_ru81.sh` | 0 | — | 고아 |
| `scripts/run_ru85_tier1_screen.sh` | 4 | 코드: `scripts/analysis/ru86_gate2.py`<br>문서: `docs/history/research_units_all.json`, `docs/history/research_units_all.md`, `docs/reports/RU-85_tier1_gate1.md` | 사용중 |
| `scripts/run_ru87_precision.sh` | 5 | 코드: `scripts/analysis/ru87_precision.py`<br>문서: `docs/history/research_units_all.json`, `docs/history/research_units_all.md`, `docs/reports/RU-87_precision_bf16_fp32.md` 등 4곳 | 사용중 |
| `scripts/run_ru90_live_arms.sh` | 3 | 문서: `docs/history/archive.md`, `docs/history/research_units_all.json`, `docs/history/research_units_all.md` | 문서만 |
| `scripts/run_ru90_shape_triple.sh` | 11 | 코드: `scripts/evaluate_pure.py`<br>문서: `docs/history/archive.md`, `docs/history/research_units_all.json`, `docs/history/research_units_all.md` 등 10곳 | 사용중 |
| `scripts/run_ru91b_legacy_32true.py` | 0 | — | 고아 |
| `scripts/run_seal10_evaluation.py` | 1 | 문서: `docs/reports/RU-92_seal10_v121_vs_v120_evaluation.md` | 문서만 |
| `scripts/run_tests.sh` | 23 | 코드: `tests/test_gf_branch.py`<br>실행설정: `talks/ops/recurring/23_regression.json`<br>문서: `docs/agent_handoff.md`, `docs/current_status.md`, `docs/history/archive.md` 등 21곳 | 사용중 |
| `scripts/run_v120_clean_loo_experiments.sh` | 6 | 코드: `scripts/eval_seal_tasks.sh`<br>문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json`, `talks/council/C-20260918-20/blackboard.json` 등 5곳 | 사용중 |
| `scripts/run_v120_seal_multi_gpu.sh` | 3 | 문서: `docs/agent_handoff.md`, `docs/history/archive.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/run_v121_loo_capacity.sh` | 2 | 문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/run_v121_primary7.sh` | 2 | 문서: `docs/history/research_units_all_manifest.json`, `docs/reports/RU-95_bagsize_sweep_primary7.md` | 문서만 |
| `scripts/run_v121_rm_screen.sh` | 2 | 문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json` | 문서만 |
| `scripts/run_v121_salience_anchor.sh` | 6 | 코드: `scripts/eval_seal_tasks.sh`<br>문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json`, `talks/council/C-20260918-20/blackboard.json` 등 5곳 | 사용중 |
| `scripts/run_v121_sh_variants.sh` | 3 | 코드: `scripts/run_ru87_precision.sh`<br>문서: `docs/history/archive.md`, `docs/history/research_units_all_manifest.json` | 사용중 |
| `scripts/run_v121_shape_screen.sh` | 4 | 코드: `scripts/run_and_wait.sh`<br>문서: `docs/agent_handoff.md`, `docs/history/archive.md`, `docs/history/research_units_all_manifest.json` | 사용중 |
