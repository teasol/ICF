# Archived sections from current_status.md

This is the running archive for fully-resolved / superseded sections that were
moved out of `docs/current_status.md` to keep the living doc compact. Each
section keeps its original heading so cross-references still resolve.

- 2026-08-02: archived §4 (v22 retrieval removal decision), §9 (2026-07-31 session), §10 (2026-08-01 session) — all superseded by later sections (§3 v24 decision, §11 v25 retirement).

---

## 4. v22 결정: retrieval 완전 제거 (2026-07-29)

### 제거 근거 (3대 가설 검증 결과)

1. **retrieval은 ICI에서 이득이 없음**: retrieval을 끈 Phase 6c가 켠 Phase 6b와 AUROC 동일(0.5454 vs 0.5481)한데 Log Loss(0.7921 vs 0.8672)와 Accuracy(0.6092 vs 0.5747)는 오히려 **더 좋음**. ICI는 fold당 context가 ~69명뿐이라 24명 선별은 가용 labeled context의 65%를 버리는 것.
2. **구현이 문서화된 설계와 달랐음**: 설계는 "query 1명당 24명 맞춤 선별"이었으나, 실제로는 두 구현 모두 query 전체에 **공용 context 1세트**를 적용 (외부 collator는 첫 query만, 모델 내부는 query 평균 사용). 공용 context는 각 query의 개별 top-24와 평균 61.1%만 겹쳤고 반응자(y=1)에서 ~50%로 최악.
3. **Phase 5 사전학습은 Phase 1의 1/4만 학습됨**: `episode_batch_size` 8→32로 올리면서 `max_epochs: 20`을 그대로 둬 optimizer step이 10,240 → 2,560으로 감소. 게다가 validation이 retrieval 없이 수행되어 `epoch=014` 체크포인트가 학습 모드와 다른 기준으로 선택됨.

무엇보다 **위 차이들이 통계적으로 구분 불가능**하다는 점이 결정적이었습니다. 검출력 없는 지표 위에 복잡한 계층을 유지할 이유가 없다고 판단해 제거했습니다.

### 제거 범위

| 대상 | 조치 |
|---|---|
| `BaseModel.extract_bag_features` / `retrieve_context_indices` / `_retrieve_context_indices_impl` / `retrieve_context_indices_per_query` | 삭제 (303줄) |
| `BaseModel.forward(retrieval_k=...)` 파라미터 | 삭제 |
| `RetrievalEvaluationEpisodeCollator` / `RetrievalSyntheticTrainingEpisodeCollator` / `SignalAwarePretrainEpisodeCollator` | 삭제 |
| `ModelInterface`의 `retrieval_k` 배선 4곳 + `_build_model` pop-list 항목 | 삭제 |
| `scripts/test.py --retrieval-k` | 삭제 |
| retrieval 계열 config 12개, launch 스크립트 3개, VRAM 벤치 2개 | `configs/archive/v21_retrieval/`, `scripts/archive/v21_retrieval/`로 이관 |
| `tests/test_feature_retrieval.py`, `tests/test_large_context_pretrain.py` | 삭제 → `tests/test_batched_episode_forward.py`로 대체 |
| `architecture_version` | 21 → **22** |

**유지된 것**: v21 aggregator/meta-classifier 전부, 4대 수학 기술, batched multi-episode forward, 세션 중 고친 DataLoader CUDA/pin_memory 수정.

**복구 지점**: retrieval 최종 상태는 git tag **`v21-retrieval-final`** 로 보존되어 있습니다 (`git show v21-retrieval-final`).

### v20 롤백이 불가능했던 이유

사용자 요청은 "v20으로 롤백"이었으나 조사 결과:
* **v20은 코드로 존재하지 않습니다.** 이 브랜치 히스토리는 v18 → v19 → **v21**이며 (`ecf6199`가 19에서 21로 직접 점프), `main`은 아직 v18입니다. `configs/archive/v20/*.yaml`는 v19 코드 위에서 돌던 **설정 파일 시리즈**일 뿐입니다.
* **v21 ≈ v19 + retrieval + 사소한 2줄**. `ecf6199`가 baseline.py에서 제거한 실질 코드는 4줄뿐이고, "v21 4대 개혁"으로 문서화됐던 기술들은 이미 v19에 있었습니다.
* 따라서 retrieval 제거는 버전 롤백이 아니라 **덧붙은 계층의 절제**로 처리하는 것이 맞았고, 사용자 승인 하에 v22 신규 버전으로 진행했습니다.

---

## 9. 2026-07-31 세션 핸드오프 — v23/v24 bag collapse family

### 이번 세션에서 확정/진행한 것

- v23-A0 (exact mean) 50-epoch **완료**: best epoch 43 `val_ce_loss 0.5912154`
  (v22 0.5946 대비 -0.0034). §3 참고.
- v24-A0 (learned projection, slot 1) 50-epoch **완료**: best epoch 45
  `val_ce_loss 0.5976237`. slot 1개 정보 손실로 훈련 val에서 v22/v23보다 높음. §3 참고.
- v24-B0 (per-token bottleneck projection, slot 12 유지) 50-epoch **완료**:
  best epoch 46 `val_ce_loss 0.5923204`. v24-A0 대비 -0.0053 개선, v22 대비 -0.0023 개선. §3 참고.
- v24-B1 (residual mean + bottleneck projection, slot 12 유지) 50-epoch **완료**:
  best epoch 41 **`val_ce_loss 0.5903045` (전체 Bag-Collapse 모델 중 최저 기록 달성)**. §3 참고.
- 구현: v24 learned projection (`26b2b27`), v24-B0 병목 (`b2fb9d0`), v24-B1 residual 병목 (`4f984ca`).
  architecture_version 23/24 분리, 모든 단위 테스트 통과.

### 다음 Action (이 세션 당시 계획, 아래 §10에서 폐기됨)

1. ~~v23-A0 (epoch 43), v24-A0 (epoch 45), v24-B0 (epoch 46), v24-B1 (epoch 41) 4개 후보 체크포인트를
   동일 pool-400, 1,000 episodes, context `40/80/160/300`에서 평가.~~
2. ~~`scripts/compare_predictions.py`로 v22(`predictions/v22_medium_baseline_pool400_curve/`)와
   episode-cluster paired delta + CI 계산.~~
3. ~~판정: overall `+0.03` 또는 target task `+0.05`가 없으면 해당 후보 폐기.~~
4. 합성 Medium+Hard 후보가 확정되기 전까지 ICI는 실행하지 않습니다. **(이 항목만 유지됨 — ICI는 계속 잠금)**

---

## 10. 2026-08-01 세션 핸드오프 — v24 확정, 평가 계획 폐기

### 이번 세션에서 확정/진행한 것

- **사용자 결정**: §9의 4종 paired 비교 계획을 실행하지 않고 폐기. "단순히 label로 bag을 나누는 구조(class-memory 압축)가 별로"라는 문제 제기에서 시작해, v24-B1(residual + bottleneck bag projection)을 train `val_ce_loss` 순위만으로 최종 v24 아키텍처로 확정.
- v22(구 기준선), v23-A0, v24-A0, v24-B0 폐기. 상세는 §3 "최종 결정 (2026-08-01)".
- `configs/train_v23_medium_bag_mean.yaml`, `train_v24_medium_bag_proj.yaml`, `train_v24_medium_bag_proj_bottleneck.yaml`을 `configs/archive/v23_v24_candidates/`로 이관. `train_v24_medium_bag_proj_residual.yaml`이 v24 production entry point.
- `docs/architecture_v23_candidates.md`에 결정 기록을 남기고 `docs/history/`로 이관 (T5-A/B/C는 필요 시 재검토할 미실행 계획으로 보존).
- Git: 이 문서 갱신 커밋은 `d668d33`. `main`/`v24` 브랜치를 이 커밋(즉 `codex/v23-bag-mean` 최종 커밋)으로 fast-forward.

### 남겨진 것 (폐기되지 않음)

- T4 Medium→Hard bridge attribution (§6) — 이번 결정과 무관하게 계속 열려 있는 질문.
- v24용 ICI config 부재 — ICI를 돌리려면 `train_v22_ici_finetune.yaml`/`train_v22_ici_scratch.yaml`에 상응하는 v24 버전이 필요.
- v22 top-level config 6개는 ICI 파이프라인이 아직 참조하므로 archive로 옮기지 않음 — v24 ICI config가 만들어지면 재검토.

### 다음 Action

1. (선택) v24용 ICI finetune/scratch config를 만들고 ICI 실행 여부를 사용자에게 다시 확인.
2. (선택) T4 Medium→Hard attribution을 v24 위에서 이어갈지 결정.
3. 새 구조 변경이 필요해지면 이번처럼 train CE만으로 확정하지 말고, 최소한 §7 평가 프로토콜의 1,000-episode paired 비교를 다시 켤지 사용자와 사전에 정할 것.
