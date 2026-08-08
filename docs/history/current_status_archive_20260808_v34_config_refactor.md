# Archived from current_status.md (2026-08-08, SS64 compaction)

SS56 is fully resolved: the config refactor landed and its durable rules now live in
`docs/agent_handoff.md` SS7 (config organization + self-contained archiving + the
reference-validation snippet). The official 50-fold restart it describes was
superseded first by SS57 (case leakage), then by SS64 (fp32 numbers are now
reference-only). Kept verbatim for provenance.

---

## 56. 2026-08-07 — config 시스템 리팩터링(v34 base·default 참조·재아카이빙) + 공식 50-fold 재시작

**상태**: **v34-1536 = PathoBench 보고용 확정 유지**. config 시스템을 **v34 base + group default
참조형**으로 재구성하고, v30/v24/v22 체인을 **자체 포함형 아카이빙**으로 정리했다. 공식 50-fold
배치는 아카이빙 회귀로 전부 실패했던 것을 config 수정으로 해결하고 **5/17 완료 → 12개 재시작**
(백그라운드).

### 1. v34 config = default 참조형 (단일 진실 공급원)

- `configs/train_v34_phase0_largectx_1536.yaml`·`_512.yaml`을 `data/model/optimizer/scheduler/
  trainer/logger/callbacks: default` 참조로 단순화. **group default를 v34-1536 해석값으로 설정**
  (`configs/{data,model,optimizer,scheduler,trainer,callbacks,logger}/default.yaml` — optimizer/
  scheduler/logger는 신규).
- `src/utils/utils.py merge_train_config`에 **`logger_overrides`·`trainer_overrides` 지원 추가**
  (experiment_name/max_epochs 등 run별·arm별 override용).
- v34-512는 dimension(512) + arm-D 레시피(batch 1, num_cells [1,32768], episodes 256, epochs 25)만
  override로 유지 (사용자 결정).
- 검증: 두 v34 config 해석 결과가 이전(자체 포함형/원본)과 **딥 이퀄**. 전체 config 141개 해석 성공.

### 2. 재아카이빙 + 아카이빙 정책

- root = **v34 3종만** (`train_v34_1536`/`_512`/`test_v34_1536_ici`). v30 5종+eval_v30 2종 →
  `archive/v30/`, v24 2종 → `archive/v24/`, v22_medium → `archive/v22/`. 이동 시 base_config
  상대경로를 `../v22/`·`../v24/`·`../v30/`·`../v18_v19/`로 보정 — **아카이브 전체 자기완결**.
- 기존 아카이브의 숨은 깨짐(ia_mil·musklike_easy_levers·v23_v24_candidates·v25·v26·v31·v32·v33,
  19개)도 모두 수정. v18_v19의 learnability 10개는 커밋 a5dfcf8에서 의도적으로 purge된 data 모듈을
  참조하는 **기존 결함**(역사 보존용, 활성/체인과 무관).
- **아카이빙 정책 신설(handoff §7 규칙 3)**: 아카이빙 config는 `base_config` 없이 **전부 인라인
  (자체 포함형)**으로 보관 → 상대경로 깨짐 원천 차단.

### 3. 공식 50-fold 재시작 (config 회귀 해결)

- 원인: 이전 배치(12:54~12:59)가 아카이빙된 `configs/train_v24_musklike_easy.yaml`을 참조해 17개
  전부 rc=1 실패 → v34 config 자체 포함/default 참조화로 해결 (smoke에서 config 해석 통과 확인).
- 배치 스크립트 신규: `scripts/run_official50_batch.sh` (17개 task, 완료분 스킵, workers
  10→6→4→2 자동 축소, per-fold 체크포인트 리쥼). 로그 `logs/official50/batch_resume.log`.
- **완료 5개(pooled)**: bc_therapy er 0.672 / grade 0.713 / her2 0.670, cptac_brca_PIK3CA 0.569,
  cptac_brca_TP53.
- **재시작(14:17 KST, 12개 백그라운드)**: lscc(3)·luad(4)·pda(1)·ucla_lung(1)·ccrcc(3).
- ⚠️ **14:17 1차 재시작은 ARID1A에서 OOM 연쇄로 중단**: `run_official_folds_parallel.py`가 worker
  실패 시 형제 worker를 종료하지 않아(고아 3개가 GPU ~166GB 점유) workers 10→6→4→2 재시도가 전부
  즉시 OOM. **러너 수정**(worker 실패 시 전체 worker kill → GPU 해제) 후 **14:26 재실행**(nohup,
  PID 723428) — ARID1A(304 슬라이드, worker당 ~50GB)는 깨끗한 GPU에서 workers=2로 수용. 완료 후
  §53 표 갱신 + SEAL 재비교.
- ARID1A 2-fold smoke는 10분 timeout으로 종료(대형 task 1-fold 평가가 10분 초과 — config 문제 아님).

### 4. 공식 50-fold 진행 (6/17 완료) + 리팩터링 최신화

- **ARID1A 완료 (6/17, 15:37)**: 50-fold mean **0.4693 ± 0.1093**, pooled **0.4616**
  (`predictions/pathobench_cptac_lscc_ARID1A_mutation_v34_1536_official50.pt`).
  이전 5-fold(§50 lscc_arid1a 0.908)와 큰 차이 — **공식 fold/코호트 프로토콜 차이**로 기록.
- **배치 일시정지 (사용자 요청)**: ARID1A 완료 직후 감시 스크립트(`/tmp/pause_after_arid1a.sh`)
  가 배치 스크립트+워커 종료. **잔여 11개**: lscc(2)·luad(4)·pda(1)·ucla_lung(1)·ccrcc(3).
  재개: `nohup bash scripts/run_official50_batch.sh` (완료분 스킵).
- **리팩터링 (폐기 분기 최신화, §56.8-9)**: 백업(태그 `repro-pre-deprecated-cleanup-20260807` +
  `src/repro_backup_20260807/`) 후 ① 죽은 메서드 3개(§56.8), ② **CCER(v31) ~570줄**, ③
  **DR-CCER(v32) ~800줄** 제거 — 각각 파라미터 시그니처 동일(220그룹/41.67M)·forward 동치
  (dense/ragged diff 0)·checkpoint strict 로드(0/0) 검증, **전체 테스트 32개 통과(148.5s)**.
  남은 폐기 분기: typed_bag(v25)·cls_token(v26)·IA-MIL·CCTS/absolute_tail·mean_pool(v23).

### 5. 다음

- 50-fold 잔여 11개 재개 → §53 표 **17개 전체 갱신** + SEAL 재비교.
- 폐기 분기 최신화 계속(typed_bag→cls_token→MIL→CCTS→mean_pool) 또는 여기서 종료.
- v34-512 학습 + 동일 평가(열린 과제 ③), v30 vs v34 PCA-per-fold 공정 비교.

---
