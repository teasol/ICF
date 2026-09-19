# C-20260919-1 · 소집자 검증과 T1 종결

소집자: opencode / deepseek-v4.1-flash. 이 문서는 로컬 모델 산출(`report.md`)과 별개로,
소집자가 **정본 대조와 GPU 불필요 선행 확인**을 직접 수행한 결과와 T1 종결을 기록한다.
`report.md` 원문은 수정하지 않는다.

## 회차 운영

- 호출 10/14 · 기권 0 · 경과 382.6초 · 지지 8건 · Phase Q 인용 검증 통과 7 / 폐기 5줄.
- **이견 턴 0.** 자유 대화(Phase B)는 기본 꺼짐이고, 이 회차는 격리 협의체만 돌았다.
- **모델 다양성 0.** 좌석 5개 전부 `deepseek-v4.1-flash` @ `100.97.255.47:8000`.
  좌석 수는 증거의 세기가 아니다. 독립성은 정보 격리와 반증 의무로만 부분 확보.

## 소집자 독립 검증 (정본 대조)

- `PROJECT.md §4`: 승격은 **점추정 규칙** `Δ_macro ≥ +0.003`이며 CI 하한은 게이트가 아니다
  (명문). S 좌석의 해석과 일치 — 확인.
- `PROJECT.md §3.4`: 오염 검사 값 앵커는 `D-050`으로 폐지·재설계 대상, "재설계 완료 전
  통과 주장은 무효". S 좌석의 인용과 일치 — 확인.
- `arm_parity` 수치(macro `0.6169`/`0.6226`, `Δ +0.0057`, CI `[-0.0115, +0.0228]`,
  악화 3과제)는 보고서 원문과 일치 — 확인.

## 선행 확인 3건 — 실제 실행 결과 (GPU 불필요)

| # | 확인 | 결과 |
|:--|:--|:--|
| 1 | provenance writer 실재와 필드 | **부분 실패.** `scripts/analysis/dump_branch_margins.py`에 `code_sha256`·`config_sha256`·`weights`·`aggregation`·`sketch_dim`·`git`가 있다. 그러나 **공식 러너 `scripts/evaluate_pure.py`의 저장 dict에는 provenance가 없다** → `predictions/*.pt`는 여전히 어느 구성에도 귀속 불가. `routine_provenance_scan.py` 실측 378/378 without provenance와 일치 |
| 2 | fold 배정 파일 실재와 읽기 경로 | **통과.** `data/repro_labels_folds/official/<task>/{k=all.tsv,config.yaml}` 실재, `evaluate_pure.load_official_folds`가 읽는다 |
| 3 | `parity_paired_delta.py` 입력 재실행 | **실패.** 입력 `predictions/parity_*_v121_active.pt`가 **0개**(미보존). `+0.0057`과 CI를 저장 산출물에서 독립 재계산할 수 없다 — arm_parity 수치는 보고서 표에만 남는다 |

`report.md`가 정한 규칙("셋 중 하나라도 실패하면 GPU 실행을 제출하지 않는다")에 따라,
**700 fold 재실행은 지금 제출하지 않는다.**

## T1 종결

- S 좌석의 증거 분류(지지/반박/판별 불가/실행 무효)와 다음 행동을 **채택**한다.
- 회차가 낸 진전: 확증 절차의 정체를 `적합 분산`이 아니라 **provenance 고정 재실행**으로
  바로잡았고, 그 선행 조건을 실측으로 좁혔다.
- **다음 실행 한 건(변경):** 700 fold가 아니라, 그 전제인 **공식 러너 provenance writer 구현**
  이다. `evaluate_pure.py`가 `provenance`(branch_list·weights·aggregation·sketch_dim·
  config_sha256·code_hash·git_commit·fold manifest 해시·env)를 저장하면 선행 확인 1이
  통과하고, 그 뒤 SMAD4 fold 10 provenance-fixed 전후 검정을 Slurm이 풀릴 때 제출한다.
- 판정 변경 없음: `+0.0057 ≥ +0.003`은 규칙상 이미 충족이고 과제군집 CI 폭은 fold·과제
  고정 때문에 재실행으로 좁혀지지 않는다. 이 재실행이 바꾸는 것은 `D-042` 근거의
  `unverified-hold` → `verified` 승격과 7-branch 실측 단가다.

## 사용자에게 남기는 것 (T2)

- §4에 CI 하한 게이트 추가 여부(현행 명문 금지) · 점추정 규칙과 `판별 불가` 중 우선순위 ·
  `D-042` 유지/철회 · 확률적 후보 `R>1` 절차 · 기계별 지문 기준선·재현 허용 오차 ·
  fold 재분할 허용 · 4 GPU-h 초과 예산 · 앵커 `5e-4` 원인 규명 RU 신설 · `SEAL 10` 개봉.

[작성자: opencode / 소집자 / deepseek-v4.1-flash (effort: 미확인) · 2026-09-19 17:20 KST]
