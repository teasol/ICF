# C-20260919-2 · 소집자 검증과 종결

소집자: opencode / deepseek-v4.1-flash. `report.md` 원문은 수정하지 않는다.

## 회차 운영

- 좌석 `P/R/A/T/S`, 호출 10/14, 경과 555.3초, 지지 6건, Phase Q 13건 중 인용 검증 4·폐기 7.
- **기권 1: A1(감사), Phase 2, `finish_reason=length`.** 추론 토큰이 32k 예산을 소진했다.
  P·S가 아니므로 회차는 유효하나, **감사 교차검증 관점이 빠졌다.** S가 지목한 그 4개 미검증
  항목을 소집자가 1단계로 대신 수행했다(아래).
- 모델 다양성 0(전원 `deepseek-v4.1-flash`). 이견 턴 0.

## 좌석 전제 정정 (카드 자료 누락)

S는 비용 항목을 `foldfit_cost`(5-branch `20.0`초/fold, 2-arm `3.89` GPU-h)로 서술해 "paired
확증은 예산 내 불가"라고 적었다. **이는 낡았다.** 카드에 RU-99/100 결과를 넣지 않은 소집자
잘못이다. 실제: 디바이스 상주화(RU-99) 후 700 fold 2-arm이 **H100에서 완주**했고(RU-100),
7-branch 단가는 **`2.0`초/fold 실측**이다. 따라서 S의 (3) 비용 전제는 무효이며, 고른 행동
(b)는 애초에 GPU-0이라 이 정정에 영향받지 않는다.

## 1단계 실행 결과 (GPU 0, 읽기 전용) — S의 4항목

산출물: [`talks/reports/2026-09-19_provenance_inventory.json`](../../reports/2026-09-19_provenance_inventory.json)

| 항목 | 결과 |
|:--|:--|
| (i) provenance writer가 `SCREEN_ONLY` 분기 밖인가 | **그렇다.** `evaluate_pure.py`의 writer 호출은 `--output` 저장 블록에서 무조건 실행된다. `ICF_SHAPE_SCREEN_ONLY`는 **어느 코드도 읽지 않는다** — 셸 스크립트가 export만 하는 죽은 경로다(`D-050`·`D-053`). |
| (ii) 선언 config vs 유효 config | **선언 config 기준.** `evaluate_pure`는 `args.config`를 그대로 파싱하며 치환 경로가 없다. `eval_seal_tasks.sh`는 `model:` 블록이 있는 학습 config를 **에러로 거부**한다(D-053 폴백 제거). |
| (iii) 저장 예측에 per-branch margin 포함? | **없다.** `per_fold` 키는 `fold·slide_id·label·probability`뿐이다. → **(a)의 오프라인 기전 진단은 이 산출물로 불가.** branch margin이 필요하면 `dump_branch_margins.py`로 따로 떠야 한다. |
| (iv) 감싸는 러너 이름·`SCREEN_ONLY` 인터페이스 | 러너는 `scripts/eval_seal_tasks.sh`, `evaluate_pure.py`를 직접 호출, `--screen-only` 인터페이스 **없음**. |

## 종결 (T1)

- S의 증거 분류와 "다음 한 수 = (b) 오염 검사 재설계, 첫 스텝은 GPU-0 4항목 점검"을 **채택**한다.
- 1단계 결과가 (b)를 막지 않는다((i)·(ii) 긍정). (iii)이 부정이라 **(a)는 보류**한다 —
  RU-100 산출물로는 3/7 악화 기전의 per-branch 재분석이 불가하다.
- S의 경고를 그대로 지킨다: 새 검사는 **귀속만** 검증하고 재현성·드리프트는 검증하지 않는다.
  값 앵커의 "교체"가 아니라 **다른 검사**이며 드리프트 검출 공백을 명시한다.
- 다음 작업(2·3단계): (b) 검사 규격 사전 등록(GPU 0)과 구현. S가 제시한 필수 필드
  (L1 구성 대조 / L2 앙상블 마진 비교 분리, 양성·음성 대조 사전 선언, 검증 범위 명시,
  D-053 계열 실패 검출 항목)를 따른다.

[작성자: opencode / 소집자 / deepseek-v4.1-flash (effort: 미확인) · 2026-09-19 19:12 KST]
