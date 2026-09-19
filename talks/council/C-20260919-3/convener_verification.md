# C-20260919-3 · 소집자 검증과 종결 (아이디어 창출)

소집자: opencode / deepseek-v4.1-flash. `report.md`·`brainstorm_transcript.md` 원문은 수정하지 않는다.

## 회차 운영

- Phase B(자유 대화) 2좌석 × 3턴: **가설 19건 · 선언된 이견 6턴 · 제안서화 0건.** 대화록은
  감사용으로만 남기고 협의체에는 가설 한 줄만 넘어갔다.
- 협의체 P/R/A/T/S, 호출 16/24, 경과 638.8초, Phase Q 17건 중 인용 검증 11·폐기 8.
- **기권 없음.** 모델 다양성 0(전원 `deepseek-v4.1-flash`).

## 소집자 0단계 실행 (GPU 0, 결과 전 코드·파일 대조)

| 확인 | 결과 |
|:--|:--|
| §3.2 집계 정의와 구현 | `trimmed_mean`은 각 브랜치 margin을 **sigmoid 확률로 바꾼 뒤 최저 1·최고 1을 절사하고 평균**한 값을 logit으로 되돌린다(`voting.py:69`, `PROJECT.md §3.2` 문구와 일치). **마진·순위가 아니라 확률을 절사한다** |
| 저장 스키마 | `predictions/margins_7b/*.pt`(RU-101)에 **per-fold per-branch margin**(7 브랜치, 50 fold)과 provenance(`config_sha256`·`code_sha256`·`weights`·`aggregation`·`sketch_dim`)가 있다. `predictions/parity_*`는 per-fold probability만 있어 branch margin이 없다 |
| readout 독립성 | 현행 config는 `trimmed_mean`이고 각 브랜치는 특징에서 **독립 계산**되어 고정 집계로 결합된다. Dual Ridge식 **결합 해가 아니다** → branch counterfactual drop은 well-defined |
| 절사 마스크 | 저장돼 있지 않으나 margin에서 재계산 가능(정렬 순서) |

## 가장 값싼 검정 실행 — 제안 3 (R이 정정한 최우선)

저장된 per-fold SJ 단독 AUROC(350 fold)을 악화 3과제와 나머지 4과제로 나눠 분포를 쟀다.

| 집단 | n | mean | sd | `\|mean−0.5\|/sd` | <0.4 | >0.6 |
|:--|--:|--:|--:|--:|--:|--:|
| 악화 3과제 | 150 | `0.493` | `0.105` | **0.07** | 0.17 | 0.13 |
| 나머지 4과제 | 200 | `0.561` | `0.110` | 0.55 | 0.09 | 0.32 |

사전 고정 규칙: 악화 3과제가 `0.5` 중심 단봉이고 편차가 sd로 설명되면 '역정보' 해석 기각.
→ **역정보 해석 기각, '무정보(잡음)' 해석 지지.** 나머지 4과제는 양의 이동. (RU-101의 "무정보"
읽기를 강화하며, TGW의 판별 불가 상태는 바꾸지 않는다.)

## 결정 (T1)

- S의 증거 분류와 다음 행동을 **채택**한다. 창출된 가설 19건은
  [`brainstorm_transcript.md`](brainstorm_transcript.md)에 원문으로 보존한다(협의체에는 한 줄만
  넘어갔다).
- 이 회차가 확인한 것: counterfactual drop이 **well-defined**이고 저장 margin으로 **GPU 0**에서
  가능하다. 이미 `RU-101`/`RU-102`의 F6-SJ·F6-SH·F5 재집계가 그 반사실의 일부다 — SJ 제거는 악화
  3과제를 개선하나 나머지 3과제를 악화시켜 **전역 제거는 macro를 낮춘다**(F6-SJ macro `0.6197` <
  `0.6226`). 즉 문제는 "SJ가 나쁘다"가 아니라 **과제 적응**이며, 그 선택 신호가 없다(`TGW` 교착과 동일).
- 새로 남는 빈칸: **절사 마스크**(어느 브랜치가 min/max로 절사되는가)와 **fold별 계수**가 저장돼
  있지 않다. 절사율 진단과 solver 진단은 이 저장이 선행 조건이다(후속 RU 후보).
- 회차의 경고를 유지: `RU-102` oracle 상한을 후보 기대 크기·방향 판정에 쓰지 않는다,
  context 셔플 null로 per-branch 인과를 판정하지 않는다, `CA-08` 경계 판정 없이 집계 변형을
  실행하지 않는다, `SEAL 10`은 열지 않는다(`D-057`).

[작성자: opencode / 소집자 / deepseek-v4.1-flash (effort: 미확인) · 2026-09-19 21:50 KST]
