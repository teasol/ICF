# 문서 간 중복 수치 전수 조사

**작업 지시**: `talks/ops/tasks/docs_numbers.md` — "같은 수치가 여러 곳에 선언처럼 적힌 곳"
을 찾아 보고. **문서는 고치지 않았고 이 보고서만 쓴다.**

- 조사 대상: `docs/**/*.md` 와 `talks/**/*.md` 총 96개 파일.
- 조사 방법: 아래 수치 계열에 대해 정규식 전수 스윕(AUROC 4자리, `%p` 증분, 상관·랭크 임계,
  게이트 실적 `n/50`, `GPU-hour`, `fold`·과제 수) + living 문서 정독.
  living 문서 8종(`README.md`·`PROJECT.md`·`agent_handoff.md`·`current_status.md`·
  `current_architecture.md`·`closed_axes.md`·`research_directions.md`·`SESSION_HANDOFF.md`)은 전문 확인.
- 판단 기준: `docs/PROJECT.md` 를 정본으로 본다. `talks/reports/`·`docs/reports/`·`docs/history/`
  ·`talks/council/` 은 기록물이므로 자기가 실측한 값의 출처가 될 수 있다
  (`agent_handoff.md` §7.1 "기록물은 단일 선언 의무에서 제외"). 문제는 **다른 문서의 값을
  옮겨 적은** 경우다.
- 구분 표기: `출처`=그 값의 실측 보고서, `복사`=출처 값을 다시 선언, `판단 불가`=구분 불가.

> 존재하는 검사 도구는 `tests/test_docs_consistency.py` 다. 이 도구는 `GUARDED_NUMBERS`
> 6개(`0.6171`·`0.6265`·`0.6972`·`0.6681`·`0.7266`·`0.7125`)만, `GUARDED_FILES` 4종
> (`README.md`·`agent_handoff.md`·`current_status.md`·`current_architecture.md`)에 대해서만
> 검사한다. `research_directions.md`·`closed_axes.md`·`talks/**` 는 **검사 대상이 아니다**
> (파일 47\~53행 주석이 명시적으로 면제). 아래 중복의 대부분은 이 사각지대에 있다.

---

## 0. 요약

- **어긋나는 값 3건**을 찾았다. 전부 §1에 모았다.
  - `talks/reports/2026-09-18_arm_parity.md` **내부에서** 같은 통계량의 CI 하한이
    `-0.0114`(§결과)와 `-0.0115`(§추가)로 갈린다. `current_status.md`·`SESSION_HANDOFF.md`
    는 `-0.0114` 쪽을 옮겨 적었다.
  - `docs/reports/RU-92...md:48` 이 5-branch 기준선을 `0.6170` 으로 적었다.
    `PROJECT.md` 는 `0.6171` 이다(0.0001 차).
  - `research_directions.md` 가 `PROJECT.md` §3.2 를 "`2.72/6 = 45.3%`"로 인용하나,
    현행 §3.2 는 `3.52/7 = 50.3%` 다(인용 대상 낡음).
- **living 문서(`PROJECT.md` 밖)의 중복 선언**이 다수다(§2). 특히 `current_architecture.md` 는
  헤더에서 "성능 수치는 선언하지 않는다"고 적고 있으면서 승격 근거·게이트 실적·패리티 macro 를
  본문에 갖고 있다.
- `talks/council/` 토론록은 정본 값을 대량 복사한다(§3). `0.6227` 442회/38파일,
  `3.52` 465회/28파일, `4.02` 303회/24파일, `4 GPU-hour` 716회/25파일 등. 기록물이므로
  형식 위반은 아니나, 정본이 바뀌면 전파 위험이 가장 큰 층이다.
- `talks/lit/` 는 우리 정본 값(`3.52/7`, 적합 분산 `2.8배`)을 논문 브리프에 인용한다(§4).
  외부 논문 수치는 우리 관측이 아니므로 판정 근거가 될 수 없다.

---

## 1. 어긋나는 값 (drift)

### 1-1. 과제군집 95% CI 하한 `-0.0114` vs `-0.0115` — **확인된 불일치**

동일 통계량(과제 7개, 평균 Δ `+0.0057`, sd `0.0185`, df=6)의 같은 구간이 한 보고서 안에서
두 값으로 적혔다.

| 위치 | 값 | 비고 |
|:---|:---|:---|
| `talks/reports/2026-09-18_arm_parity.md:51` | `[-0.0114, +0.0228]` | §"그러나 승격 근거로는 약하다" 표 |
| `talks/reports/2026-09-18_arm_parity.md:87` | `[-0.0115, +0.0228]` | §추가 "fold 수준 짝짓기" 표, **같은 통계량** |
| `docs/current_status.md:33` | `[-0.0114, +0.0228]` | 위 :51 값을 옮겨 적음 |
| `docs/SESSION_HANDOFF.md:37` | `[-0.0114, +0.0228]` | 위 :51 값을 옮겨 적음 |
| `docs/PROJECT.md:92` | `[-1.15%p, +2.28%p]` | 다른 시점(D-042) 측정의 값. 수치가 일치하나 동일 측정은 아님 |

- **출처**: `talks/reports/2026-09-18_arm_parity.md`.
- **판정**: 산술적으로는 `-0.0114` 가 맞다. `0.0185 / sqrt(7) × t(0.975,6)` = `0.0171`,
  `+0.0057 − 0.0171 = −0.0114`. 보고서 §추가의 `-0.0115` 는 **보고서 내부 오기**로 보인다.
  다만 `PROJECT.md:92` 의 `-1.15%p` 가 우연히 `-0.0115` 와 일치하므로, 어느 쪽이 현행
  정본인지 문서만으로는 단정하지 않는다(판별 불가 항목이 아니라 "산술상 -0.0114가 정합").

### 1-2. 5-branch 기준선 `0.6171` vs `0.6170`

| 위치 | 값 | 성격 |
|:---|:---|:---|
| `docs/PROJECT.md:88,128,131` | `0.6171` | 정본 선언 |
| `docs/reports/RU-92_seal10_v121_vs_v120_evaluation.md:48` | `0.6170` → `0.6226` (`+0.56%p`) | 보고서의 회고 |

- **출처**: `0.6171` (PROJECT.md). RU-92 는 자기 실측이 아니라 배경 서술로 옮겨 적은 값.
- **판정**: `0.0001` 어긋남. 같은 문장의 `+0.56%p` 는 맞으므로 반올림 또는 전사 오류로 보인다.
  RU-92 는 보고서(기록물)이므로 "출처"라기보다 **잘못 옮긴 복사**에 가깝다.

### 1-3. `research_directions.md` 의 `PROJECT.md` §3.2 인용 불일치

| 위치 | 값 |
|:---|:---|
| `docs/research_directions.md:351` | "…§3.2가 쓴 `2.72/6 = 45.3%`와 동일 기준" |
| `docs/research_directions.md:209` | "랭크 효율 임계 `2.72/6 = 45.3%`" |
| `docs/PROJECT.md:86` | "유효 랭크 `3.52 / 7` (50.3%)" |
| `docs/PROJECT.md:243` | "현행 효율은 `3.52 / 7` = `50.3%`… 통과선 eff.rank ≥ `4.02`" |

- **판정**: `research_directions.md` 가 §3.2 의 **과거 값**(6-branch `2.72`, 효율 `45.3%`)을
  현행 §3.2 로 인용한다. 현행 §3.2 는 7-branch `3.52/7`(50.3%)이므로 **인용 대상이 낡았다**.
  같은 문서 안에서도 `45.2%`(2.26/5 기준선)와 `45.3%`(2.72/6)가 병존해 표현이 흔들린다.

### 1-4. 참고: 어긋남은 아니나 값이 갈리는 지점

- `docs/PROJECT.md:88` 승격 근거 `+0.56%p` vs `talks/reports/2026-09-18_arm_parity.md:35`
  재측정 `+0.57%p`, `docs/current_status.md:31`·`docs/SESSION_HANDOFF.md:36` `+0.0057`.
  선언값과 재측정값의 차이며 서로 모순은 아니다(보고서가 "재현된다"고 명시).
- `docs/PROJECT.md:83` 기준선 `0.6227` vs 재현 macro `0.6226`
  (`docs/current_architecture.md:289`, `talks/reports/2026-09-18_arm_parity.md:33`).
  계측 반올림 차로 문서들이 이미 설명한다.

---

## 2. `PROJECT.md` 밖 living 문서의 중복 선언

`agent_handoff.md` §7.1 은 "수치를 두 곳에 적지 않는다"고 규정하나, 아래 living 문서들이
정본 값을 다시 선언한다. `tests/test_docs_consistency.py` 의 `GUARDED_FILES` 는
`README.md`·`agent_handoff.md`·`current_status.md`·`current_architecture.md` 만 검사하고,
`GUARDED_NUMBERS` 6개만 본다(§서두). **`research_directions.md`·`closed_axes.md` 는 면제**되어 있다.

### 2-1. `docs/current_architecture.md` — 문서 스스로 "선언하지 않는다"고 적은 값들

상단 주석(5\~6행): "성능 수치·비교 기준·승격 기준은 여기서 선언하지 않는다".

| 수치 | 위치 | 출처로 보이는 곳 | 판정 |
|:---|:---|:---|:---|
| 승격 근거 `+0.56%p`, 기준 `+0.3%p` | `current_architecture.md:124` | `PROJECT.md:88` | 복사 |
| 랭크 효율 `45.2% → 50.3%` | `current_architecture.md:124` | `PROJECT.md:86,243` | 복사 |
| 게이트 ① `max \|r\| = 0.418` | `current_architecture.md:151` | `docs/history/archive.md` §218 | 복사(기록 인용) |
| BS 게이트 ② `max \|r\| = 0.262`, `0.4201 ~ 0.5478`, `31/50`, `p = 0.059`, 컷 `40/50`, `p < 0.01` | `current_architecture.md:173-175` | `PROJECT.md:254-260` | 복사 |
| 패리티 macro `0.6226` vs `0.6226`, `17,723` 슬라이드 | `current_architecture.md:289` | `docs/reports/RU-91...md:82` | 복사 |

### 2-2. `docs/research_directions.md` — 후보 심사에 정본·보고서 값을 재선언

| 수치 | 위치 | 출처로 보이는 곳 | 판정 |
|:---|:---|:---|:---|
| 승격 기준 `Δ_macro ≥ +0.3%p` | `research_directions.md:15,64,311` | `PROJECT.md:176` | 복사 |
| 오염 앵커 `SMAD4 0.4421` / `PBRM1 0.5553` | `research_directions.md:193,351` | `PROJECT.md:145-146` | 복사 |
| 게이트 ① 기각선 `0.6`, 통과선 `2.72/6`, 기준선 `2.26/5 = 45.2%` | `research_directions.md:351` | `PROJECT.md:241-243` (단, `2.72/6` 은 구판) | 복사 + §1-3 불일치 |
| AKS `H_q > 0.99` 100%, `std 0.0017` | `research_directions.md:34,147` | `docs/reports/RU-85...md` | 복사 |
| MDX `max \|r\| = 0.267`, `45% → 49%` | `research_directions.md:35` | `RU-85` | 복사 |
| LID `0.314`, `0.8`, `0.051` | `research_directions.md:36` | `RU-85` | 복사 |
| RU-87 정밀도 `0.0837%p`, `0.0674%p`, 문턱 `0.072%p`, `0.638%p`, `2.579%p` | `research_directions.md:66` | `docs/reports/RU-87...md` | 복사 |
| MDX 게이트 ② `50 중 10`, `p = 0.99999720`, `k≥40` | `research_directions.md:67` | `docs/reports/RU-86...md` | 복사 |
| SW 단독 macro `0.5976` | `research_directions.md:105,243,416` | `docs/history/archive.md` §203 | 복사 |
| SH `max \|r\| = 0.418`, SHJ `0.227`, BD `0.16` | `research_directions.md:138` | `archive.md` §218·§220 | 복사 |
| SMAD4 브랜치별 `0.4212/0.4202/0.4283`, 전체풀 `0.3906/0.3729/0.3959`, CV `0.5483`, BD `0.5322`, SH `0.5434` | `research_directions.md:139-140` | RU/archive | 복사 |
| Grade `BD 0.4665`, `BM 0.6640`/`QA 0.6726`/`DS 0.7008` | `research_directions.md:153` | archive | 복사 |
| SH `+0.29%p`(4/7), SHJ `+0.32%p`(3/7) | `research_directions.md:175` | archive | 복사 |
| DS arm 기준 `0.617066`, 재집계 `0.619802`, `+0.002736`(+`0.27%p`, 6/7) | `research_directions.md:257-259` | RU-88/archive | 복사 |
| Oracle `0.6565`, ρ `-0.27`/`-0.091` | `research_directions.md:290` | archive | 복사 |
| RU-83 `7/12`, `4/12`, `0.1%p` | `research_directions.md:334` | `docs/reports/RU-83...md` | 복사 |

### 2-3. `docs/closed_axes.md` — 닫힌 축의 근거로 과거 실측값을 재선언

`closed_axes.md` 는 테스트에서 명시 면제(45\~47행)지만, 정본 밖 선언이라는 성격은 같다.
대표 항목만 적는다.

| 수치 | 위치 | 출처로 보이는 곳 | 판정 |
|:---|:---|:---|:---|
| `0.6265 → 0.5709`(`-0.0556`, 6/7) | `closed_axes.md:141` | archive §202 | 복사 |
| LR `0.5874`, 7-branch `0.6195`(`-0.0070`), SMAD4 `0.4298`(`-0.1149`) | `closed_axes.md:131-132` | archive §200 | 복사 |
| ARID1A `+0.1142`, SMAD4 `0.4290` | `closed_axes.md:124` | archive §199 | 복사 |
| ρ `-0.27`, `0.667`, `-0.091` | `closed_axes.md:160-162` | archive §212·§221 | 복사 |
| `0.25~0.93`, `1.29/3`, `RM max \|r\| = 0.690` | `closed_axes.md:188-189` | archive, RU-85 | 복사 |
| ICC `0.9996~1.0000` | `closed_axes.md:217` | archive §225 | 복사 |
| seed 폭 `0.0200` | `closed_axes.md:255` | `talks/reports/2026-09-17_ru97_gf_primary7.md` | 복사 |
| `+7.65%p`→`+0.43%p`, `\|ρ\| = 1.000`, `+2.97%p`/`-6.64%p` | `closed_axes.md:261-265` | archive §213·§223·§225 | 복사 |
| G: `H_q > 0.99`(100%), `std 0.0017`, 평균 `0.9963`, `90%`, `0.01` | `closed_axes.md:333-336` | `RU-85` | 복사 |
| H: `0.314`, `0.8`, `1.285`, `0.000`, `0.00002`, `0.051` | `closed_axes.md:349-352` | `RU-85` | 복사 |
| `exp(-2) = 0.1353` | `closed_axes.md:106` | `RU-84` | 복사 |

### 2-4. `README.md`·`agent_handoff.md`·`current_status.md`·`SESSION_HANDOFF.md`

- `README.md`: 수치 없음(문서 지도만). **정본 규칙 준수.**
- `agent_handoff.md`: 정본 수치 없음(링크만). `seed std 0.00000`(§3)과 `200 tests`(§6)만.
  **규칙 준수.**
- `current_status.md`: `+0.0057`, CI `-0.0114`, sd `0.0185`(`:31-34`),
  `1.94`/`3.89`/`31.1`/`4h`(`:46`), 중복도 `0.197`/`0.1878`(`:65`)를 선언.
  대부분 `talks/reports/2026-09-18_arm_parity.md`·`2026-09-18_foldfit_cost.md`·`2026-09-18_phase_b_ab.md`
  의 값을 옮긴 것. `GUARDED_NUMBERS` 6개에는 걸리지 않아 테스트를 통과한다.
- `SESSION_HANDOFF.md`: `+0.0057`, CI `-0.0114`, `20.0초`, `97%`, `7.8배`(`:36-40`).
  위와 같은 성격의 복사.

---

## 3. 기록물·토론록의 반복 인용 (`talks/council/`, `talks/rfc/`)

토론록은 정본 값을 좌석 프롬프트·근거로 반복 인용한다. 형식 위반은 아니나(기록물),
**정본 수치가 바뀌면 가장 넓게 오염되는 층**이다. 정규식 기준 발생 횟수:

| 정본 수치 | 발생 | 등장 파일 수 | 정본 |
|:---|---:|---:|:---|
| `0.6227` | 442 | 38 | `PROJECT.md:83` |
| `3.52` | 465 | 28 | `PROJECT.md:86` |
| `4.02` | 303 | 24 | `PROJECT.md:243` |
| `4 GPU-hour` | 716 | 25 | `PROJECT.md:55` |
| `0.6`(게이트 ① 기각선) | (부분문자열 다수) | 42 | `PROJECT.md:241` |
| `+0.3%p` | 180 | 18 | `PROJECT.md:176` |
| `0.7266`(ABMIL) | 169 | 23 | `PROJECT.md:36` |
| `0.4421`/`0.5553`(앵커) | 147/146 | 28/28 | `PROJECT.md:145-146` |
| `0.6226`(재현 macro) | 160 | 16 | `talks/reports/...arm_parity.md` |
| `50.3` | 61 | 15 | `PROJECT.md:86,243` |
| `1.29` | 42 | 16 | `closed_axes.md:189` |
| `0.6171` | 40 | 14 | `PROJECT.md:88` |
| `0.6972` | 23 | 9 | `PROJECT.md:38,129` |
| `0.690`(RM) | 20 | 10 | `PROJECT.md:246` |
| `40/50`(게이트 ②b) | 16 | 10 | `PROJECT.md:254` |

- 대표 파일: `talks/council/C-20260918-8/report.md`(정본 값을 한 문장에 모아 복사),
  `talks/council/C-20260917-3/report.md`, `talks/council/C-20260918-20/report.md`,
  `talks/council/C-20260917-4/report.md`, `talks/council/C-20260918-9/report.md`.
- `talks/rfc/RFC_2026-09-12_pure_runner_migration_and_legacy_removal.md:172,254,256`
  는 `0.6171`·`0.6226`·`0.6227` 을 선언처럼 적는다(계획 문서).
- **판단**: 이 층은 소집자가 좌석에 "무엇을 보여주는지"를 기록한 것이므로 **복사**로 분류하되,
  수치 정본 변경 시 갱신 대상 목록으로 관리할 필요가 있다. 현 시점 값 자체는 정본과 일치한다.

---

## 4. 외부 수치로 판단이 갈리는 것 (`talks/lit/`, `talks/archive/`, inbox)

- `talks/lit/2606.06458_icmil.md:51`, `talks/lit/*_digest.md` 다수가 우리 정본
  `3.52/7`, 적합 분산 `2.8배`, 승격 기준 `+0.3%p` 를 논문과 대비하며 옮겨 적는다.
  **출처는 우리 정본이므로 복사**이며, 논문 자체 수치(`0.7266` 등 외부 값)와 구분해야 한다.
- `talks/archive/inbox/**`, `talks/inbox/**` 의 옛 메시지에는 `0.62268`, `0.6972`, `0.6171`
  등이 실측·계획 보고로 남아 있다. **기록물(출처 후보)**.
- `docs/proposals/2026-09-11_tiranos_mail_rfc_import.md`: 해당 수치 계열 없음.
- SEAL 논문 기준선 `ABMIL 0.7266`·`MeanMIL 0.7125` 는 `PROJECT.md:36-37` 이 정본이며,
  이를 인용하는 것은 **목표 수치 인용**이지 우리 측정값의 중복이 아니다(선택 경계 `D-047`).

---

## 5. 커버리지와 한계

- **정독**: living 문서 8종, `talks/reports/` 6종, `docs/reports/` 계열은 수치 스윕,
  `talks/rfc/` 4종 스윕.
- **정규식 스윕만**: `docs/history/`(9파일, 1.4만 행)와 `talks/council/`(26파일).
  이들은 기록물이므로 값을 일일이 "출처/복사"로 분류하지 않았고, 정본 값과의 **불일치
  탐지**에만 사용했다.
- **판단 불가로 남긴 것**: `talks/council/` 의 어떤 수치가 회차 자체의 실측인지 정본 복사인지는
  수치만으로 구분되지 않는 경우가 있다. 예: `talks/council/C-20260918-9/report.md:48` 등의
  `0.6227`/`0.6226`/`+0.56%p` 는 좌석이 정본을 인용한 것으로 보이나, 회차 실측 여부를
  원문 맥락 없이 단정하지 않았다.
- **이번 조사에서 발견한 불일치는 §1의 3건이다.** 나머지 정본 수치는 모든 문서에서
  값이 일치했다(불일치 0). 즉 규칙 위반은 "값이 다름"보다 "같은 값을 여러 곳에 적음"의
  형태로 나타난다.

[작성자: opencode / 역할 미배정(사용자 직접 작업 지시) / deepseek-v4.1-flash (effort: 미확인) · 2026-09-18 21:05 KST]
