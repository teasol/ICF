# Current Status

> **이 파일은 "지금 무엇을 하고 있는가"만 담는다. 100줄을 넘기지 않는다.**
> 수치·기준은 [`PROJECT.md`](PROJECT.md), 닫힌 축은 [`closed_axes.md`](closed_axes.md),
> 결정 이력은 [결정 이력](history/archive.md)가 정본이다. **여기에 복사하지 않는다.**
> 종료된 RU의 상세는 [`history/research_units_all.md`](history/research_units_all.md)로 이관한다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-08 15:20 (KST) |
| **Status** | CLEAN — RU-86·87·88 종료 (Main/Idea/Coding 3에이전트 배치) |
| **Host / Node** | `nexgem-s1` · RTX A5000 8장 · driver 580.126.09 · Slurm 명령 없음 |
| **Environment** | uv venv `.venv` · Python 3.12.11 · PyTorch 2.14.0+cu130 · Lightning 2.6.5 |
| **Active Job** | 없음 · RU-86·88은 GPU 0, RU-87은 GPU 6장 3,169초 |
| **회귀 테스트** | 108 tests · `OK` (2026-09-08) |

---

## 2026-09-08 — RU-86·87·88 종료 인수인계

**세 RU를 종료했다. 셋 다 사전 등록한 기준을 결과 전에 고정하고 그대로 적용했다.**

- **RU-86** MDX 게이트 ② — **반박·종료.** 사전 등재 과제 `Grade`에서 단독 AUROC > 0.5인 fold가
  50 중 **10개**(단측 `p = 0.99999720`). 게이트 ①을 통과한 유일한 Tier 1 후보를 종료했고
  §226 Tier 1 3건이 전부 종료됐다 ([보고](reports/RU-86_mdx_gate2.md)).
  2차 탐색에서 **부호 반전**을 관측했다 — `Grade`는 50 중 40 fold가 0.5 **미만**, `KRAS`는 41 fold가
  초과이며 `|중앙값−0.5|`가 `0.0821` 대 `0.0861`로 크기가 같고 방향이 반대다. 무정보가 아니라
  **부호가 과제에 따라 뒤집히는 정보**일 수 있다(기전 미확인). 부호를 알려면 라벨이 필요하므로
  `CA-R1`과 같은 벽이며, 사후 부호 선택으로 후보를 되살리지 않았다.
- **RU-87** bf16 대 fp32 — **상쇄·종료.** 정밀도 차는 절대 AUROC fold-mean 최대 `0.0837%p`,
  대응 Δ fold-mean `0.0674%p`로 **짝지은 Δ에서 상쇄된다**. 폐기한 `최대 0.5%p` 표기의 경위는
  결정 이력 `D-037` ([보고](reports/RU-87_precision_bf16_fp32.md)).
- **RU-88** CA-R1 fold 수준 지문 패널 — **반박 5 / 판별 불가 3, 보류.** 진입 기준 충족 0/8.
  새 지문 F1(이상치 근접도)·F2(context↔query MMD)는 두 표적 모두에 대해 구간이 관심 크기
  `|ρ| = 0.3`을 배제했다. 대조군 F3(Task-Geometry 계열)에 판별 불가가 몰렸다
  ([보고](reports/RU-88_car1_fingerprint_panel.md)). **`CA-R1`은 닫지 않았다** — 과제 군집 7로는
  부재를 입증할 검정력이 없다.

**후보 큐에 `CA-R1` 계열 5건을 등재했다**([`research_directions.md` §0 P1-B](research_directions.md)
행 `B1`~`B5`). 출처는 Idea Agent 산출 제안서 6건이다
([제안서](proposals/2026-09-08_car1-label-free-task-identification.md)).

**경계 판정 2건을 사용자 판단 대기로 올렸다** ([`closed_axes.md` §3](closed_axes.md)) —
`§3-I` 컨텍스트 라벨만 쓰는 판별 통계가 `CA-06` 경계인가, `§3-J` 다중 강도 브랜치 동시 투입이
`P2-SELECTOR-CEILING`·`CA-09` 경계인가. 판단이 갈려 임의로 한쪽을 채택하지 않았다.

## 진행 중 RU

없음. 다음 RU의 질문·대조·예산은 사용자와 확정한다.

## 실행 환경 — 현재 접속 호스트

현재는 `nexgem-s1`이다. `.venv` 버전은 헤더와 일치하며 시스템 `python3`는 3.8.10이다.
`source scripts/node_env.sh`는 `.venv`와 GPU 8장을 탐지하고, GPU 0~7 CUDA matmul 정상 동작을
2026-09-07 확인했다. `sinfo`·`squeue`가 없어 **이 호스트는 직접 실행 유형**이다(`D-030`).
`nexgem`을 통해 제출할 때만 공용 [Slurm 규칙](/home/kimds/agent_rules/slurm_rules.md)이 적용된다.

### Immediate Next Command

```bash
cat docs/reports/RU-88_car1_fingerprint_panel.md
```

---

## 미해결 (Open Issues)

**사용자 판단 대기**
- **다음 연구 방향** — §226 Tier 1 3건이 전부 종료됐다. 큐의 `P1-B`(CA-R1 계열 5건)가 다음
  후보군이며 착수 순서·예산이 미결이다([`research_directions.md` §0](research_directions.md)).
- **경계 판정 2건** — `closed_axes.md` §3-I(컨텍스트 라벨 통계 대 `CA-06`)·§3-J(다중 강도 브랜치
  대 `P2-SELECTOR-CEILING`·`CA-09`). 판정 전에는 큐 `B2`·`B3`을 착수하지 않는다.
- **게이트 ② 통계량** — 게이트 ②는 단측 `AUROC > 0.5`라 '정보량'이 아니라 '**양의 방향** 정보량'을
  잰다. 부호 반전 후보는 정보를 담고도 구조적으로 탈락한다(RU-86 관측). 양측 `|AUROC−0.5|`로
  바꾸면 사후 부호 선택의 문이 열리므로 **사용자 판정이 필요하다**(큐 `B5`).
- **재개 축 활용 여부** — `CA-04`(해소 항목 ②·③ 남음)·`CA-02`(재현 선행). 착수 시점 미결이다.

**연구상의 교착**
- **과제 특화 이득을 활용할 선택 신호가 없다.** 이득은 실재하나 라벨 없이 과제를 판별할 수단이
  없다 — §221(SHJ)·§225(subsampling)가 같은 벽이다 ([`closed_axes.md` `CA-R1`](closed_axes.md)).
- **모든 판정이 `hold-out 미검증`** 이다 ([`PROJECT.md` §3.1](PROJECT.md)).
- **v115~v120 6브랜치 조합이 미검증이다.** 과거 기록은 그대로 보존하고, 현행 승격 규칙
  (대응 per-fold Δ의 과제 군집 95% 구간 하한 > `δ_min`)으로 현행 조합을 검증하는 것은 남은 과제다. **절대 macro는 과제 모집단 성능으로 주장 불가**
  (군집 SE `4.34%p`, [`PROJECT.md` §4.1](PROJECT.md)) — 대응 비교만 정밀하다.

**기술 부채**
- §226 Tier 1 3건(`AKS`·`MDX`·`LID`)이 **동일 에이전트 한 배치 산출**이며 독립 비교군이 없다.
- §226 적대적 검증 2건 미실행 (`research_directions.md`에 `미검증` 명시).
- **bf16 정밀도 부채는 실측으로 종료됐다 (RU-87).** 정밀도 간 차이는 절대 AUROC의 과제별
  fold-mean에서 최대 `0.0837%p`, 대응 Δ의 fold-mean에서 `0.0674%p`이며, 두 arm의 공통 성분으로
  **짝지은 Δ에서 상쇄된다** — `δ_min = 0.3%p`는 잡음 아래가 아니다. 다만 상쇄는 fold-mean에서만
  일어나므로 **단일 fold 수치를 근거로 쓰는 서술은 약 0.6%p 잡음을 안는다**(per-fold 95백분위
  `0.638%p`). 폐기된 `최대 0.5%p` 표기의 경위는 결정 이력 `D-037`.
- `adaptive_trimmed`의 `adaptive_tau`는 무효 인자(죽은 코드).
- `history/archive.md`에 **§199·§200 절 번호가 각각 중복** ([`closed_axes.md` §4](closed_axes.md)).

_by Claude Opus 5 (Main) on nexgem-s1 at 2026-09-08_
