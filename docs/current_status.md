# Current Status

> **이 파일은 "지금 무엇을 하고 있는가"만 담는다. 80줄을 넘기지 않는다.**
> 수치·기준은 [`PROJECT.md`](PROJECT.md), 닫힌 축은 [`closed_axes.md`](closed_axes.md),
> 결정 이력은 [결정 이력](history/archive.md)가 정본이다. **여기에 복사하지 않는다.**
> 종료된 RU의 상세는 [`history/research_units_all.md`](history/research_units_all.md)로 이관한다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-07 19:00 (KST) |
| **Status** | CLEAN — 결정 기록을 archive로 통합 · 문서 자기완결성 강제 |
| **Host / Node** | `nexgem-s1` · RTX A5000 8장 · driver 580.126.09 · Slurm 명령 없음 |
| **Environment** | uv venv `.venv` · Python 3.12.11 · PyTorch 2.14.0+cu130 · Lightning 2.6.5 |
| **Active Job** | 없음 · RU-82·83·84 모두 GPU 0 오프라인 (wall 2.8s / 15.1s / 0.5s) |
| **회귀 테스트** | 151 tests · `OK` (2026-09-07, 24.8s) |

---

## 2026-09-07 — RU-82·83·84 종료 인수인계

**세 RU를 GPU 0으로 종료했다.**
- **RU-82** λ 앙상블 — 4구성 전부 판별 불가. λ 평균은 과제 간 이질성을 흡수하지 못했다
  ([보고](reports/RU-82_lambda_ensemble.md)).
- **RU-83** 순위·크기 분리 — 가법성 지지, 크기 지배 반박(4/12). 12 cell 전부 구간이 겹쳐 판정은
  약하다 ([보고](reports/RU-83_rank_size_separation.md)). **정규화·결합 진단을 종료한다.**
- **RU-84** RBF Gram 대각 지배 — §199 기각 기전 반박, 1,050건 중 0건
  ([보고](reports/RU-84_gram_diagonal_dominance.md)).

**축 15건·결정 33건 전수 조사 완료.** 기각 근거를 분류해 `CA-04`·`CA-02`를 **재개**했다 —
둘 다 사유가 악화가 아니라 "이득 없음"이고 승격 기준이 구간 추정으로 바뀌기 전의 점추정에
근거했다. `CA-02`는 **우선순위 낮음**. `CA-06`은 6/7 방향 일치 악화로 유지. 실행 방식은
**노드 종속 원칙**으로 정리했다(s1은 Slurm 없이 직접 실행).

**문서 체계 변경.** `decisions.md`를 폐지하고 33건을 `history/archive.md` 말미의 **§결정 이력**
으로 통합했다. **living 문서의 각 항목은 자기완결적이어야 한다** — 이력 ID는 출처 표기로만 쓰고
규칙 내용을 위임하지 않으며 테스트가 검사한다. 레코드 진입 기준은 ①사용자 결정 ②번복·철회
③결과를 본 뒤 기준 변경 ④규범 변경이고 **RU 종료만으로는 레코드를 만들지 않는다.**

## 진행 중 RU

없음. 다음 RU의 질문·대조·예산은 사용자와 확정한다.

## 실행 환경 — 현재 접속 호스트

현재는 `nexgem-s1`이다. `.venv` 버전은 헤더와 일치하며 시스템 `python3`는 3.8.10이다.
`source scripts/node_env.sh`는 `.venv`와 GPU 8장을 탐지하고, GPU 0~7 CUDA matmul 정상 동작을
2026-09-07 확인했다. `sinfo`·`squeue`가 없어 **이 호스트는 직접 실행 유형**이다(`D-030`).
`nexgem`을 통해 제출할 때만 공용 [Slurm 규칙](/home/kimds/agent_rules/slurm_rules.md)이 적용된다.

### Immediate Next Command

```bash
cat docs/reports/RU-84_gram_diagonal_dominance.md
```

---

## 미해결 (Open Issues)

**사용자 판단 대기**
- **다음 연구 방향** — 자원을 신규 브랜치로 돌리기로 했다. Tier 1 3건(`AKS`·`MDX`·`LID`)의
  착수 순서·예산이 미결이며, 라벨 무관 게이트 ① 직교성 점검부터 시작한다.
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
- §217~§221의 SHJ 수치는 **bf16 경로 산출값**으로 fp32와 최대 0.5%p 차이날 수 있다.
- `adaptive_trimmed`의 `adaptive_tau`는 무효 인자(죽은 코드).
- `history/archive.md`에 **§199·§200 절 번호가 각각 중복** ([`closed_axes.md` §4](closed_axes.md)).

_by Claude Code on nexgem-s1 at 2026-09-07_
