# Current Status

> **이 파일은 "지금 무엇을 하고 있는가"만 담는다. 80줄을 넘기지 않는다.**
> 수치·기준은 [`PROJECT.md`](PROJECT.md), 닫힌 축은 [`closed_axes.md`](closed_axes.md),
> 결정 이력은 [`decisions.md`](decisions.md)가 정본이다. **여기에 복사하지 않는다.**
> 종료된 RU의 상세는 [`history/research_units_all.md`](history/research_units_all.md)로 이관한다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-07 10:54 (KST) |
| **Status** | CLEAN — RU-81 종료 · 독립 검산 및 회귀 테스트 통과 |
| **Host / Node** | `nexgem-s1` · RTX A5000 8장 · driver 580.126.09 · Slurm 명령 없음 |
| **Environment** | uv venv `.venv` · Python 3.12.11 · PyTorch 2.14.0+cu130 · Lightning 2.6.5 |
| **Active Job** | 없음 · `ru81_reg_20260907_r1` 완료 (350 folds, 14 shards) |
| **회귀 테스트** | 150 tests · `OK` (2026-09-07, 22.554s) |

---

## 2026-09-07 — RU-81 종료 인수인계

**RU-81 정규화 진단 완료.** [종료 보고](reports/RU-81_regularization_diagnosis.md)와
[RU 총람](history/research_units_all.md)에 결과·독립 검산·한계를 기록했다.
기준선 유지, λ 처방 보류. 다음 후보는 마진 순위·크기 효과 분리 진단(`D-024`).
실행 블로커 없음. 다음 단계는 후속 질문·대조·예산을 확정하고 별도 RU를 여는 것이다.
원시 예측·manifest·검산 결과·로그는 s1 로컬 `predictions/ru81_reg_20260907_r1/`와
`logs/`에 보관하며 git 비추적이다. 이식 가능한 전체 집계는 종료 보고서에 있다.

## 진행 중 RU

없음. RU-81은 종료했고 다음 연구 방향·예산은 사용자와 결정한다.
후속 진단은 아직 실행하지 않았다.

## 실행 환경 — 현재 접속 호스트

현재는 `nexgem-s1`이다. `.venv` 버전은 헤더와 일치하며 시스템 `python3`는 3.8.10이다.
`source scripts/node_env.sh`는 프로젝트 `.venv`와 GPU 8장을 탐지한다.
GPU 0~7 모두 CUDA matmul 정상 동작을 2026-09-07 확인했다.
`sinfo`·`squeue`가 없어 현재 세션에서 클러스터 상태를 조회할 수 없다.
`nexgem` 로그인 노드에서 제출할 때는 공용 [Slurm 규칙](/home/kimds/agent_rules/slurm_rules.md)을 따른다.

### Immediate Next Command

```bash
cat docs/reports/RU-81_regularization_diagnosis.md
```

---

## 미해결 (Open Issues)

**사용자 판단 대기**
- 후속 방향 선택: 마진 순위·크기 분리 진단 등 (`D-024`).

경계 4건은 모두 비저촉으로 확정했다(`D-022`, [`closed_axes.md` §3](closed_axes.md)).
후보 성능·기전은 미검증이며 별도 RU에서 다룬다.

**연구상의 교착**
- **과제 특화 이득을 활용할 선택 신호가 없다.** §221(SHJ)·§225(subsampling)가 같은 벽에 막혔다 —
  이득은 실재하나 라벨 없이 과제를 판별할 수단이 없다 ([`closed_axes.md` `CA-R1`](closed_axes.md)).
- **모든 판정이 `hold-out 미검증`** 이다 ([`PROJECT.md` §3.1](PROJECT.md)).
- **v115~v120 6브랜치 조합이 미검증이다.** 과거 기록은 보존하되(`D-021`) 새 규칙(`D-018`)으로
  현행 조합을 검증하는 것은 남은 과제다. **절대 macro는 과제 모집단 성능으로 주장 불가**
  (군집 SE `4.34%p`, [`PROJECT.md` §4.1](PROJECT.md)) — 대응 비교만 정밀하다.

**기술 부채**
- §226 Tier 1 3건(`AKS`·`MDX`·`LID`)이 **동일 에이전트 한 배치 산출**이며 독립 비교군이 없다.
- §226 적대적 검증 2건 미실행 (`research_directions.md`에 `미검증` 명시).
- §217~§221의 SHJ 수치는 **bf16 경로 산출값**으로 fp32와 최대 0.5%p 차이날 수 있다.
- `adaptive_trimmed`의 `adaptive_tau`는 무효 인자(죽은 코드).
- `history/archive.md`에 **§199·§200 절 번호가 각각 중복** ([`closed_axes.md` §4](closed_axes.md)).

_by Codex on nexgem-s1 at 2026-09-07_
