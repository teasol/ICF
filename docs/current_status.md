# Current Status

> **이 파일은 "지금 무엇을 하고 있는가"만 담는다. 80줄을 넘기지 않는다.**
> 수치·기준은 [`PROJECT.md`](PROJECT.md), 닫힌 축은 [`closed_axes.md`](closed_axes.md),
> 결정 이력은 [`decisions.md`](decisions.md)가 정본이다. **여기에 복사하지 않는다.**
> 종료된 RU의 상세는 [`history/research_units_all.md`](history/research_units_all.md)로 이관한다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-06 (KST) |
| **Status** | CLEAN — RU-79 완료 · 다음 연구 방향 설정 대기 |
| **Host / Node** | `nexgem-s1` · RTX A5000 8장 · driver 580.126.09 · Slurm 명령 없음 |
| **Environment** | uv venv `.venv` · Python 3.12.11 · PyTorch 2.14.0+cu130 · Lightning 2.6.5 |
| **Active Job** | 이번 세션에서 연구 job 실행 없음 · 원격 Slurm 상태 미확인 |
| **회귀 테스트** | 147 tests · `OK` (2026-09-06 nexgem-s1, CPU, 23.348s) |

---

## 현재 목표

**문서·진행 체계 정비 완료.** 완료 조건·예산·중단 조건은 [`PROJECT.md` §2](PROJECT.md)에 있다.

> 연구 방향은 미정이며 사용자가 설정한다(`D-020`).
> 착수 절차는 [`agent_handoff.md` §1.0](agent_handoff.md)를 따른다.
> 후보: [`research_directions.md`](research_directions.md) · 잔여 거리는 [`PROJECT.md` §1](PROJECT.md).

## 진행 중 RU

없음. `RU-79`(문서 체계)·`RU-80`(판정 설계 정밀도)은 종료했다.
결과·검증 근거는 [`history/research_units_all.md`](history/research_units_all.md)에 있다.
착수 절차는 [`agent_handoff.md` §1](agent_handoff.md), 현행 판정 기준은 [`PROJECT.md` §4](PROJECT.md)다.

## 실행 환경 — 현재 접속 호스트

현재는 `nexgem-s1`이다. `.venv` 버전은 헤더와 일치하며 시스템 `python3`는 3.8.10이다.
`source scripts/node_env.sh`는 프로젝트 `.venv`와 GPU 8장을 탐지한다.
GPU 드라이버는 실측했지만 이 호스트의 CUDA 텐서 실행은 이번 세션에서 검증하지 않았다.
`sinfo`·`squeue`가 없어 현재 세션에서 클러스터 상태를 조회할 수 없다.
`nexgem` 로그인 노드에서 제출할 때는 공용 [Slurm 규칙](/home/kimds/agent_rules/slurm_rules.md)을 따른다.

### Immediate Next Command

```bash
source scripts/node_env.sh
# 다음 연구 질문·RU 카드 초안 검토
sed -n '1,85p' docs/research_directions.md
```

---

## 미해결 (Open Issues)

**사용자 판단 대기**
- **다음 연구 방향 설정** — 위 "현재 목표" 참조.

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

_by Codex on nexgem-s1 at 2026-09-06_
