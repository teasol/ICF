# Current Status

> **이 파일은 "지금 무엇을 하고 있는가"만 담는다. 80줄을 넘기지 않는다.**
> 수치·기준은 [`PROJECT.md`](PROJECT.md), 닫힌 축은 [`closed_axes.md`](closed_axes.md),
> 결정 이력은 [`decisions.md`](decisions.md)가 정본이다. **여기에 복사하지 않는다.**
> 종료된 RU의 상세는 [`history/research_units_all.md`](history/research_units_all.md)로 이관한다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-07 16:05 (KST) |
| **Status** | CLEAN — RU-82 종료 · `N_ctx` 전제 정정(`D-026`) |
| **Host / Node** | `nexgem-s1` · RTX A5000 8장 · driver 580.126.09 · Slurm 명령 없음 |
| **Environment** | uv venv `.venv` · Python 3.12.11 · PyTorch 2.14.0+cu130 · Lightning 2.6.5 |
| **Active Job** | 없음 · RU-82는 GPU 0의 오프라인 재집계로 완료 (wall 2.79s) |
| **회귀 테스트** | 150 tests · `OK` (2026-09-07, 24.961s) |

---

## 2026-09-07 — RU-82 종료 인수인계

**RU-82 λ 앙상블 재집계 완료.** 사용자 제안(여러 λ의 로짓을 결합)을 RU-81 저장 마진으로
GPU 0에 재집계했다. 사전 선언 4구성(브랜치 내부 평균/풀 통합 × 현행 sigmoid/마진 표준화)
**전부 `판별 불가`** — 평균 Δ −0.1341 ~ +0.2083%p, 4구성 모두 구간이 0을 포함한다.
무결성 게이트 오차 0.0. 상세·가설 처리·한계는 [종료 보고](reports/RU-82_lambda_ensemble.md),
결정은 `D-025`. 경계는 `CA-08` 밖으로 확인했고 `closed_axes.md`는 변경하지 않았다.

**다음 후보는 여전히 마진 순위·크기 효과 분리 진단(`D-024`)이며, RU-82가 그 필요성을 높였다.**
A2×B1은 평균이 0에 가까운데 과제별 진폭이 4구성 중 최대였고, 원인이 순위 재배열인지
크기 재가중인지 현재 자료로 구별되지 않는다. 실행 블로커 없음.

RU-81·RU-82의 원시 예측·집계 JSON·로그는 s1 로컬 `predictions/`와 `logs/`에 있고 git 비추적이다.

## 진행 중 RU

없음. RU-82는 종료했고 다음 RU의 질문·대조·예산은 사용자와 확정한다.

## 실행 환경 — 현재 접속 호스트

현재는 `nexgem-s1`이다. `.venv` 버전은 헤더와 일치하며 시스템 `python3`는 3.8.10이다.
`source scripts/node_env.sh`는 프로젝트 `.venv`와 GPU 8장을 탐지한다.
GPU 0~7 모두 CUDA matmul 정상 동작을 2026-09-07 확인했다.
`sinfo`·`squeue`가 없어 현재 세션에서 클러스터 상태를 조회할 수 없다.
`nexgem` 로그인 노드에서 제출할 때는 공용 [Slurm 규칙](/home/kimds/agent_rules/slurm_rules.md)을 따른다.

### Immediate Next Command

```bash
cat docs/reports/RU-82_lambda_ensemble.md
```

---

## 미해결 (Open Issues)

**사용자 판단 대기**
- `CA-04`·`CA-06`의 **재개 조건 재작성** — `N_ctx ≈ 40`이 과거 합성 데이터셋 시절 값으로 확인돼
  두 축의 재개 조건이 닫은 시점에 이미 충족돼 있었다. 축은 닫힌 상태이며 실증 기각도 유지된다.
  양쪽 근거와 판정 재료는 [`closed_axes.md` §3-E·§3-F](closed_axes.md), 경위는 `D-026`.
- 후속 방향 선택: 마진 순위·크기 분리 진단의 대조 정의·예산 (`D-024`·`D-025`).
  대기 중인 GPU 0 작업은 [`research_directions.md`](research_directions.md) Phase 0의 `0-7`·`0-8`.

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

_by Claude Code on nexgem-s1 at 2026-09-07_
