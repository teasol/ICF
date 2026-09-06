# Current Status

> **이 파일은 "지금 무엇을 하고 있는가"만 담는다. 80줄을 넘기지 않는다.**
> 수치·기준은 [`PROJECT.md`](PROJECT.md), 닫힌 축은 [`closed_axes.md`](closed_axes.md),
> 결정 이력은 [`decisions.md`](decisions.md)가 정본이다. **여기에 복사하지 않는다.**
> 종료된 RU의 상세는 [`history/research_units_all.md`](history/research_units_all.md)로 이관한다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-06 (KST) |
| **Status** | WIP — 문서 체계 재구성 진행 중 |
| **Host / Node** | 로그인 노드 `nexgem` (GPU 없음) · 계산은 Slurm `batch` 파티션에 `sbatch` 제출 |
| **Environment** | uv venv `.venv` · Python 3.12.11 · PyTorch 2.14.0+cu130 · Lightning 2.6.5 |
| **Active Job** | 없음 |
| **회귀 테스트** | 147 tests · `OK` (2026-09-06 node2, 22.2s) |

---

## 현재 목표

**문서·진행 체계 정비.** 완료 조건·예산·중단 조건은 [`PROJECT.md` §2](PROJECT.md)에 있다.

> **연구 방향(다음 현재 목표) 슬롯은 비어 있다.** 정비 완료 후 사용자가 설정한다.
> 후보와 근거: [`research_directions.md`](research_directions.md) ·
> 잔여 거리는 [`PROJECT.md` §1](PROJECT.md) (SEAL hold-out 기준 ABMIL까지 **−2.94%p**).

## 진행 중 RU

| RU | 제목 | 상태 |
|:---|:---|:---|
| `RU-79` | 문서 체계 전면 재구성 및 RU 프로세스 도입 | 진행 중 |

착수 전 RU 카드 등록은 필수다 — 절차는 [`agent_handoff.md` §1](agent_handoff.md)에 있다.


## 실행 환경 — Slurm 클러스터 (2026-09-06 실측으로 정정)

`nexgem`은 **로그인 노드**이고 GPU가 없다. 모든 계산은 `batch` 파티션에 `sbatch`로 제출하며,
제출 전 사용자 승인을 받는다. 노드 스펙·선택 우선순위·로그 규약은 `/home/kimds/slurm_rules.md`가 정본이다.

- **GPU 차단 해소.** `.venv`의 `torch 2.14.0+cu130`이 `gnode6`(H100 80GB · 드라이버 595.71.05 ·
  CUDA 13.2)에서 재설치 없이 동작한다 — `cuda_available=True`, 4096² matmul 성공(job 131263).
  이전 기록의 "드라이버 550"은 로그인 노드 값이라 계산 노드에 해당하지 않았다.
- **`gnode1`~`gnode5`는 미확인.** cu130은 드라이버 ≥580을 요구한다. 처음 쓰기 전에 점검 job으로
  `torch.cuda.is_available()`를 실측한다 — `device_count()`만 보고 판단하지 않는다.
- **낮은 등급부터 쓴다.** GPU 불필요 작업은 `node1`~`node5`(GPU 없음)로 보내고, GPU가 필요하면
  A5000(`gnode1-4`) → A6000(`gnode5`) → H100(`gnode6`) 순으로 올라간다.

### Immediate Next Command

```bash
# 작업에 충분한 가장 낮은 등급의, 가장 한산한 노드를 고른다.
# GPU 트랙이면 그 노드에서 cuda_available=True를 먼저 확인한다.
sinfo -N -o "%N %C %m %G %t" -n node1,node2,node3,node4,node5,gnode1,gnode2,gnode3,gnode4,gnode5,gnode6
```

---

## 미해결 (Open Issues)

**사용자 판단 대기**
- **닫힌 축 경계 미확정 4건** — `PSW`·`TGW` / `FC` / `P3-LIMIT-CURVE` / `LSAK`.
  쟁점과 양쪽 논거는 [`closed_axes.md` §3](closed_axes.md)에 정리되어 있다.
- **다음 연구 방향 설정** — 위 "현재 목표" 참조.

**연구상의 교착**
- **과제 특화 이득을 활용할 선택 신호가 없다.** §221(SHJ)과 §225(subsampling)가 같은 벽에
  막혔다 — 이득은 실재하나 라벨 없이 과제를 판별할 수단이 없다.
  ([`closed_axes.md` `CA-R1`](closed_axes.md))
- **모든 판정이 `hold-out 미검증`** 이다 ([`PROJECT.md` §3.1](PROJECT.md)).

**기술 부채**
- §226 Tier 1 3건(`AKS`·`MDX`·`LID`)이 **동일 에이전트 한 배치 산출**이며 독립 비교군이 없다.
- §226 적대적 검증 2건 미실행 (`research_directions.md`에 `미검증` 명시).
- §217~§221의 SHJ 수치는 **bf16 경로 산출값**으로 fp32와 최대 0.5%p 차이날 수 있다.
- `adaptive_trimmed`의 `adaptive_tau`는 무효 인자(죽은 코드).
- `history/archive.md`에 **§199·§200 절 번호가 각각 중복** ([`closed_axes.md` §4](closed_axes.md)).

_by Claude Opus 5 on nexgem at 2026-09-06_
