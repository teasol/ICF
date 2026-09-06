# Current Status

> **이 파일은 "지금 무엇을 하고 있는가"만 담는다. 80줄을 넘기지 않는다.**
> 수치·기준은 [`PROJECT.md`](PROJECT.md), 닫힌 축은 [`closed_axes.md`](closed_axes.md),
> 결정 이력은 [`decisions.md`](decisions.md)가 정본이다. **여기에 복사하지 않는다.**
> 종료된 RU의 상세는 [`history/research_units_all.md`](history/research_units_all.md)로 이관한다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-06 (KST) |
| **Status** | WIP — 문서 체계 재구성 진행 중 |
| **Host / Node** | `nexgem-s1` · 8× RTX A5000 23GB · driver **550.127.08** (2026-09-06 실측) |
| **Environment** | uv venv `.venv` · Python 3.12.11 · PyTorch 2.14.0+cu130 · Lightning 2.6.5 |
| **Active Job** | 없음 |
| **회귀 테스트** | 137 tests · `OK` (2026-09-06 CPU 경로 재확인) |

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


## 🛑 BLOCKER — 이 노드에서 GPU를 쓸 수 없다 (2026-09-06 실측)

`.venv`의 `torch 2.14.0+cu130`은 CUDA 13.0 런타임(드라이버 ≥ 580)을 요구하는데 `nexgem-s1`의
드라이버는 **550.127.08 (CUDA 12.4)** 이다. `torch.cuda.is_available()`가 **False**이고
(`found version 12040`), `device_count()`는 8을 반환하나 실제 할당에서 실패한다. 이 머신의 다른
인터프리터도 전부 불가하다 — `BagPFN`·`TabPFN` 2.12.1+cu132, `TIRANOS`·`VFI` 2.13.0+cu132.

해소 경로: (a) gnode4 이동, (b) 드라이버 550에 맞춘 torch 재설치, (c) 드라이버 상향. 셋 중 하나
전까지 **350-fold 스윕·`SCREEN_ONLY` 실행은 집행 불가**하다. 오프라인 재집계와 회귀 테스트는 정상이다.
`NGPU`는 `scripts/node_env.sh`가 보고하는 값을 쓰고 문서의 과거 값을 신뢰하지 않는다.

### Immediate Next Command

```bash
# GPU 트랙에 진입하기 전 매번 1회 — 차단이 풀렸는지부터 확인한다
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1 && \
  .venv/bin/python -c "import torch;print('cuda',torch.cuda.is_available(),'| build',torch.version.cuda)"
```

`cuda True`가 나와야 GPU 트랙 진입이 가능하다. 현재 목표(문서 정비)는 GPU를 쓰지 않는다.

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
- `history/archive.md`에 **§199·§200 절 번호가 각각 중복**된다
  ([`closed_axes.md` §4](closed_axes.md)).

_by Claude Opus 5 on nexgem-s1 at 2026-09-06_
