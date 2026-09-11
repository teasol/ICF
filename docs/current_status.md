# Current Status

> **이 파일은 "지금 무엇을 하고 있는가"만 담는다. 100줄을 넘기지 않는다.**
> 수치·기준은 [`PROJECT.md`](PROJECT.md), 닫힌 축은 [`closed_axes.md`](closed_axes.md),
> 결정 이력은 [결정 이력](history/archive.md)가 정본이다.
> 종료된 RU의 상세는 [`history/research_units_all.md`](history/research_units_all.md)로 이관한다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-10 (KST) |
| **Status** | **기술 부채 8건 전수 해결 완료** · 다음 연구 방향 착수 대기 |
| **Host / Node** | `nexgem-s1` · RTX A5000 8장(23 GB) · driver 580.126.09 · Slurm 명령 없음 |
| **Environment** | uv venv `.venv` · Python 3.12.11 · PyTorch 2.14.0+cu130 · Lightning 2.6.5 |
| **Active Job** | 없음. 유휴 GPU 확인 후 필요 작업 진행 (`free_gpus.sh`) |
| **회귀 테스트** | 159 tests · `OK` (코드 변경 시 전수 검증, 문서 단독 변경 시 면제) |

---

## 현재 작업: 기술 부채 전수 해결 완료 및 후속 연구 방향 착수 준비

현재 목표였던 기술 부채 8건 해결이 전수 완료(또는 안 해도 됨 확정)되었으며 ([`PROJECT.md` §2](PROJECT.md)), 후속 연구 방향 착수를 준비한다.

### 기술 부채 현황 및 처리 상태 (8개 항목 전수 완료)

1. **§199·§200 절 번호 중복**: **해결 완료 (2026-09-10).** `history/archive.md` 4개 절 머리에 상호 안내 및 근거 축 명시 ([`closed_axes.md` §0 규칙 7](closed_axes.md)).
2. **bf16 정밀도 부채**: **해결 완료 (RU-87).** 정밀도 차는 공통 성분이므로 짝지은 Δ에서 상쇄 확인 완료 (출처: 결정 `D-037`).
3. **§226 Tier 1 독립 비교군 부재**: **안 해도 됨 확정.** RU-85~RU-90을 통해 개별 검증 및 공식 승격이 완료되어 역사적 사실로 보존됨.
4. **`adaptive_tau` 무효 인자**: **해결 완료 (오기 정정).** `history/archive.md` D-040에서 확인되었듯 `adaptive_tau`는 `config.py` 및 `voting.py`에서 정상 사용 중인 유효 파라미터임 (삭제 불요).
5. **AUROC 2대 추정기 고정 차(+8.55e-05)**: **해결 완료 (2026-09-10).** Lime이 6개 진단 스크립트(`branch_diagnostics.py` 등)의 인라인 AUROC를 공식 `src/utils/metrics.py:auroc`로 단일화 완료 (commit `8c1776c`).
6. **집계 로직 이원화 중복**: **해결 완료 (2026-09-10).** Lime이 `voting.py`에 확률 공간 공통 헬퍼(`_trimmed_mean_from_probs` 등 5종)를 추출하고 `test_pathobench.py`가 이를 직접 import하여 재사용하도록 리팩토링. `TestPathobenchAndVotingAgree`에 soft_voting 외 trimmed_mean/hard_gated/adaptive_trimmed 교차일치 테스트 3건 추가 (commit `8c1776c`).
7. **`ICF_SHAPE_SCREEN_ONLY=0` trimmed_mean 외 4종 미실측**: **안 해도 됨 확정.** 공식 운영 집계는 `trimmed_mean`이며 RU-90 층 2에서 등가성 확인 완료. 나머지 4종은 보조/닫힌 축으로 실측 불필요 판정.
8. **§226 적대적 검증 2건 미실행**: **안 해도 됨 확정.** 해당 형상 브랜치(`SH`, `SJ`)가 RU-90 거쳐 공식 7-branch 승격(`D-042`) 완료되어 과거 논증 복원 불필요 확정.

### Immediate Next Command

```bash
# [문서 작업 시] 정합성 검사만 단독 실행 (전체 회귀 스위트 불필요)
.venv/bin/python -m unittest tests/test_docs_consistency.py
# [다음 연구 착수 시] P1-B(CA-R1 계열) 또는 신규 방향 설계 (Reasoning / 사용자 판단 대기)
```

---

## 미해결 (Open Issues)

**사용자 판단 대기**
- **다음 연구 방향** — §226 Tier 1 종료. 큐의 `P1-B`(CA-R1 계열 5건) 착수 순서·예산 미결 ([`research_directions.md` §0](research_directions.md)).
- **경계 판정 2건** — `closed_axes.md` §3-I(컨텍스트 라벨 통계 대 `CA-06`)·§3-J(다중 강도 브랜치 대 `P2-SELECTOR-CEILING`·`CA-09`). 판정 전 `B2`·`B3` 착수 금지.
- **게이트 ② 통계량** — 단측 `AUROC > 0.5` 유지 vs 양측 `|AUROC−0.5|` 전환 여부 (큐 `B5`).
- **재개 축 활용 여부** — `CA-04`(해소 항목 ②·③ 남음)·`CA-02`(재현 선행).

**연구상의 교착**
- **과제 특화 이득을 활용할 선택 신호 부재**: 이득은 실재하나 라벨 없이 과제를 판별할 수단이 없음 ([`closed_axes.md` `CA-R1`](closed_axes.md)).
- **모든 판정이 `hold-out 미검증`** ([`PROJECT.md` §3.1](PROJECT.md)).
- **v115~v120 6브랜치 조합 검증은 남은 과제**: 과거 기록 보존 및 개선량 기준·불확실성 분리 정합화 (출처: 결정 `D-041`).

[작성자: Antigravity / Document Agent / 미확인 (effort: 미확인) · 2026-09-10 17:15 KST]
