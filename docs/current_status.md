# Current Status

> **이 파일은 "지금 무엇을 하고 있는가"만 담는다. 100줄을 넘기지 않는다.**
> 수치·기준은 [`PROJECT.md`](PROJECT.md), 닫힌 축은 [`closed_axes.md`](closed_axes.md),
> 결정 이력은 [결정 이력](history/archive.md)가 정본이다.
> 종료된 RU의 상세는 [`history/research_units_all.md`](history/research_units_all.md)로 이관한다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-15 (KST) |
| **Status** | **WIP** · GF(GMM-based Fisher Vector) branch 후보 파일럿 진행 중 — 배경 GMM 비교 실험 재제출 대기 |
| **Host / Node** | 로그인 노드는 `nexgem`(실제 hostname, `nexgem-s1` 아님) — **연산은 반드시 `sbatch`로 제출**(로그인 노드 직접 실행 금지 확인됨). GPU 필요 시 RTX A5000 8장 노드(`gnode1-6`), 이번 파일럿은 CPU 전용 `node3`(32 CPU, 700G mem) 사용 |
| **Environment** | uv venv `.venv` · Python 3.12.11 · PyTorch 2.14.0+cu130 · 0-parameter 순수 추론. 파일럿용으로 `scikit-learn 1.9.1`, `h5py 3.16.0` 확인됨 |
| **Active Job** | 없음 (job `140385`는 큐 대기 중 사용자 취소, `scancel` 완료) |
| **회귀 테스트** | 159 tests · `OK (16 skipped)` (코드 변경 시 전수 검증, 문서 단독 변경 시 면제) — 이번 세션은 파일럿 스크립트만 다뤄 미실행 |

---

## 현재 작업: GF(GMM-based Fisher Vector) branch 후보 — 배경 GMM 파일럿 재제출 대기

**하려는 것**: ICF의 학습-파라미터-0 branch 앙상블에 Fisher Vector 계열 pooling을 새 후보(`m_gf`)로
추가하려 합니다. 원조 Fisher Vector 이론대로 GMM 배경 분포를 **매 fold context에서 재적합하지 않고,
ICF가 쓰지 않는 PathoBench dataset feature로 한 번만 고정 적합**해 결정론을 확보하는 방향으로
설계를 좁혔습니다 (`data/fixed_gmm/`에 동결 아티팩트로 저장 예정, 아직 미생성).

**확정된 설계**:
- 배경 GMM 적합용 dataset = `COMET`(60 slide) + `MUT-HET-RCC`(1,292 slide) + `PANDA`(10,614 slide) —
  ICF 미사용 dataset 중 로컬에 UNI2 feature(`features_uni_v2/*.h5`)가 이미 있는 3개, 총 patch
  약 2,840만 개. `SEAL 10`·`Primary 7`과 완전히 분리되어 오염 위험 없음.
- patch 수 기준으로는 `MUT-HET-RCC`가 68.7%로 지배적(슬라이드 수와 반대 양상) — 그래서
  "raw pooling(자연 비율 그대로)" vs "dataset당 1M로 균형 subsample" 두 버전을 `M=8` diag-GMM으로
  각각 만들어 실제로 얼마나 다른지 먼저 확인하기로 함(파일럿).
- 브랜치 이름은 `m_gf`(GMM-based Fisher Vector) — 기존 `stream_eval.py`/`fisher_basis_probe.py`의
  Fisher **선형판별**(LDA)과 명칭 혼동 방지.

**파일럿 실행 경위 (중단 지점 상세)**:
1. 처음 로그인 노드(`nexgem`)에서 백그라운드로 직접 실행 → 이 호스트는 Slurm 로그인 노드라
   연산은 `sbatch` 필수임을 뒤늦게 확인, 즉시 취소.
2. `sbatch`로 `gnode6`(GPU 미요청) 제출 → 정상 동작 확인.
3. 사용자 요청으로 `node3`(CPU 32, mem 700G, GPU 없음)로 재제출, 파일 읽기를
   32-thread 병렬로 변경 (`job 140233`, `--time=02:00:00`).
4. `job 140233` 진행: 파일 목록 수집 → train 11,369개 파일 병렬 읽기(4,449초) → validation 읽기 →
   `X1`(raw, 3,000,985) 결합 → **`GMM1`(raw pooling) 적합 완료**(1,121초, `converged=True`,
   `n_iter=36`) → `X2`(balanced, 3,000,021) 결합 → **`GMM2`(balanced) 적합 도중 2시간 제한으로
   `TIMEOUT` 강제 종료**(15:31). `X1/X2/Xv`·`GMM1` 결과는 전부 메모리에만 있었고 디스크에
   저장하지 않아 **전부 유실**.
5. 스크립트에 체크포인트(`X1/X2/Xv` → `.npy`, `GMM1/GMM2` → `.pkl`, 있으면 로드해서 이미 끝난
   단계 재실행 생략)를 추가하고 `--time=03:00:00`으로 `job 140385` 재제출 → 큐에서 `PD`(대기) 중
   **사용자 지시로 취소**(`scancel 140385`). **현재 아무 job도 돌고 있지 않고, 캐시 디렉토리도
   아직 생성되지 않아 재실행 시 파일 읽기부터 다시 시작함.**

**산출물 위치**:
- 파일럿 스크립트(체크포인트 포함 최신본): `slurm_outputs/2026-09-15/1326/gf_gmm_pilot_compare.py`
- sbatch 템플릿: `slurm_outputs/2026-09-15/1326/gf_gmm_pilot.sbatch`
  (`--nodelist=node3 --cpus-per-task=32 --mem=700G --time=03:00:00`, GPU 미요청)
- 이전 실행 로그: `slurm_outputs/2026-09-15/1326/gf_gmm_pilot-140233.out` (GMM1 결과 포함)
- **아직 정식 RU/사전등록 이전 단계입니다** — Orca 사전등록, 게이트①(직교성)·②(정보량) 절차는
  시작 전이며, 이 파일럿은 "raw vs balanced 배경 GMM이 실제로 얼마나 다른가"만 먼저 보는
  보조 실험입니다.

### Immediate Next Command

```bash
# [파일럿 재개 시] 체크포인트 적용판 재제출 (처음부터 재읽기 필요, read~74분+GMM1~19분+GMM2 추정 15~20분)
sbatch /home/kimds/ICF/slurm_outputs/2026-09-15/1326/gf_gmm_pilot.sbatch
# 완료 후 결과 확인
cat /home/kimds/ICF/slurm_outputs/2026-09-15/1326/gf_gmm_pilot_result.json
# [문서 작업만 할 시] 정합성 검사 단독 실행
.venv/bin/python -m unittest tests/test_docs_consistency.py
```

---

## 미해결 (Open Issues)

**사용자 판단 대기**
- **다음 연구 방향** — §226 Tier 1 종료. 큐의 `P1-B`(CA-R1 계열 5건) 착수 순서·예산 미결 ([`research_directions.md` §0](research_directions.md)). GF(GMM-based Fisher Vector) branch 후보가 배경 GMM 파일럿 단계로 비공식 탐색 중이나 아직 Orca 사전등록 전 (위 "현재 작업" 참조).
- **경계 판정 2건** — `closed_axes.md` §3-I(컨텍스트 라벨 통계 대 `CA-06`)·§3-J(다중 강도 브랜치 대 `P2-SELECTOR-CEILING`·`CA-09`). 판정 전 `B2`·`B3` 착수 금지.
- **게이트 ② 통계량** — 단측 `AUROC > 0.5` 유지 vs 양측 `|AUROC−0.5|` 전환 여부 (큐 `B5`).
- **재개 축 활용 여부** — `CA-04`(해소 항목 ②·③ 남음)·`CA-02`(재현 선행).

**연구상의 교착**
- **과제 특화 이득을 활용할 선택 신호 부재**: 이득은 실재하나 라벨 없이 과제를 판별할 수단이 없음 ([`closed_axes.md` `CA-R1`](closed_axes.md)).
- **모든 판정이 `hold-out 미검증`** ([`PROJECT.md` §3.1](PROJECT.md)).
- **v115~v120 6브랜치 조합 검증은 남은 과제**: 과거 기록 보존 및 개선량 기준·불확실성 분리 정합화 (출처: 결정 `D-041`).

[작성자: Antigravity / Document Agent / 미확인 (effort: 미확인) · 2026-09-10 17:15 KST]
[작성자: GitHub Copilot / Document Agent / GLM-5.3-Flash (effort: 미확인) · 2026-09-14 17:15 KST]
[작성자: Claude Code / Platform Agent / claude-sonnet-5 (effort: 미확인) · 2026-09-15 15:40 KST]
