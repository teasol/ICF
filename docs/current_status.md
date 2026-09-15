# Current Status

> **이 파일은 "지금 무엇을 하고 있는가"만 담는다. 100줄을 넘기지 않는다.**
> 수치·기준은 [`PROJECT.md`](PROJECT.md), 닫힌 축은 [`closed_axes.md`](closed_axes.md),
> 결정 이력은 [결정 이력](history/archive.md)가 정본이다.
> 종료된 RU의 상세는 [`history/research_units_all.md`](history/research_units_all.md)로 이관한다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-16 08:45 KST |
| **Status** | **WIP** · GF(GMM-based Fisher Vector) 후보의 배경 GMM 파일럿 재개 준비. 데이터 전송·무결성 확인 완료, 실행 스크립트·의존성·사전등록 미완료 |
| **Host / Node** | `NEXGEM` · NVIDIA B200 8장(각 183,359 MiB, driver 580.95.05) · AMD EPYC 9365 72 cores · RAM 2.2 TiB. `sbatch`·`squeue` 없음 |
| **Environment** | uv venv `.venv` · Python 3.12.3 · PyTorch 2.14.0+cu130 · `h5py` 3.16.0 · `scikit-learn` 미설치 |
| **Active Job** | GF 관련 job 없음. 보이는 `tmux` 2개는 다른 저장소 `TIRANOS` 작업 |
| **회귀 테스트** | 직전 코드 기준 159 tests · `OK (16 skipped)`. 이번 세션은 데이터 검사와 문서 변경만 수행 |

---

## 현재 작업: GF(GMM-based Fisher Vector) 배경 GMM 파일럿 재개 준비

**목적**: 학습 파라미터 0개의 ICF branch 후보 `m_gf`를 검토하기 전에, 외부 배경 데이터에서
고정 적합한 GMM이 dataset 자연 비율과 dataset별 균형 표본에서 얼마나 달라지는지 탐색한다.
기존 `stream_eval.py`의 Fisher 선형판별(LDA)과 구분하기 위해 후보 이름은 `m_gf`를 사용한다.

**파일럿에서 비교하려던 구성**:
- `COMET`·`MUT-HET-RCC`·`PANDA`의 UNI2 patch feature만 사용한다. Primary 7·SEAL 10은
  배경 GMM 적합과 파일럿 선택에 사용하지 않는다.
- `M=8` diagonal GMM 두 개를 비교한다: 자연 patch 비율을 유지한 약 3M 표본과 dataset마다
  약 1M을 뽑은 균형 3M 표본.
- 이 단계는 탐색이다. GF는 후보 큐·RU·사전등록·게이트 ①·게이트 ②를 아직 시작하지 않았다.
  두 GMM의 차이를 판정할 비교 통계와 중단 조건도 실행 전에 고정해야 한다.

### 2026-09-16 — 데이터 전송 및 무결성 확인

- 데이터 루트: `/NHNHOME/BASE/kimds/Data/PathoBench/gf_background/`
- `COMET`: HDF5 60개 · 1,298,533 patches · 7.6 GiB
- `MUT-HET-RCC`: HDF5 1,292개 · 19,507,690 patches · 114 GiB
- `PANDA`: HDF5 10,614개 · 7,598,821 patches · 45 GiB
- 합계: HDF5 11,966개 · 28,405,044 patches · 166 GiB. 모든 파일에서 `features`가
  `float32`·1536차원이며, 파일 열기와 첫·마지막 row 유한값 검사를 통과했다.
- 원본 체크섬 목록이 없어 원본과 byte-for-byte 비교는 미수행이다.

### 이전 실행과 현재 차이

- 이전 `node3` job `140233`은 raw GMM 적합까지 완료했으나 balanced GMM 도중 2시간 제한으로
  종료됐다. 당시 표본·모델이 메모리에만 있어 결과는 유실됐다.
- 체크포인트를 추가한 job `140385`는 큐 대기 중 사용자 지시로 취소됐다.
- 이전 파일럿 스크립트·sbatch·로그는 `slurm_outputs/2026-09-15/1326/`에 있었다고 기록됐지만
  현재 서버와 Git에는 없다. `/home/kimds/ICF` 경로도 존재하지 않는다.
- 현재 서버에는 Slurm CLI와 `scikit-learn`이 없다. 데이터만 전송된 상태이므로 이전 명령을
  그대로 실행할 수 없다.

### Immediate Next Command

```bash
# 다음 세션의 첫 확인: 데이터·환경·누락된 실행 자산을 한 번에 확인
cd /NHNHOME/WORKSPACE/kimds/ICF && \
test -d /NHNHOME/BASE/kimds/Data/PathoBench/gf_background && \
.venv/bin/python --version && \
(.venv/bin/python -c 'import sklearn' || echo 'MISSING: scikit-learn') && \
(test -f scripts/analysis/gf_gmm_pilot_compare.py || echo 'MISSING: pilot script')
```

**재개 순서**:
1. Orca가 GF의 프로젝트 정체성(외부 데이터로 적합한 고정 GMM과 "학습 파라미터 0개") 경계,
   파일럿 질문·비교 통계·예산·중단 조건·결과별 후속 행동을 확정하고 후보 큐/RU 필요성을 판정한다.
2. 승인된 사양을 Lime이 재현 가능한 스크립트로 복구하거나 재구현한다. 표본과 GMM을 단계별로
   체크포인트하고 대용량 산출물은 Git 밖의 `gf_background/fixed_gmm/`에 저장한다.
3. Platform이 이 서버의 실행 방식을 확정하고 필요한 `scikit-learn` 환경을 준비한 뒤 파일럿을
   실행한다. 결과를 보기 전에는 비교 기준을 바꾸지 않는다.

---

## 미해결 (Open Issues)

- 후속 연구 방향은 아직 미확정이다. 큐 `P1-B` 착수 순서·예산 또는 GF 신규 탐색 중 무엇을
  현재 목표로 둘지 사용자·Reasoning 판정이 필요하다.
- GF의 고정 외부 GMM이 `PROJECT.md`의 학습 파라미터 0개 정체성과 `CA-14` 경계 밖인지 미판정이다.
- `closed_axes.md` §3-I·§3-J, 게이트 ② 통계량, `CA-04`·`CA-02` 활용 여부도 종전대로 미해결이다.
- 모든 후보 판정은 SEAL 10 hold-out 미검증 상태를 유지한다.

[작성자: Antigravity / Document Agent / 미확인 (effort: 미확인) · 2026-09-10 17:15 KST]
[작성자: GitHub Copilot / Document Agent / GLM-5.3-Flash (effort: 미확인) · 2026-09-14 17:15 KST]
[작성자: Claude Code / Platform Agent / claude-sonnet-5 (effort: 미확인) · 2026-09-15 15:40 KST]
[작성자: Codex / Reasoning Agent (Orca) / GPT-5 (effort: 미확인) · 2026-09-16 08:45 KST]
