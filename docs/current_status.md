# Current Status

> **이 파일은 "지금 무엇을 하고 있는가"만 담는다. 100줄을 넘기지 않는다.**
> 수치·기준은 [`PROJECT.md`](PROJECT.md), 닫힌 축은 [`closed_axes.md`](closed_axes.md),
> 결정 이력은 [결정 이력](history/archive.md)가 정본이다.
> 종료된 RU의 상세는 [`history/research_units_all.md`](history/research_units_all.md)로 이관한다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-16 08:52 KST |
| **Status** | **WIP** · GF(GMM-based Fisher Vector) 후보의 첫 실험인 자연 비율 대 균형 배경 GMM 비교 대기. 데이터 준비 완료, 실행 스크립트·`scikit-learn` 미준비 |
| **Host / Node** | `NEXGEM` · NVIDIA B200 8장(각 183,359 MiB, driver 580.95.05) · AMD EPYC 9365 72 cores · RAM 2.2 TiB. `sbatch`·`squeue` 없음 |
| **Environment** | uv venv `.venv` · Python 3.12.3 · PyTorch 2.14.0+cu130 · `h5py` 3.16.0 · `scikit-learn` 미설치 |
| **Active Job** | GF 관련 job 없음. 보이는 `tmux` 2개는 다른 저장소 `TIRANOS` 작업 |
| **회귀 테스트** | 직전 코드 기준 159 tests · `OK (16 skipped)`. 이번 세션은 데이터 검사와 문서 변경만 수행 |

---

## 현재 작업: GF 배경 GMM의 표본 구성 비교 실험

**연구 질문**: 고정 Fisher Vector 기저를 만들 때 전체 patch에서 자연 비율로 뽑은 GMM과
dataset별 동일 수를 뽑은 GMM이 실제로 다른 분포를 학습하는가. `MUT-HET-RCC`가 전체 patch의
68.7%를 차지하므로, 자연 비율 GMM이 한 dataset에 종속되는지 먼저 확인하려는 진단 실험이다.
후보 branch 이름은 기존 Fisher 선형판별(LDA)과 구분해 `m_gf`로 쓴다.

### 실행할 두 조건

- **GMM-Raw**: 세 dataset의 전체 patch 풀에서 자연 비율을 유지해 약 3M개를 결정론적으로 표본화한다.
  이전 실행의 실제 표본 수는 3,000,985개였다.
- **GMM-Balanced**: `COMET`·`MUT-HET-RCC`·`PANDA`에서 각각 약 1M개를 결정론적으로 표본화한다.
  이전 실행의 실제 합계는 3,000,021개였다.
- 나머지 조건은 동일하게 고정한다: UNI2 1536차원 feature, `M=8`, diagonal covariance,
  같은 seed·초기화·수렴 설정·공통 validation 표본. Primary 7·SEAL 10과 라벨은 사용하지 않는다.

### 기록할 측정값과 산출물

- 각 GMM의 `converged`, `n_iter`, wall time, 전체와 dataset별 validation 평균 log-likelihood.
- component weight와 dataset별 component responsibility/점유율. 특정 dataset 또는 component로의
  붕괴 여부를 함께 본다.
- 두 GMM의 component를 정렬한 뒤 weight·mean·diagonal covariance 차이를 기록한다.
- `X_raw`·`X_balanced`·공통 validation 표본과 각 GMM을 단계별 체크포인트하고, 요약 결과를
  JSON으로 저장한다. 대용량 산출물은 Git 밖의 `gf_background/fixed_gmm/`에 둔다.
- RU 유형은 **진단**이다. 이 결과만으로 GF를 채택·기각하거나 성능 향상·기전을 주장하지 않는다.
  실제 차이가 작으면 두 표본 정책이 유사하다고 보고 더 단순한 정책을 후속 후보로 삼고,
  차이가 크면 dataset별 validation 결과를 근거로 고정 GMM 정책을 별도 판정한다.

### 파일럿 이후 예정한 GF branch 실험

1. 진단 결과로 배경 GMM 하나를 고정하고 재학습 없이 모든 fold에서 동일하게 사용한다.
2. Lime이 Fisher Vector의 mean·variance gradient, power/L2 정규화와 ICF margin 연결을 구현·검증한다.
3. Primary 7에서 `SCREEN_ONLY=1`로 게이트 ① 직교성만 측정한다. 탈락하면 성능을 보지 않고 종료한다.
4. 통과할 때만 게이트 ② 단독 정보량을 측정하고, 채택 조건을 충족하면 공식 7-branch 대비
   8-branch 대응 Δ를 50 folds에서 평가한다. 승격 기준과 불확실성 보고는 `PROJECT.md` §4\~§5를 따른다.
5. SEAL 10은 이 전 과정에서 열지 않는다. 기전은 별도 증거 전까지 `기전 미확인`으로 둔다.

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
# Lime에 위 두 조건·측정값·체크포인트 계약으로 파일럿 실행기를 재구현하도록 전달
bash scripts/call_agent.sh lime "docs/current_status.md의 GF 배경 GMM 비교 실험을 그대로 구현하고, 실행 전 재현 명령과 자원 추정을 보고할 것"
```

Lime 구현 뒤 Platform이 현재 서버에 `scikit-learn` 환경을 준비하고 파일럿을 실행한다. 두 GMM
결과를 모두 받기 전 비교 항목을 바꾸지 않으며, 중간 실패는 체크포인트에서 재개한다.

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
[작성자: Codex / Reasoning Agent (Orca) / GPT-5 (effort: 미확인) · 2026-09-16 08:52 KST]
