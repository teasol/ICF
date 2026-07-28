# Candidate A vs Candidate B 20-Epoch Short Training Experiment & Selection

Last updated: 2026-07-26

본 문서는 BagPFN Architecture v19의 State/Covariance 병목을 해결하기 위해 수행된 **Candidate A (Gated Distance)** 및 **Candidate B (Learned Head)** 20-epoch short training 비교 실험 결과를 정리한다. 본 실험 결과를 바탕으로 Candidate B가 Architecture v20의 표준 구조로 선정되어 승격되었다.

---

## 1. 실험 배경 및 진단 (Phase 1)

Architecture v19 100-epoch production run (`epoch=052-val_ce_loss=0.5966.ckpt`) 분석 결과:
- **성능 분석**: Composition (0.8010), Combined (0.7876), Interaction (0.7581)은 높으나, State (0.6299)와 Covariance (0.6085)가 주 병목으로 확인됨.
- **진단 스크립트 실행 (`scripts/diagnose_v19_branches.py`)**:
  - `Covariance` task에서 고정 CSP relation (CovRel)의 AUROC가 0.6270으로, 기존 CovRidge (0.5700) 대비 뛰어난 성능 제공.
  - CovRidge와 CovRel 간 logit 상관관계 $r \approx 0.494$로 적절한 상보적 정보 제공 확인.
  - Context Class Separation Margin이 평균 1.1885로 수치적으로 잘 조건화되어 있음을 확인.

---

## 2. 후보 아키텍처 구조

1. **Candidate A (`gated_distance`, Config: `configs/train_v19_candidate_a_short20.yaml`)**
   - Context class margin 기반 dynamic gating을 통해 고정 CSP residual distance의 반영 비율을 제어.
2. **Candidate B (`learned_head`, Config: `configs/train_v19_candidate_b_short20.yaml`)**
   - CSP projection 저차원 feature $[d_0, d_1, d_0 - d_1, \text{sep}]$를 2-layer MLP relation head로 투영하여 trainable evidence로 변환.

---

## 3. 20-Epoch Short Training 결과 비교 (Phase 3)

NVIDIA B200 1장에서 동일한 validation bank (seed 50042)로 20-epoch 비교 학습을 진행하였다.

| Metric / Task | Baseline (100e Ep 52) | Candidate A (Gated Ep 8) | Candidate B (Learned Head Ep 13) | 개선 폭 (vs Baseline) |
|---|---:|---:|---:|---:|
| **val_ce_loss** | 0.5966 | 0.5965 | **0.5925** | **-0.0041 (최우수)** |
| **Overall AUROC** | 0.7299 | 0.7423 | **0.7513** | **+0.0214 (최우수)** |
| **Composition AUROC** | 0.8010 | **0.8044** | 0.8039 | +0.0029 |
| **State AUROC** | 0.6299 | 0.6427 | **0.6583** | **+0.0284 (최우수)** |
| **Covariance AUROC** | 0.6085 | **0.6193** | 0.6115 (CovRel: **0.6566**) | **+0.0108** |
| **Interaction AUROC** | 0.7581 | **0.7605** | 0.7527 | - |
| **Combined AUROC** | 0.7876 | 0.8042 | **0.8290** | **+0.0414 (최우수)** |

### 실행 기록 & Artifacts
- **Candidate A W&B**: <https://wandb.ai/teasol/ICF/runs/isb85jld> (`logs/20260726_v19_candidate_a_gated_20e/`)
- **Candidate B W&B**: <https://wandb.ai/teasol/ICF/runs/sv026vyq> (`logs/20260726_v19_candidate_b_head_20e/`)

---

## 4. 최종 판정 및 Architecture v20 선정 이유

1. **Candidate B의 압도적 성과**:
   - 20-epoch 학습만으로 `val_ce_loss` **0.5925** 달성 (기존 100-epoch 베이스라인 0.5966 경신).
   - `Overall AUROC` 0.7513 (+0.0214) 상승.
   - 주요 병목이었던 `State AUROC`가 0.6583 (+0.0284), `Combined AUROC`가 0.8290 (+0.0414)으로 큰 폭으로 향상됨.
2. **Architecture v20 승격 결정**:
   - Candidate B의 Learned CSP Head (`mode: learned_head`)를 표준으로 수용하여 **Architecture v20**으로 확정/통합함.
