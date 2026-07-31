# Current experiments

**Last updated**: `2026-08-01 13:20:00 KST`
**Architecture Version**: **v24 확정** (residual + bottleneck bag projection, `configs/train_v24_medium_bag_proj_residual.yaml`)

이 문서는 v22 시절에 확립된 실험 프로토콜(§0~§1, Stage 3 ICI 절차)과 실행 명령어를 설명합니다. **2026-08-01, 사용자 결정으로 v23-A0/v24-A0/v24-B0/v24-B1 4종의 1,000-episode paired 비교 평가는 폐기되고 v24-B1이 그대로 v24로 확정되었습니다** — 상세: [`current_status.md`](current_status.md) §3 "최종 결정". 아래 두 섹션(v23-A0, v24-A0/B0)은 폐기된 candidate의 기록이며 더 이상 active가 아닙니다. v21 retrieval 시대의 실험 기록은 [`history/v21_retrieval_experiments.md`](history/v21_retrieval_experiments.md)로 이관되었습니다.

## 폐기됨 (2026-08-01): v23-A0 exact bag-mean ablation

- Branch: `codex/v23-bag-mean`
- Config: `configs/train_v23_medium_bag_mean.yaml`
- Change: 각 bag의 `1 global + 36 slot-statistic + 3 tail = 40` token을
  arithmetic mean하여 1개 embedding으로 만든다.
- Context memory 입력 수: `context_bags × 40`이 아니라 `context_bags`.
- Query population 입력도 36 slot token 대신 동일한 40-token mean 1개를 쓴다.
- Direct ridge/attention global branch도 global spread 하나 대신 이 mean을 쓴다.
- abundance/covariance/rare side evidence와 최종 fusion은 유지하여 변경 범위를
  structured-token representation에 한정한다.
- 기본값은 `false`여서 v22 동작은 보존된다. 활성화 시 checkpoint
  `architecture_version=23`으로 기록되어 v22 checkpoint resume을 거부한다.
- original Medium context regime의 scratch 20 epochs 학습 완료. Mixed-context
  intervention은 섞지 않았다.
- 판정: 1,000 pool-400 episode의 context 40/80/160/300 paired 비교에서
  overall `+0.03` 또는 target task `+0.05`.
- 구현 검증: 전체 unittest 123개 통과 (`696.503s`).
- 완료 run: `20260731_155635`; best epoch 19
  `val_ce_loss=0.5933738`, checkpoint
  `checkpoints/20260731_155635/v23_medium_bag_mean/epoch=019-val_ce_loss=0.5934.ckpt`.
- ~~다음 실행: 동일 1,000 pool-400 episodes에서 v22/v23-A0 context
  `40/80/160/300` curve를 만들고 episode-cluster paired 비교한다.~~ **실행되지 않음 — 2026-08-01 폐기.**

## 폐기됨 (2026-08-01): v24-A0 / v24-B0 learned bag projection

- Branch: `codex/v23-bag-mean`
- v24-A0 config: `configs/train_v24_medium_bag_proj.yaml`
  - `project_structured_tokens: true`, slot 1 / density slot 1 → bag당 7 tokens
    (`1 global + 3 slot stats + 3 tails`)을 concat(7×512=3584) →
    `Linear(3584, 512)` → bag당 512-d 1토큰.
  - 완료 run: `20260731_182755`, 50 epochs, best epoch 45
    `val_ce_loss=0.5976237`.
- v24-B0 config: `configs/train_v24_medium_bag_proj_bottleneck.yaml`
  - 12 slot 유지 (40 tokens), 토큰별 전용 `Linear(512→64)` 40개 →
    concat(40×64=2560) → `Linear(2560, 512)` → bag당 512-d 1토큰.
    병목으로 projection 파라미터 ~2.62M (직결 40×512→512 ~10.5M 대비).
  - 완료 run: `20260731_201252`, 50 epochs, best epoch 46 `val_ce_loss=0.5923204`.
- 두 variant 모두 `architecture_version=24`. `mean_pool`(v23)과는 상호배타,
  v24-A0/B0 하위 variant는 state_dict 불일치로 구분됨.
- ~~판정: v23과 동일 — 1,000 pool-400 episode의 context 40/80/160/300 paired
  비교에서 overall `+0.03` 또는 target task `+0.05`.~~ **실행되지 않음 — 2026-08-01 폐기.**

## ✅ 확정됨 (2026-08-01): v24-B1 residual + bottleneck bag projection = v24

- Branch: `codex/v23-bag-mean` (main/v24로 fast-forward됨)
- Config: `configs/train_v24_medium_bag_proj_residual.yaml` (`project_structured_tokens: true`,
  `projection_bottleneck_dim: 64`, `projection_residual_mean: true`)
- v24-B0 구조(40 token × 전용 `Linear(512→64)` → concat 2560) +
  exact arithmetic mean(512d) residual shortcut → concat 3072 → `Linear(3072→512)`.
- 완료 run: `20260731_220100`, 50 epochs, best epoch 41 `val_ce_loss=0.5903045`
  (Bag-Collapse 계열 중 최저). Checkpoint
  `checkpoints/20260731_220100/v24_medium_bag_proj_residual/epoch=041-val_ce_loss=0.5903.ckpt`.
- **1,000-episode paired 합성 평가는 수행되지 않았습니다.** 확정은 train `val_ce_loss` 순위만으로
  사용자가 직접 결정했습니다 — 상세와 미완 항목은 [`current_status.md`](current_status.md) §3 "최종 결정".
- `_class_memories`의 label(context_labels) 기반 bag grouping/pooling은 이 확정으로도 바뀌지
  않았습니다 — [`current_architecture.md`](current_architecture.md) §3.5 참고.

> [!CAUTION]
> **v22/v23-A0/v24-A0/v24-B0는 모두 폐기되었습니다.** 확정 config는 위 `train_v24_medium_bag_proj_residual.yaml` 하나뿐이며, 나머지 config는 `configs/archive/v23_v24_candidates/`로 이관되었습니다. ICI용 v24 config는 아직 없습니다 — Stage 3(§3 아래)는 여전히 v22 config를 참조하는 옛 절차입니다.

---

## 0. 실험 전략: 합성 데이터로 결정하고, ICI는 최종 테스트에만 (2026-07-29 확정)

> [!IMPORTANT]
> **아키텍처 선택·하이퍼파라미터 튜닝·모든 반복 실험은 합성 데이터에서만 수행합니다.**
> **ICI 실데이터는 최종 확인 1회에만 사용합니다.**

### 왜 이렇게 나누는가

**실측 비교** (둘 다 95% bootstrap CI, 합성은 episode cluster 방식으로 보수적으로 계산):

| | 표본 | AUROC | 95% CI | **CI 폭** |
|---|---|---:|---|---:|
| ICI 5-fold (v21 Phase 6c) | 87명 | 0.5454 | [0.422, 0.664] | **0.242** |
| 합성 val (v22 기준선, 1,000 eps) | 1,000 episodes / 16,330 query | 0.7078 | [0.696, 0.719] | **0.021** |

합성 쪽 구간이 **약 12배 좁습니다** (0.021 vs 0.242). 게다가 합성은 `episodes_per_epoch`를 늘려 더 좁힐 수 있지만, ICI는 87명이 상한이라 아무리 seed를 늘려도 좁아지지 않습니다.

신호 자체도 합성 쪽이 훨씬 강합니다 (AUROC 0.71 vs ICI의 0.55). 즉 **아키텍처 변경이 실제로 효과가 있다면 합성에서 먼저, 더 뚜렷하게 보입니다.**

v21의 실패는 **검출력이 없는 지표 위에서 아키텍처를 반복 비교한 것**이었습니다. 또한 ICI를 반복해서 들여다보면 그 87명에 과적합됩니다 (선택 편향). ICI를 최종 테스트로 남겨두어야 그 숫자가 의미를 갖습니다.

### 규칙

| 단계 | 데이터 | 지표 | 반복 |
|---|---|---|---|
| 아키텍처 탐색·튜닝 | 합성 val | `scripts/evaluate_synthetic.py`의 AUROC + episode cluster CI | 자유롭게 |
| 최종 확인 | ICI 5 seed × 5 fold + 외부 코호트 | `scripts/evaluate_protocol.py` | **후보 확정 후 1회** |

- 합성 val에서 이긴 후보만 ICI로 넘깁니다. ICI 결과를 보고 다시 아키텍처를 고치기 시작하면 이 프로토콜은 무효가 됩니다.
- ICI 결과가 기대에 못 미쳐도 **그것 자체로는 아키텍처를 되돌릴 근거가 되지 않습니다** — §1에 따라 ±0.13 이내 변동은 이 코호트가 구분할 수 없는 범위입니다.
- 합성 데이터에서 검출력이 부족하면 `episodes_per_epoch`를 늘리면 됩니다. 이것이 ICI 대비 합성 데이터의 결정적 장점입니다.

---

## 1. 평가 프로토콜 (실험 시작 전 반드시 읽을 것)

### ① 이 코호트가 검출할 수 있는 효과 크기

ICI 코호트는 n=87 (positive 37 / negative 50)입니다. `scripts/power_analysis.py`로 측정한 검정력 (baseline AUROC 0.55, 모델 간 상관 ρ=0.7 — 실제 Phase 6b vs 6c의 Pearson ρ=0.737에서 추정):

| 실제 AUROC 향상 | 검출 확률(power) |
|---:|---:|
| +0.02 | 15% |
| +0.05 | 26% |
| +0.10 | 66% |
| **+0.15** | **92%** |
| +0.20 | 99% |

> [!IMPORTANT]
> **+0.13~0.15 AUROC 이상을 기대할 수 없는 실험은 이 코호트에서 돌릴 가치가 없습니다.** 그보다 작은 효과는 있어도 못 찾습니다(power < 80%). v21 시대에 쫓던 0.004~0.04 차이는 검출 확률이 15~26%로, 사실상 동전 던지기였습니다.
>
> 이 표가 뜻하는 것: 아키텍처를 조금씩 바꿔가며 ICI AUROC로 우열을 가리는 방식 자체가 **이 코호트에서는 작동하지 않습니다.** 큰 효과를 노리거나, 코호트를 키우거나, 합성 데이터처럼 n을 늘릴 수 있는 곳에서 판단해야 합니다.

재현: `python scripts/power_analysis.py`

### ② 세 가지 수치를 구분해서 볼 것

`scripts/evaluate_protocol.py`는 서로 다른 질문에 답하는 세 수치를 보고합니다:

| 수치 | 무엇을 재는가 | seed를 늘리면? |
|---|---|---|
| per-seed AUROC | 한 partition에서의 5-fold CV 결과 (87명 전원) | — |
| across-seed mean ± SD | partition/학습 재현성. SD가 크면 단일 seed 숫자는 못 믿음 | SD는 그대로, 평균의 표준오차는 감소 |
| pooled bootstrap CI | **코호트 자체의 표본 오차** | **줄어들지 않음** — 같은 87명을 재사용하므로 |

**핵심**: seed를 5개로 늘려도 CI는 좁아지지 않습니다. 87명이라는 한계는 사람을 더 모아야만 풀립니다. seed 확장은 "이 숫자가 partition 운에 좌우되는가"를 답할 뿐입니다.

### ③ 사용 가능한 자원 (v21 시대에는 1/5만 사용했음)

- **seed partition 5개**: `SEED42`, `SEED1234`, `SEED2026`, `SEED271828`, `SEED314159`. 각각 87명 전원을 5-fold로 정확히 한 번씩 덮는 독립 분할입니다 (CV0 기준 seed 간 val donor 겹침 1~5/18). **v21 실험은 전부 SEED42 하나만 썼습니다.**
- **외부 코호트**: `data/ICI_GSE285888_scConcept_512.pt` (26명, R 15 / NR 11). `ICIDataset(state='external')`로 로드되며 `scripts/test.py`가 `--validation-only` 없이 실행될 때 평가됩니다. **v21 실험은 이것도 쓰지 않았습니다.** 유일하게 진짜 독립적인 읽기이므로 CV 결과와 절대 합치지 말 것.

### ④ 보고 형식

`scripts/test.py`는 이제 모든 AUROC에 bootstrap 95% CI를 **자동으로** 붙여 출력합니다 (점추정치만 보고하는 것이 구조적으로 어렵게 만듦):

```
Fold 0 validation: accuracy=0.6092, AUROC=0.5454 [0.422, 0.664], log_loss=0.7921, ...
```

두 run을 비교할 때는 paired bootstrap 승률을 함께 봅니다 (0.5 = 구분 불가):

```bash
python scripts/compare_predictions.py predictions/<run_a>.pt predictions/<run_b>.pt
```

---

## 2. Context 구성 프로토콜 (v22)

**retrieval 없음.** 에피소드의 모든 context bag이 그대로 aggregator에 들어갑니다.

- 합성 데이터: `collate_synthetic_training_episode` / `collate_synthetic_evaluation_episode` — bag 전체 유지, 평가 시 클래스당 최소 1개를 context로 보호하고 나머지 20%(최대 20개)를 query로 사용.
- ICI 실데이터: `EvaluationEpisodeCollator` — fold의 train cohort 전체(~69명)를 context로 붙이고 held-out(17~18명)을 query로 마스킹.

v21의 K=24 retrieval을 제거한 근거는 [`current_status.md`](current_status.md) §4 참고.

---

## 3. 실험 파이프라인

### Stage 1: Medium 합성 사전학습
- **Config**: `configs/train_v22_medium.yaml`
- **실행**:
  ```bash
  CUDA_DEVICES=0 NPROC_PER_NODE=1 \
  TORCHRUN_BIN=/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun \
  NETRC=/NHNHOME/kimds/.netrc \
  scripts/launch_interactive_training.sh v22_medium configs/train_v22_medium.yaml
  ```
- **✅ v22 기준선 (2026-07-29 확정, 버그 수정 반영본)**: `val_ce_loss: 0.5946` @ epoch 13 / 20 epoch / 10,240 steps.
  체크포인트 `checkpoints/20260729_160643/v22_medium_fixed/epoch=013-val_ce_loss=0.5946.ckpt`, 로그 `logs/20260729_160643/v22_medium_fixed.out`.
- **참고 (v21 Phase 1)**: `val_ce_loss: 0.5921` @ 20 epoch / 10,240 steps — v22가 0.0025 이내로 재현했습니다. Phase 1도 full-context였으므로 **retrieval 제거가 사전학습 성능을 훼손하지 않았다는 확인**입니다.

> [!WARNING]
> **`episode_batch_size`를 바꾸면 `max_epochs`도 함께 조정할 것.** v21 Phase 5가 batch를 8→32로 올리면서 epoch 수를 그대로 둬 optimizer step이 10,240 → 2,560으로 줄었고, 그 결과 val_loss가 수렴하지 못한 채(0.606→0.608 평평한 노이즈) 학습이 끝났습니다. `steps = episodes_per_epoch / episode_batch_size * max_epochs`를 항상 확인하세요.

### Stage 2: Hard 합성 사전학습
- **Config**: `configs/train_v22_hard_realworld.yaml`
- **✅ v22 T3-3 완료 (2026-07-31)**: 50 epoch / 25,600 optimizer steps, best `val_ce_loss 0.6839` @ epoch 44.
- **1,000-episode 평가**: AUROC `0.5483 [0.538, 0.558]`, Log Loss `0.6920`; state `0.5167`, covariance `0.5103`.
- **Best checkpoint**: `checkpoints/20260731_035538/v22_hard_baseline/epoch=044-val_ce_loss=0.6839.ckpt`.
- **예측/로그**: `predictions/synthetic_v22_hard_baseline_1000ep.pt`; `logs/20260731_035538/v22_hard_baseline.out`.
- **v21 Phase 2 참고값**: `val_ce_loss 0.6845` — v22가 사실상 재현했습니다.

### Stage 2.5: 합성 검증 — **여기서 아키텍처를 결정합니다**

§0 전략에 따라 모든 아키텍처 비교는 이 단계에서 끝냅니다.

```bash
python scripts/evaluate_synthetic.py \
  --checkpoint checkpoints/<ts>/v22_medium/last.ckpt \
  --config configs/train_v22_medium.yaml \
  --output predictions/synthetic_<이름>.pt

# 두 후보 비교 (동일 config·seed로 평가해야 episode 구성이 일치)
python scripts/compare_predictions.py \
  predictions/synthetic_<A>.pt predictions/synthetic_<B>.pt
```

출력에는 전체 AUROC + **episode cluster bootstrap CI**, Log Loss, 그리고 5개 response task별 분해가 포함됩니다.

**✅ v22 공식 기준선 (2026-07-30, 1,000 episode)** — 새 후보는 이것과 비교합니다:

```text
AUROC     0.7078  95% CI [0.696, 0.719]   (1,000 episodes / 16,330 queries)
Log loss  0.6209

task             AUROC  CI폭  episodes
combined        0.8201  .039      213
composition     0.7729  .039      204
interaction     0.6628  .049      200
covariance      0.6216  .045      206   <- state와 동률로 최난이도
state           0.6215  .045      177   <-
```

예측 파일: `predictions/synthetic_v22_baseline_1000ep.pt`

> [!IMPORTANT]
> **반드시 `--val-episodes 1000`으로 평가하십시오.** 기본값 104개는 전체 CI 폭 0.074, task당 ~20 episode로 판정에 부적합합니다. 104개로 재던 이전 기준선(0.7466)은 폐기했고, task별 순위도 그때와 달라졌습니다(covariance가 state보다 나쁜 것이 아니라 동률). 상세: [`current_status.md`](current_status.md) §3.

> [!TIP]
> **⚠ 위 task별 수치로 아키텍처의 강약을 판단하지 마십시오.** 생성기가 task마다 다른 effect scale을 쓰기 때문에(composition 1.40 / state 0.45~1.00 / covariance 0.30~0.80) 이 순위는 난이도에 오염되어 있습니다.
> **effect scale을 통일해 재보면 진짜 약점은 `state` 하나입니다** — covariance는 composition과 동률입니다. 상세: [`current_status.md`](current_status.md) §3 T3-1.
> ```bash
> # task를 공정하게 비교하려면 scale을 통제할 것 (0.4 또는 0.7 권장 — 학습 범위 안)
> python scripts/evaluate_synthetic.py --checkpoint <ckpt> \
>   --config configs/train_v22_medium.yaml --val-episodes 400 --effect-scale 0.7
> ```
> 또한 task별 CI 폭이 0.045 수준이므로 task별로 **+0.05 이상** 차이만 신뢰하세요.

> [!IMPORTANT]
> **CI는 query가 아니라 episode 단위로 계산됩니다.** 한 episode 안의 query들은 같은 context set과 같은 생성 파라미터를 공유하므로 독립이 아닙니다. query 단위로 재표집하면 상관된 예측을 독립으로 취급해 구간이 실제보다 훨씬 좁게 나옵니다(실측 1.5배 과소). 검출력이 부족하면 `val_dataset_kwargs.episodes_per_epoch`를 늘리세요 — **episode 수가 실질적인 표본 크기입니다.**

### Tier 2 state 진단 — 🛑 종료 (2026-07-31)

1,000 validation episodes에서 현재 모델 state AUROC는 `0.6217 [0.597, 0.644]`였습니다. 모델이 실제 받는 global/slot-center 토큰의 context-label ridge probe도 `0.6196~0.6210`, raw mean 계열 관측 통계는 `0.5273~0.5578`로 현재 모델을 넘지 못했습니다. `0.8819~0.9013`은 정답 반응세포 마스크를 쓴 오라클이므로 목표가 아닙니다.

사전 판정 기준에 따라 state 구조 변경은 종료합니다. 다음 계획 실험은 v22 Hard 기준선(T3-3)입니다. 상세 표와 재현 명령은 [`current_status.md`](current_status.md) §3을 참고합니다.

### Stage 2.6: Hard 개선 방향 — Medium→Hard bridge attribution

ICI로 넘어가지 않고 Hard 붕괴 원인을 먼저 분해합니다. 순서는 재학습 없는 Hard state/covariance 접근성 감사와 matched-effect 평가, 그다음 signal scarcity → nuisance → geometry/scale → optimization의 누적 bridge ablation입니다. 전체 조합 탐색은 하지 않고 처음 성능이 무너지는 요인군만 one-factor로 좁힙니다.

판정은 1,000 episode paired cluster bootstrap으로 overall `+0.03` 또는 target task `+0.05` 이상일 때만 다음 구조/학습 후보로 인정합니다. Medium과 Hard 양쪽에서 후보가 확정되기 전까지 ICI는 계속 잠급니다. 진단 우선순위는 paired context-size curve → raw-cell 대 40-token information audit → token budget sweep → training update scaling입니다. 이 순서로 bag 압축, episode context 수, 전체 meta-training 예산을 분리합니다. 상세 변수 목록은 [`current_status.md`](current_status.md) §6을 참고합니다.

Context-size curve는 Hard best checkpoint, 동일 query, nested balanced context 10/20/40/80/160으로 완료했습니다. 991개 유효 episode에서 AUROC가 `0.5084→0.5193→0.5312→0.5505→0.5737`, log loss가 `0.7170→0.7047→0.6967→0.6900→0.6835`로 단조 개선됐습니다. 10→80의 `+0.0421`과 OOD 상한 80→160의 추가 `+0.0232`는 episode context 부족이 실제 병목임을 보여줍니다. 200회 paired episode bootstrap에서도 `P(40 > 80)=0.00`, `P(80 > 160)=0.00`으로 증가 방향이 일관됐습니다. 다만 160에서도 0.5737에 그쳐 context가 단독 원인은 아닙니다. 먼저 large-context 학습이 실제 사용 범위 40~80의 성능까지 높이는지 확인하고, 이후 raw-cell 대 현재 40-token information audit로 남은 병목을 분리합니다. 산출물은 `logs/v22_hard_context_curve_1000ep.csv`와 `predictions/v22_hard_context_curve/context_{10,20,40,80,160}.pt`입니다.

현재 Hard 학습은 B200 1장에서 episode batch 4, gradient accumulation 2(effective 8 episodes/update), BF16 mixed이며, 50~100 bags에서 query 5~12개를 빼 context 38~95개를 봅니다. 최대 1,500 cells/bag의 보수적 FP32 벤치에서도 총 172 bags는 80.8GB, 220 bags는 103.4GB peak로 통과했습니다. 추가로 batch 2에서 총 312 bags(context 300 + query 최대 12)는 73.3GB, 총 360 bags는 84.6GB peak로 통과했습니다.

따라서 large-context 후보는 episode batch 2, gradient accumulation 4로 기존 effective batch 8과 optimizer-step 수를 유지하고 최대 context 300까지 노출합니다. Context 300 고정은 실제 ICI의 약 69 context와 지나치게 다르므로 기존 38~95 범위를 충분히 포함하는 mixed/bucket sampling을 사용합니다. 평가는 고정 40/80/160/300으로 수행하며 40~80에서도 좋아져야 학습 개선으로 인정합니다. 짧은 checkpoint fine-tuning에서 신호를 확인한 뒤 동일 update/학습-bag 예산의 정식 재학습 비교로 승격하고, 이후 raw-cell audit로 남은 병목을 분리합니다.

구현 config는 `configs/train_v22_hard_context300.yaml`입니다. Training context 중심 `[40, 80, 120, 160, 180, 240, 300]`을 균등 선택하고 각 중심에 정수 `±5` jitter를 적용합니다. Query는 기존처럼 5~12개이며, dataset이 context+query 총 bag 수를 만들고 model interface가 실제 query 제거 후 context 범위 계약을 강제합니다. 최대 총 bag 수는 317입니다. 이 sampling은 train split에만 적용되며 validation/test 분포는 바뀌지 않습니다.

Batch 2/accumulation 4는 유지하되, 대형 online generation의 transient buffer 중첩을 피하기 위해 whole-batch `cuda_prefetch`와 batch 내부 parallel CUDA generation을 모두 끕니다. 두 episode는 순차 생성 후 stack되므로 실제 forward/backward와 gradient 평균은 여전히 batch 2입니다. 최대 경계 `(2, 317, 1500, 512)`, context 305의 generation+forward/backward smoke는 74.5GB로 통과했습니다. Logger는 외부 인증이 필요 없는 local CSV를 사용합니다.

Hard best epoch 44 checkpoint에서 epoch 45~49를 잇는 5-epoch fine-tuning은 완료됐습니다. Epoch별 val CE는 `0.6867/0.6889/0.6853/0.6864/0.6890`이며 best epoch 47도 기존 Hard best `0.6839`를 넘지 못했습니다. Best fine-tuned checkpoint의 pool-400, 1,000-episode AUROC는 context 40/80/160/300에서 `0.5331/0.5553/0.5842/0.5977`, log loss는 `0.6992/0.6901/0.6798/0.6762`입니다. 기존 curve는 pool 220이라 엄밀한 paired 비교가 아니므로, 동일 pool 400 원본 checkpoint 대조 평가를 PID `2530921`로 실행 중입니다.

Medium에서도 같은 context 분포를 시험하는 파생 config `configs/train_v22_medium_context300.yaml`을 추가합니다. 공식 Medium epoch 13 checkpoint에서 남은 epoch 14~19를 fine-tune하고, 원본/fine-tuned checkpoint를 동일 pool 400·context 40/80/160/300에서 비교합니다. 성공 기준은 실제 ICI 범위인 40/80에서의 paired 개선이며, state oracle-mask AUROC 0.9013과 latent 1.0은 모델이 받지 못하는 정보를 사용하므로 참고용 비현실적 상한으로만 표시합니다.

Medium 원본 checkpoint의 1,000-episode curve는 context 40/80/160/300에서 AUROC `0.6757/0.7204/0.7654/0.7989`, log loss `0.6456/0.6100/0.5817/0.5639`입니다. 즉 현재 모델 자체가 많은 labeled context를 활용해 300에서 약 0.80까지 올라갑니다.

초기 mixed-context fine-tuning은 epoch 19까지 완료됐고 best epoch 14 CE `0.5953`도 원본 `0.5946`을 넘지 못했습니다. 100 epochs로 확장했지만 train/validation이 함께 plateau하여 epoch 31에서 중단했습니다. Epoch 31 CE `0.5947`도 공식 `0.5946`과 동률입니다.

**2026-08-01 갱신**: 이 단락이 가리키던 T5-A(typed bag-preserving branch)/T5-B(distribution sketch)는 실행되지 않았습니다. 대신 v23-A0/v24-A0/v24-B0/v24-B1 bag-collapse 계열을 먼저 시험했고, v24-B1(residual + bottleneck projection)이 train CE 최저치로 v24로 확정되었습니다 — 4종 paired 비교 및 T5-A/B는 사용자 결정으로 폐기했습니다. 기록은 [`history/architecture_v23_candidates.md`](history/architecture_v23_candidates.md), 결정 근거는 [`current_status.md`](current_status.md) §3을 참고하세요. T5-A가 다루던 "label 기반 class-memory 압축" 자체는 v24에서도 그대로 남아 있으므로, 그 지점을 다시 고치려면 T5-A를 별도로 재검토해야 합니다.

### Stage 3: ICI 실데이터 — 최종 테스트 (후보 확정 후 1회)

> [!IMPORTANT]
> **2026-08-01: 후보는 v24로 확정됐지만 이 config들은 아직 v22용입니다.** ICI를 v24로 돌리려면 `project_structured_tokens: true` 등 v24 model kwargs를 반영한 `train_v24_ici_finetune.yaml`/`train_v24_ici_scratch.yaml`을 먼저 만들어야 합니다. ICI는 여전히 잠금 상태입니다 — [`current_status.md`](current_status.md) §3/§10.

- **Config**: `configs/train_v22_ici_finetune.yaml` (fine-tune), `configs/train_v22_ici_scratch.yaml` (scratch 대조군).
  fold와 seed는 config에 박아두지 않고 **`--cv` / `--seed`로 주입**합니다 (per-fold config 5개를 두던 방식은 폐기). config의 `seed: 42` / `cv: 0`은 아무것도 지정하지 않았을 때의 기본값일 뿐입니다.

- **실행 (전체 sweep: 5 seed × 5 fold = 25 run)**:
  ```bash
  # scratch
  scripts/launch_ici_protocol.sh

  # v22 사전학습 체크포인트에서 미세조정
  PRETRAINED_CKPT=/abs/path/to/v22_pretrain.ckpt scripts/launch_ici_protocol.sh

  # 일부만
  SEEDS="42 1234" scripts/launch_ici_protocol.sh
  ```
  seed 내부의 5 fold는 동시 실행하고, seed끼리는 순차 실행합니다 (GPU 25중 점유 방지). run 목록은 `logs/v22_ici_sweep_manifest.tsv`에 기록됩니다.

- **seed별 평가** (manifest의 체크포인트 경로 사용):
  ```bash
  python scripts/test.py \
    --checkpoints <해당 seed의 fold0..4 last.ckpt 5개> \
    --config configs/train_v22_ici_finetune.yaml \
    --precision bf16-mixed --validation-only \
    --output predictions/v22_ici_seed<SEED>.pt
  ```
  `--validation-only`를 빼면 외부 코호트(GSE285888) 추론까지 수행하고 fold 앙상블 결과를 함께 저장합니다.

- **집계**:
  ```bash
  python scripts/evaluate_protocol.py \
    --predictions predictions/v22_ici_seed*.pt \
    --external predictions/v22_ici_external.pt
  ```

- **참고 기준선 (v21 Phase 6c, 동일한 no-retrieval 구성, SEED42 단일)**: AUROC 0.5454 [0.422, 0.664] / Log Loss 0.7921 / Accuracy 0.6092.
  §0-①에 따라 v22가 이보다 **+0.13 이상** 좋지 않으면 이 코호트에서는 "개선됐다"고 말할 수 없습니다.

---

## 4. 합성 데이터 파라미터

`configs/data/medium.yaml` 기준 (ICI 유사 에피소드 생성):

```text
episodes_per_epoch : 4096
num_bags           : [60, 100]
num_cells          : [500, 1000]
latent_dim         : 32      output_dim : 512
class_separation   : [0.5, 1.4]
donor_shift_scale  : 0.35
rare_response_probability : 0.15    rare_response_fraction : [0.02, 0.08]
observation_noise  : 0.01
response_task_probabilities : composition / state / covariance / interaction / combined
```

ICI 실데이터: 87명 환자, donor당 1,000 세포 샘플링, 512-dim scConcept 임베딩, 5-fold CV (`data/ICI_CVOnly_scConcept_512`).

---

## 5. 검증 스위트

```bash
timeout 1500s /NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"
```

- `tests/test_base_model.py` — aggregator/meta-classifier 계약, `architecture_version == 22`
- `tests/test_model_interface.py` — 손실 항, 체크포인트 버전 게이트 (v21 거부 / v22 수용)
- `tests/test_batched_episode_forward.py` — 4D batched forward가 에피소드별 독립 forward와 일치하는지 (v21 retrieval 스위트 대체)
- `tests/test_evaluation_protocol.py` — AUROC/Log Loss/bootstrap CI 정확성, **"n이 작으면 CI가 넓어진다"는 프로토콜의 핵심 전제** 검증
- `tests/test_ici_dataset.py`, `tests/test_synthetic_variable_cells.py`, `tests/test_scheduler.py`, `tests/test_checkpoint_callback.py`, `tests/test_learnability_ladder.py`

---

## 6. 평가 도구 요약

| 스크립트 | 용도 |
|---|---|
| `scripts/power_analysis.py` | 실험 전: 이 코호트가 검출 가능한 효과 크기 확인 |
| `scripts/launch_ici_protocol.sh` | 5 seed × 5 fold sweep 실행 |
| `scripts/test.py` | 체크포인트 → 예측 파일 (AUROC에 CI 자동 부착) |
| `scripts/evaluate_protocol.py` | 다중 seed 집계 + 외부 코호트 보고 |
| `scripts/evaluate_synthetic.py` | 합성 val 평가 (episode cluster CI + task별 분해) |
| `scripts/diagnose_state_upper_bound.py` | state 관측 가능/model-input/oracle descriptor 상한 비교 |
| `scripts/compare_predictions.py` | 두 run 비교 (CI + paired bootstrap 승률, episode 있으면 cluster) |
| `src/utils/metrics.py` | 공용 지표 구현 (rank 기반 AUROC, cluster bootstrap) |
