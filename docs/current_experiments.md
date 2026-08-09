# Current experiments — CV-only 이후 (2026-08-09~)

CV-only 이전(v22~v39, 합성 중심 판정)은
[`history.md`](history.md).
**그 문서의 판정 절차는 폐기됐다** — 합성 지표로 arm을 고르는 방식이 반복적으로 실패했다.

---

## 1. 판정 기준 (2026-08-09 확정, §71-4)

> [!IMPORTANT]
> **판정은 SEAL 대상 10개 task의 macro-AUROC 평균으로 한다.**
> ```bash
> bash scripts/eval_seal_tasks.sh <gpu> <ckpt> <config> <tag> <task>...
> ```
> 대상은 `docs/seal_univ2_baseline_17tasks.csv`의 **`in_seal=yes` 10개**뿐이다 — SEAL과
> **같은 코호트·같은 공식 50-fold**로 비교 가능한 행. 나머지 7개는 SEAL에 대응 수치가 없다.

**금지 사항 (전부 실측으로 무너진 판정 방식)**

| 하지 말 것 | 근거 |
|---|---|
| **합성 val_ce로 arm 고르기** | v37은 val_ce가 더 좋았으나 50-fold는 −0.0068 (§65) |
| **합성 val_AUROC로 arm 고르기** | CV-only는 ep0 0.8885 = ep49 0.8882로 평평한데 er_status는 +0.037 (§69-6) |
| **er_status 단일로 판정** | §70이 K 증설을 "무의미(+0.004)"로 오판, 10개로는 +0.0127에 9/10 (§71) |
| **단일 측정으로 단정** | 기저 요동이 ±0.05, seed 반복 필수 (§69-3) |
| **ridge-only 진단치를 기대값으로** | K 64→128 예측 +0.016 vs er_status 실측 +0.004 (§70-2) |
| **학습 길이가 다른 arm 비교** | control은 항상 같은 epoch 수로 새로 학습 (§42-43, §65) |

---

## 2. 표준 실행

**학습** (CV-only, 50 epoch, 약 60~80분/arm, 1 GPU)
```bash
CUDA_DEVICES=<gpu> NPROC_PER_NODE=1 \
TORCHRUN_BIN=/home/aibio_3/miniconda3/envs/BagPFN/bin/torchrun \
NETRC=/NHNHOME/BASE/kimds/.netrc \
scripts/launch_interactive_training.sh <RUN_NAME> <CONFIG>
```
⚠️ **50 epoch을 유지할 것** — 합성 지표는 평평하지만 er_status는 ep11 0.6702 → ep49 0.6989로
계속 오른다(§69-6). ⚠️ **`gradient_clip_val`을 켜지 말 것** — er_status −0.0317 (§67).

**평가** (SEAL 10개, 2 GPU 분할 약 20분)
```bash
CK=<ckpt>; CFG=<config>; TAG=<tag>
bash scripts/eval_seal_tasks.sh 0 "$CK" "$CFG" "$TAG" \
  cptac_luad/{EGFR,STK11,TP53}_mutation bc_therapy/er_status &
bash scripts/eval_seal_tasks.sh 1 "$CK" "$CFG" "$TAG" \
  bc_therapy/{grade,her2_status} cptac_brca/{PIK3CA,TP53}_mutation \
  cptac_ccrcc/{BAP1,VHL}_mutation &
wait
```
⚠️ **각 arm은 자기 훈련 config로 채점**한다. CV-only는 삭제된 분기의 파라미터가 init 상태로
남아 있어 다른 config로 채점하면 미학습 분기가 주입된다.

**학습 없는 진단** (수 분, GPU 1장)
```bash
python scripts/diagnose_branch_contributions.py --checkpoint <ckpt> --config <cfg>
python scripts/diagnose_covariance_sketch.py --stages 123
```

---

## 3. 현재까지의 결과표 (SEAL 10개 macro 평균)

| arm | 10개 평균 | er_status | 비고 |
|---|---|---|---|
| SEAL ABMIL (지도학습) | **0.727** | 0.717 | 비교 상대 |
| SEAL MeanMIL (지도학습) | 0.713 | 0.712 | |
| **v41_K128** | **0.6940** | 0.7303 | 현행 최고. K=128, CV-2=128, `a=0.85π/K` |
| v41_K64 | 0.6814 | 0.7260 | K=64 |
| v38_control (전 분기, 6개) | 미측정 | 0.6994 | |
| v40_cv_only (CV-only 최초) | 미측정 | 0.6989 | K=64, `a` 고정, CV-2=32 |

**ABMIL 상회 3/10** (er_status +0.013, her2 +0.016, brca TP53 +0.018).
크게 밀리는 곳: ccrcc VHL −0.088(0.4503, 랜덤 이하), luad EGFR/TP53 각 −0.066.

---

## 4. 확정된 것 (재실험 불필요)

| 결론 | 근거 |
|---|---|
| **6개 분기 중 CV-1·CV-2만 남겨도 동률** | fold-paired −0.0005 [−0.0037,+0.0024] (§68) |
| **Q-5 population attention은 상수를 뱉는다** | AUROC 0.5000, std 0.0000 (§68-1) |
| **CV-1 제거 불가** | 안정화 유무 무관하게 학습 붕괴, 2회 재현 (§66·§67) |
| **G-2 global ridge는 무기여** | Δ −0.0004, CI가 0 포함 (§66) |
| **label-free 사영 선택은 전부 천장** | 8개 축 모두 0.68±0.03 (§69-3) |
| **차원(K)은 유효, 단 대역폭 고정 시** | 9/10 task, +0.0127 (§71-5) |
| **v36 Q1 / v37 기각** | Δ −0.0024 / −0.0001 (§65) |
| **E>1은 느리다** | E=1 60s vs E=4 86s (§68-5) |

---

## 5. 미해결 / 다음

1. **`subspace_rank` 2·4** — 진행 중. CV-2가 K와 무관하게 항상 rank개로 압축하므로 K 증설
   혜택을 못 받는다. ⚠️ rank를 올려도 MLP 입력은 여전히 스칼라 4개(§current_architecture 4).
2. **learnable 사영** — label-free 축이 전부 천장이므로 남은 유일한 정보원은 라벨.
   ⚠️ CV-1이 closed-form이라 gradient가 ridge solve를 통과해야 한다. CV-2 쪽부터 붙이는 것이
   안전하다(§66에서 ridge 제거 시 gradient 발산 전력).
3. **v40_cv_only / v38_control의 10개 채점** — §70이 "대역폭+CV-2 = +0.0271"이라 한 것이
   er_status 기준이라 10개 기준의 실제 크기를 모른다.
4. **K=256** — 차원 유효가 확인됐으므로 재검토 가치. VRAM 22%로 여유.
5. **seed 반복** — 지금까지 arm당 1 seed다.
6. **task별 편차의 원인** — 같은 TP53이 brca +0.018 / luad −0.066. 코호트 크기(112 vs 324)나
   조직 특성이 작용하는 것으로 보이나 미규명.
