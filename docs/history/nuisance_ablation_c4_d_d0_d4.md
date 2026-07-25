# C4-D 및 D0–D4 nuisance ablation 결과

## 목적과 공통 조건

C4-D를 nuisance-off 기준으로 두고 nuisance를 하나씩 단독 활성화하여 architecture v18의 실패 원인을 분리했다.

- online episode, composition-only
- medium difficulty, episode별 random nonlinear manifold
- random query, rare-effect off, CE-only
- AdamW LR `5e-4`, BF16 mixed precision, global gradient clipping `1.0`
- D0–D3는 20 epoch를 완료했고 D4는 통과 추세 확인 후 epoch 3에서 조기 종료했다. C4-D는 별도의 장기 기준 run이다.
- oracle은 labelled context의 responsive-component abundance와 label만 사용하는 1차원 ridge classifier다.
- Oracle abundance와 query label은 모델 입력이나 loss에 사용하지 않는다.

## 결과 요약

| Stage | 단독 활성 nuisance | 판정 epoch | Val accuracy | Majority | Val CE | Prior CE | Balanced accuracy | Model AUROC | Oracle AUROC | Oracle SNR | Oracle–model gap | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C4-D | 없음 | 77 | 0.8887 | 0.5936 | 0.2698 | 0.6641 | 0.8885 | 0.9466 | 1.0000¹ | 2.8142¹ | 0.0534¹ | 통과 |
| D0 | global bag shift `0.35` | 19 | 0.5677 | 0.5972 | 0.6765 | 0.6596 | 0.5688 | 0.6026 | 1.0000 | 3.1572 | 0.3974 | 실패 |
| D1 | bag×component shift `0.12` | 19 | 0.8351 | 0.5972 | 0.3559 | 0.6596 | 0.8357 | 0.9165 | 1.0000 | 3.1572 | 0.0835 | 통과 |
| D2 | response/shared fraction logit noise `0.65` | 19 | 0.8410 | 0.5972 | 0.3535 | 0.6596 | 0.8402 | 0.9192 | 0.9679 | 3.0481 | 0.0487 | 통과 |
| D3 | episode-common shared mixture variation `0.70` | 19 | 0.8905 | 0.5972 | 0.2427 | 0.6596 | 0.8898 | 0.9625 | 1.0000 | 3.1572 | 0.0375 | 통과 |
| D4 | bag-specific shared mixture variation `0.70` | 3 | 0.8592 | 0.5972 | 0.3212 | 0.6596 | 0.8574 | 0.9376 | 1.0000 | 3.1572 | 0.0624 | 통과 추세로 조기 종료 |

¹ C4-D run은 oracle logging 구현 전 실행됐다. Oracle AUROC, SNR과 gap은 현재 C4-D resolved config의 고정 validation set에서 재계산했다. C4-D model metric은 기존 C4-D run의 epoch 77 값이다. 따라서 C4-D oracle과 model의 결합 값은 참고용이며 C4-D를 제외한 D0–D4처럼 각 run에서 동시 집계된 값은 아니다.

## Run과 checkpoint

| Stage | W&B | Checkpoint |
|---|---|---|
| C4-D | [n3vvqdfm](https://wandb.ai/teasol/ICF/runs/n3vvqdfm) | `checkpoints/20260722_190229/learnability_c4_d_lr5e4/epoch=076-val_ce_loss=0.2693.ckpt` |
| D0 | [8un8ikl0](https://wandb.ai/teasol/ICF/runs/8un8ikl0) | `checkpoints/20260723_003417/learnability_d1_e20/epoch=012-val_ce_loss=0.6714.ckpt` |
| D1 | [lknvabxz](https://wandb.ai/teasol/ICF/runs/lknvabxz) | `checkpoints/20260723_010730/learnability_d2_e20/epoch=018-val_ce_loss=0.3554.ckpt` |
| D2 | [0uvxlmw8](https://wandb.ai/teasol/ICF/runs/0uvxlmw8) | `checkpoints/20260723_014321/learnability_d3_e20/epoch=015-val_ce_loss=0.3392.ckpt` |
| D3 | [luleqpsa](https://wandb.ai/teasol/ICF/runs/luleqpsa) | `checkpoints/20260723_081618/learnability_d4_e20/epoch=016-val_ce_loss=0.2365.ckpt` |
| D4 | [sbuml2ru](https://wandb.ai/teasol/ICF/runs/sbuml2ru) | `checkpoints/20260723_090441/learnability_d5_e20/epoch=003-val_ce_loss=0.3212.ckpt` |

## 해석

### D0이 유일한 단독 실패

Global bag shift를 추가해도 oracle AUROC는 `1.0`, SNR은 `3.16`으로 유지됐다. 반면 model AUROC는 `0.60`까지 하락하고 oracle–model gap은 `0.40`으로 커졌다. Generator의 responsive abundance signal이 사라진 것이 아니라 architecture가 bag마다 독립적으로 더해지는 global embedding shift를 제거하지 못한 결과다.

### D1은 제한적인 성능 저하

Bag×component shift는 model AUROC를 C4-D보다 낮췄지만 `0.9165`를 유지했다. Component별 위치 변화는 현재 slot/class-memory 경로가 상당 부분 처리할 수 있다.

### D2는 generator separability도 함께 소폭 하락

Fraction logit noise에서는 oracle AUROC가 `1.0`에서 `0.9679`로 내려갔고 model AUROC도 비슷한 방향으로 하락했다. Abundance signal 자체의 noise 증가가 주된 영향이며 architecture-specific gap은 `0.0487`로 작다.

### D3와 D4는 단독 병목이 아님

Episode-common 또는 bag-specific shared mixture variation만 켰을 때 model AUROC는 각각 `0.9625`, `0.9376`이었다. 단독으로는 slot alignment를 붕괴시키지 않는다.

## 결론과 다음 실험

우선 수정 대상은 global bag-shift invariance다. 모든 nuisance를 결합한 C4의 실패는 D0 효과가 중심일 가능성이 가장 높다. 다음 실험은 D0를 기준으로 shift scale sweep (`0.05`, `0.10`, `0.20`, `0.35`)을 수행해 failure threshold를 찾고, bag-level translation을 제거하거나 episode-relative representation을 사용하는 architecture 변경을 별도 branch에서 검증하는 것이다. Loss, synthetic difficulty 또는 oracle을 학습 신호로 바꾸면 이 진단 목적이 훼손되므로 유지한다.
