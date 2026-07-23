# B200 medium synthetic pretraining baseline

## Run identity

- Architecture: BagPFN v18
- Run: `medium_20260722_110553`
- W&B: https://wandb.ai/teasol/ICF/runs/qv31wqqy
- Config: `configs/train_medium.yaml`
- Hardware: one NVIDIA B200 (183,359 MiB)
- Precision: `bf16-mixed`
- Parameters: 6,566,579
  - Aggregator: 2,897,933
  - Meta-classifier: 3,668,646
- Outer episode batch: 8
- Optimizer steps: 51,200
- Synthetic episodes: 409,600
- Exit status: success after 100 epochs

This run started from random initialization. It did not resume the failed FP16
run or any other checkpoint.

## Optimization contract

- AdamW, initial learning rate `1e-3`
- Five-epoch linear warm-up
- Global gradient norm clipping at `1.0`
- Reduce-on-plateau monitored on `val_ce_loss`
- Final learning rate: `3.125e-5`
- Checkpoint selection monitored on `val_ce_loss`
- Single-node, single-device execution
- Eight independent episodes are evaluated on the model outer-batch axis and
  their losses are averaged into one optimizer step.

The reported total loss is not cross entropy alone:

```text
loss = ce_loss + 0.10 * ranking_loss + 0.01 * routing_balance_loss
```

Consequently, compare runs primarily by `val_ce_loss` unless their objective
weights are identical.

## Results

| Metric | Epoch | Value |
|---|---:|---:|
| Best `val_ce_loss` | 49 | 0.676737 |
| `val_loss` at best CE | 49 | 0.743555 |
| Best `val_loss` | 15 | 0.743527 |
| `val_ce_loss` at best total loss | 15 | 0.676959 |
| Final `train_loss` | 99 | 0.747957 |
| Final `train_ce_loss` | 99 | 0.681693 |
| Final `val_loss` | 99 | 0.744735 |
| Final `val_ce_loss` | 99 | 0.677699 |

Best-CE checkpoint:

```text
checkpoints/20260722_110553/medium/epoch=049-val_ce_loss=0.6767.ckpt
```

The run remained finite through all 100 epochs. FP32 ridge projection and
adaptive Cholesky solving, pre-optimizer gradient checks, post-optimizer
parameter checks, and BF16 mixed precision were active.

## Capacity diagnosis

Validation CE improved from 0.682845 at epoch 0 to a best of 0.676737, but it
did not establish a new best after epoch 49 despite repeated LR reductions.
The final branch logit standard deviations were:

| Branch | Train | Validation |
|---|---:|---:|
| Mean | 0.209890 | 0.215231 |
| Population | 0.005667 | 0.004856 |
| Tail | 0.023066 | 0.024827 |

The mean path carries most of the predictive variation. Population and tail
paths remain finite and connected but contribute much smaller logits. Train CE
also remains close to the binary random baseline of `ln(2)=0.693147`, and
there is no classic train-better-than-validation overfit gap. These observations
motivate a capacity ablation rather than changing the loss or data difficulty.

## Large-capacity comparison

`configs/train_medium_large.yaml` changes only meta-classifier capacity:

| Setting | Baseline | Large |
|---|---:|---:|
| Meta hidden dimension | 256 | 512 |
| Relation hidden dimension | 256 | 512 |
| Set layers | 1 | 3 |
| Ridge dimension | 64 | 128 |
| Class-memory tokens | 8 | 16 |
| Total parameters | 6,566,579 | 28,750,131 |

Aggregator settings, synthetic distribution, outer batch size, loss weights,
AdamW configuration, warm-up, plateau scheduler, BF16 precision, and clipping
remain unchanged. This isolates model capacity as the comparison variable.
