# Architecture v32 proposal — DR-CCER

**Status**: proposal only; not implemented or approved for training  
**Date**: 2026-08-05  
**Baseline**: v30 (`poolz_l2` + cardinality-faithful B2)  
**Working name**: **DR-CCER** — Donor-Resolved Complementary Class-Evidence Router

## 1. Executive decision

Do **not** continue by merely increasing rare slots, adding another fixed Top-K, or
raising the CCER residual scale. CCER-v2 is no longer dormant, but it has learned an
almost perfectly correlated, very small correction to v30 rather than complementary
evidence. The next architecture should therefore change both sides of the evidence
contract:

1. preserve **support-donor variation** instead of averaging every donor into one
   class/slot prototype;
2. train a cell-evidence expert to be **independently discriminative** before it can
   alter v30;
3. combine experts through a reliability-gated **convex mixture**, not an
   unconstrained additive residual;
4. train and validate it on within-episode mixed cardinalities and a dedicated sparse
   positive task before consulting Musk again.

The proposal is deliberately staged. Cheap checkpoint probes can reject the premise
before any v32 implementation or full training run.

## 2. What CCER-v2 actually established

### 2.1 Measured outcome

| Measure | v30 | CCER-v2 | Delta |
|---|---:|---:|---:|
| Synthetic AUROC, 1,000 episodes | 0.85117 | 0.85142 | +0.00025 |
| Synthetic log loss | 0.46584 | 0.46496 | -0.00088 |
| Musk AUROC | 0.85389 | 0.84697 | -0.00692 |
| Musk log loss | 0.47464 | 0.48182 | +0.00718 |
| Musk `n > 34` | 0.69841 | 0.69841 | 0.00000 |

The synthetic gain is uniformly tiny: the largest per-task AUROC change is only
`+0.00064`; no task improved by `0.001`. Prediction correlation is `0.99928` on
synthetic and `0.99311` on Musk. Only `1.21%` of synthetic classifications and
`1.96%` of Musk classifications cross the 0.5 boundary.

At epoch 18 the branch itself is numerically active (`val_ccer_v2_logit_std =
0.05184`, residual scale `0.14136`), but its approximate effective contribution is
only `0.05184 × 0.14136 = 0.00733` logit SD. This reconciles the apparently healthy
branch diagnostics with the nearly unchanged predictions.

### 2.2 Architectural interpretation

CCER-v2 solves the **gradient reachability** problem of CCER-Lite, but not the
**information complementarity** problem:

- Support slot centers are averaged across all bags of a class before comparison.
  This discards donor-to-donor consistency, dispersion, and the distinction between
  “one exceptional support donor” and “a reproducible class signal.”
- `Top-1`, `Top-4`, and `mean` summarize the query, but the final branch is still one
  shared scalar transform of routed class scores. It can cheaply imitate the margin
  already produced by v30.
- Joint CE training rewards any small calibration correction to an already strong
  v30 predictor. It does not require the new expert to solve cases v30 gets wrong.
- The training/evaluation distribution contains the five legacy response tasks, not
  the newly available `any_positive_sparse` task. Thus the rare-evidence branch is
  not evaluated on the mechanism it was introduced to fix.
- The unchanged `n > 34` result is direct evidence that the current support/query
  pooling contract does not solve large-bag tail dilution.

The reverted rare-slot `4→8` configuration should not be revived as the next step:
it increases capacity inside the same compression path without addressing any of
the five points above.

## 3. Mandatory pre-implementation probes

Run these against the existing epoch-18 checkpoint. They require diagnostic plumbing,
not retraining.

### P0 — separate branch gain from backbone drift

Produce three paired prediction files on the same episodes:

1. full epoch-18 CCER-v2;
2. epoch-18 with `ccer_v2_residual_scale = 0`;
3. original v30 checkpoint.

This decomposes the observed delta into the new branch and the 20-epoch `0.05×`
backbone update. If (1) and (2) are indistinguishable, CCER-v2 itself contributed no
ranking gain and v32 must not inherit its additive fusion.

### P1 — measure standalone evidence, not only final logits

Save per-query `ccer_v2_logits`, route scores/weights, base logits, cardinality, and
task. Report:

- standalone branch AUROC and log loss;
- correlation with v30 margin;
- AUROC conditional on v30 being wrong;
- route weights and route-specific AUROC by task and cardinality band;
- effective contribution SD after multiplication by residual scale.

### P2 — upper-bound fusion headroom

Fit a cross-validated two-feature logistic combiner using only `(v30 margin,
CCER-v2 margin)` on synthetic predictions, with folds grouped by episode. This is a
diagnostic upper bound, not a production head. If it improves AUROC by less than
`0.005`, the current CCER representation has insufficient complementary information;
router tuning is then ruled out.

## 4. Proposed v32 architecture

### 4.1 Donor-resolved support evidence

Keep encoded aligned slot centers per support bag instead of immediately computing
one class mean:

```text
support slots: [episode, support_bag, slot, dim]
labels:        [episode, support_bag]
encoded bank: [episode, class, donor, slot, hidden]
```

For each query cell and class, first pool over slots within each support donor, then
retain the donor axis. Summarize the donor distribution with robust statistics:

- median evidence;
- upper quartile evidence;
- trimmed LogMeanExp;
- donor agreement (fraction above a context-derived null threshold);
- dispersion (MAD or IQR).

This lets the model distinguish repeatable class evidence from a single support
outlier. Every support bag receives equal weight, so large donors cannot dominate the
class reference merely by having more cells.

### 4.2 Null-contrasted multi-scale query scan

For each query cell, use a symmetric binary contrast rather than only class scores:

```text
cell_margin = robust_support_score(class=1)
            - robust_support_score(class=0)
```

Scan that margin using both absolute and fractional scales:

```text
absolute: Top-1, Top-4, Top-16
fractional: Top-1%, Top-5%
dense: mean
```

Also retain the corresponding bottom-tail routes. Positive and negative effects can
then be represented without assuming that response always appears as a high-similarity
cell. Duplicate `k` values for small bags are masked rather than counted as separate
routes.

Each route is standardized against a support-derived null distribution formed by
leave-one-support-bag-out comparisons. The router receives standardized route scores,
donor agreement, support dispersion, and `log(n)`; raw cardinality alone cannot create
a class margin.

### 4.3 Independently useful evidence expert

The evidence expert produces a two-class logit and receives an auxiliary loss while
v30 is frozen:

```text
L_expert = CE(expert_logits, y)
         + 0.1 * ranking_loss
         + 0.05 * donor_consistency_loss
```

The donor-consistency term penalizes a high-confidence route supported by only one
context donor. The expert must pass a standalone synthetic gate before fusion is
trained. This prevents the zero-init output head from settling into a tiny calibration
correction.

### 4.4 Reliability-gated mixture with v30

Use a bounded mixture instead of additive residual accumulation:

```text
g = sigmoid(router(reliability_features))
final_logits = (1 - g) * stopgrad(v30_logits) + g * expert_logits
```

Reliability features are label-equivariant and query-label-free: donor agreement,
support separation/dispersion, route agreement, route margin, and `log(n)`. Initialize
the gate near zero so the exact v30 prediction is preserved. Put a small penalty on
mean gate usage during gate training; the expert must earn replacement of v30 rather
than perturb every query.

After the gate is validated, an optional final stage may unfreeze v30 at `0.01×` LR.
The default proposal keeps it frozen to preserve an interpretable attribution of any
gain.

## 5. Training distribution required by the architecture

Architecture and training distribution must change together:

1. implement **B2b within-episode cardinality mixing** so a single episode contains
   bags spanning `[1, 1024]`, matching Musk's simultaneous small/large regime;
2. add `any_positive_sparse` with an initial probability of `0.20`, reducing the five
   legacy weights proportionally rather than increasing total training steps;
3. add cardinality counterfactual pairs: generate two bags from the same latent
   population at different `n`, and penalize unexplained margin drift;
4. keep `poolz_l2`, the v30 task family, and ICI lock unchanged.

Without B2b, a cardinality-conditioned gate can learn episode-level regimes but is
never forced to compare small and large query bags against the same support context.

## 6. Staged experiment and stop gates

### Stage A — representation gate (10 epochs, v30 frozen)

- Train the donor-resolved expert only.
- Evaluate on a fixed 1,000-episode mixed-cardinality suite containing all six tasks.
- **Continue only if** `any_positive_sparse` AUROC improves by at least `+0.03` over
  v30 and standalone expert AUROC is at least `0.70` overall.

### Stage B — mixture gate (10 additional epochs)

- Train only the reliability router; keep both experts frozen.
- **Continue only if** paired overall synthetic AUROC improves by `>= +0.01`, its
  episode-bootstrap 95% CI excludes zero, and no legacy task drops by more than
  `0.01`.
- Require a non-degenerate gate: mean `g` between `0.05` and `0.40`, and clearly
  higher `g` on examples where the expert is correct and v30 is wrong.

### Stage C — full synthetic confirmation

- Repeat Stage B with seeds `42`, `1234`, and `2026`.
- Require positive paired delta for all three seeds and mean delta `>= +0.01`.
- Report cardinality bands and size-bias correlation on the synthetic suite.

### Stage D — one Musk read

Only after Stage C passes, run Musk once and require:

- overall AUROC `>= 0.87` as an intermediate promotion threshold;
- `n > 34 >= 0.75`;
- no cardinality band worse than v30 by more than `0.02`;
- improved or equal log loss.

Musk `0.95` remains the long-term target, not a credible single-step acceptance gate.
ICI remains locked.

## 7. Minimal implementation surface

If P0–P2 justify implementation, keep it isolated and reversible:

- `src/models/baseline.py`: donor-resolved expert and mixture gate behind empty-by-
  default config fields;
- `src/datasets/synthetic_data.py` and collator path: ragged/bucketed B2b episodes and
  sparse task sampling;
- `src/modules/model_interface.py`: expert, gate, and consistency losses plus
  diagnostics;
- `scripts/evaluate_synthetic.py`: optionally persist auxiliary expert outputs;
- new config `configs/train_v32_dr_ccer.yaml`, warm-started weight-only from v30;
- targeted tests for label equivariance, query-label isolation, donor permutation
  invariance, background append behavior, duplicate-route masking, exact v30 init,
  and dense/list equivalence.

Do not modify the v30 defaults, v30 checkpoint contract, or ICI configs.

## 8. Recommended immediate next action

Implement only P0–P2 diagnostic export first. The decisive question is not whether a
larger CCER branch can move logits, but whether donor-resolved cell evidence contains
ranking information that is absent from v30. A negative answer retires the CCER family
before another costly architecture run; a positive answer gives a measured basis for
DR-CCER and its mixture gate.
