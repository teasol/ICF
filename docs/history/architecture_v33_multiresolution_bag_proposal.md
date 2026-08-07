# Architecture v33 proposal — MR-BagPFN

**Status**: proposal only; no implementation or training started  
**Date**: 2026-08-05  
**Baseline**: v30 (`poolz_l2` + B2 log-uniform cardinality)  
**Working name**: **MR-BagPFN** — Multi-Resolution BagPFN with Cardinality Counterfactual Views

## 1. Executive decision

Retire the CCER family and move the next investigation from **new cell-to-support
similarity** to **multiple views of the already successful v30 bag representation**.
The next candidate must not be implemented until two data controls and one frozen-v30
headroom probe pass:

1. v30 unchanged on the six-task mix, isolating `any_positive_sparse` exposure;
2. v30 unchanged with true within-episode mixed cardinality (B2b), isolating the
   train/test cardinality-structure gap;
3. a no-training multi-resolution probe showing that views of the same bag add ranking
   information beyond its full-view v30 margin.

Only a positive third result opens v33 architecture work. This ordering is the main
proposal: the project should no longer build a branch first and discover afterward
that its representation has zero standalone information.

## 2. Evaluation of completed v32b

### 2.1 Outcome against its registered gates

| Test | Result | Gate | Verdict |
|---|---:|---:|---|
| P0 branch contribution | full `0.87679`, zeroed `0.87681`, v30 `0.87654` | measurable ranking gain | fail |
| P1 CCER-v2 standalone AUROC | `0.51055` | discriminative | fail |
| P2 fusion delta | `-0.00034` | `>= +0.005` | fail |
| P3 donor-feature fusion delta | `+0.00000` | `>= +0.005` | fail |
| DR-CCER Stage-A expert CE | `0.6931` throughout | standalone AUROC `>= 0.70` and sparse gain | fail |
| Stage-A expert logit SD | collapsed to `0.003–0.006` | non-degenerate expert | fail |

The gate remaining at `g≈0.018` was correct behavior: the expert never earned control.
The slightly lower total validation CE (`0.4307`) cannot be compared with the old
five-task v30 CE because v32b used a different six-task distribution. It is not evidence
for the architecture.

### 2.2 What is falsified

The evidence jointly rejects:

- CCER-v2 additive residual tuning;
- donor averaging as the primary cause of CCER failure;
- donor median/quartile/agreement/dispersion as missing complementary features;
- more slots, more fixed Top-K routes, or a larger residual scale as justified next
  steps;
- Stage-B router training, because both the routed representation and its donor-resolved
  replacement have zero measured headroom.

The 100-episode probe is smaller than the proposed 1,000 episodes, so it cannot establish
a small positive effect precisely. Here that limitation does not rescue CCER: P1 is at
chance, P2 is negative, P3 is exactly zero, and an independent 10-epoch expert run
collapsed to random CE. All four signals agree.

### 2.3 What remains open

The result does **not** reject:

- the existing v30 rare/population branches, which contribute to the confirmed v30;
- training v30 on `any_positive_sparse` without a new branch;
- B2b, because all synthetic bags inside one current episode still share a single `n`;
- multi-resolution bag summaries, because CCER compared cells to new support prototypes
  rather than reusing the proven v30 bag representation at several sampling scales.

## 3. Hypothesis for v33

Musk presents bags with `n=1…1044` simultaneously against a heterogeneous support
context. B2 samples `n` across episodes, but within each synthetic episode every bag has
the same cardinality. The model can therefore solve training episodes without learning
how evidence changes when two bags under the same context have different sampling
resolution.

For a large bag, a single compressed v30 token can also hide whether its margin is
stable across the population or is driven by a small subpopulation. Deterministic
subsamples and partitions of that same bag expose this information while using the
v30 encoder that already transfers to Musk. Unlike CCER, this hypothesis has a direct
zero-training test.

## 4. Phase 0 — factor-separated data controls

Use identical seeds, episode counts, optimizer steps, and fixed evaluation streams.
The factor matrix is:

| Arm | Task mix | Cardinality structure | Architecture |
|---|---|---|---|
| A | legacy five-task | B2: one `n` per episode | v30 |
| B | six-task, sparse `0.20` | B2 | v30 |
| C | legacy five-task | B2b: per-bag mixed `n` | v30 |
| D | six-task, sparse `0.20` | B2b | v30 |

Arm A reuses the committed v30 result when the evaluation stream matches. B and C are
the indispensable main effects; D measures interaction. Do not introduce v33 modules
in this phase.

### B2b implementation contract

- Draw one `n_b ~ LogUniform[1,1024]` per bag, not per episode.
- Do not zero-pad the cell axis before bag statistics. Use the existing list/ragged path
  or cardinality buckets and preserve exact per-bag masks.
- Keep context/query split and latent episode parameters identical across arms.
- Report optimizer steps and total generated cells; equal epochs alone are not an equal
  compute comparison.

### Phase-0 gates

- Six-task evaluation: report all six task AUROCs with episode-cluster bootstrap CIs.
- Legacy overall regression may not exceed `0.01` and its paired CI must include zero.
- The sparse task must reach AUROC `>= 0.75` before it is treated as a useful training
  lever.
- On cardinality-counterfactual pairs, B2b must reduce the full-vs-subsample margin drift
  by at least `20%` relative to B2 without reducing overall AUROC.

If B and D improve only `any_positive_sparse` but not a single non-sparse diagnostic,
record the data result but do not infer Musk transfer yet.

## 5. Phase 1 — frozen-v30 multi-resolution headroom probe

Run the best Phase-0 checkpoint without updating weights. For every query bag, create
deterministic, label-free views:

```text
full bag
disjoint partitions at target size 16
disjoint partitions at target size 64
nested uniform subsamples at 25%, 50%
```

Small bags use only unique valid views; never duplicate cells to manufacture a target
size. Partition membership is determined by a stable bag/episode seed so evaluation is
reproducible.

Pass every view through the unchanged v30 bag/episode path and export:

- full-view margin;
- mean, maximum, median, and IQR of view margins;
- margin slope versus `log(view_n)`;
- class prediction agreement and entropy across views;
- full-to-view token cosine dispersion.

Fit an episode-grouped cross-validated logistic combiner using `(full margin + view
features)`. This is a diagnostic upper bound, not the proposed production head.

### Phase-1 gate

Proceed only if all are true on a fixed 1,000-episode suite:

- paired overall AUROC delta `>= +0.01`;
- episode-bootstrap 95% CI excludes zero;
- at least one of `n<=4` or `n>34` improves by `>= +0.02` on the synthetic cardinality
  bands;
- no legacy task regresses by more than `0.02`;
- gains repeat for at least two deterministic view seeds.

If this gate fails, retire multi-resolution architecture work. The next action would be
generator/domain alignment, not another neural branch.

## 6. Proposed architecture (only after Phase 1 passes)

### 6.1 Shared v30 view encoder

Apply the same v30 structured population aggregator and bag projection to every view.
Weights are shared; no new cell-to-support similarity encoder is introduced.

```text
bag views -> shared v30 bag encoder -> [view, 512-d bag token]
```

The full-bag token is always present and remains the exact baseline path.

### 6.2 Multi-resolution consensus token

Summarize view tokens with four label-free statistics:

- exact full-view token;
- masked mean of view tokens;
- coordinatewise robust dispersion;
- attention-pooled extreme view, with attention conditioned on `log(view_n)` and view
  disagreement.

Project only the three auxiliary summaries through a bottleneck and add them as a
zero-initialized residual to the full-view token:

```text
h_mr = h_full + alpha * W([mean_views, dispersion, extreme_view])
alpha_init = 0
```

This guarantees exact v30 logits at initialization and makes ablation attribution
straightforward. Unlike CCER, the candidate must first show frozen-feature headroom.

### 6.3 Sampling-reliability head

Predict a scalar reliability from view agreement, token dispersion, `log(n)`, and
support-context cardinality range. It may scale the multi-resolution residual but may
not directly add a class logit. Cardinality therefore controls confidence in added
evidence, not the predicted class.

### 6.4 Training loss

```text
L = CE(final_logits, y)
  + 0.10 * ranking_loss
  + 0.05 * consistency_loss
  + 0.01 * residual_usage_penalty
```

`consistency_loss` applies to dense/compositional counterfactual pairs and to negative
sparse bags. Do **not** force every positive sparse subview to equal the full bag: a
subview can legitimately omit the causal instance. No latent responsive-instance mask
may enter model inputs, gates, or inference-time features.

For positive sparse bags, aggregate all disjoint views before applying bag-level CE so
their union covers the original bag. This preserves the multiple-instance semantics
without oracle cell labels.

## 7. Training stages and stop gates

### Stage A — residual only, v30 frozen (10 epochs)

- Train the multi-resolution projection and reliability scale.
- Require paired synthetic delta `>= +0.01`, CI excluding zero.
- Effective residual SD must exceed `0.02` but remain below `30%` of full-margin SD.
- Zeroing the residual after training must remove at least `80%` of the measured gain;
  otherwise the gain is not attributable to v33.

### Stage B — representation adaptation (10 epochs)

- Unfreeze the shared bag projection at `0.01x` LR; keep earlier v30 cell encoder frozen.
- Require positive paired delta over Stage A and no task regression beyond `0.02`.

### Stage C — confirmation

- Seeds `42`, `1234`, `2026`; fixed 1,000-episode paired evaluation.
- Positive delta for every seed and mean delta `>= +0.01`.
- Report accuracy, AUROC, log loss, four cardinality bands, prediction-size correlation,
  and view-seed sensitivity.

### Stage D — one Musk read

Only after Stage C passes:

- overall AUROC must exceed v30 `0.8539` and log loss may not worsen;
- no band may regress by more than `0.02`;
- report paired bootstrap, not overlap of independent CIs;
- treat `0.95` as the long-term target, not a single-run promotion threshold.

ICI remains locked.

## 8. Minimal implementation surface

- `src/datasets/synthetic_data.py`: per-bag cardinalities and nested/partition view
  generation without latent-mask leakage;
- `src/modules/data_interface.py`: ragged or bucketed B2b collation, never unmasked zero
  padding;
- `src/models/baseline.py`: shared view encoding and zero-init consensus residual behind
  empty-by-default `mr_bag_*` fields;
- `src/modules/model_interface.py`: staged freezing, consistency term, and attribution
  diagnostics;
- `scripts/probe_multiresolution_headroom.py`: Phase-1 frozen-v30 probe;
- configs for the B/C/D factor arms and, only after the probe passes,
  `configs/train_v33_multiresolution_bag.yaml`;
- tests for ragged/dense equivalence, padding invariance, view permutation invariance,
  deterministic view construction, exact v30 initialization, no query-label leakage,
  and residual-zero attribution.

Do not change v30 defaults or checkpoint markers. Archive v32 code/config only after all
references remain loadable; reproducibility code may remain inactive until then.

## 9. Immediate next action

Implement **only Phase 0 arms B and C first**. They answer the two unresolved causal
questions at the lowest cost: whether sparse-task exposure helps the proven v30 model,
and whether episode-internal cardinality heterogeneity is the real transfer gap. Build
the Phase-1 probe only after those results select a checkpoint. Do not implement the
v33 residual architecture before frozen-v30 headroom is measured.
