# v23 architecture improvement candidates

## Evidence carried forward from v22

1. More inference context helps without retraining.
   - Medium original checkpoint: context 40/80/160/300 AUROC
     `0.6757/0.7204/0.7654/0.7989`.
   - Hard original checkpoint: `0.5284/0.5483/0.5734/0.5839`.
2. Mixed-context training does not fix the ceiling.
   - Hard gain at context 40/80/160/300 was only
     `+0.0047/+0.0070/+0.0108/+0.0138`.
   - Medium train loss and validation CE plateaued. Epoch 31 CE `0.5947`
     remained effectively tied with the official `0.5946`.
3. Rare-cell selectors are not informative.
   - Geometry-based and learned selectors were near AUROC 0.5 and their
     precision matched the base rare-cell rate.
4. The 0.90 state oracle uses `responsive_instance_mask`; the model never
   receives this information. It is not a directly trainable target.

These results reject “insufficient epochs” as the primary explanation and
show that the architecture can use more bags but cannot extract all useful
within-bag and bag-label structure.

## Current information bottlenecks

### 1. Structured context loses bag identity

Each bag produces 40 structured tokens:

- 1 global token;
- 12 slots × center/spread/rare = 36 tokens;
- 3 tail tokens.

The structured population branch then concatenates all tokens from all bags
of one class, removes their bag boundaries, and compresses the entire class
into 8 learned memory tokens. It does not add explicit token-type, slot-index,
tail-fraction, or bag-identity embeddings before this compression.

Consequences:

- the model cannot preserve correlations between center, spread, rare, and
  tail summaries belonging to the same donor;
- a center token and rare token are distinguished only by their values;
- 20 or 150 labelled bags per class both end as the same 8-token memory;
- query raw cells are compared with memory tokens built from aggregated bag
  statistics, creating a semantic mismatch.

### 2. The strongest direct bag-label branch sees only global spread

`global_shape_classifier` already performs direct per-bag ridge and
query-to-context attention without fixed class-memory compression. However,
with centered bag representation its input is only the 512-dimensional global
standard-deviation vector. Multimodal state, population proportions, local
tails, and cross-statistic relationships are not available to this strong
branch.

### 3. Rare information is selected without label guidance

Episode anchors are built from 32 central-to-tail samples per context bag by
unlabelled farthest-point selection. Slot rare states use assignment-weighted
distance, and tail states use novelty from those anchors. In Hard episodes the
responsive fraction is only 0.5–3%, so the relevant population can be missed
before the label-context relation is learned.

### 4. Evidence fusion is mostly globally gated

Global, population, tail, covariance, and interaction paths are combined with
learned global residual scales. The gate does not explicitly condition on
context size, class separation, ridge conditioning, attention entropy, or
query uncertainty. It cannot reliably down-weight a failed rare path for one
episode while using it for another.

## Recommended implementation order

### T5-A0 — Exact mean baseline (active)

Before adding typed token embeddings, run the cheapest bag-boundary ablation:

- average the existing 40 structured tokens to exactly one vector per bag;
- retain every labelled bag as a separate memory input;
- apply the same mean to each query bag;
- keep the original Medium context distribution and train from scratch.

This intentionally discards token identity and within-bag higher-order structure.
Its purpose is attribution, not an assumption that mean pooling is optimal. If it
beats v22, loss of bag boundaries was more harmful than the crude within-bag
average. If it fails, the result does not reject bag preservation; proceed to the
typed learned pooling in T5-A.

Implementation: `mean_pool_structured_tokens: true`,
`configs/train_v23_medium_bag_mean.yaml`, branch `codex/v23-bag-mean`.

### T5-A — Typed, bag-preserving structured context branch

Reuse the existing 40-token aggregator first, so this experiment isolates
episode-level context compression from cell-level bag compression.

1. Add learned embeddings for token type, slot index, and tail fraction.
2. Pool the typed tokens **within each bag** into a structured bag embedding.
3. Feed each labelled bag embedding directly to a shared
   `RidgeResidualMetaClassifier`, preserving every bag until the
   query-to-context relation is computed.
4. Add the new logits as a zero/small-initialized residual beside the existing
   global branch. Keep the old class-memory path for an ablation.

Required ablation:

- global-spread branch only;
- existing fixed 8-token class memory;
- new typed bag-preserving direct branch;
- direct branch + existing paths.

Interpretation:

- improvement means the second context compression and loss of bag identity
  were the main bottleneck;
- no improvement means useful signal was already lost inside the bag
  aggregator.

### T5-B — Distribution-preserving multi-resolution bag sketch

Run only if T5-A is insufficient.

Add an anchor-independent descriptor computed from observable cells:

- learned low-dimensional projections;
- per-projection mean/spread and quantiles or top/bottom-k summaries;
- explicit 0.5%, 1%, 3%, 5%, and 10% tail resolutions;
- projection to one typed bag embedding consumed by the same direct
  label-conditioned branch.

This removes reliance on an unlabelled rare-cell selector and preserves weak
distribution shifts without using oracle cell membership.

### T5-C — Episode-adaptive evidence gating

After a useful structured branch exists, replace global scalar fusion with a
small gate conditioned on:

- context count per class;
- context class separation;
- attention entropy;
- ridge condition/regularization statistics;
- disagreement between global, structured, rare, and covariance logits.

The gate should start near the current residual weights and report its weights
for every evaluation.

## Changes not recommended as the first move

- More epochs: train and validation already plateau together.
- Context 300 only: real ICI context is about 69.
- Increasing class-memory tokens 8→16/32 alone: it does not restore bag
  boundaries or token semantics.
- Increasing slots 12→24/48 alone: the current selector is near random for
  rare responsive cells.
- Training toward oracle-mask 0.90: that target uses information unavailable
  at inference.

## Acceptance protocol

1. Train each architecture candidate with the original Medium context regime
   first; do not mix architecture and context interventions.
2. Evaluate the same 1,000 pool-400 episodes at context 40/80/160/300.
3. Require paired overall `+0.03` or target-task `+0.05` before promotion.
4. Confirm the promoted candidate on Hard and verify that gains remain at
   context 40/80.
5. Keep ICI locked until a candidate passes both Medium and Hard.
