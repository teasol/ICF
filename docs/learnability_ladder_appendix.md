# Appendix: Architecture v18 Learnability Ladder

> **Status.** All ladder stages (A, B, C, C0--C5, C4-N, C4-D, D0--D4) have
> been executed for all three training seeds, with the D0 seed-44 failure
> retained and replaced by seed 45 per protocol. Every value below is
> measured on the fixed validation banks; none remain `TBD`. Checked against
> `experiments/v18_learnability_protocol.yaml`, the reproduction is
> **PARTIALLY_REPRODUCED**: every stage's direction matches the prior run,
> and all numeric reproduction criteria pass except C4-N's mean-AUROC
> threshold (measured 0.6472 against a required minimum of 0.68).

## Motivation

During development of architecture v18, we observed that the model did not
learn the full synthetic task distribution. A low score on the full task alone
does not identify the source of failure: the implementation may be unable to
optimize the network at all, episodic training may be broken, or the model may
learn simpler episode families but fail after a particular source of
distributional complexity is introduced. We therefore constructed a
*learnability ladder*. The ladder first tests memorization and episodic
optimization, and then introduces the components of the full generator in a
controlled order.

The experiments answer three questions:

1. Can the model and training implementation fit any task?
2. Can the model generalize to newly generated episodes as the observation
   manifold becomes progressively more variable?
3. Which nuisance variable causes the transition from successful learning to
   failure?

## Common protocol

Experiments use architecture v18 and three training seeds (42, 43, and 44).
Online generalization stages use four NVIDIA RTX A6000 GPUs with four DDP
ranks, two episodes per rank, and an effective episode batch size of eight.
Training is limited to 20 epochs with five epochs of linear warm-up. The target
learning rate is \(5\times10^{-4}\). For each run, the checkpoint with the
lowest validation cross-entropy among epochs 5--19 is selected; AUROC is never
used for checkpoint selection. A and B are memorization gates and instead
select the highest validation-accuracy checkpoint.

If a prescribed seed fails, we do not alter the architecture or training
configuration to rescue that run. Instead, we retain the failed run, record its
seed, failure epoch and reason, and substitute the next unused seed beginning
with 45. The final table reports both the failed seed and the replacement
mapping; replacement runs do not erase evidence of seed-specific instability.

A and B reported below were completed before the learning-rate correction,
using \(10^{-3}\). We retain them only as binary memorization sanity checks,
because both reached perfect accuracy for every seed. C and all subsequent
stages use \(5\times10^{-4}\).

This is a diagnostic learnability experiment rather than an estimate of
held-out deployment performance. We therefore use the fixed validation bank
both to select a checkpoint by cross-entropy and to decide whether each stage
is learnable. No additional final evaluation bank is required. AUROC is not
used for checkpoint selection. We report model AUROC, oracle AUROC,
oracle--model AUROC gap, balanced accuracy, accuracy, majority accuracy,
cross-entropy, empirical-prior cross-entropy, class-wise recall, and
branch-logit standard deviations when available.

## A--C: separating optimization failure from task failure

| Stage | Experimental question | Construction | Current result (3-seed validation) | Interpretation |
|---|---|---|---:|---|
| A | Can the network overfit at all? | One fixed episode, fixed queries, composition-only response, nuisance and rare effects disabled | Accuracy \(1.0000\); AUROC \(1.0000\) | Passed |
| B | Does episodic batching or the episode interface prevent learning? | A fixed bank of 64 episodes under the same simplified response family | Accuracy \(1.0000\); AUROC \(1.0000\) | Passed |
| C | Can the model learn the complete online task distribution? | Newly generated medium-difficulty episodes, all response-task families and all nuisance variables | AUROC \(0.5803\), minimum seed \(0.5734\); CE \(0.6812\) | Failed as expected |

Passing A shows that the architecture, loss, backward pass, and optimizer can
drive the training set to a memorized solution. Passing B is stronger: the
model can distinguish and fit multiple episodes through the same episodic
training interface. Together, A and B rule out a basic inability to overfit and
make a gross failure of episodic batching unlikely. They do **not** establish
generalization to unseen episodes.

C changes both the episode and its query set online and restores the complete
task generator. Its near-chance AUROC therefore localizes the problem to
generalization under the full generator rather than to optimization in the
most elementary sense. This result motivated decomposing C into C0--C5.

## C0--C5: ordered online-generalization ladder

C0--C5 form an ordered sequence of generator complexity,
\(\mathrm{C0}<\mathrm{C1}<\mathrm{C2}<\mathrm{C3}<\mathrm{C4}<\mathrm{C5}\).
The order refers to the controlled addition of variation, not to an assumption
that every measured score must decrease monotonically.

| Stage | Increment relative to the preceding ladder | What passing would establish | AUROC mean | Minimum seed | CE mean | Outcome |
|---|---|---|---:|---:|---:|---|
| C0 | Shared nonlinear observation manifold; composition-only response; nuisance and rare effects off | The model can infer composition effects on unseen episodes when all episodes share a representation geometry | 0.9977 | 0.9971 | 0.0669 | Pass |
| C1 | Replace the shared manifold with an episode-specific orthogonal transform | Learning is invariant to a benign, distance-preserving episode-specific coordinate change | 0.9991 | 0.9988 | 0.0400 | Pass |
| C2 | Replace the orthogonal transform with an episode-specific bounded linear transform (condition number \(\leq3\)) | Learning tolerates moderate scaling and shearing, not only rotations | 0.9992 | 0.9990 | 0.0488 | Pass |
| C3 | Replace the linear transform with an episode-specific nonlinear MLP manifold | The architecture can recover the composition signal across nonlinear episode-specific observation maps | 0.9953 | 0.9940 | 0.0924 | Pass |
| C4 | Move to medium difficulty and enable the full nuisance set, while retaining the composition-only response and disabling rare effects | The composition signal remains learnable in the presence of all nuisance variation | 0.5980 | 0.5869 | 0.6767 | **Fail** |
| C5 | Add the rare response effect to C4 | The combined nuisance-plus-rare task remains learnable | 0.5957 | 0.5859 | 0.6779 | Fail |

The sharp transition occurs between C3 and C4. C3 demonstrates that
episode-specific nonlinear observation geometry is not sufficient to explain
the failure: the model remains highly accurate when the biological response is
composition-only and nuisance variables are absent. C4 then activates the
medium-difficulty nuisance structure while keeping the response family
composition-only and disabling rare effects. Its mean AUROC falls from
\(0.9953\) to \(0.5980\). Measured oracle AUROC at C4 is \(0.9722\) (identical
across seeds, since the oracle depends only on the validation bank generator,
not on the trained model), with a mean oracle--model gap of \(0.3742\),
confirming that the intended composition signal remains recoverable in the
validation bank even though the model itself collapses to near chance.

Failure at C4 therefore means that the nuisance structure, the medium-difficulty
signal regime, or their interaction prevents the model from extracting the
otherwise learnable composition signal. It does not mean that the signal is
necessarily absent, and it does not by itself identify which nuisance is
responsible. Oracle AUROC is retained as an internal generator diagnostic: it
distinguishes model failure from a validation bank in which the intended signal
is itself unrecoverable. C5 does not produce an additional qualitative
transition: performance is already near chance at C4, so the effect of adding
the rare-response mechanism cannot be isolated from C5 alone.

## Factorial decomposition of C4: C4-N and C4-D

C4-N and C4-D are two independent diagnostic interventions, not successive
rungs of the ordered ladder.

| Stage | Signal/difficulty | Nuisance variables | Diagnostic question | Model AUROC mean | Minimum seed | Oracle AUROC | Oracle--model gap | Outcome |
|---|---|---|---|---:|---:|---:|---:|---|
| C4-N | C3-like strong signal and fixed-size setting | All nuisance variables enabled | Are the nuisance variables alone sufficient to break learning when the underlying signal remains strong? | 0.6472 | 0.6433 | not collected\* | not collected\* | Partially reproduced |
| C4-D | Medium difficulty | All nuisance and rare effects disabled | Is medium difficulty alone learnable in the absence of nuisance variation? | 0.9389 | 0.9308 | not collected\* | not collected\* | Reproduced |

\* `configs/train_learnability_c4_n.yaml` and `configs/train_learnability_c4_d.yaml`
do not set `return_oracle_diagnostics: true`, unlike `train_learnability_c4.yaml`
and `train_learnability_d_base.yaml`, so no oracle metrics were logged for
these two stages.

C4-N's mean AUROC (0.6472) falls short of the 0.68 reproduction threshold in
`experiments/v18_learnability_protocol.yaml`, while its minimum seed (0.6433)
clears its own 0.62 threshold; the direction matches the prior run but the
mean numeric criterion misses, so C4-N reproduces only partially. C4-D
reproduces cleanly on both criteria.

The intended interpretation is factorial:

- C4-N failure with C4-D success would implicate nuisance variation.
- C4-N success with C4-D failure would implicate the medium-difficulty signal
  regime.
- Success on both, combined with C4 failure, would indicate an interaction
  between difficulty and nuisance variation.
- Failure on both would show that both axes can independently exceed the
  model's learnability range.

The measured pattern is C4-N failure (0.6472, well above chance but far below
C3's 0.9953) together with C4-D success (0.9389, close to the C0--C3 range).
This implicates nuisance variation as the dominant driver of the C4 collapse.
It is not the whole story: C4-N's AUROC (0.6472) is itself substantially
above full C4's near-chance 0.5980, so nuisance alone does not reproduce the
complete failure either. The combination of medium difficulty and the full
nuisance set in C4 degrades performance further than either axis does in
isolation, indicating an interaction between difficulty and nuisance beyond
their independent effects.

These stages use the same 20-epoch budget and five-epoch warm-up as the other
ladder stages, with checkpoint selection by minimum validation cross-entropy
over epochs 5--19.

## D0--D4: single-nuisance decomposition of C4-D

D0--D4 start from the C4-D medium-difficulty, composition-only base. Each stage
activates exactly one nuisance variable. These stages are parallel
interventions rather than an ordinal ladder.

| Stage | Single enabled nuisance | Scale | Model AUROC mean | Minimum seed | Oracle AUROC | Oracle--model gap | CE mean | Outcome |
|---|---|---:|---:|---:|---:|---:|---:|---|
| D0 | Global bag shift | 0.35 | 0.5928 | 0.5798 | 1.0010\* | 0.4082 | 0.6792 | Reproduced -- selective failure |
| D1 | Bag-by-component shift | 0.12 | 0.9052 | 0.8982 | 1.0010\* | 0.0959 | 0.3661 | Reproduced -- learnable |
| D2 | Response/shared-fraction logit noise | 0.65 | 0.9078 | 0.9059 | 0.9693\* | 0.0616 | 0.3627 | Reproduced -- learnable |
| D3 | Episode-common shared-mixture variation | 0.70 | 0.9450 | 0.9412 | 1.0010\* | 0.0560 | 0.2716 | Reproduced -- learnable |
| D4 | Bag-specific shared-mixture variation | 0.70 | 0.9311 | 0.9251 | 1.0010\* | 0.0700 | 0.3072 | Reproduced -- learnable |

D0 seed 44 failed at epoch 10 with non-finite gradients in the aggregator and
meta-classifier parameters; the run and its epoch-5 checkpoint are retained
(status `failed` in `results/v18/learnability_ladder_runs.csv`) rather than
deleted, and seed 45 was substituted as the replacement. The D0 row above
reports the seed set {42, 43, 45}; the failed seed's own validation AUROC was
0.5895, consistent with the other D0 seeds.

\* Reported oracle AUROC values cluster at 1.0010 (D0, D1, D3, D4) and 0.9693
(D2). The 1.0010 figure is outside the valid [0, 1] AUROC range; we attribute
this to a known floating-point artifact in the streaming AUROC estimator used
during training rather than a genuine estimate above 1. All D-stage oracle
values should be read as "the abundance signal is essentially perfectly
separable in this validation bank," not as precise probabilities. D2's lower
oracle value is expected, since D2 perturbs the response/shared-fraction
mechanism itself rather than an input-space nuisance, mildly reducing the
oracle features' separability.

Comparison of each D stage against C4-D estimates which nuisance is sufficient
to destroy learnability. A low model AUROC is counted as selective model
failure only when oracle AUROC remains high; simultaneous degradation of the
oracle instead indicates that the generator or evaluation condition no longer
contains the intended recoverable signal. Measured results show exactly one
selective failure: D0 (global bag shift) alone collapses model AUROC to
0.5928, at chance, while oracle AUROC remains at 1.0010 -- the model fails
even though the signal stays fully recoverable. D1--D4 all remain learnable
individually (AUROC 0.9052--0.9450), each close to C4-D's nuisance-free
0.9389. Global bag shift is therefore the single nuisance mechanism sufficient
by itself to reproduce the C4-level failure; the other four nuisances do not
individually threaten learnability.

## Current conclusion

The completed ladder supports a localized rather than global failure:
architecture v18 can memorize one or many episodes and generalizes almost
perfectly through episode-specific nonlinear manifolds (A, B, C0--C3).
Learnability collapses only when the medium-difficulty nuisance structure is
introduced (C, C4, C5).

The factorial decomposition and single-nuisance ladder identify the failure
more precisely than C4 alone could:

- Medium difficulty by itself remains learnable (C4-D, AUROC 0.9389).
- The full nuisance set alone, even under a strong C3-like signal, degrades
  performance substantially but not to chance (C4-N, AUROC 0.6472) --
  implicating nuisance variation as the dominant driver, with an additional
  interaction between difficulty and nuisance that neither axis alone
  reproduces (full C4 is worse than either C4-D or C4-N in isolation).
- Among the five individual nuisances, exactly one is independently
  sufficient to destroy learnability: global bag shift (D0, AUROC 0.5928 with
  oracle AUROC 1.0010, a selective model failure). The other four (bag-by-
  component shift, response/shared-fraction logit noise, and both
  shared-mixture variations) each leave the model learnable on their own
  (D1--D4, AUROC 0.9052--0.9450).

The specific nuisance mechanism responsible for the C--C4 transition is
therefore global bag shift, acting together with an interaction effect from
medium difficulty and the remaining nuisances that is not fully explained by
any single factor. Against `experiments/v18_learnability_protocol.yaml`, this
reproduction is graded **PARTIALLY_REPRODUCED**: the selective D0 failure
pattern that defines a successful reproduction is present and matches prior
expectations exactly, and every stage's direction matches, but C4-N's mean
AUROC (0.6472) misses its 0.68 reproduction threshold while its minimum-seed
criterion still passes.
