# Architecture v32b proposal — DR-CCER (revised)

**Status**: proposal only; staged implementation and training to follow
**Date**: 2026-08-05
**Baseline**: v30 (`poolz_l2` + cardinality-faithful B2)
**Working name**: **DR-CCER** — Donor-Resolved Complementary Class-Evidence Router
**Revision note**: v32b is the critical-review revision of
[`architecture_v32_dr_ccer_proposal.md`](architecture_v32_dr_ccer_proposal.md). It keeps the
v32 diagnosis (CCER-v2 is a tiny, near-degenerate correction) and the rejection of
capacity-only next steps, but fixes five structural weaknesses found in the review:

1. the probes did not test the donor-resolved premise (P3 added);
2. the n>34 mechanism was still unexplained (error decomposition added);
3. generator and architecture were bundled (factor separation restored);
4. several gates were statistically unenforceable (thresholds fixed);
5. the mixture ignored logit-scale asymmetry (standardization added).

---

## 0. Executive decision

Do **not** train a full v32 architecture yet. The only defensible next move is the
**cheap probe set P0–P3 + the "v30-on-new-task-mix" baseline** because:

- P2 measures the *upper bound* of any fusion of the current CCER-v2 representation
  with v30. If it is below `+0.005`, the CCER evidence family is retired before another
  costly run — the highest-probability, lowest-cost outcome given the measured
  correlation (`0.99928` synthetic / `0.99311` Musk) and v28's finding that
  bag-summary-level fusion changes moved the model at most `±0.02`.
- P3 is new: it tests the **donor-resolved premise itself** (the §4.1 mechanism) with
  zero training, using the existing epoch-18 checkpoint's per-donor evidence. P0–P2
  cannot reject or confirm this premise, so a v32 architecture was premature without it.
- The **data effect must be isolated** before the architecture effect. v30 retrained on
  the 6-task mix (incl. `any_positive_sparse`) is both the Stage-A gate baseline
  ("+0.03 over v30" is currently undefined) and the transfer check for the sparse
  mechanism, which has no Musk instance annotation and a 2/2 synthetic→real failure
  precedent (§23 rawstats, §24 IA-MIL).

Full DR-CCER training (Stage A→D) proceeds **only if** P0–P3 and Phase 1 pass their gates.

---

## 1. What the review established about v32

| v32 claim | Review verdict |
|---|---|
| CCER-v2 ≈ tiny correction (`0.00733` logit SD, corr `0.999`) | confirmed; branch/backbone decomposition (P0) still required |
| Donor-averaging of support prototypes is the cause | **unverified** — 3 competing causes (donor averaging, info redundancy with v30 bag stats, 20-epoch step budget). P1/P2 measure the current rep only |
| `n>34` unchanged ⇒ pooling contract fails tail dilution | **mechanism not established**; "not solved" ≠ "this is the cause". n>34 error decomposition needed |
| Skip rare-slot `4→8`, skip another Top-K | confirmed — capacity without information |
| Musk 11–34 band regression `0.958→0.933` (−0.025) | **was hidden in v32's table**; the largest per-band change, must be in P0's decomposition |
| `any_positive_sparse` is "newly available" | confirmed — implemented in generator, unused by v31 config |

The review also flagged: the 0.95 goal's binding constraint is the **small-bag band**
(n≤4, 0.800 after v30, involved in 46% of pairs), which the donor-resolved architecture
does not target; gates at `±0.01` (legacy task) and `n>34 ≥ 0.75` are below/above the
noise floor (task CI width `0.045`; n>34 band has 7 positives, CI width `0.61`).

---

## 2. Stage 0 — mandatory probes (no training, ~1 GPU-hour)

All run against the existing CCER-v2 epoch-18 checkpoint
(`checkpoints/20260805_123630/v31_ccer_v2/epoch=018-val_ce_loss=0.4438.ckpt`) and the v30
best (`checkpoints/20260804_132334/v30_cardinality_poolz_l2/epoch=048-val_ce_loss=0.4442.ckpt`)
on a fixed 1,000-episode mixed-cardinality synthetic suite (all six tasks).

### P0 — decompose branch gain vs backbone drift + n>34 / 11–34 band decomposition
Three prediction files on the same episodes: (1) full CCER-v2, (2) CCER-v2 with the
`ccer_v2` residual contribution removed (`logits − scale·ccer_v2_logits`, available from
the auxiliary dict — no retraining), (3) original v30. Report overall and **all four
Musk bands / synthetic bands**, especially `n>34` and `11–34`. If (1)≈(2), the branch
contributed no ranking gain and additive fusion is not inherited by v32.

### P1 — standalone evidence
Persist per-query `ccer_v2_logits`, route weights/scores, base logits, cardinality, task.
Report standalone branch AUROC/log loss, correlation with the v30 margin, AUROC
conditional on v30 being wrong, route-specific AUROC by task and cardinality band,
effective contribution SD after residual-scale multiplication.

### P2 — fusion headroom (upper bound)
Episode-grouped cross-validated two-feature logistic combiner on `(v30 margin,
CCER-v2 margin)`. **Gate**: if the paired overall AUROC gain `< +0.005`, the current
CCER representation has insufficient complementary information and the CCER family is
retired. (Explicitly: this rejects the *current* representation, not the donor-resolved
one — that is P3's job.)

### P3 — donor-agreement headroom (tests the §4.1 premise directly)
Recompute per-donor class evidence from the epoch-18 checkpoint's own encoders (per-donor
class prototypes instead of the bag-averaged prototype), then derive donor-resolved
features: **median, upper-quartile, trimmed LogMeanExp, donor-agreement fraction,
MAD/IQR** of the query-cell-to-donor-prototype similarity, plus `log(n)`. Fit a
cross-validated combiner on `(v30 margin + donor features)`. **Gate**: if the paired gain
`< +0.005`, donor-resolved pooling contains no ranking information absent from v30, and
§4.1 is falsified regardless of P2.

### n>34 error decomposition (folded into P0/P1)
Report, for the n>34 band: margin distribution, which pair types (pos-big vs neg-big)
drive the loss, and the correlation of the per-band error with support-prototype
separation. This keeps the large-bag failure mechanism an *open diagnostic* instead of an
architectural bet.

**Stage-0 gate**: proceed to Phase 1 only if P2 or P3 shows headroom `≥ +0.005`. Otherwise
retire the CCER family and revisit data-centric levers (small-bag exposure, task mix).

---

## 3. Phase 1 — v30 on the new task mix (data-effect isolation + Stage-A baseline)

Retrain **v30 unchanged** (no architecture change) on the 6-task distribution:
`any_positive_sparse` at probability `0.20`, with the five legacy weights scaled
proportionally. **Note**: unlike v32 §5, legacy task weights are *reduced proportionally*
only to keep the total episode budget identical; this deliberately follows the v31 rule
"do not arbitrarily reduce legacy probability" by renormalizing rather than zeroing, and
any legacy regression is measured, not assumed. This run provides:

1. the measured `any_positive_sparse` baseline for Stage A's `+0.03` gate;
2. the first synthetic→Musk transfer check of the sparse mechanism (Musk read at the end
   of Phase 1 — a *development* read, documented as such; the confirmatory Musk read
   remains Stage D);
3. the data-effect arm of the factor-separation matrix.

**Phase-1 gate**: if the sparse task improves in synthetic *but Musk does not move*, the
sparse mechanism does not transfer and Stage A's sparse gate is not credible — stop
before building the expert.

**Phase 1b (B2b) — within-episode cardinality mixing (deferred from Stage A)**:
true per-bag `n ∈ [1,1024]` within an episode requires a ragged/bucketed training path
that is a separate pipeline change. It is **deliberately not bundled** into Stage A so the
architecture effect is measurable. It is added only when a Stage-A/B pass justifies it,
and it is the fix for the "same support context, different query sizes" regime the
cardinality gate needs.

---

## 4. Phase 2 — DR-CCER architecture (only if Stage 0 + Phase 1 pass)

### 4.1 Donor-resolved support evidence
Per-donor class prototypes from pre-projection aligned slot centers, retained on the
donor axis:
```text
support slots: [episode, support_bag, slot, dim]   (center tokens)
donor evidence per class: [episode, class, donor, hidden]
```
Query-cell evidence pools over slots within each donor, then summarizes the **donor
distribution** with robust statistics: median, upper-quartile, trimmed LogMeanExp,
donor-agreement fraction (above a support-derived null), and MAD/IQR. **Donor
reliability weighting** (effective cell count per donor) is used so tiny donors do not
dominate the robust statistics; the per-donor *mean* is not shrunk (v30 B3 finding).

### 4.2 Null-contrasted multi-scale query scan
Symmetric binary contrast `cell_margin = robust_support(class=1) − robust_support(class=0)`;
scan with absolute (`top-1/4/16`), fractional (`top-1%`, `top-5%`), dense `mean`, and
**bottom-tail** routes. Duplicate `k` for small bags are masked, not counted as separate
routes. Each route is standardized against a leave-one-donor-out support null; the router
receives standardized routes, donor agreement, dispersion, and `log(n)`.

### 4.3 Independently useful evidence expert
Two-class logit; trained with its own loss while v30 is frozen:
```text
L_expert = CE(expert_logits, y) + 0.10·ranking + 0.05·donor_consistency
```
Zero-init output head (exact v30 logits preserved at init). **Stage-A gate adds
sparse-specific discrimination AND decorrelation from v30** — a standalone overall AUROC
`≥ 0.70` alone is satisfiable by a v30-imitation, which is precisely the CCER-v2 failure
to avoid.

### 4.4 Reliability-gated convex mixture (with logit standardization)
```text
s = (v30_logits − μ_v30)/σ_v30,  e = (expert_logits − μ_exp)/σ_exp   (eval-time stats, per episode)
g = sigmoid(router(reliability_features)),  g_init ≈ 0
final_logits = (1 − g)·stopgrad(s) + g·e
```
Reliability features are label-equivariant and query-label-free: donor agreement, support
separation/dispersion, route agreement/margin, `log(n)`. A small penalty on mean gate
usage is scheduled (linear warm-up) so the gate can open. After gate validation, an
optional final stage may unfreeze v30 at `0.01×` LR; the default keeps it frozen.

### 4.5 Stage gates (corrected)
| Stage | Requirement | Correction vs v32 |
|---|---|---|
| A (expert, 10 ep, v30 frozen) | sparse-task AUROC `+0.03` over Phase-1 v30 baseline **and** standalone expert `≥ 0.70` **and** `|corr(expert resid, v30 resid)| < 0.9` | sparse baseline now measured; decorrelation added; "overall 0.70" no longer sufficient |
| B (router, 10 ep) | paired overall synthetic `+0.01`, episode bootstrap CI excludes 0; **legacy tasks within `±0.03`**; mean `g ∈ [0.05, 0.40]`; higher `g` where expert correct & v30 wrong | legacy `±0.01` was unenforceable (CI width 0.045) |
| C (3 seeds) | positive paired delta on all 3 seeds, mean `+0.01` | unchanged |
| D (one Musk read) | overall `≥ 0.87`; `n>34` **CI excludes v30's point estimate**; no band `> 0.02` worse; log loss ≤ v30 | single `n>34 ≥ 0.75` on 7 positives was a coin flip (CI width 0.61) |

**Small-bag alignment**: every Stage reports the n≤4 / 5–10 / 11–34 / n>34 stratified
AUROC. The 0.95 goal requires small bags `≈ 0.90`; DR-CCER is only accepted if it does
not regress n≤4 and the Phase-1/1b data changes carry the small-bag lever.

---

## 5. Training distribution (Phase 2+)
1. **6-task mix** with `any_positive_sparse` at `0.20` (legacy renormalized).
2. `any_positive_sparse` generator: `m ∈ {1,2,3}` shifted cells in positive bags, with
   `m ≤ n_b` for small bags (sparse regime is not defined when `m = n_b`).
3. Keep `poolz_l2`, the v30 task family, and ICI lock unchanged.
4. B2b within-episode mixing is added at Phase 1b (not bundled with Stage A).

---

## 6. Implementation surface (isolated, reversible)
- `src/models/baseline.py`: donor-resolved expert + standardized mixture gate behind
  empty-by-default config fields (`dr_ccer_*`); CCER-v2/v30 paths untouched.
- `src/modules/model_interface.py`: expert/gate/consistency losses, diagnostics, and a
  `dr_ccer_` param group (base LR) vs frozen/`0.05×` backbone.
- `configs/train_v32_dr_ccer.yaml`: 6-task mix + `dr_ccer_*` flags, warm-start weight-only
  from v30 best.
- Tests: label equivariance, query-label isolation, donor permutation invariance,
  duplicate-route masking, exact v30 init, dense/list equivalence, gate `g∈[0,1]`.
- Do not modify v30 defaults, v30 checkpoint contract, or ICI configs.

---

## 7. Immediate next action
1. Write Stage-0 probe script (`scripts/probe_v32_headroom.py`) covering P0–P3 + band
   decomposition; run it; record results in `current_status.md`.
2. If Stage-0 gate passes: implement Phase 1 (6-task mix config, v30 retrain) and measure
   the sparse baseline + a development Musk read.
3. Only then implement Phase 2 (DR-CCER) and run Stage A.

A negative Stage-0 result is a **successful, cheap retirement** of the CCER family, not a
failure: it converts the v32 question into a data-side question (task mix / small-bag
exposure) with a measured basis.
