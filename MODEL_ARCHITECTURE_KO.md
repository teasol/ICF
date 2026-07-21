# ICF Base Model Architecture v18

## Episode-conditioned population aggregation

```text
Label-free context cells
    ├─ soft k-means density anchors (8)
    └─ density residual rare anchors (4)
    → shared episode population coordinate system

Each context/query bag
    ├─ exact all-cell mean
    ├─ slot abundance
    ├─ exact all-cell mean token [512]
    ├─ 12 × 3 slot tokens
    │   ├─ population center
    │   ├─ feature-wise spread
    │   └─ within-population rare state
    ├─ 12 × explicit abundance/dispersion metadata
    └─ 3 × 512-D novelty tail token (1% / 5% / 15%)
        ↓ structured representation 유지
```

Population anchor는 context cell만 사용해 만들며 label과 query cell은 anchor 생성에
사용하지 않습니다. 따라서 query batching이나 query label이 context representation을
바꾸지 않습니다. Instance와 context 순서에도 불변입니다.

Slot encoder는 모든 anchor에 공유됩니다. Center token뿐 아니라 weighted feature std와
각 population 내부에서 assignment×distance가 큰 상위 5% rare-state token을 별도로
유지합니다. Tail 크기는 고정 개수가 아니라 `ceil(fraction × instances)`입니다.

Tail encoder, centered cosine, metadata normalization은 FP32 안정 경로에서 계산합니다.
Class prototype 차이가 거의 0인 경우에도 smooth norm floor를 사용해 AMP gradient가
finite하도록 보장합니다.

- 100 cells: 1 / 5 / 15
- 1,000 cells: 10 / 50 / 150
- 6,347 cells: 64 / 318 / 953

## Class-memory meta classifier

```text
class context structured tokens
    → shared inducing cross-attention
    → 8 memory tokens per class

mean_logits = class-balanced ridge + donor-set cross-attention residual
population_logits = query multi-stat slots ↔ class memory cross-attention
rare_logits = all query cells ↔ class memory top 1/5/10/20% evidence
interaction = shared MLP(mean, population, rare, pairwise interactions)

logits = mean_logits
       + population_gate × population_logits
       + rare_gate × rare_logits
       + fusion_gate × interaction
```

Class memory encoder, relation scorer, rare evidence head, fusion MLP는 모든 class에
공유됩니다. Class embedding을 사용하지 않으므로 label을 바꾸면 memory와 output column만
같이 permutation됩니다. Query의 모든 instance를 사용해 class별 evidence를 계산하며,
fraction pooling이므로 bag instance 수에 따라 선택 개수가 자동으로 변합니다.

Population/rare residual gate는 각각 0.10/0.05의 하한을 갖는 sigmoid gate입니다. 따라서
학습 중 전문 경로가 최종 prediction에서 완전히 끊기는 exact-zero 상태가 되지 않습니다.

## Training objective

```text
loss = main CE/ranking
     + episode-level population slot usage balance penalty
```

경로별 auxiliary CE/ranking은 사용하지 않습니다. Pairwise ranking weight는 0.10이며,
같은 episode의 positive query score가 negative query score보다 높도록 학습합니다.
Checkpoint와 plateau scheduler는 `val_ce_loss`를 기준으로 선택합니다.

## Stationary ICI-like training distribution

- episode당 60~100 donors
- donor당 500~1,000 cells
- 4~10 shared populations
- 높은 donor shift와 donor별 population mixture variation 유지
- composition/state/covariance/interaction/combined response
- effect multiplier `0.6~1.8`을 epoch 0부터 고정 분포로 sampling
- curriculum 없음
- 2~8% rare response population

Bag mean prototype은 donor variation 때문에 약 0.5이지만 generator의 실제 response
population fraction은 composition AUROC 약 0.84, combined 약 0.82입니다. 즉 global
mean만으로는 풀기 어렵고 population-aligned representation으로는 식별 가능한 문제를
의도합니다.

## ICI all-cell evaluation

Donor별 1,203~6,347개 cell을 sampling/padding 없이 variable-length list로 전달합니다.
한 fold의 context로 population anchor를 한 번 만들고 모든 validation query를 같은
좌표계에서 동시에 평가합니다.
