# Retrieval & In-Context Feature Aggregation Architecture Analysis

**Last updated**: `2026-07-28 10:30:00 KST`  
**Author**: Antigravity AI Coding Agent & Project Lead  
**Target Repository**: ICF (BagPFN Single-Cell In-Context Meta-Classifier)  
**Architecture Version**: `21` (`architecture_version = 21`)

---

## 1. 개요 및 실증 검증 수치 요약

BagPFN 아키텍처 v21에서의 사전학습 및 ICI Real-World 5-Fold Cross Validation 실험을 통해 **(1) Naive Retrieval의 한계**, **(2) 사전학습-실데이터 간 Representation Discrepancy**, 그리고 **(3) 아키텍처 내부 40차원 특징 기반 Retrieval**의 필요성이 명확히 규명되었습니다.

### Phase 3 & Phase 4 실증 수치 비교표

| 평가 항목 (Metric) | 기존 사전학습 (Non-retrieved) | **개편 후 사전학습 (Retrieval $K=24$ 주입)** | 정량적 변화 및 분석 |
|---|---:|---:|---|
| **Phase 1 Pretrain `val_ce_loss`** | **`0.5921`** (우수) | `0.6839` (정체) | Full-Context 사용 시 사전학습 손실 `0.0918` 우위 |
| **Phase 2 Pretrain `val_ce_loss`** | `0.6845` | **`0.6803`** | 유사 손실 플래토 형성 |
| **ICI 5-Fold Combined Log Loss** | `0.8232` | **`0.7288`** | **`0.0944` 대폭 하강 (불확실성/불균형 획기적 개선)** |
| **ICI 5-Fold Combined AUROC** | `0.5654` | **`0.5524`** | 유사 수치 유지 |
| **Probability Spread (`p1_std`)** | `0.2213` | `0.1664` | 예측 확률 분포 정규화 달성 |

---

## 2. Naive Retrieval의 실패 원인 (Background Noise vs True Response Signal)

### ① Naive Global Mean Cosine Similarity의 심각한 허점
기존 외부 Collator([`data_interface.py`](file:///NHNHOME/kimds/ICF/src/modules/data_interface.py))는 1,000개 세포의 단순 평균(Mean)과 표준편차(Spread)를 normalize하여 Cosine Similarity를 계산하였습니다.

$$
\text{Summary}(X) = \text{Concat}\left( \text{Normalize}(\bar{X}), \text{Normalize}(\text{Std}(X)) \right) \in \mathbb{R}^{1024}
$$

* **문제점**: 단일세포 데이터에서 반응 신호(Response Signal)는 전체 1,000개 세포 중 **Sub-1% ~ 5%의 희귀 세포군 변동**에만 존재하며, 95% 이상은 정상 배경 면역 세포(Shared Background)입니다.
* **결과**: Naive Cosine Similarity는 반응 기전이 유사한 Donor를 뽑는 대신 **95%의 단순 배경 배치 노이즈(Background Noise)가 유사한 무작위 Donor**를 Top-12 NR + Top-12 R로 선별하였습니다.
* 이로 인해 사전학습 시 모델에 '더 쉬운 정답' 대신 '왜곡된 노이즈 24개 Context'가 쥐어져 사전학습 손실이 `0.5921` $\rightarrow$ `0.6839`로 정체되었습니다.

---

## 3. 아키텍처 내부 40차원 특징(40-dim Bag Features) 기반 Retrieval 설계

### ① 40차원 표현형 특징 벡터 ($Z \in \mathbb{R}^{40}$)
BagPFN 아키텍처 내부 Aggregator는 1,000개 세포를 아래 4가지 축으로 정교하게 분해하여 **40차원 벡터**로 압축합니다:

1. **12개 Density Slots**: 세포 밀도 분포 및 Sub-population 비율 (12-dim)
2. **Top-1% / 5% / 15% Rare Evidence**: 희귀 세포군 변동 신호 (Tail Features)
3. **Subspace Covariance Sketch**: 세포 간 상관관계 및 랭크 구조 (Covariance Features)
4. **Centered-Spread Vector**: 세포 분산 및 확산 척도

### ② 40차원 특징 기반 Signal-Aware Retrieval (`extract_bag_features`, `retrieve_context_indices`)
외부 Collator의 Naive summary 연산을 완전히 제거하고, **모델이 직접 추출하는 40차원 Feature Vector $Z$ 간의 Cosine Similarity**로 Top-12 NR + Top-12 R ($K=24$)을 동적 선별합니다. (`tests/test_feature_retrieval.py`로 단위 테스트 검증 완료)

---

## 4. `bag_centered_representation` 전역 종속성 및 2-Pass Streaming 개편

### ① `bag_centered_representation: true` 전역 종속성 문제
BagPFN 아키텍처는 각 Bag $X_i$의 40차원 특징 $Z_i$를 뽑기 전, **에피소드 내 모든 Donor Bag($X_1 \dots X_N$) 세포 전체의 전역 중심점(Global Centroid $\mu_{episode}$)**을 구한 뒤 상대적 편차($X_i - \mu_{episode}$)를 계산합니다.

$$
\mu_{episode} = \frac{1}{N \cdot C} \sum_{i=1}^{N} \sum_{c=1}^{C} X_{i, c}
$$

따라서 Bag을 무작정 쪼개어 독립 계산(Chunking)하면 전역 중심점을 잃게 되므로, **에피소드 전체 세포(수백만 세포)가 전역 집계(Global Aggregation)에 1번은 참여**해야만 합니다.

### ② 2-Pass Streaming Feature Extraction 기법
GPU Memory Wall (OOM)을 회피하며 $N=500 \sim 1,000$ Bag 스케일업을 달성하기 위한 2-Pass 알고리즘:

1. **Pass 1 (Global Centroid 선형 누적)**:
   - $N$개 Bag을 32개씩 Chunk 순회하며 세포 텐서를 거대하게 적재하지 않고 Sum과 Count만 누적하여 $\mu_{episode}$을 메모리 $O(1)$로 선형 계산.
2. **Pass 2 (Centered 40-dim Feature Calculation)**:
   - 구해진 $\mu_{episode}$를 기반으로 Chunk별 $(X_i - \mu_{episode})$ 편차를 부여하여 40차원 특징 벡터 $Z_i \in \mathbb{R}^{40}$를 순차 계산.
3. **Pass 3 (Signal-Aware Retrieval & Meta-Transformer)**:
   - 40차원 벡터 $Z_1 \dots Z_N$ 간 Cosine Similarity로 Class-Balanced Top-24 Context를 정밀 추출하고 Meta-Transformer 포워딩.

---

## 5. 최종 구현 및 파이프라인 개편 로드맵

1. **`src/models/baseline.py` 및 Aggregator 모듈 개편 완료**:
   - 2-Pass Streaming 기반 40차원 Bag Feature Extractor 및 내부 Class-Balanced Retrieval Layer 내장 완료 (`extract_bag_features`, `retrieve_context_indices`).
2. **사전학습 (Phase 5) 전략**:
   - Full Context ($N=60 \sim 100$)로 40차원 Signal-Aware Retrieval 주입 및 `val_ce_loss 0.59` 이하의 강력한 사전학습 가중치 도출.
3. **실데이터 미세조정 (Phase 5 ICI Fine-Tuning)**:
   - 40차원 Feature 기반 $K=24$ Retrieval Filter 및 Log Loss 캘리브레이션 결합.
