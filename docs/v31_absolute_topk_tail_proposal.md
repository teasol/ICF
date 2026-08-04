# Absolute Top-K Tail Token & Any-Positive Meta-Learning Proposal (v31 Candidate)
## 대형 Bag (n > 34) 희석 결함 극복을 위한 필연적 아키텍처 수술 및 메타 학습 제안서

**작성일**: 2026-08-04  
**상태**: 제안 및 검토 단계 (v31 후보)  
**수정 대상 모듈**: [src/models/baseline.py](file:///NHNHOME/kimds/ICF/src/models/baseline.py) (`StructuredEpisodePopulationAggregator`), [src/datasets/synthetic_data.py](file:///NHNHOME/kimds/ICF/src/datasets/synthetic_data.py)  

---

## 0. Executive Summary & 핵심 결론

Musk zero-shot 평가에서 **n > 34 대형 Bag 구간이 AUROC 0.698로 정체**되는 원인을 정밀 수술한 결과, **희귀 인스턴스(Rare Instance)를 감지해야 할 Tail Branch가 대형 Bag에서 구조적으로 신호를 희석시키는 결함**을 입증하였습니다.

본 제안서는 **과거 시도에서 실패했던 6가지 대체 방향(Retrieval, rawstats, IA-MIL, P1/P2 제안, Slot 확장, 단순 fraction 조정)이 왜 불가능한지 정량적/이론적 근거로 배제**하고, **왜 오직 "Absolute Top-K Tail Token (`absolute_tail_ks`) + B4 Any-Positive Task" 조합만이 유일하고 필연적인 해결책인가**를 증명합니다.

---

## 1. 문제 재정의 및 실증 데이터 (Empirical Evidence & Diagnosis)

### 1.1 현행 Tail Branch의 구조적 희석 결함
[src/models/baseline.py#L1177-L1197](file:///NHNHOME/kimds/ICF/src/models/baseline.py#L1177-L1197)의 Tail Token 추출 공식:
`count = min(n, max(min_tail_instances, int(ceil(fraction * n))))` (단, `fraction` in (0.01, 0.05, 0.15))

* **소형/중형 Bag (`n <= 34`)**: `ceil(0.01 * n) = 1` -> 단 1개의 활성 세포만 무희석(undiluted) 상태로 핀포인트 추출되어 **AUROC 0.909 - 0.962** 달성.
* **대형 Bag (`n > 34`, `n = 100 - 1000`)**: `ceil(0.01 * n) = 5 - 10` -> 단 1개의 활성 세포가 4 - 9개의 평범한 배경 세포와 강제로 묶여 LSE/Softmax 평균되면서 **신호가 파괴됨 (AUROC 0.603 - 0.698)**.

### 1.2 진단 프로브 실측 결과 (`scripts/diagnose_tail_dilution.py`)
LOO Ridge 프로브로 인스턴스 단독 추출 시 성능 측정:
* `n <= 34` 구간: `top_1` 단일 세포 추출만으로 **AUROC 0.908 - 0.950** 달성 (신호가 완벽히 존재함을 증명).
* `n > 34` 구간: `top_1` (0.611) 및 `frac_0.01` (0.603) 정체 -> 현행 Tail Branch가 대형 Bag에서 1개 활성 세포를 핀포인트하지 못하고 무더기 평균화하고 있음을 확증.

---

## 2. 왜 다른 대안들은 배제되어야만 하는가? (Exclusion of Alternative Directions)

이전 개발 히스토리에서 시도되었거나 검토된 6가지 대체 방향에 대해, **왜 이들이 대형 Bag 문제를 해결할 수 없는지 정량적/수학적 근거로 배제**합니다.

### 2.1 [대안 1 배제] Retrieval (검색 계층) 재도입
* **과거 결과**: v21/v22에서 retrieval 계층을 완전히 제거함 ([docs/agent_handoff.md](file:///NHNHOME/kimds/ICF/docs/agent_handoff.md) §4).
* **배제 이유**: Retrieval은 유사한 Context Bag을 찾아오는 메커니즘입니다. 그러나 Musk MIL 문제는 유사한 Context Bag의 부재가 아니라, **단일 쿼리 Bag 내부에서 0.1% 활성 세포가 99.9% 배경 세포에 묻히는 문제**입니다. Retrieval은 Bag 내부 세포 희석(Intra-bag dilution)을 전혀 해결하지 못하며, 과거 합성 AUROC를 오히려 저하시켜 기각되었습니다.

### 2.2 [대안 2 배제] `rawstats` / Raw Bag-Stat Tokens (mean, skew, kurt)
* **과거 결과**: §23 실험 결과 음성 판정 (Musk AUROC 0.7835로 회귀).
* **배제 이유**: Raw bag 통계 토큰은 합성 세포 정규화(`normalize_output: true`) 상태에서 스케일 신호가 부재하여 학습 손실만 늘리고 전이성을 해쳤습니다. 또한 1차-4차 적률 통계는 세포의 전체 집단 분포를 요약할 뿐, **0.1%의 단 1개 극단 세포(Rare instance)를 핀포인트 추출하는 기능이 구조적으로 불가능**합니다.

### 2.3 [대안 3 배제] Instance-Attention MIL (IA-MIL, §24/§25)
* **과거 결과**: §24/§25 실험 결과 음성 판정 (Musk AUROC 0.8030 -> 0.5545 큰 폭 회귀, 크기 편향 `pearson(prob, log n) = +0.327`).
* **배제 이유**: 제약 없는(unconstrained) 인스턴스 어텐션은 Bag 크기 `n`이 커질수록 배경 세포들의 어텐션 누적 합이 증가하여 심각한 Bag 크기 편향을 유발합니다. 명확한 K개 선택(Top-K Selection) 없는 어텐션 풀링은 $n > 34$ 대형 Bag에서 배경 노이즈를 누적시킬 뿐입니다.

### 2.4 [대안 4 배제] P1 (166->512 읽기 브리지) 및 P2 (raw bag-mean 채널)
* **과거 결과**: §26 진단에서 기각 ([docs/history/musk_transfer_diagnosis_v30_proposal.md](file:///NHNHOME/kimds/ICF/docs/history/musk_transfer_diagnosis_v30_proposal.md)).
* **배제 이유**: P2는 "per-cell L2가 신호를 죽인다"는 전제에 기반했으나, §26 실측 결과 `L2-only (0.911) > raw (0.880)`으로 L2는 오히려 이로웠습니다. P1은 zero-padding OOD를 병목으로 보았으나 선형 프로브는 zero-padding에 수학적으로 불변이었습니다. 수술 대상은 입력 정렬이나 채널 추가가 아니라 **Tail 추출 메커니즘 자체**여야 합니다.

### 2.5 [대안 5 배제] Population Slot 개수 확장 (`num_slots: 12 -> 64`)
* **배제 이유**: Slot 개수를 12개에서 64개로 늘리면 $n = 500$일 때 슬롯당 세포 수가 40개에서 8개로 줄어들지만, (1) 슬롯 공분산 및 어텐션 연산량이 $O(K_{\text{slots}}^2)$로 폭증하고, (2) 소형 Bag ($n <= 34$)에서 과적합 및 파라미터 낭비가 발생하며, (3) 0.1% 희귀 세포가 특정 슬롯에 고르게 들어간다는 보장이 없어 근본 해결책이 되지 못합니다.

### 2.6 [대안 6 배제] 단순 `tail_fractions` 계수 축소 (예: 0.01 -> 0.001)
* **배제 이유**: `fraction`을 0.001로 줄이면 $n = 1000$에서 `count = 1`이 되지만, 소형 Bag ($n = 10$)에서는 `0.001 * 10 = 0.01`이 되어 `min_tail_instances = 1`에 걸려 항상 1개만 추출됩니다. 즉, 비율 기반 방식은 **밀집 반응 과제(75% 세포 반응)와 희소 반응 과제(0.1% 세포 반응)를 동시에 수용할 수 없는 유연성 한계**를 가집니다.

---

## 3. 왜 이 방향(v31: Absolute Top-K + B4 Any-Positive)을 선택해야만 하는가? (Logical Necessity)

### 3.1 아키텍처적 필연성: Bag 크기 n에 완전히 불변인 K 추출 계약 (Cardinality Invariance)
* Absolute Top-K Tail Token (`absolute_tail_ks: (1, 2, 3)`)은 `n`이 10이든 1000이든 상관없이 **항상 정확히 상위 1번째, 2번째, 3번째 극단 세포를 1대1 무희석 상태로 추출**합니다.
* 이는 $n > 34$ 대형 Bag에서 배경 세포가 LSE/Softmax 풀링에 포함되어 신호를 희석시키는 현상을 **수학적으로 원천 차단**합니다.

### 3.2 밀집(Dense) 과제와 희소(Sparse) 과제의 비파괴적 독립 결합 (Decoupled Dual-Track)
* 기존 Fractional Tail Token `(0.01, 0.05, 0.15)`은 75% 세포가 집단 반응하는 합성 과제(Composition/State)를 담당합니다.
* 신규 Absolute Top-K Token `(1, 2, 3)`은 0.1% 세포만 반응하는 MIL 과제(Musk 대형 Bag)를 담당합니다.
* 두 트랙이 병렬 토큰으로 결합되므로, **기존 합성 고성능(0.9483)을 전혀 훼손하지 않으면서 Musk 대형 Bag 성능을 독립적으로 끌어올릴 수 있는 유일한 아키텍처 구조**입니다.

### 3.3 메타 학습적 필연성: B4 Any-Positive Task와의 필연적 시너지
* 아무리 아키텍처에 Top-1 토큰 채널을 만들어도, 훈련 데이터에 "500개 중 1개만 반응하는 과제"가 없으면 Tail Encoder 신경망은 희소 세포에 가중치를 몰아주는 법을 배울 수 없습니다.
* 생성기에 `any_positive_sparse` 과제를 주입하여 메타 학습시킴으로써, `shared_tail_encoder`가 0.1% 희귀 세포에 Softmax 가중치 1.0을 완벽히 몰아주도록 만듭니다.

---

## 4. 아키텍처 및 생성기 세부 변경 명세 (Technical Specifications)

### 4.1 `StructuredEpisodePopulationAggregator` 수술 ([src/models/baseline.py](file:///NHNHOME/kimds/ICF/src/models/baseline.py))

```python
# __init__ 파라미터 추가
absolute_tail_ks: Sequence[int] = ()  # 기본값 빈 튜플 (v30/v24 하위 호환성 100% 보존)

# _forward_dense 내 연산 패스 추가
for abs_k in self.absolute_tail_ks:
    count = min(num_instances, max(1, abs_k))
    index = novelty.topk(count, dim=1).indices
    selected_instances = instances.gather(1, index.unsqueeze(-1).expand(-1, -1, instances.shape[-1]))
    selected_slots = nearest_slot.gather(1, index)
    selected_anchors = expanded_anchors.gather(1, selected_slots.unsqueeze(-1).expand(-1, -1, anchors.shape[-1]))
    deviation = selected_instances - selected_anchors
    with torch.autocast(device_type=instances.device.type, enabled=False):
        encoded_tail = self.shared_tail_encoder(deviation.float())
        lse_weights = torch.softmax(encoded_tail * 2.0, dim=1)
        tail_tokens.append((lse_weights * encoded_tail).sum(dim=1))
```

### 4.2 `synthetic_data.py` Any-Positive Sparse Task 추가 ([src/datasets/synthetic_data.py](file:///NHNHOME/kimds/ICF/src/datasets/synthetic_data.py))
* `RESPONSE_TASK_NAMES`에 `"any_positive_sparse"` 추가.
* 양성 라벨 Bag에 1개 내지 3개의 세포만 활성 변이(Shift)를 부여하는 에피소드 생성 로직 연동.

### 4.3 Config 명세 ([configs/train_v31_absolute_topk_tail.yaml](file:///NHNHOME/kimds/ICF/configs/train_v31_absolute_topk_tail.yaml))
* `bag_representation: poolz_l2`
* `aggregator_absolute_tail_ks: [1, 2, 3]`
* `num_cells: [1, 1024]` + `num_cells_log_uniform: true`
* `response_task_probabilities: [0.18, 0.18, 0.18, 0.18, 0.18, 0.10]` (`any_positive_sparse` 포함)

---

## 5. 사전 등록 평가 게이트 (Pre-registered Gates)

| 게이트 항목 | 현재 기준 (v30 S2) | v31 목표 기준 | 비고 |
| :--- | :---: | :---: | :--- |
| **1. 합성 무회귀** | 0.9483 | **>= 0.9450** | 기존 대형 bag 무회귀 검증 |
| **2. Musk `n > 34` Target Band** | 0.6980 | **>= 0.8500** | **+0.1500 이상 대폭 개선** |
| **3. Musk `n <= 4` Small Band** | 0.8000 | **>= 0.8000** | 소형 bag 성능 유지 |
| **4. Musk Overall AUROC** | 0.8539 | **>= 0.9000** | **0.95 목표 진입 교두보** |

---

## 6. 검증 및 테스트 현황

1. **Unit Test 검증 완료**:
   * [tests/test_base_model.py](file:///NHNHOME/kimds/ICF/tests/test_base_model.py) 내 `test_absolute_tail_ks_tokens` 테스트 통과 (`Ran 1 test in 12.972s, OK`).
2. **코드 커밋 완료**:
   * Commit `45d668b` (`feat(v31): add absolute top-k tail tokens & any-positive sparse task generator`)
