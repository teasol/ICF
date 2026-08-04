# Absolute Top-K Tail Token & Any-Positive Meta-Learning Proposal (v31 Candidate)

**작성일**: 2026-08-04  
**상태**: 제안 및 검토 단계 (v31 후보)  
**수정 대상 모듈**: [src/models/baseline.py](file:///NHNHOME/kimds/ICF/src/models/baseline.py) (`StructuredEpisodePopulationAggregator`), [src/datasets/synthetic_data.py](file:///NHNHOME/kimds/ICF/src/datasets/synthetic_data.py)  

---

## 0. Executive Summary

Musk zero-shot 평가에서 **n > 34 대형 Bag 구간이 0.698로 정체**되는 원인을 규명한 결과, **Rare Instance를 포획해야 할 Tail Branch가 대형 Bag에서 자기 역할을 못하고 붕괴하는 결함**을 발견하였습니다.

1. **결함 메커니즘**: 현행 Tail Branch는 추출 개수를 `count = ceil(fraction * n)` (fraction = 0.01, 0.05, 0.15) 방식으로 계산합니다.
   * `n <= 34` 소형 Bag: `ceil(0.01 * n) = 1` -> 단 1개의 가장 희귀한 세포만 무희석(undiluted) 추출하여 **AUROC 0.909 - 0.962** 달성.
   * `n > 34` 대형 Bag (`n = 100 - 1000`): `ceil(0.01 * n) = 5 - 10` -> 단 1개의 활성 세포가 4 - 9개의 평범한 배경 세포와 섞여 LSE/Softmax 평균되면서 **신호 희석 발생 (AUROC 0.603 - 0.698)**.
2. **해결 방안 (Absolute Top-K Tail Tokens)**:
   * Bag 크기 `n`에 비례하지 않고 **항상 절대적 Top-1, Top-2, Top-3 극단 인스턴스를 무희석 상태로 핀포인트 추출하는 `absolute_tail_ks` 토큰**을 아키텍처에 추가합니다.
   * 이에 대응하여 합성 생성기에 **Any-Positive (500개 중 1개만 활성) 과제**를 추가하여 Tail Encoder가 단일 희귀 세포에 가중치를 1.0으로 몰아주도록 메타 학습시킵니다.

---

## 1. 정량적 증거 및 진단 (Empirical Evidence)

[scripts/diagnose_tail_dilution.py](file:///NHNHOME/kimds/ICF/scripts/diagnose_tail_dilution.py) 프로브 측정 결과:

* **소형/중형 Bag (`n <= 34`)**: `top_1` 단일 세포 추출만으로 **AUROC 0.908 - 0.950** 달성.
* **대형 Bag (`n > 34`)**: `top_1` 추출 시 0.611, `frac_0.01` 추출 시 0.603으로 정체.
* **원인**: 현행 구조는 `fraction` 기반으로만 Tail을 뽑으므로 대형 Bag에서 무더기 추출(5 - 75개) 후 평균화하여 신호를 파괴함.

---

## 2. 아키텍처 변경안 (Proposed Architecture Changes)

### 2.1 `StructuredEpisodePopulationAggregator` 모듈 확장

[src/models/baseline.py](file:///NHNHOME/kimds/ICF/src/models/baseline.py)의 `StructuredEpisodePopulationAggregator` 클래스에 신규 파라미터 `absolute_tail_ks`를 추가합니다.

```python
# __init__ 파라미터 추가
absolute_tail_ks: Sequence[int] = (1, 2, 3)
```

#### 연산 로직 (`_forward_dense` 내 추가):
1. **Novelty 정렬**: 기존과 동일하게 Population Anchor와의 거리를 기준으로 세포별 Novelty 점수 계산:  
   `novelty = 1.0 - nearest_similarity`
2. **Absolute Top-K 추출**: `n`의 크기와 상관없이 절대 순위 `k`가 1, 2, 3 인 인스턴스를 추출:
   * `k = 1`: `novelty.argmax()` (가장 극단적인 단 1개 세포)
   * `k = 2`: `novelty.topk(2)` 상위 2개 세포
3. **토큰 생성**: 추출된 세포의 앵커 대비 편차(`deviation`)를 `shared_tail_encoder`에 통과시켜 absolute tail token 생성:  
   `abs_tail_token_1`, `abs_tail_token_2`, `abs_tail_token_3`
4. **토큰 결합**: 기존 Fractional Tail Token `(0.01, 0.05, 0.15)` 뒤에 Absolute Top-K Token을 병렬 결합하여 출력.

### 2.2 기존 기능과의 호환성 및 파괴적 변경 없음 (Backward Compatibility)
* 기존 Fractional Tail Token 및 Population Slot 토큰은 그대로 유지되므로, 집단 반응 데이터(Composition/State) 처리 능력에 영향이 없습니다.
* Config 제어로 `absolute_tail_ks` 기본값은 `()` (비활성)으로 두고, v31 config에서만 `(1, 2, 3)`을 명시하여 v30 재현성을 보존합니다.

---

## 3. 메타 학습 데이터 생성기 확장 (B4 Any-Positive Task)

[src/datasets/synthetic_data.py](file:///NHNHOME/kimds/ICF/src/datasets/synthetic_data.py) 생성기에 **Any-Positive Sparse Task**를 추가합니다.

* **설계**: `n = 500`인 대형 Bag 생성 시, 양성 라벨 Bag에는 **정확히 1개 내지 3개의 세포만 활성 변이(Shift)를 부여**하고 나머지 997개 세포는 음성 배경 세포로 유지합니다.
* **효과**: Tail Encoder가 `lse_weights = softmax(encoded_tail * 2.0)` 계산 시 background 세포의 가중치를 0.0으로 누르고, 단 1개의 활성 세포에 가중치 1.0을 몰아주는 방식을 메타 학습하게 됩니다.

---

## 4. 사전 등록 평가 게이트 (Pre-registered Gates)

| 게이트 항목 | 현재 기준 (v30 S2) | v31 목표 기준 | 비고 |
| :--- | :---: | :---: | :--- |
| **1. 합성 무회귀** | 0.9483 | **>= 0.9450** | 기존 대형 bag 무회귀 검증 |
| **2. Musk `n > 34` Target Band** | 0.6980 | **>= 0.8500** | **+0.1500 이상 대폭 개선** |
| **3. Musk `n <= 4` Small Band** | 0.8000 | **>= 0.8000** | 소형 bag 성능 유지 |
| **4. Musk Overall AUROC** | 0.8539 | **>= 0.9000** | **0.95 목표 진입 교두보** |

---

## 5. 단계별 실행 계획 (Action Plan)

1. **Step 1: v30 Medium 기준 학습 완료**
   * [configs/train_v30_medium_bag_proj_residual.yaml](file:///NHNHOME/kimds/ICF/configs/train_v30_medium_bag_proj_residual.yaml) 백그라운드 50-epoch 학습 실행 및 참조 수치 확보.
2. **Step 2: v31 Absolute Top-K Tail 코드 구현**
   * `baseline.py` 내 `absolute_tail_ks` 모드 추가 및 단단한 unit test 작성 ([tests/test_base_model.py](file:///NHNHOME/kimds/ICF/tests/test_base_model.py)).
3. **Step 3: B4 Any-Positive 과제 생성기 연동 및 v31 학습 구동**
   * Config `configs/train_v31_absolute_topk_tail.yaml` 작성 및 백그라운드 구동 후 게이트 검증.
