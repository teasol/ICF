# Musk 0.95 달성 — 아키텍처 개선 Proposal

**작성일**: `2026-08-04`
**상태**: 제안 (미구현) — 사용자 판단 대기
**목표**: UCI Musk2 (Musk) zero-shot AUROC **0.95** (현재 최고 **0.822**)
**관련**: [`current_status.md`](../current_status.md) §25, 아카이브 §21~§24 ([`archive.md`](archive.md))

---

## 0. Executive Summary

현재 Musk zero-shot 최고는 **AUROC 0.822** (`--preprocess raw`, v24 musklike-easy
체크포인트, 102 bag leave-one-out). 목표 0.95까지 **+0.13**이 필요합니다.

진단(§22, 학습 없이 확립)에 따르면 Musk의 핵심 신호는 **bag 평균(원자 descriptor
스케일)** 인데, 현재 모델 입력 파이프라인(`_bag_view` = bag 별 centering + per-cell
L2 정규화)이 이 **bag 평균을 통째로 제거**합니다. ridge 천장 측정에서 raw mean+var
**0.829** vs center+L2(현재 모델 입력) **0.554** — 즉 **현재 모델이 입력에서 가장
중요한 신호를 버리고 들어옵니다.** 여기에 166→512 **zero-padding**(OOD 브리지)까지
겹쳐 두 가지 독립 병목이 존재합니다.

이 proposal은 **학습 없이 즉시 가능한 측정 개선(P0)** 과, 진단에 직접 대응하는
**두 아키텍처 변경(P1 읽기 브리지, P2 bag-mean 보존 채널)**, 그리고 IA-MIL 실패
교훈을 반영한 **선택 실험(P3)** 을 제안합니다. **모든 아키텍처 변경은 합성 데이터
우선 검증** (실험 전략 §0) 후 Musk에 적용합니다.

---

## 1. 배경 & 진단 근거 (측정된 사실만)

### 1.1 지금까지의 Musk 결과

| 체크포인트 | 입력 표현 | Musk AUROC [95% CI] | Log loss |
|---|---|---|---|
| v24 Medium (§21) | zero-pad + bag_view | 0.7766 [0.667, 0.878] | 0.5833 |
| musklike-easy (§22) | zero-pad + bag_view | 0.8030 [0.705, 0.889] | 0.5439 |
| musklike-easy + `--preprocess raw` (§22) | zero-pad + **raw mean 보존** | **0.8217** [0.733, 0.904] | **0.511** |
| musklike-easy + IA-MIL (§24, 폐기) | zero-pad + bag_view | 0.5545 [0.440, 0.672] | 6.33 |

### 1.2 입력 표현 병목 (LOO ridge 천장 프로브, §22)

| 정규화 조합 | ridge AUROC |
|---|---:|
| **raw mean + var** | **0.829** |
| L2 only | 0.742 |
| centering only | 0.567 |
| **center + L2 (= 현재 모델 입력)** | **0.554** |
| 인스턴스 약지도 → bag max·mean·softmax pooling | **~0.90** |

→ **bag 평균(스케일)이 가장 정보량이 큰 단일 신호인데, 현재 전처리가 이를 삭제.**
`--preprocess raw`가 zero-shot에서 0.822를 낸 것도 이 신호를 되살린 결과입니다.

### 1.3 코드 상의 제약 (핵심)

- `StructuredPopulationMetaClassifier._bag_view` (`src/models/baseline.py:618`):
  `bag_centered_representation=True`(기본)면 인스턴스를 `bag - bag_mean` 후
  `F.normalize` → **bag 평균이 정보 소실**.
- 생성자 검증 (`baseline.py:528-537`): `bag_centered_representation`(centered 뷰)과
  `use_raw_mean_branch`(raw-mean 뷰)는 **상호배타** — 두 뷰를 동시에 켜면 ValueError.
  즉 현재 코드로는 "centered 뷰 + raw bag-mean 채널" 병행이 불가능합니다.
- `test_musk.py`: 166차원을 512로 **zero-padding** (OOD 브리지, 학습 시맨틱 없음 —
  스크립트 docstring에 명시). `--preprocess raw`는 학습 없는 config 오버라이드.

### 1.4 실패로 닫힌 지렛대 (반복 방지)

- **지렛대 2 — raw bag-stat token**(mean/skew/kurt, §23): 합성 동률(0.9522)이나 실제
  Musk 0.7835 → 음성. (이것은 "통계 토큰을 구조화 토큰에 추가"였고, 아래 P2는
  "입력 표현의 bag 평균 보존"이라는 다른 계층입니다 — 혼동 주의.)
- **지렛대 3 — Phase 1 IA-MIL**(§24): rare 판별 유의 열위(P=1.00), Musk 큰 회귀
  (0.8030→0.5545) → **2026-08-04 폐기**. 복잡한 작업적응 인스턴스 어텐션은 과적합.
  P3는 이 교훈(단순화 + 강한 정규화 + 사전 게이트)을 반영.

---

## 2. 목표 정의 & 성공 기준

- **일차 지표**: Musk2 zero-shot AUROC **≥ 0.95** (102 bag, **5-seed 평균**, LOO).
  단일 seed의 CI 폭(~0.17)은 판정 불가능하므로 5-seed 앙상블을 기준으로 함.
- **회귀 방지 (아키텍처 변경의 필수 조건)**:
  - 합성 musklike-easy 1,000-ep AUROC **≥ 0.94** (현재 0.951~0.952 무회귀)
  - 합성 Medium **≥ 0.70** (현재 v24 계열 천장 유지)
- **ICI**: 여전히 잠금 — 이 제안은 합성 + Musk로 판정하고, ICI는 (해제 시) 최종 1회만.

---

## 3. 제안 3축

### P0 — 측정 강화 (학습 없음, 즉시, 리스크 0)

- **5-seed 예측 앙상블**: 같은 체크포인트로 seed만 바꿔 LOO 확률 평균. n=102 분산을
  상당히 축소(CI 폭 ~0.17 → ~0.10 이하 예상), point estimate 안정화.
- **`--preprocess raw` 고정**: 현재 최고 표현(bag-mean 보존)을 기본으로.
- **모델 앙상블**: musklike-easy(0.8030) + raw(0.8217) 등 서로 다른 체크포인트 평균.
- **산출물**: `predictions/musk_ensemble_*.pt` + bootstrap CI.
- **판정**: 앙상블 AUROC가 0.85~0.88을 못 넘으면 아키텍처 변경(P1/P2)이 필수임을
  정량화하는 게이트. 학습 없이 즉시 실행 가능 — **모든 후속 실험의 기준선 확정**.

### P1 — Phase 2 읽기 브리지 (166→512 학습 입력 projection) — 핵심 제안

**문제**: zero-padding은 화학 descriptor 열에 가중치 시맨틱이 없어 OOD. 브리지 없이는
Musk 신호가 모델의 학습된 입력 공간과 정렬되지 않음.

**설계**: 모델 입력단에 학습 가능한 **`InputReadBridge`** 추가 (기본 OFF — 완전 호환).

```text
Musk instance x ∈ R^166
  -> InputReadBridge: LayerNorm(166) -> Linear(166->H) -> GELU -> Linear(H->512)   (H=256)
  -> 이후 기존 v24 파이프라인 (bag_view / slot / covariance / fusion) 그대로
```

**학습 경로 (3안, 사용자 판단 필요)**:

| 안 | 학습 데이터 | 방식 | 과적합 위험 |
|---|---|---|---|
| **1A** | 166차원 합성 Musk-like 데이터 | 브리지+백본 공동 사전학습 → Musk zero-shot | 낮음 |
| **1B** | Musk 102 bag LOO | **브리지+소형 어댑터만** 미세조정, 백본 freeze | 중 (파라미터 최소화로 억제) |
| **1C** | 기존 합성(512) + Musk | 브리지는 Musk에서, 백본은 유지 | 중 |

- **1A가 프로토콜(합성 우선)과 가장 부합**: 브리지가 "화학 시맨틱"을 학습하도록
  **166차원 Musk-like 합성 생성기**를 추가하고, 그 위에서 브리지(±백본)를 사전학습.
  이후 Musk zero-shot. 브리지가 합성에서 검증되므로 ICI/OOD 일반화 관점도 안전.
- **1B**는 n=102 과적합을 최소화하기 위해 브리지 파라미터를 아주 작게(Linear 166→512
  단층 + LayerNorm) 유지하고, 검증은 반드시 LOO로만.
- **구현 포인트**: `StructuredPopulationMetaClassifier.__init__`에 `input_read_bridge`
  분기 (`use_input_read_bridge: bool = False`). 512차원 입력이면 identity (기본 무변화).
- **예상 효과**: bag-mean 보존(P2)과 직교적으로 **입력 정렬**을 제공. 단독으로는
  0.85 부근, P2와 결합 시 0.88~0.92 전망 (ridge 천장 0.829 + 인스턴스 풀링 ~0.90).

### P2 — bag-mean 보존 입력 채널 (진단에 직접 대응) — 우선 검증 권장

**문제**: §1.2에서 center+L2가 bag 평균(최고 신호)을 삭제. 현재 raw-mean 모드는
centered와 **상호배타**(`baseline.py:528-537`)라 "두 뷰 병행"이 불가능.

**설계**: centered 뷰(기존, 모델이 학습한 표현)를 유지한 채, **raw bag-mean을 추가
잔차 채널**로 병행:

```text
기존: _bag_view -> centered+L2 인스턴스, summary=global_spread (변경 없음)
추가: raw bag-mean μ_i = mean_j(x_ij) ∈ R^512
      -> (v24 bag projection과 동일 패턴으로) bag-mean 토큰 1개 생성
      -> logits += sigmoid(bag_mean_residual_logit) * bag_mean_logits   (잔차 채널)
```

- `use_raw_mean_branch`의 "진단 전용" 제약을 해제하고, centered 뷰와 **공존**하도록
  `_bag_view`/생성자 검증을 완화 (기존 검증은 진단 모드 보호용이었으므로, 새
  `preserve_bag_mean_channel` 플래그를 추가해 기본 OFF로 호환 유지).
- **학습**: musklike-easy(및 Medium)에서 scratch 50 epoch, 합성 1,000-ep + Musk 측정.
- **핵심 장점**: 1) 합성에서 바로 검증 가능(차원 이슈 없음), 2) `--preprocess raw`가
  이미 zero-shot에서 0.822를 낸 신호를 **학습된 1급 채널**로 승격, 3) 회귀 위험이
  가장 낮음(잔차 채널 + 기본 OFF).
- **위험**: bag-mean이 합성 데이터에선 유익하지 않을 수 있음(합성은 centered 뷰가
  이미 분리 가능). 따라서 **합성 musklike-easy 무회귀(≥0.94)를 필수 게이트**로 하고,
  실질 이득은 Musk에서 판정. 합성에서 이득 없어도 Musk 회귀 없으면 채택 가능(사용자
  판단 — 목표가 Musk 0.95이므로).

### P3 (선택) — 인스턴스 스코어링 + 단순 풀링 재검토 (IA-MIL 교훈 반영)

**근거**: §22 천장 ~0.90은 인스턴스 약지도 + bag max·mean·softmax 풀링. IA-MIL은
**복잡한 작업적응 MLP 어텐션**이 과적합해 실패.

**설계**: IA-MIL과 달리 **최소한의 단순 스코어러**로 제한:
- 인스턴스 1층 스코어 (선형 또는 얕은 MLP, 드롭아웃 강화)
- softmax(+max) 풀링 → bag 임베딩 → 잔차 logit
- residual 초기 scale을 아주 작게(0.02~0.05) 시작

**필수 사전 게이트 (IA-MIL 실패 반복 방지)**: 합성 **rare-response 판별기**에서
baseline보다 유의 우위(P<0.95 방향)가 확인되기 전에는 Musk에 적용하지 않음. 게이트
실패 시 **즉시 중단** (P1/P2만 진행).

---

## 4. 실험 계획 (단계별, 합성 우선)

| 단계 | 내용 | 판정 기준 | 산출물 |
|---|---|---|---|
| **S0** | P0 실행: 5-seed 앙상블 + raw preprocess | 앙상블 AUROC 확정, CI 보고 | `predictions/musk_ensemble_*.pt` |
| **S1** | P2 구현 + musklike-easy/Medium scratch 50ep | 합성 무회귀(≥0.94/≥0.70), Musk ≥0.84 | config/ckpt/predictions |
| **S2** | P1 구현 (1A 우선): 166차원 합성 생성기 + 브리지 사전학습 | 브리지 합성 검증 → Musk zero-shot | 브리지 ckpt, Musk 예측 |
| **S3** | P2 + P1 결합 | Musk 5-seed 평균 AUROC 목표 | 최종 ckpt |
| **S4** | (선택) P3 게이트 후 적용 | rare 판별 우위 없으면 중단 | — |
| **S5** | 최종: ICI (잠금 해제 시) 1회 | 프로토콜 §0 | — |

**모든 비교**: `--val-episodes 1000` 합성 + `compare_predictions.py` (episode cluster
bootstrap). Musk는 5-seed 평균 + bootstrap CI로 판정.

---

## 5. 위험 & 열린 질문 (사용자 판단 필요)

1. **n=102 과적합**: P1 브리지/어댑터 파라미터를 최소화하고 LOO로만 검증. 1A(합성
   사전학습)가 이 위험을 가장 잘 회피.
2. **166차원 합성 생성기**: 1A를 위해 화학 descriptor(166)를 흉내 내는 Musk-like 합성
   생성기가 필요 — 유효성 검증(분리성 ridge)을 선행해야 함.
3. **합성-Musk 분포 이동**: 합성에서 잘 돼도 Musk에서 안 될 수 있음 — P2/P1 모두
   Musk에서 최종 판정. 목표가 Musk 0.95이므로 사용자 동의 하에 Musk 중심 평가 허용.
4. **ICI 잠금**: 이 제안은 합성 + Musk로 판정하며 ICI에는 영향 없음(잠금 유지).
5. **우선순위 확인**: ① Musk 0.95를 ICI보다 우선? ② Musk 미세조정(1B) 허용 범위?
   ③ P2(합성 검증 가능, 빠름) 먼저 진행? ④ P1 브리지 학습 데이터(1A)에 투자?
6. **측정 한계**: 102 bag에서 0.95 달성은 분산이 큼 — 5-seed 평균이 정직한 기준.

---

## 6. 요약

| 제안 | 유형 | 학습 필요 | 합성 검증 가능 | 예상 (Musk) | 리스크 |
|---|---|---|---|---|---|
| **P0** 앙상블/측정 | 없음 | ❌ | — | 0.82~0.85 (지표 안정화) | 0 |
| **P1** 166→512 읽기 브리지 | 아키텍처 | ✅ (1A 합성) | 부분 | 0.85~0.90 | 중 (OOD/과적합) |
| **P2** bag-mean 보존 채널 | 아키텍처 | ✅ | ✅ (직접) | 0.85~0.90 | 낮 |
| **P3** 단순 인스턴스 풀링 | 아키텍처 | ✅ | ✅ (게이트) | (게이트 후) | 중 (IA-MIL 교훈) |

**권장 실행 순서**: **P0 → P2 → P1 → (P3 게이트)**. P0는 즉시, P2는 합성에서 먼저
검증 가능하므로 첫 아키텍처 변경으로, P1은 목표 달성의 핵심 브리지로 후속.
**P2 + P1 조합이 0.95 달성의 가장 유망한 경로** (입력 신호 보존 + 입력 정렬 동시 해결).
