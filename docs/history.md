# Project History — Design Decisions, Trade-offs & Lessons Learned

**Last updated**: `2026-08-09` (CV-only 전환, SEAL 10개 task 판정 기준)

> [!NOTE]
> 이 문서는 `docs/history/` 디렉터리에 흩어져 있던 과거 기록(딥다이브 분석서, 옛
> 아키텍처 설계안, 폐기된 proposal, 과거 세션 아카이브)을 **통합·요약**한 단일
> history 문서다. 각 항목은 원본 파일명과 작성 시점을 출처로 병기한다. 개별 원문은
> git 이력에 보존되어 있다(`docs/history/` 삭제 커밋의 부모 커밋).
>
> 현재 스펙/프로토콜은 Living 문서(`agent_handoff.md`, `current_status.md`,
> `current_architecture.md`, `current_experiments.md`, `README.md`)가 우선이다.
> 아래 내용은 **왜 그런 결정이 내려졌는가**와 **다시 열지 말아야 할(또는 재개 시
> 전제로 삼아야 할) 결론**을 기록한 것이다.

---

## 0. 이 문서가 대체하는 원본 (source map)

| 원본 파일 (모두 `docs/history/` 하위) | 성격 |
|---|---|
| `archive.md` | running archive — `current_status.md`에서 해결·폐기된 §4~§63의 이관본 (v22 결정, 세션 핸드오프, v23~v26, v28~v34, §54~§63 정리 작업) |
| `learnability_ladder.md`, `nuisance_ablation_c4_d_d0_d4.md`, `medium_b200_baseline.md`, `v19_acceptance_protocol.md`, `candidate_a_b_comparison.md`, `architecture_v18.md`, `architecture_v19.md` | v18/v19/v20 시대 — learnability ladder, nuisance ablation, acceptance protocol, Candidate A/B 선정 |
| `synthetic_data_and_tasks.md`, `retrieval_architecture_analysis.md`, `v20_scalability_plan.md`, `v21_retrieval_investigation.md`, `v21_retrieval_experiments.md` | 합성 생성기 규칙, v20/v21 retrieval 시대 |
| `branch_structure.md` | 브랜치/버전 관리 정책, v18→v22 미적용 수정 2건 |
| `v23_bag_mean_50e_session.md`, `v23_v24_bag_collapse_candidates.md`, `architecture_v23_candidates.md` | v23/v24 bag-collapse family |
| `covariance_tier1_diagnosis.md`, `architecture_v28_analysis_ceiling_and_gates.md` | Tier 1(covariance) 진단, 0.70 정보 상한 분석과 v26/v27/v29 게이트 |
| `architecture_v26_proposal_ec_moe_rejected.md`, `architecture_v27_proposal_ac_icar_rejected.md`, `architecture_v29_proposal_sp_sat_rejected.md` | 미구현 폐기된 외부 제안서 3건 |
| `musk095_architecture_proposal.md`, `musk_transfer_diagnosis_v30_proposal.md` | Musk 0.95 로드맵 — 1차 제안과 그 반박(CFMT/v30) |
| `v31_absolute_topk_tail_proposal.md`, `v31_ccer_proposal.md`, `architecture_v32_dr_ccer_proposal.md`, `architecture_v32b_dr_ccer_proposal.md` | v31 CCTS/CCER, v32/v32b DR-CCER proposal |
| `architecture_v33_multiresolution_bag_proposal.md`, `current_status_archive_20260808_v33_armC.md` | v33 MR-BagPFN proposal + Phase 0 arm B/C saga (§41~§48) |
| `architecture_v34_v39_pre_cvonly.md`, `experiments_pre_cvonly.md`, `architecture_v35_tokenonly_chunked_query_proposal.md`, `architecture_v36_q1_structured_population_proposal.md`, `architecture_v37_context_adaptive_aggregation_proposal.md` | CV-only 전환 이전의 architecture/experiments 명세와 v35/v36/v37 proposal |
| `current_status_archive_20260808_v34_config_refactor.md` | §56 config 시스템 리팩터링 |

---

## 1. 버전 관리 정책 & 브랜치 구조 (ADR)

> 출처: `branch_structure.md` (2026-08-02), `archive.md` §4.

- **semver(0.1.0/0.2.0)는 폐기.** `architecture_version` **정수 하나**만 쓴다.
  - `architecture_version`은 체크포인트에 `model._architecture_version` 텐서로 저장되고
    `ModelInterface.on_load_checkpoint`가 다르면 **로딩을 거부**한다 → 구조 변경 시마다
    정수를 올린다. 문자열/semver로 바꾸면 게이트가 깨지므로 금지.
  - 브랜치 이름을 버전 정수에 맞춘다 ("어느 브랜치의 ckpt가 어디에 로드되는가"가 명확).
- **v20은 코드로 존재한 적이 없다.** `configs/archive/v20/*.yaml`은 v19 코드 위에서 돈
  설정 파일 시리즈일 뿐. **v21은 브랜치 없이 히스토리로만 존재**(`ecf6199`~`d8c2b2b`),
  retrieval 최종 상태는 태그 `v21-retrieval-final`.
- 태그: `arch-v18-learnability-baseline`(v18 ladder 동결), `v21-retrieval-final`,
  `v25-typed-bag-final`(v25 폐기 직전), `v30`, `v34`.
- ⚠️ **v18 → v22 미적용 수정 2건** (2026-07-29 확인, 사용자 결정으로 보류):
  - `c05ff8d` ridge-residual **Cholesky backward 안정화** — 거의 특이 행렬에서 forward는
    성공해도 backward non-finite. 첫 시도부터 adaptive jitter를 더하는 수정이 v22에
    미반영(`baseline.py:1482`). 우선순위 높음.
  - `835b726` **rank-local CUDA episode 생성** — `torch.device("cuda")`가 thread-local이라
    중첩 생성 worker가 전부 device 0 해석. `_generation_device()`가 v22에 없음(단일 GPU라
    잠재). DDP 전환 시 터짐.

## 2. 평가 방법론 — 검증된 불변식 (모든 실험의 전제)

> 출처: `v21_retrieval_investigation.md` §4-⑧ (2026-07-29), `archive.md` §6,
> `covariance_tier1_diagnosis.md` (2026-07-31), `experiments_pre_cvonly.md`,
> `current_status.md` §65/§69/§71.

- **n=87 ICI 단일 코호트에서는 어떤 아키텍처 차이도 검출 불가능** (AUROC CI 전부 [0.42,0.68],
  paired bootstrap 승률 0.5±0.03). "Phase 4가 최선" 같은 당시 판단은 **노이즈 추적**이었다.
  이후 모든 비교는 **bootstrap CI + paired 비교**를 필수로 함.
- **합성 val 지표(val_ce·val_AUROC)로 arm을 고르지 말 것** — 3~4차례 반복 실증으로 확정
  (§65 v37: val_ce는 좋았으나 50-fold −0.0068; §69-6: 합성 AUROC ep0=ep49 평평한데
  er_status +0.037; §70-4: val_ce 0.3455가 더 나쁜데 er_status +0.027). **판정은 공식 50-fold로만.**
- **판정 기준은 SEAL 10개 task macro 평균** (§71-4, 2026-08-09). er_status 단일 task 기준의
  모든 arm 선택(K·대역폭·CV-2·CV-only)은 **er_status에 과적합된 설계일 위험이 실재** —
  10개로 넓히니 "SEAL 상회"가 무너졌다(3/10).
- **오라클 상한을 목표로 삼지 말 것**: descriptor가 `responsive_instance_mask`나 latent
  파라미터를 쓰는지 먼저 확인. `observed_` 접두사가 "latent 아님"을 뜻할 뿐 **세포 선택은
  오라클이 정한다**.
- **effect scale 통일 없이 task별 AUROC로 우열 판단 금지** (T3-1). 기본 config의 task별 표는
  생성기 난이도에 오염.
- **val episode 1,000개 필수** (104개는 CI 폭 0.074로 판정 불가; task별은 2,000개 이상).
- **학습 길이가 다른 arm 비교는 교란**: control은 항상 같은 epoch 수로 새로 학습.
- **선형 ridge 천장은 기대치 산출 지표로 쓰지 말 것**: §28에서 부호가 양쪽 다 반대
  (합성 천장 최악이 실현 최고, Musk 천장 최고가 실현 최악). 방향 탐색용으로만.

## 3. 정밀도/수치 안전 계약 (2026-08-08 강제)

> 출처: `archive.md` §63 (2026-08-08), `agent_handoff.md` §3.4.

- **bf16-mixed 필수**: 공분산 sketch 역행렬의 FP16 계수 오버플로/NaN 방지. 2026-08-08
  **이전 수치는 전부 fp32 산출물로 참고용**(실측 이동: er_status fold1 0.4348→0.5130).
- 선언만 되고 강제되지 않던 상태를 `configs/trainer/default.yaml`에 `precision: bf16-mixed`
  고정 + `tests/test_precision_contract.py`(활성 entry point + 선택 가능한 trainer group 전부
  검사, 우회 불가)로 **실제 강제**.
- ⚠️ 확정 ckpt 재현 주의: **v34-1536·v35-16384는 fp32 학습** — 지금 재실행하면 bf16으로
  돌아가 재현 안 됨. 역사적 재현은 `trainer_overrides.precision: 32-true`.
- **평가도 bf16-mixed 강제**(§64): `eval_autocast` 단일 정의, fp16은 ValueError. 단
  `configs/test_*.yaml`·`configs/archive/`는 검사 제외(의도적 — 평가 강제 시 모든 공식 수치가 이동).

## 4. v18/v19/v20 — learnability ladder와 대형 bag-shift 불변성

> 출처: `learnability_ladder.md`, `nuisance_ablation_c4_d_d0_d4.md`,
> `v19_acceptance_protocol.md`, `candidate_a_b_comparison.md`, `architecture_v18.md`,
> `architecture_v19.md`, `medium_b200_baseline.md` (2026-07-22~07-26).

- **Ladder**: A(단일 episode 과적합) → B(고정 64-bank) → B2(random query) → C(online new
  episode) → manifold ladder(C0~C3) → C4-N/C4-D → D0~D4 nuisance ablation.
  - A/B/B2/C0~C3 통과, **원래 medium 조건 C는 실패**(CE가 prior보다 높음) — manifold 변화만으론
    실패를 설명 못 함.
  - **D0(global bag shift)가 유일한 단독 실패**(model 0.60, oracle 1.00 유지, gap 0.40) —
    architecture가 bag별 global embedding shift를 제거하지 못함이 v18 실패 원인. D1~D4는
    통과(0.92~0.96).
- **v19**: bag-centered view(centered delta + L2) + 40 structured token(global spread 1 +
  slot center/spread/rare 36 + tail 3). raw mean/raw bag mean을 분류에 사용 금지.
  acceptance protocol(v19_acceptance_protocol.md)에 구조·수치·성능 게이트(C4-D ≥0.92,
  D0 ≥0.85)를 사전 고정 — "결과 후 threshold 변경 금지" 원칙의 시작.
- **Candidate A(Gated Distance) vs B(Learned Head) 20-epoch 비교 → B 승격으로 v20** (2026-07-26):
  B가 val_ce 0.5925, overall AUROC 0.7513(+0.0214), 특히 State +0.0284.
- **B200 medium baseline** (`medium_b200_baseline.md`, v18, 6.57M params): best val_ce 0.6767.
  mean branch만 실질 logit std 보유, population/tail은 미미 → capacity ablation 동기.

## 5. v21 retrieval 시대 — 3대 가설과 제거 결정

> 출처: `retrieval_architecture_analysis.md`, `v21_retrieval_experiments.md`,
> `v21_retrieval_investigation.md`, `v20_scalability_plan.md` (2026-07-27~07-29).

- **Naive Retrieval 실패**: 1,000세포 평균/표준편차 cosine은 반응 신호(<1~5% 희귀 세포)를
  놓치고 95% 배경 노이즈가 유사한 donor를 선별 → 사전학습 val_ce 0.5921→0.6839 정체.
- **Signal-Aware Retrieval**: aggregator가 이미 계산하는 40-token 구조화 요약을 flatten해
  class-balanced Top-24 선별. anchor 안정성(chunk 구성 의존) 문제를 "anchor는 항상 전체 pool에서
  1회"로 해결.
- **3대 가설 검증 (§4-⑧, 2026-07-29)**:
  1. **retrieval 자체는 ICI에서 이득 없음** — Phase 6c(off) AUROC 0.5454 = 6b(on) 0.5481
     (동전 던지기)인데 Log Loss·Accuracy는 오히려 off가 우세. fold당 context ~69명에서
     K=24는 가용 context 65%를 버림.
  2. **구현이 설계와 다름** — "query 1명당 맞춤형"이 실제로는 전 query 공용 context(첫 query
     or 평균). 평균 공용은 각 query의 개별 top-24와 61.1%만 겹치고 반응자에서 ~50%.
  3. **사전학습 부족** — Phase 5는 `episode_batch_size` 8→32 + `max_epochs` 20 유지로
     **optimizer step이 1/4**(10,240→2,560), validation도 retrieval 없이 수행돼 ckpt 선택 오염.
- **결정**: 복잡한 계층을 검출력 없는 지표 위에 유지할 이유가 없음 → **v22에서 retrieval
  계층 전부 제거**(303줄 삭제, 태그 `v21-retrieval-final`로 보존). "v20 롤백" 요청은 v20이
  코드로 없어 절제로 처리.
- **교훈**: 사전학습·미세조정 context 선별 분포 불일치가 성능을 크게 깎는다(Phase 6 0.5081).

## 6. v22~v24 — T4/T1/T2/T3 진단과 bag-collapse family

> 출처: `archive.md` §4/§6/§9/§10, `covariance_tier1_diagnosis.md`,
> `v23_bag_mean_50e_session.md`, `v23_v24_bag_collapse_candidates.md`,
> `architecture_v23_candidates.md` (2026-07-29~08-01).

- **합성 생성기 규칙**은 `synthetic_data_and_tasks.md`에 정밀 명세 (medium: 60–100 bags,
  500–1000 cells, 5 task 확률 0.40/0.30/0.05/0.05/0.20, episode마다 새 random MLP manifold,
  bag 순서·query label 무작위, 클래스 개수 shortcut 제거). 이 문서는 generator를 바꿀 때마다
  갱신할 기준.
- **T4 context-size curve** (Hard): context 10→160 AUROC 0.5084→0.5737 단조 증가, paired
  P(40>80)=P(80>160)=0.00 → labeled context 부족이 주요 병목. mixed-context 학습은 +0.005~0.014로
  미미.
- **T1(covariance) Tier 종료**: 세포 선택이 무작위(기하 3종·학습 1종 모두 ~0.50), 슬롯 capture
  0.155/fragmentation 0.963, **bag 라벨만으로는 purity 0.128**(판정 ≤0.15, `f5ddbf5`에 사전 고정).
  모델(0.612)은 진짜 관측 상한(0.570)을 이미 초과. **covariance는 effect scale 통일 시 composition과
  동률 → 원래 약점은 state.**
- **0.70 정보 상한 발견** (`architecture_v28_analysis_ceiling_and_gates.md`, 2026-08-02):
  **학습 파라미터 0개의 closed-form ridge가 v24 slot 충분통계로 overall 0.700** — 9.45M 파라미터
  모델(0.708)과 task별 ±0.02 동률. 즉 v22~v25의 CE 정체(0.5903~0.5976, 폭 0.0073)는 아키텍처
  탐색 실패가 아니라 **비지도 population-slot 요약의 정보 상한**이다. 올바른 분할(oracle 2-slot
  0.93)을 관측·bag 라벨로 찾는 경로는 세 번 독립적으로 닫힘.
- **v23-A0(exact mean)/v24-A0(learned proj, slot1)/v24-B0(per-token bottleneck)/v24-B1(residual+
  bottleneck)** 학습 완료. **사용자 결정(2026-08-01): 1,000-episode paired 비교 없이 v24-B1을
  train CE 순위로 v24 확정**(best 0.5903). `architecture_v23_candidates.md` 상단에 사후 기록.
  ⚠️ 이 결정 때문에 §2 프로토콜(paired 비교)을 건너뛴 선례 — 이후로는 사전 합의 필요.
- **v25(T5-A, typed bag-preserving) 폐기 (2026-08-02)**: Medium paired는 맥락 의존 trade-off
  (v25 @ctx40 우세, v24-B1 @ctx300 압도, 80/160 구분 불가), Easy Δ+0.0033 → 승격 기준 미달.
  **작은 context(40) 우세는 ICI(~69) 관점에서 참고 가치.**

## 7. v26/v27/v29 미구현 폐기 + CLS-token(v26) — 학습 없는 게이트 문화

> 출처: `archive.md` §16~§20, `architecture_v26_proposal_ec_moe_rejected.md`,
> `architecture_v27_proposal_ac_icar_rejected.md`, `architecture_v29_proposal_sp_sat_rejected.md`,
> `architecture_v28_analysis_ceiling_and_gates.md` (2026-08-02~08-03).

- 외부 작성 제안서 3건(EC-MoE, AC-ICAR, SP-SAT)을 코드·실측으로 비판 → **E2 게이트(oracle gating
  상한 delta 0.0000)로 v26/v27 routing 폐기 확정**, E7(population-slot 지도 상한 purity 0.335,
  INCONCLUSIVE)로 v29 폐기. **v27 Riemannian branch는 실행 불가**(512×512 batched `eigh` step당
  1.99s = v24 대비 7~15배, 인접 고유값 간격 2.97e-7로 bf16 backward 불안정).
- **핵심 반증**: 40→1 압축이 정보를 파괴한다는 전제는 v22(40t, 0.5946) < v24(1t, 0.5903)로 반증.
  softmax simplex 융합(Σg=1)은 지배 항 global 계수를 1.0→0.2~0.5로 줄여 logit 붕괴.
- **대안(사용자 제안) CLS-token pooling = `architecture_version=26`** 구현·학습: raw cell 전체를
  학습된 CLS cross-attention(셀 축, O(N))으로 요약해 41번째 token으로 추가. **배치 경로의 3중
  인라인 복제 버그**(`_all_structured_tokens_batched`로 통합) 발견 — 실제 벤치에서만 잡힌 유형.
- **v26 = v24 동률(0.5908) → 폐기.** `diagnose_cls_attention.py` probe: 학습된 cross-attention이
  **균등(lift 1.003×)** → "readout 용량(1→24개)"이 아니라 "**관측 manifold에서 반응 세포 신원에
  접근 불가**"가 병목(0.70 상한의 실체). 세포 신원 접근 경로 4번째로 닫힘.
- **§19 정규화 천장**: centered 0.685 > current 0.636 > whiten 0.597. "지울 패턴을 배운다"(whitening)
  는 cross-feature 공분산이 반응 신호 자체라 더 나쁨. **0.70 상한의 실체는 "centered bag의 평균+
  분산"이지 12-slot 구조가 아님.**
- **§20 no-L2 ablation 음성**: 전체 모델은 global_summary(spread)·covariance로 magnitude를 이미
  공급받아 L2 제거 이득 없음(val_ce 0.5925 > v24 0.5903).

## 8. "0.70 한계 = 생성기 lossiness" 가설과 Musk 0.95 로드맵

> 출처: `musk095_architecture_proposal.md`, `musk_transfer_diagnosis_v30_proposal.md`,
> `archive.md` §21~§27 (2026-08-03~08-04).

- **Musk-like easy 데이터에서 모델이 근완벽 분류(1,000-ep AUROC 0.9510, val_ce 0.2552)** →
  가설 확정: **Medium의 0.70 천장은 아키텍처가 아니라 생성기의 lossiness**.
- **Musk 실제 전이 병목 = per-bag centering의 rank 결핍**: n=1→0벡터, n=2→대척쌍, median n=12→
  166차원의 ≤11차원 그림자. raw 0.975 vs center 0.475(n≤4). **"표현 문제"와 "cardinality 문제"는
  같은 결함.**
- **musk095 1차 proposal 기각** (P1/P2): "raw > L2 → per-cell L2가 신호를 죽인다"는 **전제 역전**
  (L2 0.911 > raw 0.880, λ 전부 일관). zero-padding은 선형이라 수학적으로 불변(P1 근거 부재).
  raw bag-mean 채널 추가(P2)보다 **centering 제거**가 정답. 합성은 `normalize_output:true`라
  스케일 신호가 없어 P2를 합성에서 검증 불가.
- **raw-stat token(mean/skew/kurt) 음성** (§23): 합성 동률(0.9522)이나 실제 Musk 0.7835로
  centered(0.803)보다 낮음 — 합성 단위-cell 분포에 과적합, Musk descriptor로 전이 안 됨.
- **IA-MIL 음성** (§24): 합성 무회귀(0.9520)지만 rare 판별에서 유의 열위(P=1.00), 실 Musk
  0.5545 급락. `use_instance_attention_mil`은 **기본 OFF 유지**.
- **Musk 0.95는 아직 미달** — v30 이후 최고 0.8539, n>34(0.698)가 최약 구간. 0.95의 통계 검증
  가능성 자체도 n=102에서 의문 제기됨(§7.1). **Musk는 이제 "완전히 untouched" confirmatory
  test가 아니므로 transfer development benchmark로 취급하고 별도 최종 확인 데이터를 잠가야 함**
  (v31 proposal이 명시).

## 9. v30 확정 — B1(poolz_l2) + B2(cardinality-faithful)는 상호 필수

> 출처: `archive.md` §28, `experiments_pre_cvonly.md`, `musk_transfer_diagnosis_v30_proposal.md`
> (2026-08-04).

- **B1(poolz_l2)**: per-bag centering을 context-pool 대각 표준화로 대체. B1 단독(S1)은 음성 —
  magnitude 보존 표현이 **bag 크기와 교란**(corr(prob,log n) +0.327), 소형 +0.04 / 대형 −0.21
  "구간 교환".
- **B2(log-uniform [1,1024] cardinality)**: 에피소드 간 크기 변동 학습. **B2 단독은 학습 불가**
  (legacy는 n=1을 0벡터로 → NaN 그라디언트, 프로젝트 가드 발동).
- **B1+B2 = v30 확정 (2026-08-04)**: Musk 0.8539(최고 경신), n≤4 0.475→0.800(paired Δ+0.325,
  CI 0 제외, P=0.997), n>34 0.667→0.698 유지, 합성 무회귀 0.9483, corr(prob,log n) +0.059.
  **진단이 예측한 지점에 정확히 국소화된 첫 양성 결과.**
- 코드 기본 `bag_representation`은 `legacy` 유지(기본값 플립은 기존 ckpt/테스트 재평가를 깨뜨림) —
  **조용한 의미 변경 방지 원칙**.

## 10. v31 CCER / v32·v32b DR-CCER — "활성화"와 "상보 정보"의 구분

> 출처: `v31_absolute_topk_tail_proposal.md`, `v31_ccer_proposal.md`,
> `architecture_v32_dr_ccer_proposal.md`, `architecture_v32b_dr_ccer_proposal.md`,
> `archive.md` §30~§38 (2026-08-04~08-05).

- **v31 CCTS** (cardinality-calibrated tail scan): 고정 Top-K는 cardinality-invariant하지 않고
  대형 bag에서 null maximum이 커지는 문제를 겨냥. **미채택** — `frac_0.15`가 `top_1`보다 좋아
  absolute Top-K 우월성 미지지, instance-level annotation 부재.
- **CCER-Lite**: support class를 직접 지지하는 class-conditioned router. **branch 묻힘**(기여 ~1e-4).
- **CCER-v2**: projection 전 aligned slot-center prototype + 독립 support/query encoder +
  zero-init head + route floor 0.30. branch는 활성(잔차 0.141, logit_std 0.052)이나
  **v30과 상관 0.999, 실효 기여 0.00733 logit SD** — 합성 +0.00025, Musk −0.00692로 v30 미달.
- **교훈**: branch "활성화(gradient 도달)"와 "상보 정보 학습"은 다르다. branch가 v30과 거의
  정확히 상관된 작은 보정을 학습했다면 capacity 확대는 무의미.
- **v32/v32b DR-CCER**: donor-resolved support + independently discriminative expert +
  reliability-gated mixture. **P0~P3 게이트 전부 음성** (P1 standalone 0.51055, P2 fusion
  −0.00034, P3 donor-feature +0.00000, expert CE 0.6931 무작위) → **CCER 계열 실증적 폐기**.
  v32b가 v32의 구조적 결함(donor 전제 미검증, n>34 미해명, generator/architecture 묶임,
  게이트 통계 비강제, logit-scale 비대칭)을 지적한 것이 원인 분리의 본보기.

## 11. v33 MR-BagPFN Phase 0 — 데이터 컨트롤부터

> 출처: `architecture_v33_multiresolution_bag_proposal.md`,
> `current_status_archive_20260808_v33_armC.md` §41~§48 (2026-08-05~08-06).

- **원칙**: 새 아키텍처 구현 전에 (1) v30 on six-task mix, (2) v30 + B2b(에피소드 내 mixed
  cardinality), (3) frozen-v30 multi-resolution headroom probe를 먼저. **"branch를 먼저 만들고
  나중에 representation이 무정보임을 발견하는" 순서 금지.**
- **arm B(six-task)** gate 미달: `any_positive_sparse` 0.6747 (<0.75).
- **arm C(legacy+B2b)** gate 미달: legacy overall 회귀 +0.0373 (50ep) → top-up(8×A6000 DDP,
  episode-match 150ep) 후에도 **+0.0412로 회복 안 됨** → 과소학습 편향 가설 기각, **B2b 데이터
  자체가 회귀 원인**. Musk n>34 개선(0.698→0.849)·PathoBench lscc 개선은 실질 신호이나 v30이
  전반 우위 → **v30 유지, v33 미채택.**
- **운영**: B2b ragged batch는 `episode_batch_size=1` 필요 → **패딩 배칭** 구현(§44, commit
  `568c5f8`, 처리량 5.8→16 ep/s) — ragged 평가/학습의 지속 기반. `test_b2b.py`/`test_ragged_batching.py`
  → 나중에 `tests/history/legacy_*`로 이관.

## 12. v34 large-context — PathoBench 보고용 확정 모델

> 출처: `architecture_v34_v39_pre_cvonly.md` (2026-08-08, CV-only 전환 전 명세),
> `archive.md` §50~§56, `current_status_archive_20260808_v34_config_refactor.md` (2026-08-06~08-07).

- **v34-1536 = PathoBench 보고용 확정 (2026-08-07)**: `input_dim=1536`, `[1,8192]` log-uniform,
  slot MLA 저랭크 affinity·slot_std 분산 트릭·배치 population candidates·정규화 통합 —
  전부 수치 동일(byte-identical)한 MLA 계열 효율화. best val_ce 0.4419 (1024ep×50, fp32).
- **평가**: PathoBench **17개 이진 task** (전처리: train-only PCA 1536→512, 전체 타일, all-context,
  ragged per-episode — 패딩 dense는 69GB OOM). §51 정정: **로컬 `cptac_ccrcc_{er,grade,her2}`가
  `bc_therapy` 복사본** — 실제 CPTAC-CCRCC는 미평가. 실측 6개 데이터셋·14개 유효 task.
- **Musk(§50)**: `--pad-mode tile`(166×9+42) 0.858 vs zero-pad 0.822.
- **ICI 5-seed 0.512±0.027 = 랜덤** — ICI 잠금 유지, PCA-per-fold 미지원으로 v30과 직접 비교 보류.
- **config 시스템 리팩터링 (§56, 2026-08-07)**: v34 base + **group default 참조형**,
  `merge_train_config`에 `logger_overrides`/`trainer_overrides` 지원. 아카이브 config는 **base_config
  없이 전부 인라인(자체 포함형)** — 상대경로 깨짐 원천 차단. 아카이브 19개 숨은 깨짐도 수정.
- **case leakage 진단 (§57)**: 기존 5-fold는 slide-level split이라 case 82/108이 fold 간 분산
  (lscc_arid1a 0.908 부풀림). **공식 50-fold(case-disjoint) 0.462가 정직한 값(실질 랜덤)** —
  재개 안전.

## 13. v35 데이터 단독 arm — 정확 스트리밍 축약 (rev.2)

> 출처: `architecture_v35_tokenonly_chunked_query_proposal.md` (2026-08-07, rev.2),
> `archive.md` §61.

- rev.1의 결정 3건 중 **①rare branch 제거·③context/query 분리 대형화는 폐기 권고**, ②는
  "근사 평균"이 아니라 **정확 충분통계 축약**(online softmax + top-k merge, 수치 동일)으로 재설계.
- 코드 대조로 반증된 주장들(교훈): `global_summary`는 1차 모멘트가 아니라 **표준편차**;
  `covariance_matrix` 보정식은 `_bag_view`가 버리는 chunk 평균을 요구해 구현 불가;
  `_population_candidates`는 bag당 32개 고정이라 chunk 분할 시 대형 bag이 anchor를 지배;
  query 위치는 dataset이 아니라 훈련 스텝이 뽑아 dataset이 알 수 없음.
- 동기 자체 반증: context 2,000 tile cap → pooled AUROC **−0.0019**뿐. **학습 arm은 데이터만
  바꾼 단독 arm**(num_cells [1,32768] + log-uniform power 1.5로 Musk 밴드 보존).
- 구현된 것: **bag 단위** 스트리밍(`stream_eval_bags`, 수치 동일, peak VRAM 40,990→18,930 MiB).
  chunk 단위(bag 내부)는 미구현.

## 14. v36 Q1 / v37 — 죽은 모듈을 고치던 두 arm (기각)

> 출처: `architecture_v36_q1_structured_population_proposal.md`,
> `architecture_v37_context_adaptive_aggregation_proposal.md` (2026-08-08),
> `current_status.md` §62/§65.

- **원래 v36(40→1 압축 해제)** 전제 3건이 반증: ①합성 bag cell은 exchangeable이라 sequential
  chunk에 학습 신호 없음, ②"선택 기제 부재"는 사실이 아님(`_instance_attention_mil_logits` 존재,
  §24 기각은 §31 측정 6이 무효), ③region 수는 15가 아니라 median 3.4개. → 좌표 미사용 결정.
- **§62-2 실측**: `project_structured_tokens: true`에서 `_projected_bag_tokens`가 구조 token
  40개를 **라벨 정보 전에** 1개로 압축 → `_population_memory_logits` routing softmax가 길이 1
  축에 걸려 `population_slot_weights (Q,1) 전부 1.0` — **ABMIL형 선택 기제가 구현돼 있으나 무력**.
  P0-slots probe: 이 압축이 버리는 정보는 EGFR +0.1597 / STK11 +0.1577.
- **§68-1 실측**: Q-5 population attention은 **상수를 뱉는다**(AUROC 0.5000, std 0.0000) — v36/v37
  은 상수를 뱉는 모듈에 더 좋은 입력을 넣은 것이라 Δ≈0이 당연.
- **결과(§65)**: Q1 fold-paired **−0.0024**, v37 **−0.0001** — 둘 다 +0.005 게이트 미달. **라벨
  조건화는 미검정 레버로 잔존.**
- **핵심 교훈**: 새 아키텍처 arm을 설계하기 전에 `scripts/diagnose_branch_contributions.py`로 그
  분기가 **살아 있는지** 먼저 확인할 것.

## 15. ridge ablation / 안정화 레버 — G-2 무기여, clipping 금지

> 출처: `current_status.md` §66/§67, `agent_handoff.md` (2026-08-09).

- **ridge ablation(v38)**: 세 closed-form ridge를 독립 제거하는 플래그
  (`meta_enable_global_ridge` G-2 / `meta_enable_abundance_ridge` P-2 /
  `meta_enable_covariance_ridge` CV-1, 기본 true). 각 플래그는 ridge 하나만 0으로 하고 분기
  residual은 남김. ⚠️ **ablation된 ridge 파라미터는 gradient를 받지 않아 init 상태로 남음 —
  반드시 같은 플래그로 평가.**
  - **G-2는 무기여**(Δ −0.0004, CI 0 포함). **P-2·CV-1은 제거 시 학습 붕괴**(non-finite/발산) —
  "ridge가 정보를 뺏는다" 가설 기각.
- **clipping 금지 (§67)**: `gradient_clip_val: 1.0` → er_status **−0.0317**. non-finite가 없던
  arm에 불안정을 만듦. `nonfinite_gradient_policy: zero`는 no-op이라 안전. **clipping은 기본 off.**

## 16. CV-only 전환 (§68) — 6분기 중 4개가 죽은 분기

> 출처: `current_status.md` §68, `architecture_v34_v39_pre_cvonly.md` (아카이브 배너).

- **분기 기여도 진단**(`diagnose_branch_contributions.py`, v38_control): CV-1 0.9052 / CV-2 0.8867 /
  P-2 0.6254 / G 0.5949 / **Q-5 0.5000(상수)** / R 0.5196. FINAL 0.9199.
- **CV-only 계약**: `meta_covariance_only: true`면 `final = cov_res·CV-1 + cov_rel_res·CV-2`만
  남고 나머지는 **계산조차 안 됨**(죽은 key는 zeros가 아니라 부재 — KeyError가 정상, 0으로 채우지 말고
  분기 가드). 전 분기 모델과 er_status 50-fold **동률(−0.0005)**, 훈련 forward 16.91→2.85 ms
  (5.9×), peak VRAM 50.5→14.7 GB(3.4×), epoch 98→60s.
- **융합이 해가 되는 경우**: CV-1 제거 시 모델 0.7929가 자기 최고 분기 CV-2 0.8706보다 나쁨 —
  약한 분기가 좋은 분기를 끌어내림.
- **첫 skip 구현 오류는 테스트가 잡음**: `_forward_dense`의 `instances`는 이미 pool 표준화된
  값인데 다시 계산해 dense/ragged가 2.4e-2 어긋남. `test_skip_matches_a_full_branch_model_...`
  가 방지선. 등가성은 end-to-end(epoch 0–4 val_ce 소수 4자리 일치)로 확인.
- `meta_bag_aggregation`(v37 계열)은 CV-only에서 무의미(bag token 소비처 부재).

## 17. covariance sketch 기하 진단과 v41 — 이득은 대역폭·CV-2

> 출처: `current_status.md` §69/§70/§71 (2026-08-09).

- **sketch 구성**: `centered_delta (N×1536) --P--> (N×64) --> 64×64 공분산 --> 상관행렬 -->
  shrinkage 0.1 --> 상삼각 2080 --> CV-1 ridge`. `P[d,k]=QR(sin(a·d·k)+cos(b·(d+1)·k))`,
  `a=0.019, b=0.011` 하드코딩, `persistent=False`(결정적 공식 — 랜덤이면 DDP seed 정합 필요).
- **label-free 축 8개 전부 무효**(0.68±0.03 천장): 랜덤 vs 사인, PCA(분산 15배 보존), Sobol QMC,
  앨리어싱, 사다리 간격 `a` — 모두 무차이. **위상 seed가 결정적**(0.65~0.75, 폭 0.10).
- **차원만 유효 — 단 대역폭을 고정해야 보인다**: `a = 0.85π/K`로 고정 시 K 16→256에서 단조 증가
  (0.62→0.70), std 단조 감소. 기존 "K=64 포화"는 대역폭(`a·K`) 교란 오독. **현행 K=64는 최적 아님.**
- **v41**: K=64 + 대역폭 정규화 + CV-2 차원 K 연동 → er_status 0.7303(+0.031). **이득은 차원이
  아니라 대역폭·CV-2**(K 고정 arm +0.0271, 차원 증설 추가 +0.0043, CI 0 포함). K=64 arm을 함께
  돌린 것이 귀속을 가능하게 함. **두 손잡이는 아직 분리 안 됨(§70-3).**
- **§71 SEAL 10개 평가 — 일반화 실패**: v41_K128 평균 0.6940 vs ABMIL 0.727(−0.033), 상회 3/10.
  er_status는 10개 중 **가장 유리한 task**. ccrcc VHL 0.4503은 랜덤 이하, TP53도 brca +0.018 /
  luad −0.066로 코호트 의존. **판정 기준을 10개 macro 평균으로 변경(필수).**
- **합성 지표 불신이 네 번째·다섯 번째 확인**; seed 반복 필수(단일 측정 요동 ±0.05).

## 18. 운영·인프라 교훈

> 출처: `archive.md` §12~§16/§24/§28/§43/§54~§56, `v21_retrieval_investigation.md`,
> `current_status_archive_20260808_v33_armC.md` §43/§44.

- **NCCL P2P hang (gnode5)**: 8×A6000에서 `dist.barrier()` hang. `NCCL_P2P_DISABLE=1`만 통과 →
  `launch_interactive_training.sh`에 기본 적용. (SHM 전송, 단일 노드 NVLink 없음.)
- **launcher wrapper가 torchrun child보다 먼저 종료** → wrapper PID kill로 GPU가 안 풀림
  (실측 153GB 잔존). **프로세스 그룹**(`kill -TERM -$pgid`)으로 죽일 것.
- **`pgrep -f "scripts/train.py"` 대기 루프는 자기 자신에 매칭**돼 영원히 안 끝남 — launcher 로그 +
  프로세스 부재를 함께 확인하거나 `/proc/<pid>/comm`으로 실제 python/torchrun만 필터.
- **순차 큐 레이스**: launcher가 detached worker를 즉시 반환 → pgrep이 "GPU free" 오판 →
  병렬 OOM. `wait_launched_training_done`(스폰 폴링 + 완료 블록)으로 수정.
- **스크립트 삭제 전 tests의 `from scripts.* import` 의존성 확인** — `diagnose_context_size.py`
  삭제 후 테스트 import 깨짐 → `a5dfcf8^`에서 복원.
- **checkpoint purge 교훈**: 폐기 run의 수치는 문서에, 경로는 삭제됨을 기록. gitignore 대상 정리.
- **PathoBench 러너**: worker 실패 시 형제 worker를 안 죽여 GPU 166GB 고아 — 실패 시 전체 kill.
  `--tmp-dir` 새로 써서 fp32 캐시로 인한 정밀도 혼합 방지.
- **배치·평가 캐싱 가드**: context 캐싱은 `context_mode=="all"` && `context_limit is None`일 때만
  사용(아니면 자동 폴백) — `--max-tiles`/`--context-max-tiles` 주면 shared generator가 쿼리마다
  전진해 부표본이 달라짐.
- **다중 위치(연구실/집/노트북) 동기화**: 같은 테스트를 중복 구동해 load 72까지 급등한 사례 —
  세션마다 Living 문서 + `git log`만으로 이어받을 수 있게 기록(이 프로젝트의 SSOT 문화).

## 20. CV-2 손잡이 소진과 계보 B(Encoder+Ridge)의 일반화 실패

> 출처: `current_status.md` §72~§79 (2026-08-09/10).

- **학습이 평가용 ragged 경로를 타고 있었다**: `_episode_losses`가 단일 에피소드도
  `forward_episode_batch`(dense)로 보내야 한다. 이전에는 `self.model(...)`(bag마다 Python 루프)을
  불러 74.2 → 31.3 ms/step, **2.4배 느렸다**. `tests/test_training_uses_dense_path.py`가 고정.
  **범인이 아닌 것들**(전부 측정으로 배제): 로깅(이미 epoch 단위, 지표 계산 3%), CPU 비동기
  생성(GPU 3.2 ms vs CPU 2,579 ms — **805배**), 프리페치 깊이(depth 1 > depth 3).
- **소스 prune (−11,285줄)**: config로 끄기만 했던 5개 분기를 삭제해 `baseline.py`가
  5,685 → 2,224줄. 근거는 v41_K128 ckpt의 파라미터별 gradient 실측 — 43,198,660개 중 gradient를
  받는 것이 **229개**뿐이었다. ⚠️ **prune 이전 ckpt는 현재 트리로 strict 로드가 깨진다** —
  `8caa96c` 고정 worktree `/NHNHOME/BASE/kimds/ICF_pre_prune` **유지 필수**.
- **대규모 삭제 전에는 출력을 fixture로 녹화할 것**: `tests/fixtures/cvonly_golden.pt`가 실제로
  1 ulp 차이를 잡아냈고, 추적 끝에 "수식은 동일, 텐서 정렬이 달라져 커널 선택이 바뀜"으로
  규명됐다. fixture는 **도달 가능한 가중치만** 담을 것(전체는 691MB가 git에 들어간다).
- **아키텍처에 대한 사실을 config 키에 두지 말 것**: VRAM 가드가 `meta_covariance_only`를 읽고
  있었는데 그 키를 지우자 조용히 6층 추정으로 회귀해 가드가 무력화됐다. 이제 모델이
  `vram_activation_layers`로 직접 선언한다.
- **CV-2는 병목이 아니다 (§75·§76·§77)**: margin activation(identity −0.017),
  subspace_rank(±0.001), head 구조(paired_head −0.0003) 셋 다 SEAL 10개 macro를 못 움직였다.
  v43(T→34.0)과 v44(T→2.84)는 **10개 task 전부 셋째 자리까지 일치** — CV-2의 출력 스케일은
  성능과 무관하다. 다만 `paired_head`는 **라벨 대칭성을 구성으로 보장**한다(`learned_head`는
  4.4e-2로 깨져 있다). 그 대가로 출력단 bias는 `h(a,b,s)−h(b,a,s)`에서 상쇄돼 gradient를 받지
  않는다(정상, 테스트가 명시적으로 고정).
- **CV-1의 dual(kernel) ridge는 옳다 (§78)**: 이유는 "bag 수 ≪ 특징 수"이지 instance와 무관하다.
  설계행렬이 (bag 60~133 × 8,256)이고, instance(N)와 임베딩 차원(1536)은 이미 공분산으로 요약돼
  사라진 뒤라 "N > d니 kernel 불필요" 논리는 적용되지 않는다. 실측 dual 0.33 ms vs primal
  9.8 ms — **30배**.
- **계보 B (Encoder+Ridge) — 기각 (§79)**: label-free 사영 8종이 전부 0.68 천장이므로 **라벨을
  보는 사영**이 유일한 미시험 축이었다. ⚠️ **v50~v52의 설계 오류를 반복하지 말 것** — 세포끼리
  attend하지 않는 inducing-point 인코더였고, 근거는 "셀-셀 attention은 에피소드당 2.7e10 쌍이라
  불가"였는데 **쌍의 개수를 실행 불가로 번역한 것이 오류**다(flash 계열 커널은 attention 행렬을
  만들지 않아 메모리가 O(N), 최악 구성도 3.13 GiB). 그 오판이 bag 기술자를 256개 숫자로 좁혔다
  (sketch는 8,256). **그러나 재설계도 SEAL을 올리지 못했다** — 셀 간 attention과 16,384차원으로
  합성 val_auroc는 0.784 → 0.849로 올랐는데 SEAL은 **0.6619 → 0.6526으로 내려갔고**, 합성에서
  가장 좋았던 v54가 SEAL에서 가장 나빴다(0.6219). **문제는 용량이나 구조가 아니라 일반화다.**
- **attention 백엔드**: B200(sm_100)에서 **cuDNN이 flash보다 빠르다**(bag 85×2,836셀 6.5 vs
  17.7 ms; 100×8,192셀 55.6 vs 145.3 ms). FlashAttention-3는 Hopper(sm_90) 타깃이라 사용 불가.

## 21. 합성 데이터 축 — per-bag cardinality, factorized/XOR, manifold 교체

> 출처: `current_status.md` §81~§85 (2026-08-10/11).

- **per-bag cardinality + padding 계약 (§81)**: episode 안에서도 bag마다 cell 수를 독립 draw한다.
  training collate는 4,096까지 zero-pad하고 초과 bag은 매번 `randperm` subsample한다. mask 밖
  padding은 모델 통계/attention에서 제외해야 한다(`cell_mask`/`bag_mask`). batch 1도 반드시 이
  dense masked 경로를 탄다. **분포가 달라졌으므로 이전 arm의 연장으로 비교하지 않는다.**
- **factorized response / XOR arm 전수 기각 (§82·§83)**: `response_dim`,
  `responsive_population_count`, `label_rule(single|xor)`, `random_causal_factors`,
  `separate_nuisance_rng`를 추가했다. XOR label은 두 causal factor bit가 다를 때 1이며 각 단일
  factor는 label과 marginally independent하다. 결과 v57 0.6127 / v58 0.5530 / v59 0.5616 /
  v60 0.6090 / v61 orthogonal 0.6157 — **모두 v41 0.6940 미달**로 승격하지 않았다.
  ⚠️ v41 CV-only로 잘못 시작한 `logs/20260810_143000/`은 폐기다.
- **OOM 교훈 (§84)**: v62부터 per-bag raw cardinality를 `[256,8192]`(log-uniform power 2.0)로
  하고 **4,096 cap을 dense episode 생성 전에** 적용한다. 이전에는 raw Nmax로 전 bag을 먼저 생성해
  OOM이 났다.
- **v62–v66 hybrid (§85)**: Linear-16 + CV-1 K128 등 hybrid arm과 4-pop DDP8을 완주했으나
  canonical을 넘지 못했다. 이 시기에 branch 명칭(CV/DD/CT)이 확정됐다.

## 22. Canonical CV / DD / CT와 relation head 계보 (v70~v77)

> 출처: `current_status.md` §86~§89 (2026-08-11), 현행 스펙은 `current_architecture.md` F·G·H.

- **canonical CV 승격 (§86)**: "CV branch"는 covariance 단독이 아니라 **fixed-projection centered
  covariance upper triangle + 중심화 전 raw bag mean**이다(K128/1536-d에서 8,256+1,536=9,792).
  두 block은 context-only로 각각 독립 center/scalar-RMS 정규화하고 padding을 제외한다.
  covariance-only 0.6630 → CV 0.6667(+0.0037), ICI 5-seed 0.5381 → 0.5449(+0.0068)로 두 도메인
  방향이 양수였다(ICI CI는 0.5를 포함하므로 실세계 통과 주장은 하지 않는다). v66은 기각.
- **DD (§87)**: 같은 covariance에서 support label로 whitening한 rank-1 dispersion 방향을 만들고
  bag을 그 방향의 log variance scalar로 줄여 standardized squared distance `D0,D1`을 만든다.
  학습 파라미터가 없다. DD-only 0.5862, `0.5p_CV+0.5p_DD` 0.6441로 약했고 `(p_CV+0.1p_DD)/1.1`이
  0.6688로 +0.0021이었으나 **고정 가중치는 task별 신뢰도 차이를 처리하지 못했다** — 그 답이
  relation MLP(v70)다.
- **v70~v74 (§88)**: feature extractor와 ridge/DD 계산을 모두 고정하고 마지막 MLP만 학습한다.
  CT는 평균·covariance가 아니라 **판별적 cell-state의 bag-level abundance**를 측정한다 —
  support cells에서 label-free farthest-point로 후보 16개를 만든 뒤 bag-level abundance의 표준화
  class 차이로 label-0/1 token을 대칭 선택한다(query label은 어느 단계에서도 보지 않는다).
  v71(CV+MLP) 0.6667, v72 0.6709, **v73(+full-dimensional Magnitude Distance) 0.6473으로 실패** —
  Woodbury로 shrinkage Fisher direction을 정확히 계산했지만 raw bag mean이 이미 canonical CV에
  포함되므로 Magnitude는 넣지 않는다. v74(CV+DD+CT, 449 params) 0.6731로 활성 baseline이 됐다.
- **v76 learnable P (§89)**: P(1536×128)를 CV ridge gradient로 학습해 fixed-P v74 0.6731 →
  0.6748. 매 forward thin-QR로 `PᵀP=I`를 보장한다. DD는 현재 P의 covariance를 읽되 **DD→P
  gradient는 차단**된다(그 이유와 v78의 우회는 §23 참조).

## 23. v77 파생 arm 전수 기각과 판정 프로토콜 전환

> 출처: `current_status.md` §90~§100 (2026-08-12). 활성 baseline·판정 절차는
> `current_status.md` §98/§99와 `current_architecture.md` Active 절이 SSOT.

- **synthetic 난이도 축이 유일하게 먹힌 레버 (§90·§91)**: ClassSep을 baseline `[1.0,2.0]` 0.6748에서
  Medium 0.6823 / Mild 0.6853 / **Hard `[0.2,0.8]` 0.6873** / Very-hard 0.6823으로 스윕했다.
  이 축은 단봉·매끄러워 실효과로 읽힌다. 사용자 결정으로 Hard를 **canonical v77**로 승격했다
  (§98). 이는 **데이터/실험 baseline 버전 승격이지 텐서 graph 변경이 아니므로** 내부
  `architecture_version=54`를 유지한다. 과거 v77이라 부른 `PopulationTokenResidualModel`(0.6750)은
  **retired provisional v77-pop-residual**로만 표기하고 내부 version 55는 replay용으로 보존한다.
- **파생 arm 전수 기각 (§92~§98)**: latent 2/4/8/16/32 = 0.6775/0.6781/0.6771/0.6662/**0.6873**,
  MLP bank M=128~4096 = 0.6697/0.6726/**0.6779**/0.6751/0.6649, 50:50 fresh-linear+MLP-1024
  0.6755, learned ridge λ+logit scale 0.6840, large-ragged 2k–16k warm-start 0.6885.
  nonlinear manifold 계열은 fresh orthogonal을 넘지 못한다. **infinite fresh linear의 orientation
  일반화가 반복 가능한 nonlinear manifold 학습보다 낫다**는 것이 이 시기의 결론이다.
  ⚠️ large-ragged 첫 launch는 CUDA prefetch가 다음 대형 generator buffer와 현재 ragged forward를
  겹쳐 GPU 3을 180.5/183.4 GiB까지 밀어 즉시 죽었다 — 그 arm만 `cuda_prefetch: false`로 해결.
- **판정 프로토콜 전환 (§99, 사용자 지시)**: §71에서 판정이 er_status 단일 → SEAL 10-task macro로
  넓어질 때 **pairing과 CI가 함께 넘어오지 않아**, 이후 모든 arm이 점추정 macro끼리의 차이로
  판정됐다. task 내 fold 산포가 ±0.09인데 판정 대상 Δ는 0.001~0.012다. 앞으로는 **fold별 차이를
  통계 단위로** 삼는다(`scripts/compare_arms_paired.py`, GPU 불필요 — 저장된 예측만 읽는다).
  이 방식으로 §98 판정 4건을 재검증해 **전부 유지**됐고 CI 폭은 0.004~0.008로 좁았다.
  새로 드러난 것 둘: ⓐ large-ragged는 동률이 아니라 **재분배**(5개 task가 CI 0 제외로 상승,
  BAP1 −0.0179가 상쇄), ⓑ latent sweep의 비단조성은 **fold 노이즈가 아니다**(L16−L8 −0.0108,
  L32−L8 +0.0103, 둘 다 CI 0 제외) → 남은 후보는 학습 seed 노이즈이며 **pairing으로는 줄일 수
  없다**(두 학습 run에 공유 난수가 없어 상쇄할 공통항이 없다).
- **eigen 미분 금지 (§100)**: DD의 rank-1 방향은 **어느 arm에서도 미분하지 않는다**. eigh backward가
  `1/(λ_i−λ_j)`를 갖고, `+shrinkage·trace·I`는 모든 고윳값을 **균일하게 밀어 간격을 바꾸지 않으며**
  (forward `rsqrt`만 보호), 방향 선택은 hard argmax라 불연속이다. 게다가
  `nonfinite_gradient_policy: zero` 때문에 이 실패는 **조용하다** — 학습이 완주하고 SEAL도 나오므로
  §66의 함정("Δ≈0이 가설 기각인지 경로 미개방인지 구분 불가")이 재현된다. **gradient finite·nonzero를
  테스트로 단정할 것.** v78은 방향을 고정하고 이차형식 `log(fᵀ C_b f)`만 미분한다. 무가중이면 DD가
  P gradient를 CV의 **52배**(median, range 21–103)로 지배하고 거의 직교하므로
  `dd_projection_gradient_weight`로 균형을 맞춘다.

---

---

## 19. 다시 열면 안 되는 결론 / 재개 시 전제 (quick reference)

1. **retrieval은 ICI 규모에서 이득 없음**(v22 제거 확정). 후보 pool이 크거나 노이즈 donor가 해로울
   때만 의미.
2. **세포 선택은 bag 라벨로 학습 불가**(purity 0.128; held-out LDA 0.697은 세포 라벨 상한).
   관측·bag 라벨로 반응 세포 신원을 찾는 경로는 네 번 닫힘.
3. **비지도 population-slot 요약의 정보 상한 ≈0.70** — 토큰 구성·융합·routing 변경은 ±0.02 안.
   데이터가 분리 가능하면(0.951) 모델은 근완벽 — 천장은 데이터 lossiness가 원인.
4. **magnitude 보존 표현은 bag 크기와 교란** — B1과 B2는 상호 필수(단독은 구간 교환/NaN).
5. **CCER/DR-CCER 계열 폐기** — branch 활성 ≠ 상보 정보.
6. **Q-5 population attention은 상수** — v36/v37이 실패한 이유. 새 arm 전에 branch 활성 확인.
7. **clipping 금지**, **bf16-mixed 필수**, **합성 val 지표 불신**, **판정은 SEAL 10개 macro 평균**.
8. **er_status 단일 task 기준의 arm 선택은 과적합 위험** (§71).
9. **v28 이득 곡선**: 선택 순도 0.40이면 covariance +0.107(실현 가능 순도), 오라클 1.00이면 +0.33
   — 부분 개선도 즉시 값이 남.
10. **Musk 0.95 미달** — n>34(0.698) 최약, small-bag 밴드가 0.95 목표의 binding constraint.
    Musk는 transfer development benchmark로 취급, 별도 최종 확인 데이터 잠금 필요.

## 24. current_status.md §91·§98·§100~§102 아카이브 (2026-08-13, §106 시점)

> 출처: `current_status.md` 동일 번호 절 (2026-08-12 작성). 전부 종결됐거나 §104~§106이
> 대체했다. ⚠️ §91 표의 Medium 0.6823은 **오기**이며 실제는 0.6881이다(§105-3), 그리고
> §106-2가 **Medium > Hard (+0.0053, 4 seed)**로 결론을 갱신했다. §98이 정한 0.6873은
> `epoch=048` val-best이고 활성 baseline은 `epoch 49 = 0.6880`이다(§104).

### 91. 2026-08-12 — ClassSep sweep 완료: Hard 0.6873, seed 반복 전 승격 보류

> [!WARNING]
> **아래 표에 오기가 있고 결론이 §105-3에서 정정됐다.** Medium `[0.5,1.4]`는 0.6823이 아니라
> **0.6882**(ep48)/**0.6881**(ep49)다 — 0.6823은 Very-hard 값을 잘못 복사한 것이다. 정정하면
> sweep은 단봉이 아니라 **Medium ≡ Hard의 평탄한 고원**이고(Δ +0.0001 [−0.0022,+0.0024]),
> 네 난이도 arm의 폭 0.0038은 seed std 0.0051보다 작다. **"Hard가 최적"은 성립하지 않는다.**
> 살아남는 결론은 "ClassSep을 `[1.0,2.0]`에서 조이면 +0.011~+0.015"까지다.

`scripts/run_v76_classsep_sweep.py`가 GPU 0–3에서 Mild → Hard → Very-hard를 모두 50 epochs
학습하고 각 validation-best checkpoint를 공식 SEAL 10-task로 평가했다. 09:07 Very-hard의
마지막 artifact까지 생성됐고 runner/DDP/eval 프로세스는 모두 종료됐다. 산출물은
`checkpoints/20260812_v76_classsep_sweep/`, 학습 로그는
`logs/20260812_v76_classsep_sweep/`, task별 평가는 `logs/official50/*_v76_classsep_*_best.log`다.

| ClassSep | 범위 | SEAL macro | Δ vs v76 |
|---|---|---:|---:|
| baseline | `[1.0,2.0]` | 0.6748 | — |
| Medium | `[0.5,1.4]` | 0.6823 | +0.0075 |
| Mild | `[0.8,1.7]` | 0.6853 | +0.0105 |
| **Hard** | **`[0.2,0.8]`** | **0.6873** | **+0.0125** |
| Very-hard | `[0.1,0.5]` | 0.6823 | +0.0075 |

Hard와 동일 SEAL 50-fold 지도학습 baseline의 task별 비교는 다음과 같다.

| task | Hard | ABMIL | Δ ABMIL | MeanMIL | Δ MeanMIL |
|---|---:|---:|---:|---:|---:|
| bc_therapy er_status | 0.7023 | 0.717 | −0.0147 | 0.712 | −0.0097 |
| bc_therapy grade | 0.7227 | 0.770 | −0.0473 | 0.751 | −0.0283 |
| bc_therapy her2 | 0.6908 | 0.663 | **+0.0278** | 0.684 | **+0.0068** |
| cptac_brca PIK3CA | 0.5746 | 0.595 | −0.0204 | 0.544 | **+0.0306** |
| cptac_brca TP53 | 0.8083 | 0.801 | **+0.0073** | 0.787 | **+0.0213** |
| cptac_luad EGFR | 0.7714 | 0.830 | −0.0586 | 0.777 | −0.0056 |
| cptac_luad STK11 | 0.8703 | 0.908 | −0.0377 | 0.873 | −0.0027 |
| cptac_luad TP53 | 0.6621 | 0.751 | −0.0889 | 0.735 | −0.0729 |
| cptac_ccrcc BAP1 | 0.6320 | 0.693 | −0.0610 | 0.720 | −0.0880 |
| cptac_ccrcc VHL | 0.4385 | 0.538 | −0.0995 | 0.542 | −0.1035 |
| **macro** | **0.6873** | **0.7266** | **−0.0393** | **0.7125** | **−0.0252** |

Hard는 ABMIL을 2/10, MeanMIL을 3/10 task에서 상회한다. HER2와 BRCA TP53에서는 둘 다
상회하고 PIK3CA에서는 MeanMIL을 상회하지만, LUAD TP53과 CCRCC BAP1/VHL이 큰 잔여 약점이다.
ABMIL/MeanMIL은 task-label 지도학습이고 Hard는 in-context 모델이므로 직접 수치 비교 시 학습
프로토콜 차이를 명시한다.

**판정/다음 단계**: Hard는 현재 최고 ClassSep 후보이나 모든 arm이 seed 1회뿐이어서 활성 baseline은
v76으로 유지한다. 다음 작업은 Hard `[0.2,0.8]`를 동일 50-epoch·validation-best·SEAL 10-task
절차로 seed 반복하고, +0.0125 상승의 재현성과 task별 편차를 확인한 뒤 승격 여부를 정하는 것이다.

### 98. 2026-08-12 — Hard v76을 canonical v77 baseline으로 승격

사용자 결정에 따라 지금까지 `Hard v76`이라 부른 실험을 **v77 Hard orthogonal**로 명확히
이름 붙이고 활성 baseline으로 승격했다.

- canonical config: `configs/train_v77_hard_orthogonal_1536.yaml`
- canonical checkpoint:
  `checkpoints/20260812_v76_classsep_sweep/hard/epoch=048-val_ce_loss=0.1697.ckpt`
- data: ClassSep `[0.2,0.8]`, fresh orthogonal manifold, latent 32, per-bag 256–8,192,
  training cap 4,096
- model: `CovarianceMeanLearnablePDDCTMLPModel`, P(1536×128) + CV/DD/CT 12→32→1 head,
  trainable **197,057**
- official SEAL 10-task macro: **0.6873**

이 승격은 **데이터/실험 baseline 버전**의 변경이지 텐서 graph 변경이 아니다. 따라서 모델의
내부 `architecture_version=54`는 기존 checkpoint strict-load 호환을 위해 유지한다. 과거에
v77이라 부른 `PopulationTokenResidualModel`은 성능 0.6750으로 기각됐으므로 앞으로
**retired provisional v77-pop-residual**로 표기한다. 그 모델의 내부 version 55도 replay용으로
유지한다.

완료된 파생 실험은 다음처럼 판정한다.

| arm | SEAL macro | Δ vs v77 | 판정 |
|---|---:|---:|---|
| **v77 Hard orthogonal** | **0.6873** | — | **active baseline** |
| large-ragged 2k–16k warm-start (epoch 34) | 0.6885 | +0.0012 | 사실상 동률, 파생 실험 유지 |
| learned ridge λ + logit scale | 0.6840 | −0.0033 | 기각 |
| MLP bank best (M=1024) | 0.6779 | −0.0094 | 기각 |
| 50:50 fresh-linear + MLP-1024 | 0.6755 | −0.0118 | 기각 |

v41_K128 0.6940은 여전히 역사적 전체 최고지만 활성 개발 baseline은 사용자 결정에 따라
v77 0.6873이다. 현재 ICF 학습·평가 프로세스는 없으며, 이후 모든 새 arm은 v77을 control로
비교한다. 이번 갱신에서는 아직 열려 있는 최근 가설과 재현 근거가 상호 참조되므로 별도
section archive는 하지 않았다.

### 100. 2026-08-12 — v78: DD quadratic-form gradient path (구현 완료, 미실행)

v77은 P를 **CV ridge 목적으로만** 학습하는데 DD는 그 P가 만든 covariance를 읽는다. subspace가
한 소비자에 맞춰 최적화되고 다른 소비자는 그것을 물려받는 구조다. v78은 DD에도 P 설계
발언권을 준다.

### 1. eigen 미분은 하지 않는다 — 사용자 지적이 옳았다

`_dd_distance_features`의 방향 계산에는 문제가 둘 있어서 `no_grad`를 그냥 벗기면 안 된다.

1. **`eigh` backward의 `1/(λ_i − λ_j)`**. eigh가 두 번([L730/L734] 구 기준) 있고 고유벡터 항이
   고윳값 간격의 역수를 갖는다. ⚠️ **기존 shrinkage가 이걸 막지 못한다** —
   `+ dd_shrinkage · trace · I`는 모든 고윳값을 같은 양 밀어 **간격을 그대로 둔다**. `clamp_min`도
   forward `rsqrt`를 지킬 뿐이다. 128×128 pooled covariance는 스펙트럼 어딘가에 반드시 촘촘한
   군집이 있다.
2. **hard argmax**. `eigenvectors[:, eigenvalues.abs().argmax()]`는 선택에 gradient가 없고, 상위
   2개 `|λ|`가 교차하면 방향이 **점프**한다.

게다가 이 실패는 **조용하다** — `nonfinite_gradient_policy: zero`가 non-finite를 0으로 치환하므로
학습이 완주하고 SEAL도 나온다. §66의 함정("Δ≈0이 가설 기각인지 경로 미개방인지 구분 불가")이
그대로 재현된다.

**따라서 v78은 방향을 미분하지 않는다.** 방향 계산을 `_dd_direction`으로 분리해 항상 `no_grad`에
두고(어느 arm에서든), gradient는 그 방향을 소비하는 **이차형식** `z_b = log(fᵀ C_b f)`로만 P에
도달한다. f는 에피소드별 상수다. `∂f/∂P`를 버린 부분 gradient이며 방향을 현재값에 고정한
alternating 스킴이다. 이 리팩터는 forward 값을 바꾸지 않는다(`no_grad`는 수치 무관).

### 2. 필수였던 발견 — 무가중 DD는 CV를 대체해버린다

1536-d/K=128에서 6 에피소드 측정한 P gradient 기여도:

| episode | 0 | 1 | 2 | 3 | 4 | 5 | median |
|---|---:|---:|---:|---:|---:|---:|---:|
| DD/CV norm 비 | 90.5 | 75.6 | 21.0 | 23.3 | 102.9 | 29.2 | **52.4** |

`cos(grad_CV, grad_CV+DD) = −0.068`로 거의 직교하며 부호도 음수 쪽이다. 즉 flag를 그냥 켜면
"DD에 발언권을 준다"가 아니라 **P를 DD에 넘기고 CV 신호를 덮어쓴다** — v77이 fixed-P(v74
0.6731)보다 +0.0142 얻은 그 학습을 지우는 것이다. 그 상태로 지면 §66 함정에 다시 걸린다.

그래서 `dd_projection_gradient_weight`를 도입했다. `_ScaleGradient`(forward 정확한 identity,
backward에 weight 곱)를 DD로 들어가는 covariance에 적용한다. **v78 arm은 0.02 ≈ 1/52**로
median 에피소드에서 두 경로를 맞춘다. 비 자체가 5배 변동하므로 모든 에피소드를 맞출 수는
없다.

### 3. 계약

- config: `configs/train_v78_dd_projection_1536.yaml` (v77 canonical 상속, Hard `[0.2,0.8]`,
  orthogonal, DDP4 GPU 0–3, bf16, 50 epochs)
- runner: `scripts/run_v78_dd_projection.py`, tag `v78_dd_projection_best`
- **파라미터 추가 0개, shape 변경 0개** → trainable 197,057 유지, `architecture_version=54` 유지,
  v77 checkpoint와 strict-load **양방향** 호환. flag는 backward 그래프만 넓힌다.
- 기본값 `train_dd_projection: false` → v77 동작 보존. v74는 학습 P가 없으므로 클래스 속성으로
  opt-out 고정.

### 4. 검증

- 신규 테스트 5개 (`tests/test_set_transformer_ridge.py`): ⓐ 기본 off + flag on/off **forward
  동일**, ⓑ `_dd_direction`이 두 arm 모두에서 `requires_grad=False`·`grad_fn=None`
  (eigh가 backward에 없음), ⓒ P gradient가 finite·nonzero이고 **control과 다름**(조용한 null
  방지), ⓓ weight 0.25/0.5의 DD 기여가 정확히 선형이고 weight 0은 control과 일치, 음수는
  ValueError, ⓔ 파라미터 수 동일 + strict-load 양방향.
- 전체 suite **149 tests** (신규 5개 포함). 실패 1건은 **기존** 실패로 이 변경과 무관 —
  `tests/test_mlp_manifold_bank.py`(`7de8b70`)가 BagPFN env에 없는 `pytest`를 import한다.
- config: 전체 `base_config` 참조 검증 141개 통과(failing 0), numeric-type·precision 계약 7개 통과.
- CUDA bf16 smoke (60 bags × 4,096 cells, GPU 0): logits finite, loss 0.5859, P grad finite·nonzero
  (norm 3.99e-01), head grad finite, peak allocation **2.37 GiB**.

### 5. 다음 Action

1. v78 실행 후 **fold-paired Δ + CI**로 v77 대비 판정(§99):
   `python scripts/compare_arms_paired.py --baseline v76_classsep_hard_best --arm v78_dd_projection_best`
2. Δ가 양수이고 CI가 0을 제외하면 seed 반복으로 config 수준 주장으로 승격. 음수여도 이번에는
   weight로 두 경로를 맞춰뒀으므로 "DD가 P를 잡으면 안 된다"는 해석이 성립한다.
3. 미실행 상태다. 실행 전 working tree의 mode 변경 16개를 정리할 것.

### 101. 2026-08-12 — 문서 압축: §2~§97 본문을 history.md로 아카이빙

`current_status.md`가 3,652줄까지 커져 새 세션이 읽을 수 없는 상태였다. 사용자 지시로
`history.md`로 가야 할 본문을 정리했다.

- **3,652 → 811줄**. 전문으로 남긴 것은 §0(요약), §91(활성 baseline 증거표), §98·§99·§100·§101뿐이다.
- 69개 섹션(§2~§97)의 본문을 **결론 1–2줄 + `history.md` 포인터 스텁**으로 교체했다.
  **헤딩은 하나도 지우지 않았다** — 다른 Living 문서가 § 번호로 참조하기 때문이다.
- `history.md`에 빠져 있던 시대를 4개 절로 채웠다(기존 §1–§19는 v41/§71 시대까지만 덮고 있었다):
  - **§20** CV-2 손잡이 소진과 계보 B의 일반화 실패 (구 §72~§79)
  - **§21** 합성 데이터 축 — per-bag cardinality, factorized/XOR, manifold 교체 (구 §81~§85)
  - **§22** Canonical CV / DD / CT와 relation head 계보 v70~v77 (구 §86~§89)
  - **§23** v77 파생 arm 전수 기각과 판정 프로토콜 전환 (구 §90~§100)
- §0이 **2026-08-04(v30 시대) 내용으로 stale**했다 — v22/v24 기준선과 이미 폐기된 Action Plan을
  "새 세션은 여기부터"로 제시하고 있었다. 현재 baseline·판정 방법·열린 과제로 다시 썼다.
- §80("다음 세션이 할 일", 2026-08-10)도 소진돼 §99-5/§100-5를 가리키는 포인터로 교체했다.
- 래핑돼 두 개의 `## ` 줄로 쪼개져 있던 §16 헤딩을 하나로 합쳤다.

**검증**: ⓐ `agent_handoff.md`/`current_architecture.md`/`current_experiments.md`/`README.md`가
참조하는 모든 `§N`이 `current_status.md`에 헤딩으로 존재함을 스크립트로 확인(누락 0),
ⓑ 모든 스텁이 가리키는 `history.md §N`이 존재함을 확인(누락 0).

**아카이빙 기준**(handoff §6.2): 지속 관리 가치가 있는 결론(ADR·설계 이유·트레이드오프·레슨런)은
`history.md`에 요약하고, 개별 파일은 `docs/history/` 폴더에 두지 않으며 원문은 git 이력에 보존한다.
따라서 이 커밋 이전 본문이 필요하면 `git show <이 커밋>^:docs/current_status.md`를 쓸 것.

### v78 진행 상황

문서 정리 중에도 §100의 v78 arm이 계속 돌았다. runner PID/PGID `2164419`(PPID 1, 완전 이탈),
GPU 0–3, DDP rank 4개. 약 18초/epoch(기존 Hard arm과 동일), first-step peak allocated 10.68 GiB.
epoch 26에서 val_ce 0.1717로 v77 best 0.1697에 근접 중이다. 완료 후 판정은 §100-5대로
fold-paired Δ + CI다.

### 102. 2026-08-12 — configs 정리: 루트 67개 → 2개

§101이 부채로 기록만 해둔 config 정리를 실행했다. handoff §7의 "루트에는 활성 entry point만"
규칙이 지켜지지 않아 루트에 종결된 v40~v76 arm이 67개 쌓여 있었다.

### 1. 결과

- **루트는 2개만 남는다**: `train_v77_hard_orthogonal_1536.yaml`(canonical),
  `train_v78_dd_projection_1536.yaml`.
- **v77을 자체 포함형으로 인라인**했다. 이전 체인
  `v77 → v76_classsep_hard → v76_cv_learnable_p_dd_ct_mlp → v74 → v70 → v69`를 인라인해 그 6개를
  아카이브해도 v77이 단독 실행된다 — v34가 §56에서 받은 처리와 같다.
- 종결된 arm **64개**를 시대별로 이관하고 전부 `base_config` 없는 자체 포함형으로 변환했다
  (§7.3): `v34_largectx/`(3), `v40_v45_cvonly/`(8), `v50_v54_encoder/`(5),
  `v57_v61_data_arms/`(5), `v62_v68_hybrid/`(12), `v69_v76_relation/`(30), `v77_pop_residual/`(1).

### 2. 검증 — merged 결과 동일성이 핵심 안전망이었다

이동 전 **entry point 183개의 merged config 해시를 스냅샷**해두고 작업 후 대조했다.
결과 **CHANGED: 0, MISSING: 0**. 전체 테스트는 149개, 실패는 기존 1건
(`test_mlp_manifold_bank.py`의 `pytest` import)뿐이다.

### 3. 참조 검증에서 배운 것 두 가지 (handoff §7에 반영)

**ⓐ 문자열 검색만으로는 참조를 못 찾는다.** 여러 테스트가
`REPO_ROOT / "configs" / "<name>.yaml"`처럼 경로를 **조립**하므로 `configs/<name>` 문자열이
파일에 아예 없다. `configs/` prefix로만 grep해 1차 치환했더니 **테스트 5개에서 35 errors**가
났다(`test_covariance_sketch_knobs`, `test_paired_relation_head`, `test_ridge_ablation`,
`test_training_uses_dense_path`). **basename으로도 grep**해야 한다.

**ⓑ fixture 내부에 config 경로가 저장돼 있다.** `tests/fixtures/cvonly_golden.pt`가 config
경로를 키로 들고 있어 4개 subTest가 깨졌다. **fixture를 재생성하면 pre-prune 기록이 현재
출력으로 대체돼 그 fixture의 존재 이유가 사라진다**(§73). 그래서 재생성하지 않고
`test_cvonly_golden._resolve`가 기록된 경로가 없으면 basename으로 폴백하도록 고쳤다
(단일 매치가 아니면 에러).

### 4. 검증 명령의 구멍을 막았다

handoff §7의 문서화된 검증 명령이 `if 'base_config' not in p.read_text(): continue`로 걸러서
**삭제된 module fragment 때문에 깨진 config를 못 잡고 있었다**. 실제로
`configs/archive/v18_v19/` 10개가 `a5dfcf8`에서 삭제된 `configs/data/learnability_*.yaml` 등을
참조해 로드 불가인데 "failing: 0"을 통과했다. 그 줄을 없애고 module 조각 디렉터리만 제외하도록
바꿨다.

⚠️ **이 10개는 고치려다 되돌렸다.** fragment를 `a5dfcf8^`에서 복구해 인라인하면 10개가
전부 로드되지만, 이 파일들은 **v19~v33 config 50개의 base**이고 인라인하면 group 참조가
해석된 dict로 바뀌어 **자식의 group override 병합 의미가 달라져 그 50개의 merged 결과가 전부
바뀐다**. 스냅샷 대조가 이것을 잡아냈다(`CHANGED: 50`). 따라서 **`failing: 10`이 정상
기준선**이며, 새 작업이 이 수를 늘리지 않는지만 확인한다. 폐기된 v18/v19 아키텍처의 재현
기록이므로 실사용 영향은 없다.

### 5. v78 결과 (§100 판정)

문서 정리 중 v78이 완주했다. 공식 SEAL 10-task macro **0.6869**, v77 대비 fold-paired
**Δ −0.0004, 95% CI [−0.0021, +0.0013]** — **CI가 0을 포함해 구별 불가**, 상승 5/10 task다.

| task | Δ | 95% CI |
|---|---:|---|
| bc_therapy her2_status | +0.0068 | [+0.0034, +0.0102] |
| cptac_brca PIK3CA | −0.0098 | [−0.0202, −0.0021] |
| cptac_luad TP53 | −0.0038 | [−0.0065, −0.0011] |

**판정: 기각.** 그리고 이번에는 **가설 기각으로 읽을 수 있다** — 기제가 실제로 P에 도달함을
테스트로 단정했고(control과 gradient가 다름) weight로 두 경로 크기를 맞춰뒀으므로, §66의
"Δ≈0이 가설 기각인지 경로 미개방인지 구분 불가" 함정에 걸리지 않는다.
**결론: DD에 P 설계 발언권을 줘도 subspace 품질이 개선되지 않는다.**

### 6. 다음 Action

1. **seed 반복** — 여전히 미측정이고 이제 더 급하다. v77/v78의 Δ가 −0.0004라 realization
   노이즈 크기를 모르면 "동률"의 의미를 확정할 수 없다. L8/L16/L32 각 2 seed로 §99-3의
   ⓐ/ⓑ를 가른다.
2. BAP1 large-bag 붕괴 진단(§99-2), VHL 랜덤 이하 진단(§0 열린 과제 2).

---

## 25. v83 Linear Head 승격 & v84 Deep-Head 기각 (2026-08-13, §108~§110)

> 출처: `docs/agent_handoff.md` (구 baseline 및 기각 callout 아카이빙), `current_status.md` §108~§110.

_by Gemini 3.7 Flash on gnode3 at 2026-08-19 15:45:00_

### 1. v83 Linear Head 승격 (§109)

- **배경**: relation head의 비선형성(GELU 은닉층) 필요성을 검증하기 위해 `ct_head_hidden_dims: []`로 hidden layer와 GELU를 없애고 `12→32→1`에서 bare `Linear(12, 1)`로 축소 (trainable parameters: 197,057 → 196,621).
- **결과**: SEAL 10-task macro 0.6905 / 0.6896 / 0.6774 / 0.6944 → mean **0.6880** (seed std 0.0074).
- **판정**: seed-paired Δ = +0.0045, t ≈ 1.15로 §107-3 게이트(4/4 부호 일치 + |t|≥2.5) 미달 상태였으나, "뚜렷하진 않아도 상승으로 보는 것이 타당하다"는 사용자 결정으로 승격됨.

### 2. v84 Deep-Head 기각 (§110)
- **배경**: head 심화 가설 검증 (`ct_head_hidden_dims: [32, 32]`, `12→32→32→1`, GELU 2개, trainable 198,113).
- **결과**: 4 seed mean **0.6777** (seed std 0.0018).
- **판정**: v82 대비 Δ −0.0057 (t ≈ −3.63), v83 대비 Δ −0.0102 (t ≈ −3.61)로 4/4 시드 일치 유의미한 하락 → **기각**.
- **결론**: Relation head는 얕게 만들면 미판정(§108), 깊게 만들면 명확히 손해(§110)이므로 `Linear(12, 1)` 형태가 적정선으로 확인되었으며 이 축의 추가 탐색은 종료됨.



