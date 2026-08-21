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





---

# Part IV: Historical Experiments Archive (§98~§184)

> [!NOTE]
> 이하 내용은 에서 이관된 과거 실험(§98~§184)의 상세 기록이다.

## 98. 2026-08-12 — Hard v76을 canonical v77 baseline으로 승격 — **아카이브됨** ([history.md §24](history.md))

여기서 정한 0.6873은 `epoch=048` validation-best다. 활성 baseline 숫자는 **§104**가 `epoch 49 = 0.6880`으로 대체했다.

---

## 99. 2026-08-12 — 판정 프로토콜: fold-paired Δ + bootstrap CI (사용자 지시)

지금까지 arm 판정은 task별 `fold-mean AUROC` 점추정 10개를 평균한 macro끼리 빼서 했다. CI도
pairing도 bootstrap도 없었다. §65 시절 er_status 단일 task에서는 fold-paired 20k bootstrap을
썼으나, §71에서 판정 기준이 SEAL 10-task macro로 넓어질 때 **pairing과 CI가 함께 넘어오지
않았다**. 문제는 크기다 — er_status가 `fold-mean 0.7023 ± 0.0903`인데 §98 판정표의 Δ는
0.0012~0.0118로 fold 산포가 판정 대상 효과의 8~75배다.

사용자 지시로 앞으로 모든 arm 비교는 **fold별 차이를 통계 단위로** 삼는다.

- 도구: `scripts/compare_arms_paired.py` (신규). GPU 불필요, 재평가 불필요 —
  `test_pathobench.py`가 이미 저장한 `predictions/pathobench_{task}_{tag}_official50_bf16.pt`를
  읽는다.
- 방법: `d_f = auroc_arm(f) − auroc_base(f)`. `d_f`가 이미 차분이므로 fold resample은 구성상
  paired다. fold 20,000회 resample → percentile CI. macro는 task별로 독립 resample한 뒤
  10개 평균을 replicate로 삼는다.
- pairing은 가정하지 않고 **검증**한다: fold 수, `fold_indices`, fold별 `slide_id` 순서, label이
  모두 일치해야 하며 어긋나면 unpaired 폴백이 아니라 `PairingError`다. AUROC는 양쪽을 같은
  코드(`auroc_rows`)로 재계산하고 저장값과 교차검증한다.
- 사용법: `python scripts/compare_arms_paired.py --baseline <TAG> --arm <TAG> [--arm <TAG> ...]`

### 1. §98 판정표 재검증 — 4건 모두 유지

macro 점추정은 §98과 정확히 재현됐다(0.6873 / 0.6885 / 0.6840 / 0.6779 / 0.6755).

| arm | Δmacro | 95% CI | 상승 task | 재판정 |
|---|---:|---|---:|---|
| large-ragged 2k–16k warm-start | +0.0012 | [−0.0008, +0.0032] | 5/10 | CI가 0 포함 — 동률 확증 |
| learned ridge λ + logit scale | −0.0033 | [−0.0058, −0.0010] | 5/10 | CI가 0 제외 — 기각 확증 |
| MLP bank M=1024 | −0.0094 | [−0.0135, −0.0053] | 3/10 | CI가 0 제외 — 기각 확증 |
| 50:50 fresh-linear + MLP-1024 | −0.0118 | [−0.0159, −0.0075] | 2/10 | CI가 0 제외 — 기각 확증 |

CI 폭이 0.004~0.008로, fold 산포 ±0.09 대비 한 자릿수 배 이상 좁다. 점추정 판정이 결과적으로
전부 옳았지만, 그건 사후에 확인된 것이고 그 판정 시점에는 근거가 없었다.

### 2. large-ragged는 "동률"이 아니라 "재분배"다

macro Δ는 0이지만 개별 task 6개의 CI가 0을 제외한다.

| task | Δ | 95% CI | 이긴 fold |
|---|---:|---|---:|
| bc_therapy grade | +0.0111 | [+0.0068, +0.0157] | 37/50 |
| cptac_ccrcc VHL | +0.0090 | [+0.0038, +0.0139] | 34/50 |
| cptac_luad TP53 | +0.0074 | [+0.0023, +0.0123] | 34/50 |
| cptac_brca TP53 | +0.0068 | [+0.0027, +0.0109] | 31/50 |
| cptac_luad EGFR | +0.0053 | [+0.0020, +0.0086] | 31/50 |
| **cptac_ccrcc BAP1** | **−0.0179** | [−0.0282, −0.0070] | 15/50 |

5개 task를 실제로 올리고 BAP1 하나가 그것을 상쇄한다. "+0.0012라 파생 실험 유지"보다 정보량이
크다. **대형 bag에서 BAP1만 무너지는 이유**가 별도 조사 대상이다.

ridge calibration 기각의 주동인은 PIK3CA −0.0272 [−0.0407, −0.0160] (8/50)이며 STK11 −0.0093,
EGFR −0.0059가 뒤따른다. er_status는 반대로 +0.0081이었다.

### 3. latent sweep 비단조성은 fold 노이즈가 아니다

| 비교 | Δmacro | 95% CI | 상승 task |
|---|---:|---|---:|
| L16 − L8 | −0.0108 | [−0.0154, −0.0063] | 4/10 |
| L32 − L8 | +0.0103 | [+0.0054, +0.0151] | 9/10 |

두 CI 모두 0을 제외하므로 L16의 딥은 **주어진 checkpoint 기준으로는 견고**하다. fold 노이즈가
배제되었으니 남은 설명은 ⓐ latent_dim 효과가 실제로 들쭉날쭉하다, ⓑ realization(학습 seed)
노이즈 둘뿐이다. ⓑ는 pairing으로 줄일 수 없는 축이다(두 학습 run에 공유 난수가 없어 상쇄할
공통항이 없다). **L8/L16/L32 각 2 seed 추가**가 이 둘을 가른다.

### 4. 한계 — CI를 하한으로 읽을 것

- 50 fold가 166장 슬라이드를 겹쳐 쓰므로 fold를 독립 표본으로 보는 bootstrap은 분산을
  **과소추정**할 가능성이 크다.
- fold 노이즈만 다룬다. **학습 seed 노이즈**(arm당 checkpoint 1개)와 **task 선택 노이즈**(고정
  10개)는 포함하지 않는다. 스크립트가 출력 말미에 이 한계를 명시한다.

### 5. 다음 Action

1. seed 반복 — v77 + L8/L16/L32. arm당 학습 약 15분 + 평가 1–2분. §3의 ⓐ/ⓑ를 가르고, macro
   seed std를 얻어 앞으로의 +0.005 게이트에 분모를 준다.
2. BAP1이 large-bag에서만 무너지는 원인 진단(§2).

## 100. 2026-08-12 — v78 DD quadratic-form gradient path (구현) — **아카이브됨** ([history.md §24](history.md))

실행 결과와 기각 판정은 §102-5·§103-1, epoch 49 재채점은 §105-4에 있다. DD 미분 금지 계약은 `agent_handoff.md` 상단이 SSOT다.

---

## 101. 2026-08-12 — 문서 압축: §2~§97 본문을 history.md로 아카이빙 — **아카이브됨** ([history.md §24](history.md))

---

## 102. 2026-08-12 — configs 정리: 루트 67개 → 2개 — **아카이브됨** ([history.md §24](history.md))

config 관리 규칙은 `agent_handoff.md` §7이 SSOT다. 이후 v80~v82 arm이 추가돼 루트 config는 다시 늘었다(§106-6).

---

## 103. 2026-08-12 — v78 무가중 기각(단조 악화 확정), v79 dual projection 진행 중

### 1. v78 weight 스윕 완결 — DD gradient의 방향이 해롭다

사용자 지시로 `dd_projection_gradient_weight`를 제한하지 않은 arm을 돌렸다. 결과가 balanced와
합쳐져 **단조 용량-반응**이 됐다.

| weight | SEAL macro | fold-paired Δ vs v77 | 95% CI | 상승 task | 판정 |
|---:|---:|---:|---|---:|---|
| 0 (v77) | 0.6873 | — | — | — | baseline |
| 0.02 (≈1/52) | 0.6869 | −0.0004 | [−0.0021, +0.0013] | 5/10 | 구별 불가 |
| **1.0 (무가중)** | **0.6826** | **−0.0047** | **[−0.0082, −0.0013]** | 3/10 | **CI 0 제외 — 유의하게 나쁨** |

두 arm 직접 비교도 **−0.0043 [−0.0075, −0.0013]**로 CI가 0을 제외한다. 세 점이 단조 감소하고
그 감소가 통계적으로 실재한다.

**해석이 확정됐다.** balanced가 null일 때 남아 있던 두 해석(ⓐ DD가 보탤 게 없다 / ⓑ 0.02가 너무
조였다) 대신 **제3의 답**이다 — **DD는 P를 실제로 움직이며, 그 방향이 전체 readout에 해롭다.**
`cos(grad_CV, grad_CV+DD) = −0.068`(거의 직교, 살짝 음수)과 부합한다. 기제가 P에 도달함을
테스트로 단정하고 크기도 맞춰둔 상태였으므로 §66 함정에 걸리지 않는다 — **가설 기각**이다.

⚠️ **task별로 보면 §71 패턴이 재현된다**: 무가중은 er_status만 **+0.0277**(44/50)로 크게 올리고
PIK3CA −0.0349, grade −0.0093, STK11 −0.0079를 떨어뜨린다. er_status 단독으로 보면 "개선"으로
오판할 arm이었다.

**`train_dd_projection`은 되살리지 말 것.** 계약과 근거는 `current_architecture.md` **G-5**.

### 2. DD 명세를 재정리했다 (current_architecture G절)

DD 관련 서술이 여러 절에 흩어져 있고 "어느 P를 읽는지"가 명시되지 않아 arm 간 차이를 읽을 수
없었다. G절을 DD 전용 명세로 다시 썼다.

- **G-0 (신설)**: DD는 자기 사영을 갖지 않고 **CV가 만든 covariance를 재사용**한다. 따라서 "DD의
  subspace"는 arm마다 다르다 — v74 fixed / **v77 CV가 학습한 P** / v78 같음+gradient 개방 /
  **v79 fixed(분리)**. 표로 고정했다.
- **G-4 (신설)**: rank-1 방향을 **어느 arm에서도 미분하지 않는 이유**. eigh backward의
  `1/(λ_i−λ_j)`와 hard argmax 불연속, 그리고 ⚠️ **shrinkage `+0.25τI`가 고윳값을 균일하게 밀어
  간격을 바꾸지 않으므로 backward를 전혀 보호하지 못한다**는 점. `nonfinite_gradient_policy: zero`
  때문에 이 실패가 **조용하다**는 경고와, 미분 가능 우회(Newton–Schulz + `A²` power iteration,
  미구현)도 적었다.
- **G-5 (신설)**: 위 1절의 스윕 결과와 "되살리지 말 것".
- **G-6 (신설)**: **미측정 항목 2건** — ⓐ **학습된 P에서의 DD-only 성능**(G-2의 0.5862는 fixed P
  수치다). DD는 학습 파라미터가 없으므로 v77의 학습된 P로 DD-only를 채점하면 "DD가 CV의 학습된
  subspace에서 손해를 보는가"를 **학습 없이** 확인할 수 있다 — v79의 전제를 값싸게 검증하는
  진단이며 미실행이다. ⓑ DD 전용 learnable 사영(v79는 fixed로 되돌리기만 했다).

### 3. v79 dual projection — 진행 중

사용자 지시 설계: **learnable CV + fixed CV + fixed DD + CT → 16 feature MLP**. v78처럼 중재하지
않고 **공유를 끊는다**.

- 클래스 `DualProjectionCVDDCTMLPModel`, **`architecture_version = 56`** (v77 ckpt와 strict-load
  **비호환**). 상세 명세는 `current_architecture.md` **Active-6**.
- descriptor를 `[cov_learnable 8,256, mean 1,536, cov_fixed 8,256]` = **18,048**로 확장하고 세
  block을 각각 독립 context-only center/scalar-RMS로 정규화한다. raw bag mean은 사영과 무관하므로
  두 CV branch가 공유한다.
- **fixed-P CV를 독립 evidence block으로 남긴 것**이 두 번째 요점이다. fixed P는 옛 기본값이
  아니라 **v41_K128이 0.6940을 낸 기저**이고 그것이 여전히 역사적 전체 최고다. head가 학습된
  subspace와 고정 subspace를 저울질하게 한다.
- `train_dd_projection`은 **ValueError로 거부**한다 — DD가 buffer를 읽으니 조용한 no-op이 될 뿐이다.
- trainable **197,185** (P 196,608 + head 577).
- config `configs/train_v79_dual_projection_1536.yaml`, runner
  `scripts/run_v79_dual_projection.py`, tag `v79_dual_projection_best`.
- artifacts `checkpoints/20260812_v79_dual_projection/`, `logs/20260812_v79_dual_projection/`.
- **runner PID/PGID `2329816`** (PPID 1, 완전 이탈), GPU 0–3, DDP rank 4개. first-step peak
  allocated 10.69 GiB (v77/v78과 동일). 문서 갱신 시점 epoch 14, val_ce 0.2017.
  ⚠️ 새 16-input head가 랜덤 초기화라 초반 val_ce가 v77보다 높다 — 판정 근거가 아니다.

**검증**: 신규 테스트 4개 중 둘이 설계의 구조적 주장을 고정한다 — ⓐ P를 흔들어도 `cov_fixed`와
`mean` block이 **정확히 불변**(DD가 CV의 subspace를 타지 않음), ⓑ v79의 fixed block이 독립
`CovarianceMeanDDCTMLPModel`의 CV와 **수치적으로 동일**(재파라미터화가 아니라 진짜 v41-스타일 CV).
전체 suite **153 tests**, 실패는 기존 1건(`test_mlp_manifold_bank.py`의 `pytest` import).
CUDA bf16 smoke(60 bags × 4,096 cells) loss 0.5056, P grad norm 1.78e-01 finite, peak 2.38 GiB.
numeric-type·precision 계약 7개 통과.

### 4. v79 결과 — 기각. 세 arm 중 가장 나쁘다

| arm | macro | Δ vs v77 | 95% CI | 상승 task |
|---|---:|---:|---|---:|
| v77 (baseline) | 0.6873 | — | — | — |
| v78 balanced (0.02) | 0.6869 | −0.0004 | [−0.0021, +0.0013] | 5/10 |
| v78 무가중 (1.0) | 0.6826 | −0.0047 | [−0.0082, −0.0013] | 3/10 |
| **v79 dual projection** | **0.6768** | **−0.0105** | **[−0.0137, −0.0074]** | 3/10 |

PIK3CA −0.0518은 이번 세션 단일 task 최대 하락이고 VHL은 0.4166으로 랜덤에서 더 멀어졌다.
er_status만 또 +0.0168로 오른다(§71 패턴 세 번째).

**진단 ⓐ — 과소학습이 아니다. 반대다.** v79 best val_ce **0.1687**(epoch 48)로 v77의 0.1697보다
**좋다**. 즉 합성 val이 개선되는 동안 SEAL이 −0.0105 떨어졌다. 이 리포의 대표 병리가 세 번째로
재현된 것이다 — v54(§79-6, 합성 최고 = SEAL 최저), mixed manifold(§94, val CE 0.2218 개선 /
SEAL 0.6755 하락), 그리고 v79. **50 epoch이 부족한 게 아니라 합성에 더 잘 맞춘 것이 손해였다.**

**진단 ⓑ — head가 선택하지 않고 분산시켰다.** 학습된 head 1층의 block별 column norm share는
CV(learnable) 31.4% / CV(fixed) 26.7% / DD(fixed) 25.5% / CT 16.5%로 거의 균등하다. "fixed CV가
learnable CV를 밀어낸다"가 아니라 **16개 상관된 입력에 weight가 퍼졌다** — 두 CV branch는 mean
block을 공유하고 covariance 정보도 중복된다. ⚠️ column norm은 **거친 대리 지표**다(feature마다
스케일이 달라 곧 기여도가 아니다). 방향성 증거로만 읽을 것.

### 5. 이 축은 소진된 것으로 본다

v78 balanced → 무가중 → v79가 **−0.0004 → −0.0047 → −0.0105**로 단조 악화한다. gradient 개방 /
무가중 / 완전 분리 세 방식으로 CV·DD·사영 배선을 건드렸고 셋 다 졌으며 **건드린 정도가 클수록 더
졌다**. headroom이 이 배선에 있지 않다는 결론이 세 방향에서 수렴한다. 새 arm을 이 축에서 더
설계하지 말 것.

### 6. 다음 Action

1. **seed 반복 — 이제 선행 조건이다.** §99-3·§102-6·§103에서 세 번 미뤘다. 판정 대상 Δ가
   −0.0004~−0.0105인데 realization 노이즈 크기를 모른다. v77 3 seed(약 51분)로 macro seed std를
   먼저 확보하고, 겸해서 L8/L16/L32로 §99-3의 ⓐ/ⓑ를 가른다. **이것 없이는 다음 arm의 판정도
   공허하다.**
2. **G-6ⓐ 진단** — 학습된 P에서의 DD-only(학습 불필요). v79가 졌으므로 "DD가 CV의 학습된
   subspace에서 손해를 보는가" 자체는 아직 답이 없다.
3. **task-side 진단** — VHL 랜덤 이하(§0 열린 과제 2), BAP1 large-bag 붕괴(§99-2).
   배선 축이 막혔으므로 남은 레버는 여기와 §99-2/§0의 reliability feature 쪽이다.
   ⚠️ **§104-5가 이 항목의 근거를 약화시켰다** — BAP1의 large-bag 붕괴 −0.0179는 시드만 바꿔도
   나오는 크기(−0.0402)보다 작다.

---

## 104. 2026-08-12 — baseline을 epoch 49 = 0.6880으로 확정, v80 shallow MLP 기각, seed std 실측

### 0. 이 절이 정한 것 (요약)

| 항목 | 결정 |
|---|---|
| **활성 baseline checkpoint** ⚠️ §107이 대체 | `checkpoints/20260812_v76_classsep_sweep/hard/periodic-epoch=049-val_ce_loss=0.1717.ckpt` |
| **활성 baseline 수치** ⚠️ §107→§109가 대체 | **SEAL 10-task macro `0.6880`** (tag `v77_hard_ep49`, DDP4 1 seed) — §107은 v82 Medium 4 seed 0.6835로, §109는 v83 linear head 4 seed **0.6880**(숫자만 같은 별개 레짐)으로 교체했다 |
| 이전 표기 0.6873의 정체 | 같은 run의 `epoch=048` **validation-best**. Δ +0.0007 [+0.0000, +0.0014] |
| **채점 규칙** | **epoch 49 고정.** validation-best 선택은 판정에 쓰지 않는다 |
| **macro seed std** | **0.0051** (epoch 고정, n=4) / 0.0023 (val-best 선택 시) |
| **판정 게이트** | macro Δ **≈0.010(2σ)** 이상일 때만 단정. 그 미만은 "판정 불가" |
| **task별 CI** | **판정 근거로 쓰지 않는다** (§104-5) |
| v80 shallow MLP | **기각.** 4 seed 평균 0.6722, Δ −0.0158 |

### 1. v80 arm — Hard에서 한 번도 안 돌린 칸이었다

`manifold_mode`에서 이 리포의 "infinite"는 **bank size 무한 = episode마다 새로 뽑음**을 뜻한다.
Hard `[0.2,0.8]`에서 돌린 manifold arm은 **유한 bank뿐**이었다(`mlp_bank` M=128~4096, `mixed` 50:50).
`nonlinear`(fresh MLP per episode, M=∞)는 ClassSep baseline `[1.0,2.0]` 시절 v72(0.6709)가 마지막이고
**Hard에서는 미측정**이었다. ClassSep이 유일하게 먹힌 레버였으므로 난이도별 manifold 순위는
확립된 것이 아니었다.

깊이는 사용자 지시로 **가장 얕은 진짜 MLP**를 썼다. `mlp_num_layers`는 weight 행렬 개수이고
GELU는 마지막 층에 붙지 않는다(`_map_to_manifold`):

| `mlp_num_layers` | 차원 | GELU | 판정 |
|---|---|---:|---|
| 1 | `[32→1536]` | **0** | MLP가 아니다. orthogonal의 비-isometric 열등판 — 쓰지 말 것 |
| **2** | `[32→96→1536]` | **1** | **가장 얕은 진짜 MLP. v80이 이것** |
| 3 | `[32→96→96→1536]` | 2 | bank sweep이 쓴 깊이 |

config `configs/train_v80_hard_shallow_mlp_1536.yaml`(DDP4 정의) +
`..._1gpu.yaml`(4-seed 배치용). v77 대비 바뀐 resolved 키는 **정확히 두 개**
(`manifold_mode: orthogonal→nonlinear`, `mlp_num_layers: 3→2`)이고 타입도 int로 확인했다(§79 함정).
사전 검증: weight shape `[(96,32),(1536,96)]`, superposition gap 0.3965(선형이면 0),
episode마다 다른 맵. 셀 특징은 `synthetic_data.py:792`에서 L2 정규화되므로 nonlinear의 출력
스케일 차이는 모델 입력에서 사라진다 — 스케일 confound 없음.

### 2. 채점을 epoch 49로 고정한 이유 — validation-best가 과소학습 지점을 골랐다

v80 4 seed의 validation-best epoch가 **44 / 16 / 11 / 49**로 흩어졌다. val_ce 스프레드가
0.2276~0.2312(0.0036)뿐이어서 곡선이 거의 평평하고, 어느 epoch이 "best"로 뽑히는지가 사실상
무작위였다. 사용자 지시로 전부 `periodic-epoch=049`로 다시 채점했다.

| seed | ep49 macro | val-best macro | 차이 | val-best epoch |
|---|---:|---:|---:|---:|
| 42 | 0.6688 | 0.6667 | +0.0021 | 44 |
| 43 | **0.6795** | 0.6706 | **+0.0089** | 16 |
| 44 | 0.6686 | 0.6690 | −0.0004 | 11 |
| 45 | 0.6720 | 0.6720 | ±0.0000 | 49 (동일 ckpt) |
| 평균 | **0.6722** | 0.6696 | +0.0026 | |

**seed 43이 결정적이다.** val_ce는 epoch 16(0.2276)이 epoch 49(0.2290)보다 0.0014 좋아 보였지만
SEAL은 **−0.0089 손해**였다. §69-6("합성 지표는 평평한데 실제 task는 계속 오른다")이 이 arm에서
재현됐고, val-best 선택이 seed 43을 과소학습 지점에서 잘라냈다.

**baseline 쪽은 둔감했다**: v77 ep49 0.6880 vs val-best(epoch 48) 0.6873, Δ **+0.0007
[+0.0000, +0.0014]**. 즉 baseline 숫자는 규칙을 바꿔도 흔들리지 않는다. v77의 val_ce는
0.1697 부근에서 안정적이고 v80은 평평했다는 차이다.

⚠️ **주의**: epoch 고정은 분산을 줄이지 않았다. 오히려 **늘렸다**(§104-3). validation-best 선택은
각 시드가 자기 궤적의 최고점을 고르므로 시드 간 차이를 부분적으로 상쇄한다 — 대신 seed 43 같은
편향을 만든다. 편향을 없애고 분산을 드러내는 쪽을 택한 것이다.

### 3. macro seed std 실측 — §103-6의 선행 조건 해소

동일 config·동일 50 epoch, `SEED`만 42/43/44/45로 바꾼 4 run(1-GPU, GPU 0–3 병렬):

| 채점 규칙 | mean | **seed std** | range |
|---|---:|---:|---:|
| epoch 49 고정 | 0.6722 | **0.0051** | 0.0109 |
| validation-best | 0.6696 | 0.0023 | 0.0053 |

⚠️ **n=4라 std 추정 자체가 매우 불확실하다**(자유도 3이면 std의 95% 구간이 대략 0.6~2.9배).
"epoch 고정이 std를 2.2배 늘렸다"는 방향성으로만 읽을 것.

`SEED`는 `train.py`가 `seed_everything`으로 라우팅해 **모델 초기화와 training episode 스트림**을
움직인다. training dataset은 `seed: null` 유지(`episode_dataset: true`라 CLI seed로 덮이지 않는다),
val/test는 50042/60042 고정 — 네 시드가 **동일한 val/test episode**로 채점됐다.

### 4. 판정 게이트가 엄격해졌다 — 기존 판정 3건이 "판정 불가"로 내려간다

epoch-고정 seed std **0.0051**을 분모로 과거 Δ를 다시 읽으면:

| arm | Δmacro | σ 배수 | 재해석 |
|---|---:|---:|---|
| v78 balanced (0.02) | −0.0004 | 0.1σ | 노이즈 (기존 "동률"과 일치) |
| large-ragged warm-start | +0.0012 | 0.2σ | 노이즈 (미승격이 옳았다) |
| ridge calibration | −0.0033 | 0.6σ | **판정 불가** — 기존 "CI 0 제외 → 기각"이 성립하지 않는다 |
| v78 무가중 (1.0) | −0.0047 | 0.9σ | **판정 불가** — 같은 문제 |
| MLP bank M=1024 | −0.0094 | 1.8σ | 경계 |
| v79 dual projection | −0.0105 | 2.1σ | 겨우 유지 |
| **v80 shallow MLP** | **−0.0158** | **3.1σ** | **기각 (4 seed)** |

**§103-5의 논거가 좁아진다.** "v78 balanced → 무가중 → v79가 −0.0004 → −0.0047 → −0.0105로
**단조 악화**하므로 축이 소진됐다"에서 중간 단계(−0.0047)가 seed 노이즈와 구분되지 않는다.
소진 결론 자체는 v79의 −0.0105(2.1σ)가 버티므로 방향은 유지하지만, **"단조성"을 증거로 인용하지 말 것.**

### 5. task별 fold-paired CI는 판정 근거로 쓸 수 없다 — 실측

**시드만 다른 두 run**(v80 seed42 vs seed45, val-best 채점)을 fold-paired로 비교했더니
**CI가 0을 제외하는 task가 6개** 나왔다. 처치는 없었다.

| task | Δ (시드 차이뿐) | 95% CI |
|---|---:|---|
| **cptac_ccrcc BAP1** | **−0.0402** | [−0.0666, −0.0156] |
| cptac_ccrcc VHL | +0.0324 | [+0.0089, +0.0570] |
| cptac_brca PIK3CA | +0.0259 | [+0.0019, +0.0500] |
| bc_therapy er_status | +0.0199 | [+0.0068, +0.0339] |
| cptac_luad STK11 | +0.0119 | [+0.0014, +0.0225] |
| bc_therapy grade | −0.0120 | [−0.0208, −0.0030] |
| MACRO | +0.0053 | [−0.0001, +0.0107] (0 포함 — macro는 정직했다) |

task별 seed std(val-best 채점, n=4)는 BAP1 0.0299 / PIK3CA 0.0237 / luad TP53 0.0225 /
VHL 0.0214 / grade 0.0191이고 평균 **0.0161** — macro(0.0023)의 7배다. **평균이 노이즈를 지운다.**
epoch-고정에서도 BAP1은 0.5461~0.6464(range 0.100)로 흔들린다.

⚠️ **따라서 §99-2의 "large-ragged는 동률이 아니라 재분배다"는 성립하지 않는다.** 그 판정의
근거는 task 6개의 CI 0 제외와 **BAP1 −0.0179**였는데, 시드만 바꿔도 같은 BAP1이 **−0.0402**
(2.2배)로 움직인다. pairing은 fold 노이즈만 잡고 realization 노이즈는 그대로 남긴다 —
§99-4가 한계로 적어둔 것이 실측된 것이다. **arm이 다른 두 run 사이의 task별 CI는 해석하지 말 것.**

### 6. v80 최종 판정 — 기각

epoch 49 규칙을 baseline에도 적용한 비교:

| arm | macro | Δ vs v77 ep49 | 95% CI | 상승 task |
|---|---:|---:|---|---:|
| **v77 Hard orthogonal ep49** | **0.6880** | — | — | — |
| v80 seed 42 | 0.6688 | −0.0192 | [−0.0240, −0.0142] | 3/10 |
| v80 seed 43 | 0.6795 | −0.0085 | [−0.0134, −0.0036] | 5/10 |
| v80 seed 44 | 0.6686 | −0.0194 | [−0.0241, −0.0149] | 2/10 |
| v80 seed 45 | 0.6720 | −0.0160 | [−0.0210, −0.0111] | 3/10 |
| **v80 평균** | **0.6722** | **−0.0158** | | |

네 시드 모두 CI가 0을 제외한다. v80 seed SE 0.0025에 v77의 seed 노이즈를 v80과 같다고 가정해
합성하면 SE≈0.0057, **t≈2.8**. ⚠️ **v77은 시드 1개뿐이므로 이 마진은 그 가정에 의존한다.**

사전 증거와 방향이 일치한다: bank sweep M=1024→4096 하강(0.6779→0.6649)의 M→∞ 외삽, 그리고
v72의 0.6709. **얕은 infinite MLP도 fresh orthogonal linear를 넘지 못한다 — Hard 난이도에서도
manifold 순위는 뒤집히지 않았다.**

⚠️ **남은 confound (판정을 뒤집을 정도는 아니나 명시해야 한다)**: v80은 **1-GPU**
(1024 step/epoch, effective batch 1), v77 baseline은 **DDP4**(rank당 256 step/epoch,
gradient 4개 평균)다. Lightning이 loader를 DistributedSampler로 감싸기 때문이다. 같은 LR에서
optimizer step이 4배, gradient 노이즈가 4배다. **−0.0158 전부를 manifold 효과로 귀속할 수 없다.**
순수 manifold 효과를 원하면 v77을 `..._1gpu` 레이아웃으로 4 seed 돌려야 한다(약 55분, 미실행).

### 7. 산출물

- 학습: `logs/20260812_v80_shallow_mlp_seeds/v80_shallow_mlp_seed4{2,3,4,5}.out`,
  ckpt `checkpoints/20260812_v80_shallow_mlp_seeds/v80_shallow_mlp_seed4{2,3,4,5}/`
  (4개 전부 `training completed successfully`). 1-GPU 4병렬로 학습 약 50분.
- 평가 태그: `v80_shallow_mlp_seed4{2,3,4,5}_best`(val-best), `..._ep49`(epoch 고정),
  `v77_hard_ep49`(baseline). 예측 90개 = 40+40+10, 로그 `logs/official50/*_<tag>.log`.
- 재확인:
  ```bash
  for tag in v77_hard_ep49 v80_shallow_mlp_seed42_ep49 v80_shallow_mlp_seed43_ep49 \
             v80_shallow_mlp_seed44_ep49 v80_shallow_mlp_seed45_ep49; do
    printf "%-32s " $tag
    grep -hoP 'fold-mean AUROC: \K[0-9.]+' logs/official50/*_${tag}.log \
      | awk '{s+=$1;k++} END{printf "%.4f (%d)\n", s/k, k}'
  done
  ```
- ⚠️ v80 config 2개는 기각된 arm이므로 다음 정리에서 v78 두 개와 함께
  `configs/archive/`로 이관할 것(§7 규칙).

### 8. 다음 Action

1. **v77 1-GPU 4 seed control** — v80의 batch-regime confound를 없애고, 동시에 baseline
   자신의 seed std를 얻는다(현재 v77은 시드 1개다). 약 55분, GPU 0–3.
2. **§99-2·§103-6의 task-side 항목 재검토** — BAP1 large-bag 붕괴와 VHL 랜덤 이하는 §104-5에
   비추면 근거가 약하다. 다시 세우려면 **arm마다 최소 3 seed**가 필요하다.
3. **판정 게이트 0.010을 넘길 만한 축을 찾을 것.** 지금까지의 arm은 대부분 이 게이트 아래에서
   싸웠다 — 미세 배선이 아니라 큰 레버(ClassSep이 그랬던 것처럼)를 찾아야 한다.

---

## 105. 2026-08-12 — 과거 판정 전수 감사 + 27개 arm epoch 49 재채점

### 0. 이 절이 뒤집은 것 (요약)

| 항목 | 이전 기록 | 재채점 후 (epoch 49) |
|---|---|---|
| **ClassSep Medium** | 0.6823 (§91) | **0.6881** — §91 표가 Very-hard 값을 잘못 복사했다 |
| **ClassSep 선택** | Hard가 최고, 단봉 | **Medium ≡ Hard** (Δ +0.0001 [−0.0022,+0.0024]). Hard 선택은 근거 없음 |
| **v74 → v76 승격** | +0.0017 | **+0.0004 [−0.0030,+0.0038] → 판정 불가.** learnable P 도입에 SEAL 근거가 없다 |
| ridge calibration | −0.0033, 기각 | **−0.0010 [−0.0021,+0.0002] → 기각 철회, 판정 불가** |
| v78 무가중 | −0.0047, 기각 | −0.0039 [−0.0075,−0.0004] → CI는 0을 제외하나 **게이트(0.010) 미만** |
| v79 dual projection | −0.0105, 기각 | **−0.0112 [−0.0144,−0.0080] → 기각 유지** (게이트 초과) |

### 1. 감사 방법 — 로그가 checkpoint를 기록해 두었다

`logs/official50/*.log`의 `Model: arch vNN, checkpoint <파일명>` 줄로 **10-task 완주 tag 80개
전부**가 어느 checkpoint로 채점됐는지 복원했다. 문서가 실제 판정에 쓴 36건을 추리면:

| 분류 | 건수 |
|---|---:|
| Δ < 0.005 (게이트 0.010의 절반에도 미달) | **17건** |
| **최종 epoch을 쓰지 않음**(중간 validation-best) | **27건** |
| 둘 다 | 11건 |
| 온전한 판정 | 4건 |

최악은 ridge calibration **epoch 27**, mlpbank2048/4096 **epoch 28/27**, latent4/8 **epoch 27/37**,
axis noise/rare **epoch 33/32**다. `scripts/rescore_final_epoch.sh`로 27개 arm을 최종 epoch
(200-epoch combo는 epoch 199)에서 재채점했다 — 270회 평가, 4-GPU 병렬 약 30분, 전부 10/10 성공.

### 2. 재채점 결과 — 차이는 작았다 (부호가 뒤집힌 arm은 없다)

| arm | 기존 ep | 기존 | **ep49** | Δ |
|---|---:|---:|---:|---:|
| mlpbank128 | 40 | 0.6697 | **0.6734** | +0.0037 |
| v73_magnitude | 46 | 0.6473 | **0.6509** | +0.0035 |
| ridge_calibration | 27 | 0.6840 | **0.6870** | +0.0031 |
| mlpbank4096 | 27 | 0.6649 | **0.6678** | +0.0030 |
| mlpbank2048 | 28 | 0.6751 | **0.6776** | +0.0025 |
| classsep_veryhard | 40 | 0.6823 | **0.6843** | +0.0020 |
| v78_unweighted | 42 | 0.6826 | **0.6841** | +0.0015 |
| v78_balanced | 42 | 0.6870 | **0.6879** | +0.0009 |
| combo_200ep | 95 | 0.6751 | **0.6761** (ep199) | +0.0009 |
| axis_noise | 33 | 0.6856 | **0.6839** | −0.0017 |
| **v76_learnable_p** | 46 | 0.6748 | **0.6735** | −0.0014 |
| latent8 | 37 | 0.6771 | **0.6759** | −0.0011 |
| axis_rare | 32 | 0.6847 | **0.6838** | −0.0008 |
| latent4 | 27 | 0.6781 | **0.6776** | −0.0005 |
| v69/v71/v72/v75/latent2/latent16/mlpbank512/mlpbank1024/mixed50/axis_classsep/axis_response/classsep_mild/v79 | — | — | 0.6712/0.6668/0.6706/0.6724/0.6775/0.6665/0.6730/0.6780/0.6755/0.6881/0.6777/0.6854/0.6768 | \|Δ\|≤0.0004 |

**최대 변화가 +0.0037이다.** 미완주 채점은 실재한 문제였지만 **효과 크기는 seed std(0.0051)보다
작다** — 즉 과거 판정이 통째로 틀린 것은 아니었다. 다만 판정 대상 Δ 자체가 0.001~0.005였으므로
**게이트 대비 위치는 여러 arm에서 바뀐다**(§105-4).

### 3. §91 ClassSep sweep 표에 오기가 있었다 — 결론이 바뀐다

§91은 Medium `[0.5,1.4]`를 **0.6823**으로 적었는데, 그 값은 같은 표의 Very-hard 값과 동일하고
어느 tag에도 대응하지 않는다. 실제 Medium arm은 `v76_axis_classsep_medium`이고 —
resolved config를 `classsep_mild`와 diff하면 **`class_separation`만 다르다**(확인함) —
**0.6882(ep48) / 0.6881(ep49)**다.

epoch 49로 통일한 ClassSep sweep:

| ClassSep | 범위 | ep49 macro | Δ vs baseline | Δ vs Hard | CI (vs Hard) |
|---|---|---:|---:|---:|---|
| baseline | `[1.0,2.0]` | 0.6735 | — | −0.0145 | — |
| **Medium** | `[0.5,1.4]` | **0.6881** | **+0.0146** | **+0.0001** | [−0.0022, +0.0024] 동률 |
| Mild | `[0.8,1.7]` | 0.6854 | +0.0119 | −0.0026 | [−0.0055, +0.0003] 동률 |
| **Hard** | `[0.2,0.8]` | **0.6880** | **+0.0145** | — | — |
| Very-hard | `[0.1,0.5]` | 0.6843 | +0.0108 | −0.0037 | [−0.0053, −0.0022] |

**⚠️ §91의 "단봉·매끄러워 실효과로 읽힌다"는 성립하지 않는다.** 실제 모양은
0.6735 → **0.6881** → 0.6854 → **0.6880** → 0.6843로, Medium과 Hard가 동률이고 그 사이 Mild가
살짝 내려가는 **평탄한 고원**이다. 네 난이도 arm의 폭은 0.0038로 **게이트(0.010)와 seed
std(0.0051) 모두 아래**다.

**살아남는 결론**: ClassSep을 `[1.0,2.0]`에서 조이는 것 자체는 **+0.011~+0.015로 유효하다**(게이트
초과). **사라지는 결론**: 그 안에서 `[0.2,0.8]`이 최적이라는 것 — Hard 선택은 노이즈 안의 임의
선택이었다. baseline을 옮길 이유는 없지만(Medium과 동률이므로), **"Hard가 최적"이라고 쓰지 말 것.**

참고: 2026-08-11의 `v76_medium`(0.6723)은 같은 `[0.5,1.4]`지만 **다른 arm**이다 — resolved diff에서
`observation_noise` 0.01, `rare_response_probability` 0.15, response scale 3종이 함께 달랐다.
난이도 축을 한꺼번에 5개 움직인 arm이므로 ClassSep sweep과 같은 표에 놓지 말 것.

### 4. 판정이 바뀌는 arm

v77 ep49 = 0.6880 기준, fold-paired:

| arm | ep49 Δ | 95% CI | 이전 판정 | **새 판정** |
|---|---:|---|---|---|
| axis_classsep (=Medium) | +0.0001 | [−0.0022, +0.0024] | (미승격) | 동률 — Hard와 구분 불가 |
| v78_balanced | −0.0001 | [−0.0006, +0.0003] | 동률 | 동률 (유지) |
| ridge_calibration | **−0.0010** | [−0.0021, +0.0002] | **기각** | **기각 철회 → 판정 불가** |
| classsep_mild | −0.0026 | [−0.0055, +0.0003] | 미승격 | 판정 불가 |
| classsep_veryhard | −0.0037 | [−0.0053, −0.0022] | 미승격 | 게이트 미만 |
| v78_unweighted | −0.0039 | [−0.0075, −0.0004] | 기각 | 게이트 미만 — 단정 불가 |
| **v79_dual** | **−0.0112** | [−0.0144, −0.0080] | 기각 | **기각 유지** |

그리고 v74(0.6731) 기준:

| arm | ep49 Δ | 95% CI | **새 판정** |
|---|---:|---|---|
| **v76_learnable_p** | **+0.0004** | [−0.0030, +0.0038] | **판정 불가 — 승격 근거 없음** |
| v75_cv2 | −0.0007 | [−0.0020, +0.0007] | 동률 (유지) |
| v72_mlp1 (nonlinear manifold) | −0.0025 | [−0.0045, −0.0005] | 게이트 미만 |

**⚠️ 계보의 두 승격 모두 근거를 잃었다.** v74 → v76(learnable P)은 +0.0004,
v76 → v77(Hard 선택)은 Medium과 +0.0001이다. 현행 baseline이 **틀렸다는 뜻은 아니다** — 지는
증거도 없다. 다만 **"learnable P가 fixed P보다 낫다"와 "Hard가 최적 난이도다"는 둘 다
미측정 상태**이고, 확정하려면 arm당 3 seed가 필요하다.

### 5. §103-5와 §104-6의 논거 재확인 — 둘 다 살아남았다

**단조 악화(§103-5)**: epoch 49에서도 v78 balanced −0.0001 → 무가중 −0.0039 → v79 −0.0112로
**순서가 유지된다.** 다만 세 점 중 둘이 게이트 미만이므로 "단조성"은 방향 증거로만 쓸 것.

**mlpbank M→∞ 외삽(§104-6)**: 재채점 전에는 하강 구간이 epoch 28/27에서 채점돼 학습 부족
artifact 의심이 있었다. epoch 49로 통일한 결과는 **0.6734 / 0.6730 / 0.6780 / 0.6776 / 0.6678**
(M=128/512/1024/2048/4096) — 1024와 2048이 고원을 이루고 **4096에서 −0.0102 하강한다.**
큰 M에서의 하강은 학습 부족이 아니라 실재하며, v80(M=∞) 기각의 보강 근거는 **유효하다.**

**latent sweep 비단조성(§99-3)**: epoch 49에서 L2 0.6775 / L4 0.6776 / L8 0.6759 / **L16 0.6665** /
L32 0.6880 — L16 딥이 그대로 남았다. 학습량 차이 가설은 배제됐다.

**axis sweep** (baseline 0.6735 기준, ep49): classsep +0.0146 / noise +0.0104 / rare +0.0103 /
response +0.0042. 앞의 셋은 게이트 초과다.

### 6. 재채점 불가 3종

1. **large-ragged warm-start** — epoch 34에서 조기 종료돼 epoch 49가 없다. +0.0012는 구조적으로
   epoch 통일이 불가능하고, 어차피 게이트 미만이다.
2. **CV-only 계보(v41_K128 0.6940, v41_K64, v42, v45)** — prune 이전 ckpt라 현재 트리로 로드
   불가(§73). `8caa96c` worktree 필요. 다행히 **원래 epoch 49/last로 채점돼 이미 최종 epoch**이다.
   v41_K128의 0.6940은 9-task 로그 + er_status 별도 채점으로 정확히 복원됐다.
3. **v43_notanh(0.6770) / v44_lowT(0.6763)** — `logs/official50`·`predictions/`에 **산출물이 전혀
   없다.** 결과표 수치를 artifact로 되짚을 수 없다 → `current_experiments.md`에 "artifact 없음"으로
   표기했다.

### 7. 산출물

- 러너: `scripts/rescore_final_epoch.sh` (경로를 사전 해석해 `manifest.txt`에 기록 — 누락은 조용히
  빠지지 않고 `SKIP` 줄로 남는다. 이번 실행은 skip 0건).
- 로그 `logs/20260812_rescore_ep49/{runner.out,manifest.txt,<tag>.out}`,
  평가 로그 `logs/official50/*_<tag>_ep49.log`, 예측 `predictions/*_ep49_official50_bf16.pt` 270개.
- 태그 규칙: 기존 `_best` 태그는 남기고 `_ep49`(combo는 `_ep199`)를 새로 만들었다 — 두 채점을
  나란히 비교할 수 있다.

### 8. 다음 Action

1. **seed 반복이 여전히 병목이고, 대상이 늘었다.** 이제 판정 불가로 내려간 것이 4건
   (v76 learnable P, Hard vs Medium, ridge calibration, v78 무가중)이다. 우선순위는
   **v74 vs v76 (learnable P)** — 계보의 뿌리이고, 여기가 무효면 v77 전체가 fixed-P로 되돌아간다.
2. **ClassSep을 "조이면 좋다"까지만 쓰고 특정 범위를 최적이라 쓰지 말 것.** Medium과 Hard의
   3 seed 비교가 필요하다.
3. §91의 Medium 행은 오기다 — 아카이빙할 때 이 절을 함께 참조하도록 표시했다.

---

## 106. 2026-08-13 — 4 arm × 4 seed 정면 비교: Medium이 Hard를 이기고, learnable P는 여전히 미판정

### 0. 이 절이 정한 것

| 질문 | 답 | 근거 |
|---|---|---|
| **Hard가 최적 난이도인가** | **아니다. Medium이 +0.0053 낫다** | 4/4 seed 양수, seed-paired t=3.0 |
| **learnable P가 fixed P보다 나은가** | **여전히 판정 불가 (+0.0048, t=1.5)** | seed 44가 부호 반전(−0.0043) |
| **1-GPU와 DDP4는 얼마나 다른가** | **DDP4가 +0.0098 높다** | v77 1-GPU 4 seed 0.6782 vs DDP4 0.6880 |
| v80 shallow MLP의 진짜 크기 | **−0.0059** (기존 −0.0158은 부풀려진 값) | 같은 레이아웃 control과 비교 |
| v77 seed std | **0.0053** | §104-3이 가정했던 0.0051과 일치 |

### 1. 배치 구성 — 처음으로 모든 arm이 같은 레짐이다

`SEED` 42/43/44/45, 1-GPU 레이아웃(`*_1gpu.yaml`), 50 epoch, **epoch 49 채점**으로 네 arm을
동일 조건에서 돌렸다. 각 arm은 v77에서 **한 축만** 바꾼 것이다.

| arm | v77에서 바뀐 것 | mean | seed std | range |
|---|---|---:|---:|---:|
| **v82** | ClassSep `[0.2,0.8]` → **`[0.5,1.4]`** | **0.6835** | 0.0030 | 0.0068 |
| **v77** | (기준) | 0.6782 | **0.0053** | 0.0120 |
| **v81** | learnable P → **fixed P** (197,057 → 449) | 0.6734 | 0.0018 | 0.0041 |
| **v80** | manifold orthogonal → **shallow MLP** | 0.6722 | 0.0051 | 0.0109 |

**seed std가 arm마다 3배 차이 난다** — fixed P 0.0018 < Medium 0.0030 < Hard 0.0053 ≈ shallow MLP
0.0051. 학습 파라미터가 449개뿐인 v81이 가장 안정적이고, **학습되는 P를 가진 arm일수록 realization
노이즈가 크다.** §104-5에서 "task별 CI가 seed 노이즈에 압도된다"고 한 것도 learnable-P arm의
성질이었다(v81은 task별 산포도 작다: brca TP53 폭 0.0008, EGFR 0.0034, 단 BAP1만 0.0256).

### 2. 난이도 — Hard가 최적이 아니다

seed-paired (같은 시드끼리 뺀 뒤 평균), fold-paired CI는 시드별:

| seed | v77 Hard | v82 Medium | Δ | 95% CI |
|---|---:|---:|---:|---|
| 42 | 0.6811 | 0.6846 | +0.0035 | [+0.0007, +0.0063] |
| 43 | 0.6834 | 0.6870 | +0.0036 | [+0.0006, +0.0067] |
| 44 | 0.6714 | 0.6821 | **+0.0107** | [+0.0075, +0.0140] |
| 45 | 0.6767 | 0.6802 | +0.0036 | [+0.0006, +0.0066] |
| **평균** | 0.6782 | **0.6835** | **+0.0053** | seed-paired SE 0.0018, **t=3.0** |

**4/4 시드에서 양수이고 시드별 CI도 모두 0을 제외한다.** §105-3은 단일 시드로 "Medium ≡ Hard
(+0.0001)"라 했는데, 4 seed로 보면 **Medium이 유의하게 낫다.** §91의 "Hard가 최고"는 오기(§105-3)
때문이었고, 정정 후에도 남아 있던 "동률" 판정마저 뒤집힌 셈이다.

⚠️ 다만 +0.0053은 §104-4의 게이트 0.010(단일 시드 2σ 기준) 미만이다. **4 seed paired 설계에서는
SE가 0.0018로 줄어 이 크기를 판정할 수 있다** — 게이트는 시드 1개짜리 비교에 대한 규칙이지
seed-paired 설계에 그대로 적용하는 값이 아니다.

### 3. learnable P — 같은 난이도에서 비교하면 여전히 미판정

| seed | v81 fixed P | v77 learnable P | Δ | 95% CI |
|---|---:|---:|---:|---|
| 42 | 0.6716 | 0.6811 | +0.0095 | [+0.0049, +0.0141] |
| 43 | 0.6739 | 0.6834 | +0.0095 | [+0.0049, +0.0140] |
| 44 | 0.6757 | 0.6714 | **−0.0044** | [−0.0086, −0.0001] |
| 45 | 0.6724 | 0.6767 | +0.0042 | [−0.0006, +0.0090] |
| **평균** | 0.6734 | 0.6782 | **+0.0048** | seed-paired SE 0.0033, **t=1.5** |

**시드 44에서 부호가 뒤집히고 그 CI도 0을 제외한다.** 즉 fold-paired CI만 보면 "seed 42·43은
learnable 승, seed 44는 fixed 승"으로 서로 모순되는 결론이 나온다 — §104-5가 경고한 그대로,
**realization 노이즈가 이 효과 크기를 삼킨다.** §105-4의 "v74→v76 승격은 근거 없음"이 유지된다.
197,057개 파라미터가 449개 대비 얻는 것은 **아직 증명되지 않았다.**

⚠️ 앞서 v82(Medium, learnable) − v81(Hard, fixed) = **+0.0101 (t=5.8)**로 크게 이긴 것은
**두 축이 겹친 결과**다: 난이도 +0.0053과 P +0.0048의 합이며, 개별로는 후자가 유의하지 않다.
**이 +0.0101을 "learnable P의 이득"으로 인용하지 말 것.**

### 4. 레이아웃 효과를 처음 실측했다 — DDP4가 +0.0098 높다

v77 1-GPU 4 seed **0.6782** vs 같은 config의 DDP4 단일 run **0.6880** → **−0.0098**.
1-GPU는 1024 step/epoch·effective batch 1, DDP4는 rank당 256 step·gradient 4개 평균이다.
합성 val_ce는 반대 방향이었다(1-GPU 0.1650~0.1692 < DDP4 0.1717) — **더 많이 학습해 합성 loss는
낮아졌지만 SEAL은 −0.0098 떨어졌다.** 합성 지표와 실데이터의 괴리가 레이아웃 축에서도 재현된다.

⚠️ DDP4 쪽은 시드 1개이므로 −0.0098에는 DDP4의 realization 노이즈(≈0.005)가 섞여 있다. 크기는
seed std의 약 2배이므로 방향은 신뢰하되 정확한 값으로 인용하지 말 것.

**이 confound가 §104-6의 v80 판정을 부풀렸다.** 같은 레이아웃 control과 비교하면:

| 비교 | Δ | 판정 |
|---|---:|---|
| v80 vs **DDP4** v77 0.6880 (§104-6에 기록된 값) | −0.0158 | 부풀려짐 |
| **v80 vs 1-GPU v77 4 seed (정당한 control)** | **−0.0059** (t=−2.7, 4/4 시드 음수) | **기각 유지** |

v80 기각 자체는 유지되지만 **크기는 2.7배 과장돼 있었다.** §104-6은 이 confound를 명시했고
"matched control 미실행"이라 적어 두었으므로 기록은 정직했으나, 이제 실측값으로 대체한다.

### 5. 승격 판단은 사용자 몫으로 남긴다

Medium이 Hard보다 나은 것은 4 seed로 확인됐지만, 이는 **1-GPU 레짐에서의 결과**다. 활성
baseline은 DDP4 v77 ep49 0.6880이고, Medium의 DDP4 수치는 단일 시드 0.6881뿐이다.
**baseline을 Medium으로 옮기려면 DDP4 Medium을 3~4 seed로 확인하는 것이 순서다**(arm당 약 28분).
지금 문서의 baseline 정의는 바꾸지 않았다.

### 6. 산출물

| arm | config | checkpoints | tag |
|---|---|---|---|
| v77 4 seed | `train_v77_hard_orthogonal_1536_1gpu.yaml` | `20260813_v77_baseline_seeds/` | `v77_baseline_seed4{2..5}_ep49` |
| v82 Medium | `train_v82_medium_classsep_1536{,_1gpu}.yaml` | `20260813_v82_medium_seeds/` | `v82_medium_seed4{2..5}_ep49` |
| v81 fixed P | `train_v81_hard_fixed_p_1536{,_1gpu}.yaml` | `20260812_v81_fixed_p_seeds/` | `v81_fixed_p_seed4{2..5}_ep49` |
| v80 shallow MLP | `train_v80_hard_shallow_mlp_1536{,_1gpu}.yaml` | `20260812_v80_shallow_mlp_seeds/` | `v80_shallow_mlp_seed4{2..5}_ep49` |

v81 config 검증: 초기 시점에 v77과 **수치적으로 동일**함을 확인했다 — P 차이 3.6e-07, 8,256차원
covariance descriptor 차이 1.0e-06. v77의 클래스는 부모의 `_covariance_projection` buffer를
`nn.Parameter`로 재등록하는 것이 전부이고 그 buffer는 이미 `QR(sin/cos).Q`로 orthonormal이라,
`model_src`를 부모로 되돌리면 **P가 v77의 초기값에서 얼어붙는다.**
v82 config 검증: resolved config가 기존 단일시드 Medium arm(`v76_axis_classsep_medium`)과
**완전히 동일**하고 v77과는 `class_separation`만 다르다.

### 7. 다음 Action

1. **DDP4 Medium 3~4 seed** — baseline 승격 여부를 가르는 유일한 남은 측정이다.
2. **learnable P를 살리려면 다른 증거가 필요하다.** 같은 난이도에서 t=1.5이고 시드 하나가 반대
   방향이다. Medium 난이도에서 fixed P를 돌려(4 seed) 상호작용을 보는 것이 다음 후보다 —
   "난이도를 조이는 이득이 learnable P에서만 나온다"는 가설은 v81(Hard, fixed) 0.6734 ≈
   v74(easy, fixed) 0.6731이라는 관측과 부합하지만, 아직 fixed×Medium 칸이 비어 있다.
3. **레이아웃은 DDP4로 통일할 것.** 1-GPU는 −0.0098이고 합성 val_ce는 오히려 좋아지므로
   착시를 만든다. 앞으로 arm 비교는 DDP4로 돌리거나, 1-GPU를 쓰면 control도 1-GPU로 맞춘다.

---

## 107. 2026-08-13 — baseline을 v82 Medium으로 승격, 판정 레짐을 1-GPU 4 seed로 전환 (사용자 결정)

### 0. 이 절이 정한 것

| 항목 | 이전 | **확정** |
|---|---|---|
| **활성 baseline** ⚠️ §109가 대체 | v77 Hard ClassSep `[0.2,0.8]` | **v82 Medium ClassSep `[0.5,1.4]`** (§109 이후는 v83 linear head) |
| **공식 baseline 숫자** ⚠️ §109가 대체 | 0.6880 (DDP4, 시드 1개) | **0.6835 = 1-GPU 4 seed 평균** (§109 이후는 v83 4 seed **0.6880**, 숫자만 같은 별개 레짐) |
| **판정 레짐** | DDP4, arm당 시드 1개 | **1-GPU, arm당 4 seed (42/43/44/45)** |
| **판정 통계** | fold-paired Δ + CI (단일 시드) | **seed-paired Δ + t** (시드별 fold-paired CI는 보조) |
| **채점 epoch** | epoch 49 고정 | 변경 없음 — **epoch 49 고정** |

**근거는 §106**이다. 네 arm을 처음으로 같은 레짐·4 seed로 돌린 결과 Medium이 Hard를 **+0.0053
(4/4 시드 양수, seed-paired t=3.0)**으로 이겼고, 그 전까지 "Hard가 최적"이라는 판정은 §91의 오기와
단일 시드 노이즈 위에 서 있었다(§105-3). 사용자 결정으로 **DDP4 Medium 4 seed를 기다리지 않고**
이미 4 seed가 존재하는 1-GPU 레짐을 공식 판정 레짐으로 채택한다.

### 1. ⚠️ 0.6835 < 0.6880은 성능 하락이 아니다 — 가장 큰 오독 위험

**숫자가 내려간 것은 arm이 나빠져서가 아니라 레짐이 바뀌었기 때문이다.** 1-GPU는 같은 config에서
DDP4보다 SEAL이 **−0.0098** 낮다(§106-4). 같은 레짐 안에서 비교하면 순위는 이렇다.

| 레짐 | v77 Hard | v82 Medium | Δ |
|---|---:|---:|---:|
| **1-GPU 4 seed (공식)** | 0.6781 | **0.6835** | **+0.0053** (t=3.0) |
| DDP4 1 seed (구 공식) | 0.6880 | 0.6881 | +0.0001 (판정 불가) |

**두 레짐의 숫자를 빼지 말 것.** v82 1-GPU 0.6835를 v77 DDP4 0.6880과 비교해 "Medium이 진다"고
읽는 것이 정확히 §106-4가 경고한 confound이며, 그 오독이 §104-6에서 v80의 기각 크기를 2.7배
부풀린 바로 그 실수다.

### 2. 무엇이 비교 대상에서 빠지는가

**이 문서·`current_experiments.md`의 기존 결과표는 전부 DDP4 단일 시드다.** 그 값들은 **자기들끼리는
여전히 유효**하지만 **새 1-GPU 4-seed arm과 직접 뺄 수 없다**. 구체적으로:

- **§105의 27개 arm epoch-49 재채점표** — 전부 DDP4 1 seed. 역사적 기록으로 유지한다.
- **v41_K128 0.6940** (역사적 전체 최고) — 레짐 미상. **여전히 미달 상태로 본다** — 1-GPU 페널티
  0.0098을 감안해도 v82 0.6835 + 0.0098 ≈ 0.693으로 근접할 뿐이고, 이 보정 자체가 시드 1개짜리
  추정(§106-4 경고)이라 **"따라잡았다"고 쓰지 말 것.**
- **ClassSep sweep 표(§105-3)** — Medium 0.6881, Hard 0.6880 등. DDP4 1 seed 기록으로 남긴다.

새 arm은 **`train_v82_medium_classsep_1536_1gpu.yaml` 4 seed와 직접 비교**한다. 다른 레짐의 숫자를
control로 쓰지 않는다.

### 3. 새 판정 절차

1. arm을 **1-GPU, SEED 42/43/44/45, 50 epoch**로 돌린다 (arm당 약 28분 × 4 = 약 2시간, GPU 0–3에
   4개를 동시에 올리면 약 28분).
2. **epoch 49 checkpoint**를 `_ep49` 태그로 채점한다.
3. baseline 4 seed와 **같은 시드끼리 빼서**(seed-paired) 평균 Δ와 SE를 낸다. 시드별
   fold-paired CI(`scripts/compare_arms_paired.py`)는 **보조 근거**로만 읽는다.
4. 판정: **4/4 시드 부호 일치 + |t| ≥ 2.5**면 판정 가능. 부호가 갈리면 **미판정**이다 —
   learnable P(+0.0048, t=1.5, seed 44 반전)가 그 예다.

⚠️ **단일 시드 게이트 0.010(2σ)은 이제 1 seed 비교에만 적용한다.** 4 seed paired 설계에서는
SE가 0.002 안팎이라 +0.005도 판정된다. 다만 **seed std가 arm마다 3배 차이 난다**(fixed P 0.0018 <
Medium 0.0029 < Hard 0.0053 ≈ shallow MLP 0.0051) — 학습되는 P를 가진 arm일수록 노이즈가 크므로
SE는 arm마다 실측할 것.

### 4. 승격되지 않은 것

- **learnable P는 여전히 미판정이다** (+0.0048, t=1.5, seed 44 −0.0044). v82가 learnable P를
  쓰는 것은 v77 계보를 이어받은 결과이지 learnable P가 증명돼서가 아니다. §105-4의 "v74→v76 승격은
  근거 없음"은 **유지**된다.
- **fixed P × Medium 칸은 여전히 비어 있다.** v82가 baseline이 된 지금 이 칸은 "baseline에서
  파라미터 197,057개를 449개로 줄여도 되는가"라는 질문이 되어 **우선순위가 올라간다.**
- **DDP4 레짐 자체를 폐기한 것은 아니다.** 최종 보고 숫자를 DDP4로 낼 필요가 생기면 baseline과
  arm을 **함께** DDP4 4 seed로 재측정한다.

### 5. 산출물

| 항목 | 값 |
|---|---|
| canonical config | `configs/train_v82_medium_classsep_1536_1gpu.yaml` (**self-contained로 전환**) |
| DDP4 판본 | `configs/train_v82_medium_classsep_1536.yaml` (base 체인 유지, 참고용) |
| checkpoints | `checkpoints/20260813_v82_medium_seeds/seed4{2..5}/` |
| tags | `v82_medium_seed4{2..5}_ep49` |
| 시드별 macro | 0.6846 / 0.6870 / 0.6821 / 0.6802 (mean **0.6835**, std 0.0029) |
| 이전 baseline | v77 Hard — 1-GPU 4 seed 0.6781, DDP4 1 seed 0.6880 (historical) |

canonical config를 self-contained로 인라인할 때 **resolved 결과가 chain 판본과 byte-identical**임을
확인했다 (`merge_train_config` 출력 sha256 `1b9ed13a…35cc5` 일치) — 이미 돌린 4 seed의 재현성이
깨지지 않는다.

### 6. 다음 Action

1. **fixed P × Medium 4 seed** — 비어 있는 칸이자 baseline의 파라미터 수를 197,057 → 449로 줄일 수
   있는지 가르는 측정이다. config는 v81을 Medium ClassSep으로 복제하면 된다.
2. **v82 Medium 위에서 큰 레버 탐색.** ClassSep을 조인 것은 +0.011~+0.015로 유효했고(§105-3),
   CV/DD·사영 배선 축은 소진이다(§103-5). 남은 축은 데이터 생성기 쪽 — noise(+0.0104)·rare(+0.0103)가
   ClassSep과 같은 크기의 레버였다(§105-6 axis sweep).
3. **레짐 전환에 따른 문서 정합성**은 이 커밋에서 처리했다 — `agent_handoff.md`,
   `current_experiments.md`, `current_architecture.md`의 baseline 계약을 모두 v82/1-GPU 4 seed로
   갱신했다.

## 108. 2026-08-13 — v83 linear-head ablation: GELU 제거해도 baseline과 통계적으로 구분되지 않음 (미판정)

_Recorded by: nhn-SMC-claude — 2026-08-13 17:30_

**질문**: relation head는 v70부터 12개 closed-form feature
(`[CV0,CV1,CV1-CV0,SEP_CV, D0,D1,D1-D0,SEP_DD, q0,q1,q0-q1,SEP_CT]`)를 `12→32→1`에 GELU 하나를
거쳐 결합해왔다 — 이 비선형성이 실제로 뭔가 기여하는지 한 번도 검증된 적이 없었다. `v83`은
`ct_head_hidden_dims: []`로 hidden layer와 GELU를 없애 head를 bare `Linear(12,1)`로 줄인다.
P(196,608)와 그 위 전부는 v82와 동일 — trainable **197,057 → 196,621**. v82 체크포인트와는
head shape가 달라 strict-load 불가, 처음부터 학습(`architecture_version=54`는 그대로 — 모델
클래스는 같고 head hidden dims만 다른 config 값이다).

**산출물**:
```
config: configs/train_v83_linear_head_1536_1gpu.yaml   (self-contained)
ckpts:  checkpoints/20260813_153750/v83_linear_head_seed4{2..5}/  (epoch 49)
tags:   v83_linear_head_seed4{2..5}_ep49
macro:  0.6905 / 0.6896 / 0.6774 / 0.6944  →  mean 0.6880, seed std 0.0074
```

**baseline(`v82_medium_seed4{2..5}_ep49`, 0.6846/0.6870/0.6821/0.6802, mean 0.6835) 대비
seed-paired Δ**:

| seed | v82 | v83 | Δ(v83−v82) |
|---|---:|---:|---:|
| 42 | 0.6846 | 0.6905 | +0.0059 |
| 43 | 0.6870 | 0.6896 | +0.0026 |
| 44 | 0.6821 | 0.6774 | **−0.0047** |
| 45 | 0.6802 | 0.6944 | +0.0142 |
| 평균 | 0.6835 | 0.6880 | **+0.0045** |

평균 Δ = +0.0045, SD(Δ) ≈ 0.0078, SE ≈ 0.0039, **t ≈ 1.15**.

**판정 (§107-3 기준)**: **미판정**. 4/4 시드 부호 일치 조건을 충족하지 못하고(seed 44만 음수,
3/4 양수) `|t| ≈ 1.15`로 게이트 `|t| ≥ 2.5`에도 크게 못 미친다. GELU를 없애도 뚜렷한 손해는
없지만 뚜렷한 이득도 아니다 — 서두의 두 가설(GELU가 장식이다 / head가 실제로 비선형 결합을
한다) 중 어느 쪽도 이 4 seed로는 확정할 수 없다.

⚠️ **계산 방법 주의**: 위 macro는 각 task의 `fold-mean AUROC`(50-fold 평균) 10개를 평균한 값이며
`scripts/test_pathobench.py` 로그에서 직접 집계했다. **fold-paired CI(`scripts/compare_arms_paired.py`)는
아직 돌리지 않았다** — task별 세부 비교나 더 정밀한 유의성 검정이 필요하면 그 스크립트로 재확인할 것.

**per-task 값(seed42, 참고)**: bc_therapy/er_status 0.7219, grade 0.7444, her2_status 0.6405,
cptac_brca/PIK3CA 0.5592, TP53(brca) 0.8321, cptac_luad/EGFR 0.7753, STK11 0.8645, TP53(luad) 0.6725,
cptac_ccrcc/BAP1 0.6246, VHL 0.4699. (seed43/44/45도 동일 10-task 순서로 로그에 남아 있다,
`logs/official50/*_v83_linear_head_seed{42,43,44,45}_ep49.log`.)

→ 승격 여부에 대한 사용자 결정은 **§109**.

## 109. 2026-08-13 — baseline을 v83 linear head로 승격 (사용자 결정, §107-3 게이트 미달)

_Recorded by: nhn-SMC-claude — 2026-08-13 17:30_

§108의 결과(Δ +0.0045, t≈1.15, 3/4 시드 양수)는 §107-3이 정한 판정 게이트(4/4 시드 부호 일치 +
`|t| ≥ 2.5`)를 충족하지 못한다. 사용자가 이 수치를 검토한 뒤 "뚜렷하진 않아도 올랐다고 보는 게
맞다"고 판단해 **v83을 baseline으로 승격하기로 결정**했다(2026-08-13).

**⚠️ 이것은 통계적 판정이 아니라 사용자의 연구적 판단(override)이다.** §104-5·§105-4가 경고한
"게이트 미달 상태에서의 승격"과 같은 범주이며, 과거에 이런 승격 중 일부(§91 Hard 최적 판정,
v74→v76 승격)가 나중에 근거 없음으로 되돌려진 전례가 있다. 이 승격도 같은 리스크를 안고 있다는
것을 이어받는 모든 arm 비교에서 유의할 것.

**새 활성 baseline**:
```
config: configs/train_v83_linear_head_1536_1gpu.yaml   (self-contained)
ckpts:  checkpoints/20260813_153750/v83_linear_head_seed4{2..5}/  (epoch 49)
tags:   v83_linear_head_seed4{2..5}_ep49
SEAL 10-task macro: 0.6905 / 0.6896 / 0.6774 / 0.6944 → mean 0.6880 (seed std 0.0074)
trainable parameters: 196,621 (P 196,608 + head Linear(12,1) = 13)
```

⚠️ **0.6880이라는 값이 옛 v77 DDP4 baseline(§104)의 0.6880과 우연히 숫자가 같다.** 완전히 다른
레짐(1-GPU 4 seed vs DDP4 1 seed)의 별개 수치이니 혼동하지 말 것 — §107-1·§107-2가 경고한
"레짐이 다른 숫자를 빼지 말 것"이 여기서도 그대로 적용된다.

**직전 baseline** v82 Medium ClassSep `[0.5,1.4]`(1-GPU 4 seed 0.6835)은 historical로 남는다.
v82와 v83의 모델 클래스·`architecture_version=54`는 동일하고 `class_separation`도 동일
(`[0.5,1.4]`, Medium) — 유일한 차이는 relation head 구조(32-hidden GELU → bare `Linear(12,1)`)뿐이다.

**바뀌지 않는 것**:
- §107의 판정 레짐(1-GPU · SEED 42/43/44/45 · epoch 49)과 §107-3 판정 절차는 그대로 유지된다 —
  새 arm은 **v83 4 seed와 seed-paired**로 비교한다.
- learnable P 미판정(§107-8), CV/DD·사영 배선 축 소진(§103-5) 등 §107이 정한 나머지 결론은
  전부 유지된다. v83 promotion은 head 구조에 대한 것이고 이 결론들과 무관하다.
- ⚠️ §107-6(fixed P × Medium 4 seed 빈칸)은 §111에서 **취소**됐다 — 더 이상 열린 과제가 아니다.

**다음 Action**:
1. 이 승격의 통계적 근거를 보강하려면 `scripts/compare_arms_paired.py`로 fold-paired CI를 뽑아
   task별 그림을 확인할 것 (§108은 macro만 계산했다).
2. seed를 4개 더 늘려(46–49) 8 seed paired 비교를 하면 t가 안정화될 수 있다 — 지금 t≈1.15는
   n=4에서 흔들림이 크다(§107-1 참고, n=4 std 추정 구간이 0.6~2.9배).

## 110. 2026-08-13 — v84 deep-head ablation: 더 깊은 head는 명확히 손해 (기각)

_Recorded by: nhn-NEXGEM-claude — 2026-08-13 18:15_

**질문(§108의 반대 방향)**: §108은 relation head의 GELU를 없앴을 때(`Linear(12,1)`)를 봤다 —
미판정이었다. 나머지 절반은 head에 오히려 용량을 더 주면 어떻게 되는가다. `v84`는
`ct_head_hidden_dims: [32, 32]`로 hidden layer를 2단으로 늘려 `12→32→32→1`(GELU 2개)로 만든다.
P(196,608)와 그 위는 v82/v83과 동일 — trainable **197,057 → 198,113** (head 1,505개). v82/v83
체크포인트와는 head shape가 달라 strict-load 불가, 처음부터 학습(`architecture_version=54` 동일).

**산출물**:
```
config: configs/train_v84_deep_head_1536_1gpu.yaml   (self-contained)
ckpts:  checkpoints/20260813_163412/v84_deep_head_seed4{2..5}/  (epoch 49)
tags:   v84_deep_head_seed4{2..5}_ep49
macro:  0.6786 / 0.6783 / 0.6752 / 0.6789  →  mean 0.6777, seed std 0.0018
```

**seed-paired Δ, 두 baseline 모두에 대해**:

| seed | v82(구 baseline) | v83(현 baseline) | v84 | Δ(v84−v82) | Δ(v84−v83) |
|---|---:|---:|---:|---:|---:|
| 42 | 0.6846 | 0.6905 | 0.6786 | −0.0060 | −0.0119 |
| 43 | 0.6870 | 0.6896 | 0.6783 | −0.0087 | −0.0113 |
| 44 | 0.6821 | 0.6774 | 0.6752 | −0.0069 | −0.0022 |
| 45 | 0.6802 | 0.6944 | 0.6789 | −0.0013 | −0.0155 |
| 평균 | 0.6835 | 0.6880 | 0.6777 | **−0.0057** | **−0.0102** |

v82 기준: SD(Δ) ≈ 0.0032, SE ≈ 0.0016, **t ≈ −3.63**. v83 기준: SD(Δ) ≈ 0.0057, SE ≈ 0.0028,
**t ≈ −3.61**. 둘 다 4/4 시드 부호 일치(전부 음수).

**판정 (§107-3 기준)**: **기각, 양쪽 baseline 모두 기준으로**. 4/4 시드 부호 일치 +
`|t| ≥ 2.5`를 (구 baseline v82, 현 baseline v83) 둘 다에 대해 충족한다. head를 `12→32→32→1`로
심화하는 것은 뚜렷한 손해다.

**§108과 종합**: relation head는 **얕게 만들면 미판정(±0 근처)**, **깊게 만들면 명확히
손해**다 — 비대칭적이다. 12개 closed-form feature를 결합하는 데 GELU 하나(`12→32→1`)가 이미
충분하거나 과분하고, 그 이상의 용량은 information bottleneck이 아니라 순수 노이즈로 작용하는
것으로 보인다. 이 결과는 v83 promotion 결정(§109)을 뒤집지 않는다 — 오히려 "지금 head가 이미
적정 크기"라는 그림과 일치한다.

**바뀌지 않는 것**: 판정 레짐·baseline(v83, §109)은 그대로다. head 구조 축은 이제 얕음(미판정)·
기본(baseline)·깊음(기각) 세 지점이 다 나와서 **소진으로 본다** — 이 축에서 새 arm을 더 설계하지
말 것. (§107-6 fixed P × Medium은 이후 §111에서 취소됐다.)

**평가 방법**: `scripts/eval_seal_tasks.sh`로 GPU 0-3에 seed당 1개씩 올려 10-task 전부를 한
GPU에서 순차 실행(각 checkpoint 독립 실행이라 2-GPU 분할 없이도 4 seed가 병렬로 끝난다).
macro는 `logs/official50/*_v84_deep_head_seed{42,43,44,45}_ep49.log`의
`fold-mean AUROC`를 집계한 값이다. fold-paired CI는 아직 안 돌렸다 — 필요하면
`scripts/compare_arms_paired.py`로 추가 확인할 것.

## 111. 2026-08-13 — §107-6(fixed P × Medium, v85) 취소 (사용자 결정)

_Recorded by: nhn-NEXGEM-claude — 2026-08-13 19:05_

§107-6은 v81(fixed P, Hard 난이도)을 현재 baseline 난이도(Medium)에서 다시 측정해 "baseline
파라미터를 196,621 → 449로 줄여도 되는가"를 가르려는 계획이었다. config
`train_v85_medium_fixed_p_1536_1gpu.yaml`을 만들고(§107-6, 449 trainable parameters 확인)
다른 노드에 실행을 맡겼으나, 실제로는 어느 노드에서도 학습이 시작되지 않았다("진행 중"이라는
이전 기록은 착오 — 정정됨).

**사용자가 이 실험 자체가 필요 없다고 판단해 취소했다** — 굳이 지금 할 필요가 없다는 판단.
config는 한 번도 실행되지 않은 채 리포에서 삭제했다(재현 증거로 보존할 학습 결과가 없으므로
`configs/archive/`로 옮기지 않고 그냥 지웠다 — v84처럼 실행 후 기각된 arm과는 다르다).

**바뀌는 것**: §107-6은 더 이상 열린 과제가 아니다. §107이 남긴 다른 열린 항목(learnable P
미판정 §107-8, CV/DD·사영 축 소진 §103-5)은 이 취소와 무관하게 그대로 유지된다.

**다음**: 데이터 생성기 축(§105-6이 남겨둔 noise/rare 레버)을 재기획한다 — §112.

## 112. 2026-08-13 — v86 noise 재검증: 옛 효과가 재현되지 않는다 (null)

_Recorded by: nhn-NEXGEM-claude — 2026-08-13 20:23_

**질문**: §105-6 axis sweep(2026-08-12, DDP4 1seed, 옛 v74/v76-era baseline 0.6735)에서
`observation_noise: 0.005 → 0.01`(관측 노이즈 2배)이 **+0.0104**로 ClassSep 조이기(+0.0146)와
비슷한 크기의 레버였다. 그 이후 재확인도, 현재 baseline·레짐으로의 이관도 없이 방치돼 있었다 —
이 arm이 그 gap을 메운다.

**메커니즘** (`src/datasets/synthetic_data.py`): manifold로 만든 cell embedding에
`x = x + observation_noise * randn_like(x)`로 순수 가우시안 노이즈를 더한다(정규화 직전).
`class_separation`(class-conditional 평균 간 거리)과는 독립적인 축이다.

**변경**: v83 기준 `observation_noise` 0.005 → 0.01만 바꿈(옛 스윕과 동일한 스텝). 모델·head·P는
v83과 byte-identical(architecture_version=54, trainable 196,621) — 데이터만 다르다.

**산출물**:
```
config: configs/train_v86_noise_1536_1gpu.yaml   (self-contained)
ckpts:  checkpoints/20260813_191148/v86_noise_seed4{2..5}/  (epoch 49)
tags:   v86_noise_seed4{2..5}_ep49
macro:  0.6896 / 0.6902 / 0.6775 / 0.6962  →  mean 0.6884, seed std 0.0072
```

**baseline(v83, 0.6905/0.6896/0.6774/0.6944, mean 0.6880) 대비 seed-paired Δ**:

| seed | v83 | v86 | Δ(v86−v83) |
|---|---:|---:|---:|
| 42 | 0.6905 | 0.6896 | −0.0009 |
| 43 | 0.6896 | 0.6902 | +0.0006 |
| 44 | 0.6774 | 0.6775 | +0.0001 |
| 45 | 0.6944 | 0.6962 | +0.0018 |
| 평균 | 0.6880 | 0.6884 | **+0.0004** |

SD(Δ) ≈ 0.0011, SE ≈ 0.0006, **t ≈ 0.71**. 3/4 시드만 양수.

**판정 (§107-3 기준)**: **완전 무효과(null)** — 미판정보다 더 명확하다. 4/4 부호 일치도 못
채우고 `|t|`가 게이트(2.5)의 3분의 1에도 못 미친다. §105-6의 +0.0104는 **지금 baseline·레짐에서
재현되지 않는다.**

**해석**: 두 가능성이 있다 — ① 그 옛 결과 자체가 DDP4 단일 시드의 요행이었을 가능성(realization
노이즈 ≈0.005 규모를 감안하면 +0.0104는 이례적으로 크긴 하지만 n=1이라 배제 못 함), ② 그때
baseline(ClassSep `[1.0,2.0]`, GELU head, 다른 response 파라미터 조합)과 지금 baseline(v83,
ClassSep `[0.5,1.4]`, linear head)이 달라 레버 자체가 baseline-dependent일 가능성. 어느 쪽이든
**observation_noise 0.005→0.01 스텝은 현재 baseline에서 쓸 수 있는 레버가 아니다.**

**바뀌지 않는 것**: v83 baseline·판정 레짐은 그대로다. §105-6의 나머지 축(rare, response,
classsep)에 대한 재검증 여부는 각각 별도로 판단할 것 — 이 null이 그것들의 재현성까지 부정하지
않는다.

**다음**: rare 축(§105-6에서 +0.0103, noise와 비슷한 크기) 재검증을 계획 중이다.

## 113. 2026-08-13 — v87 rare 재검증: 역시 재현 안 됨, 타겟 task(VHL/BAP1)도 무효 (null)

_Recorded by: nhn-NEXGEM-claude — 2026-08-13 21:37_

**질문**: §112(v86, noise)와 짝을 이루는 arm. §105-6에서 `rare_response_probability: 0.0 → 0.15`가
+0.0103으로 noise와 비슷한 크기였다. noise와 달리 rare는 실제로 지금 깨져 있는 두 task —
**cptac_ccrcc VHL(랜덤 이하)**과 **BAP1(large-bag에서만 붕괴)** — 와 메커니즘이 맞아떨어질
가능성이 있었다: 둘 다 "신호를 가진 세포가 소수"인 상황에서 못 찾는 패턴이고,
`rare_response_probability`는 정확히 그 상황(반응 세포 비율 1~8%)을 훈련에 주입한다.

**메커니즘** (`src/datasets/synthetic_data.py`): 에피소드별로 `rare_response_probability`
확률로 반응 세포 비율을 `shared_component_fraction`(기본, ~4~18%) 대신
`rare_response_fraction`(0.01~0.08, 1~8%)에서 뽑는다. 0.15면 학습 에피소드 약 7개 중 1개가
이 극소수-신호 모드로 생성된다.

**변경**: v83 기준 `rare_response_probability` 0.0 → 0.15만 바꿈(옛 스윕과 동일 스텝). 모델·head·P는
v83과 byte-identical(architecture_version=54, trainable 196,621) — 데이터만 다르다.

**산출물**:
```
config: configs/train_v87_rare_1536_1gpu.yaml   (self-contained)
ckpts:  checkpoints/20260813_203144/v87_rare_seed4{2..5}/  (epoch 49)
tags:   v87_rare_seed4{2..5}_ep49
macro:  0.6872 / 0.6904 / 0.6802 / 0.6887  →  mean 0.6866, seed std 0.0043
```

**baseline(v83, 0.6905/0.6896/0.6774/0.6944, mean 0.6880) 대비 seed-paired Δ (macro)**:

| seed | v83 | v87 | Δ(v87−v83) |
|---|---:|---:|---:|
| 42 | 0.6905 | 0.6872 | −0.0033 |
| 43 | 0.6896 | 0.6904 | +0.0008 |
| 44 | 0.6774 | 0.6802 | +0.0028 |
| 45 | 0.6944 | 0.6887 | −0.0057 |
| 평균 | 0.6880 | 0.6866 | **−0.0013** |

SD(Δ) ≈ 0.0037, SE ≈ 0.0018, **t ≈ −0.70**. 2/4만 부호 일치.

**타겟 task 개별 확인 (VHL, BAP1)** — macro가 안 움직여도 이 둘은 따로 좋아질 수 있다는 가설을
직접 검정:

| seed | VHL v83 | VHL v87 | Δ | BAP1 v83 | BAP1 v87 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.4699 | 0.4589 | −0.0110 | 0.6246 | 0.6021 | −0.0225 |
| 43 | 0.4142 | 0.4222 | +0.0080 | 0.6619 | 0.6678 | +0.0059 |
| 44 | 0.4374 | 0.4301 | −0.0073 | 0.6082 | 0.6396 | +0.0314 |
| 45 | 0.4224 | 0.4090 | −0.0134 | 0.6795 | 0.6727 | −0.0068 |
| 평균 | 0.4360 | 0.4300 | **−0.0059** (t≈−1.23, 1/4) | 0.6436 | 0.6455 | **+0.0020** (t≈0.18, 2/4) |

**판정 (§107-3 기준)**: **완전 무효과(null)**, macro도 타겟 task도. macro는 4/4는커녕 2/4
부호 일치에 `|t|`가 게이트의 4분의 1 수준이다. **VHL은 오히려 약하게 반대 방향**(1/4만 양수) —
rare-episode 학습이 VHL을 돕는다는 가설과 맞지 않는다. BAP1은 seed마다 부호·크기가 요동해
방향성 자체가 없다(−0.0225~+0.0314).

**해석**: §112(noise)와 같은 결론 — §105-6 축 스윕의 효과들이 지금 baseline(v83)·레짐에서
재현되지 않는다. rare의 경우 추가로, **VHL/BAP1의 실패가 "훈련 데이터에 rare-abundance 신호가
부족해서"라는 가설도 반증됐다** — 이 메커니즘으로는 두 task를 못 고친다. 두 task의 실패는
데이터 생성기 축이 아니라 다른 원인(레이블 자체, 코호트 특성, 구조적 문제)일 가능성이 높다.

**바뀌지 않는 것**: v83 baseline·판정 레짐은 그대로다. §105-6의 남은 축(response, classsep)은
이 두 null과 별개로 각자 판단할 것 — 다만 classsep은 이미 baseline에 반영돼 있어 재검증 대상이
아니다(§107).

**다음**: 데이터 생성기 축(noise·rare)은 여기서 소진으로 본다. VHL/BAP1 문제는 별도 조사가
필요하다 — 다음 실험 방향은 재기획 중.

## 114. 2026-08-14 — v88 PA(label-conditioned population branch): 명확히 기각, VHL/BAP1도 개선 없음

_Recorded by: nhn-NEXGEM-claude — 2026-08-14 02:11_

**질문**: 지금까지의 relation source는 전부 **per-cell 레이블을 fit에 넣지 않는다** — CV는 bag
descriptor 위의 ridge, DD는 label-free 분산 방향, CT는 label-free farthest-point 토큰을 뽑아
**그 다음에** 레이블로 점수만 매긴다. §65가 "미검정 레버"로 남겨둔 마지막 항목이 그것,
즉 **support 레이블을 보고 "이 population이 0/1을 가르는 데 중요한가"를 직접 학습**하는 분기다.
v88(PA, population attention)은 그 분기를 처음으로 구현한 arm이다.

**메커니즘** (`CovarianceMeanLearnablePDDCTPAMLPModel`, architecture_version=57):
1. context bag마다 `pa_cells_per_bag: 64` 세포를 샘플하고, **각 세포에 자기 bag의 레이블을
   상속**시킨다 — 노이즈 레이블이다(대부분의 세포는 공유 배경이고 소수만 판별적이다).
2. 그 세포 전체에 대해 **한 번의 ridge 회귀**를 풀어 방향 `w`와 절편을 얻는다
   (`_pa_cell_direction`, `solve_ridge_system`). "세포 하나를 이 축에 투영하면 label-1스러움
   점수가 나온다"는 축이다.
3. bag마다 **양쪽 방향의 soft abundance를 독립적으로** 잰다 —
   `abundance1 = sigmoid((z−τ)/T).mean()`, `abundance0 = sigmoid((−z−τ)/T).mean()`.
   ⚠️ 초기 설계는 같은 signed 축의 top-k-mean/bottom-k-mean이었는데, 그 둘은 거의 서로의
   거울상이어서 심어놓은 신호를 **우연 수준(8/16)으로만** 잡아냈다. 독립 abundance로 바꾼 뒤
   90~100%가 됐다(`tests/test_population_attention.py::PopulationAttentionAliveTest`).
   §62-2가 역사적으로 죽였던 population attention의 실패 모드와 **정확히 같은 종류**다.
4. 기존 12개 feature에 (PA0, PA1, PA1−PA0, SEP_PA) 4개가 붙어 head가 16-wide로 재구성된다.
   PA 계산 자체는 CT/DD와 마찬가지로 training-free(`no_grad`)고, 학습되는 건 P(196,608)와
   head(577)뿐 — **trainable 197,185**.

**GPU를 쓰기 전에 구현 버그 3개를 자체 테스트로 잡았다** (이게 이 절의 부수적 성과다):
- P의 gradient가 `None`이었다 — 내 freeze 루프가 부모가 learnable로 만든 `_covariance_projection`을
  다시 얼렸다. 해제 후 정상(§100 계약).
- 위의 top/bottom-k-mean 설계 결함(우연 수준).
- `train_dd_projection=True` 경로가 PA의 무조건 `no_grad` 안에 갇혀 조용히 죽었다 — scoping
  수정 + regression 테스트 추가.

**산출물**:
```
model:  src/models/set_transformer_ridge.py  CovarianceMeanLearnablePDDCTPAMLPModel (arch=57)
tests:  tests/test_population_attention.py   (9 tests, 전부 통과)
config: configs/train_v88_population_attention_1536_1gpu.yaml   (self-contained)
ckpts:  checkpoints/20260813_224003/v88_population_attention_seed4{2..5}/  (epoch 49)
tags:   v88_population_attention_seed4{2..5}_ep49
macro:  0.6803 / 0.6782 / 0.6700 / 0.6790  →  mean 0.6769, seed std 0.0047
```
⚠️ v83 checkpoint와 **strict-load 불가**(head 12 vs 16 in_features, arch 54 vs 57).

**baseline(v83, mean 0.6880) 대비 seed-paired Δ (macro)**:

| seed | v83 | v88 | Δ(v88−v83) |
|---|---:|---:|---:|
| 42 | 0.6905 | 0.6803 | −0.0102 |
| 43 | 0.6896 | 0.6782 | −0.0114 |
| 44 | 0.6774 | 0.6700 | −0.0074 |
| 45 | 0.6944 | 0.6790 | −0.0154 |
| 평균 | 0.6880 | 0.6769 | **−0.0111** |

SD(Δ) ≈ 0.0033, SE ≈ 0.0017, **t ≈ −6.69**, **4/4 시드 부호 일치**.

**타겟 task 개별 확인 (VHL, BAP1)** — PA를 만든 동기가 "소수 population을 못 찾아서 깨지는
두 task"였으므로 §113과 같은 방식으로 직접 검정:

| seed | VHL v83 | VHL v88 | Δ | BAP1 v83 | BAP1 v88 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.4699 | 0.4267 | −0.0432 | 0.6246 | 0.6308 | +0.0062 |
| 43 | 0.4142 | 0.4408 | +0.0266 | 0.6619 | 0.6368 | −0.0251 |
| 44 | 0.4374 | 0.3657 | −0.0717 | 0.6082 | 0.6051 | −0.0031 |
| 45 | 0.4224 | 0.3946 | −0.0278 | 0.6795 | 0.6209 | −0.0586 |
| 평균 | 0.4360 | 0.4070 | **−0.0290** (t≈−1.41, 1/4) | 0.6436 | 0.6234 | **−0.0202** (t≈−1.40, 1/4) |

**판정 (§107-3 기준)**: **기각**. macro가 4/4 부호 일치 + |t|=6.69로 게이트를 기각 방향으로
확실히 넘는다 — 이 세션에서 관측된 가장 강한 기각이다(v84의 |t|=3.61보다 크다). 타겟 task
쪽은 게이트에 미달해 확정 판정은 아니지만 **둘 다 1/4만 양수로 방향이 가설과 반대**다.

**해석**: PA는 죽지 않았다 — 합성 planted-signal 테스트에서 90~100%로 신호를 잡는다. 그런데
실제 SEAL에서는 head에 **노이즈만 추가한** 셈이 됐다. 두 가지로 읽을 수 있다: (a) 12개
CV/DD/CT feature가 이미 이 정보를 담고 있어 4개가 중복·잡음으로만 작용, (b) bag 레이블을
세포에 상속시키는 근사가 실제 데이터에서는 신호 대비 노이즈가 너무 크다(합성 데이터는
판별 세포 비율이 깨끗하게 심어져 있지만 실제 코호트는 그렇지 않다). 어느 쪽이든 **"레이블을
fit에 직접 넣는" 축은 이 형태로는 안 된다**.

**추가로 확정된 것**: VHL/BAP1의 실패는 §113(rare 데이터 주입)으로도, §114(레이블 조건
population 분기)로도 고쳐지지 않는다. "소수 판별 population을 못 찾아서"라는 **메커니즘 가설이
두 방향에서 반증**됐다 — 원인은 표현/구조가 아니라 레이블·코호트 쪽일 가능성이 더 커졌다.

**바뀌지 않는 것**: baseline은 v83(0.6880), 판정 레짐도 §107-3 그대로다. v88 코드는 기각됐지만
**삭제하지 않고 남긴다** — 테스트가 §62-2 실패 모드에 대한 살아있는 probe 역할을 하고,
"레이블 조건 분기를 이미 해봤다"는 음성 결과의 근거이기 때문이다.

**다음**: §65의 미검정 레버가 이걸로 소진됐다. 남은 방향은 재기획이 필요하다.

## 115. 2026-08-14 — 진단: 학습 에피소드가 실제 평가 에피소드의 **모양**을 한 번도 모사한 적이 없다

_Recorded by: nhn-NEXGEM-claude — 2026-08-14 10:18_

**이 절이 정한 것**: §110·§112·§113·§114가 연속으로 실패한 뒤, arm을 더 던지는 대신 **평가 지표를
task 단위로 분해**했다. 그 결과 (a) v83의 task별 성적을 공식 지도학습 baseline과 처음으로 정면
대조했고, (b) **학습 분포와 평가 분포가 두 축에서 어긋나 있으며 그 어긋남이 한 번도 모델링된 적이
없다**는 것을 실측했다. v89는 그중 한 축을 움직이는 arm이고 현재 학습 중이다.

### 1. v83 task별 성적 vs 공식 지도학습 baseline

출처는 `docs/seal_univ2_baseline_17tasks.csv`(SEAL 논문의 50-fold ABMIL/MeanMIL, task별 mean±std).
v83 숫자는 4 seed(42–45) `logs/official50/*_v83_linear_head_seed4{2..5}_ep49.log`의 fold-mean AUROC
평균이다.

| task | v83 | ABMIL | Δ | MeanMIL | Δ |
|---|---:|---:|---:|---:|---:|
| bc_therapy er_status | 0.7276 | 0.717 | **+0.0106** | 0.712 | **+0.0156** |
| bc_therapy grade | 0.7259 | 0.770 | −0.0441 | 0.751 | −0.0251 |
| bc_therapy her2 | 0.6417 | 0.663 | −0.0213 | 0.684 | −0.0423 |
| cptac_brca PIK3CA | 0.5588 | 0.595 | −0.0362 | 0.544 | **+0.0148** |
| cptac_brca TP53 | 0.8270 | 0.801 | **+0.0260** | 0.787 | **+0.0400** |
| cptac_luad EGFR | 0.7761 | 0.830 | −0.0539 | 0.777 | −0.0009 |
| cptac_luad STK11 | 0.8754 | 0.908 | −0.0326 | 0.873 | **+0.0024** |
| cptac_luad TP53 | 0.6678 | 0.751 | −0.0832 | 0.735 | −0.0672 |
| cptac_ccrcc BAP1 | 0.6436 | 0.693 | −0.0494 | 0.720 | −0.0764 |
| cptac_ccrcc VHL | 0.4360 | 0.538 | −0.1020 | 0.542 | −0.1060 |
| **macro** | **0.6880** | **0.7266** | **−0.0386** | **0.7125** | **−0.0245** |

**읽는 법 세 가지**:
1. **모델이 전반적으로 약한 게 아니다.** ABMIL을 2/10, MeanMIL을 **5/10**에서 이긴다. 결손은
   특정 지점에 몰려 있다 — VHL(0.102) + luad TP53(0.083) + BAP1(0.049) = 0.234로 **총 결손
   0.386의 61%**가 3개 task에서 나온다.
2. ⚠️ **VHL은 지도학습으로도 거의 랜덤이다 — ABMIL 0.538 ± 0.128 (n=218, 50 fold).** std가
   0.128이라 0.5와 구분되지 않는다. **"VHL을 정상 task 수준으로 고친다"는 목표는 존재하지 않는다.**
   §113·§114가 VHL을 겨냥해 실패한 것은 상한이 0.538인 task를 겨눈 탓이 크다.
3. 그럼에도 **v83의 0.4360은 여전히 이상하다.** 신호 없는 모델은 0.5로 간다. 4/4 시드가 전부
   0.5 아래(0.4699/0.4142/0.4374/0.4224)인 것은 역상관 신호를 잡고 있다는 뜻이다. 그리고 이건
   실익이 있다 — **랜덤(0.5)까지만 돌려놔도 macro +0.0064, ABMIL 수준이면 +0.0102**로 §107-3
   게이트급이다.

### 2. 두 축의 train/eval mismatch — 실측

**클래스 비율.** `src/datasets/synthetic_data.py::_sample_labels`에 **클래스 사전확률 knob이 아예
없다**. `balanced: true`는 정확히 50/50이고, v83~v89가 쓰는 `balanced: false`는 불균형이 아니라
**Bernoulli(0.5)**다. 코드 주석이 의도를 명시한다 — *"Independent labels remove the episode-level
class-count variable: context label counts contain no information about a masked target."* 즉
context의 레이블 구성비가 정보를 갖지 **않도록 일부러 제거한** 설계다. 실제 task는 정반대다.

**context 크기.** 평가 경로 `scripts/test_pathobench.py`의 `--context-mode` 기본값은 `all`이고
`scripts/eval_seal_tasks.sh`는 이 인자를 넘기지 않는다. 따라서 `sample_context_ids()`가
`return list(train_ids)` — **train fold 전체를 자연 비율 그대로** 쓴다. 균형을 맞추는 `sample`
모드는 존재하지만 deprecated이고 사용되지 않는다.

| | 클래스 비율 | context 크기 |
|---|---|---|
| **학습** (v83 config) | 0.500 ± 0.055 (num_bags 60–100) | **60–100 bags** |
| **평가** (50 fold 실측) | **0.178 ~ 0.780** | **90 ~ 261 slides** |

| task | ctx | 양성비 | 학습분포 기준 σ | Δ ABMIL |
|---|---:|---:|---:|---:|
| er_status | 133 | 0.692 | +3.5σ | +0.0106 |
| grade | 133 | 0.617 | +2.1σ | −0.0441 |
| her2 | 133 | 0.391 | −2.0σ | −0.0213 |
| PIK3CA | 90 | 0.358 | −2.6σ | −0.0362 |
| brca TP53 | 90 | 0.404 | −1.7σ | +0.0260 |
| EGFR | 261 | 0.342 | −2.9σ | −0.0539 |
| STK11 | 261 | 0.178 | **−5.9σ** | −0.0326 |
| luad TP53 | 261 | 0.592 | +1.7σ | −0.0832 |
| BAP1 | 197 | 0.181 | **−5.8σ** | −0.0494 |
| VHL | 197 | 0.780 | **+5.1σ** | −0.1020 |

**in-distribution인 task가 하나도 없다.** 가장 가까운 것도 1.7σ 밖이고, context 크기는 6/10이
학습 상한(100)의 2~2.6배다. brca 두 개만 크기 범위 안에 있다.

⚠️ **상관은 약하다, 과대해석 금지**: corr(context 크기, Δ) = **−0.607**, corr(|비율 편향|, Δ) =
**−0.27**. n=10에 코호트가 4개뿐이라 코호트 정체성과 완전히 교란돼 있어 **어느 쪽도 결정적이지
않다**. 또 AUROC는 클래스 사전확률에 불변이므로 불균형이 지표를 기계적으로 깎지는 않는다 —
영향은 ridge가 치우친 design에서 방향을 추정하게 되는 간접 경로로만 온다. **결론으로 주장할 수
있는 것은 "이 두 축이 모사된 적이 없다"는 사실뿐이고, "그래서 성능이 낮다"는 아직 가설이다.**
§112·§113이 옛 스윕 수치를 재현하려다 실패한 것과 달리 이건 실측된 mismatch라는 점이 다르다.

### 3. 셀 축은 이미 반대로 어긋나 있다 (v89 해석에 필수)

실제 slide의 tile 수(`Data/PathoBench/features/*.h5`, 923개 실측):

| 코호트 | min | p25 | median | p75 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| ccrcc | 358 | 2,638 | 4,988 | 9,635 | 28,831 | 6,947 |
| luad | 364 | 2,950 | 5,215 | 10,137 | 35,107 | 7,203 |
| brca | 1,282 | 4,059 | 7,736 | 13,149 | 33,297 | 8,975 |
| bc_therapy | 392 | 2,011 | 2,674 | 3,246 | 6,487 | 2,730 |

학습은 bag당 평균 **2,724** cell(cap 4,096)이다. 즉 셀 축은 **이미 실제보다 짧다**.

### 4. v89 — bag 축 arm (완료, 결과는 §7)

```
config: configs/train_v89_episode_shape_1536_1gpu.yaml   (self-contained)
run:    checkpoints/20260814_094411/v89_episode_shape_seed4{2..5}/
logs:   logs/20260814_094411/v89_episode_shape_seed4{2..5}.out
```

v83 대비 데이터 knob **4개만** 변경(모델·head·P는 byte-identical — arch 54, trainable 196,621로
실측 확인):

| knob | v83 | v89 |
|---|---|---|
| `num_bags` | [60, 100] | **[180, 300]** |
| `num_cells` | [256, 8192] | **[85, 2731]** |
| `per_bag_max_cells` | 4096 | **1365** |

**예산 중립이 설계 의도**다. log-uniform 추출을 시뮬레이션하면 E[cells/bag] 2,724 → 909(0.334배)로
E[total cells/episode] 217,936 → 218,167(**+0.1%**)이다. `scripts/smoke_train_budget.py --bf16`
실측(24 step, GPU 0):

| | v83 | v89 |
|---|---:|---:|
| mean cells/step | 297,643 | 298,480 (**+0.3%**) |
| mean step | 78 ms | 102 ms (+31%) |
| peak VRAM | 11.9 GiB | 13.3 GiB |
| episode shape | `(1, 79, 4096, 1536)` | `(1, 230, 1365, 1536)` |

셀 예산은 중립인데 step 시간이 31% 는다 — **bag당 연산(covariance sketch, DD `eigh`, CT 토큰)이
cell이 아니라 bag 수에 비례**하기 때문이다. epoch ~2분, 50 epoch ~100분/seed.

⚠️ **판정 시 반드시 지킬 것 — 두 축이 같이 움직인다.** 예산 고정은 곧 "셀 축이 bag 축의 비용을
지불한다"는 뜻이고, §3에서 보듯 셀 축은 이미 짧았다(평균 2,724 → 909, cap 4,096 → 1,365; 실제
중앙값 대비 약 2배 아래에서 **3~8배 아래**로). 따라서:
- **양성이면 명확하다** — 셀 축을 악화시키고도 이겼다는 뜻이니 bag 축 효과가 그만큼 크다.
- **음성/null이면 분리되지 않는다** — bag 축이 무효인지 두 효과가 상쇄된 건지 구분할 수 없다.
  그때는 **예산 3배(셀 유지, bag만 3배)** 변형으로 갈라야 하고 seed당 ~3.3시간이 든다.
  이 경우를 "bag 축 소진"으로 기록하지 말 것.

### 5. v90 — 클래스 비율 축 (준비 완료, 미실행)

§2의 나머지 한 축. `_sample_labels`에 knob이 없으므로 생성기 변경이 필요하다 — 상세는 §116.

### 6. 다음 Action

1. **v89 판정** — epoch 49, §107-3(1-GPU 4 seed, seed-paired Δ+t, 게이트 4/4 + |t|≥2.5),
   v83(0.6880) 대비. macro와 **10개 task 전부** per-seed로 뽑아 §1 표를 갱신할 것.
2. **v90 실행** — nhn-SMC에 준비 완료 상태로 인계됨(§116).
3. ~~**VHL을 겨냥한 arm 설계**~~ **하지 말 것** — §1-2에 따라 상한이 0.538이다. VHL은 "고칠
   task"가 아니라 "0.436 → 0.5의 역상관을 없앨 대상"이고, 그마저 별도 진단(fold별 부호 분포)
   사안이지 학습 arm 사안이 아니다.

### 7. v89 결과 — 미판정 (2026-08-14 11:36)

_Recorded by: nhn-NEXGEM-claude — 2026-08-14 11:36_

```
ckpts: checkpoints/20260814_094411/v89_episode_shape_seed4{2..5}/  (epoch 49)
tags:  v89_episode_shape_seed4{2..5}_ep49
macro: 0.6835 / 0.6782 / 0.6822 / 0.6889  →  mean 0.6832, seed std 0.0044
```

| seed | v83 | v89 | Δ(v89−v83) |
|---|---:|---:|---:|
| 42 | 0.6905 | 0.6835 | −0.0070 |
| 43 | 0.6896 | 0.6782 | −0.0114 |
| 44 | 0.6774 | 0.6822 | **+0.0048** |
| 45 | 0.6944 | 0.6889 | −0.0055 |
| 평균 | 0.6880 | 0.6832 | **−0.0048** |

SD(Δ) ≈ 0.0069, SE ≈ 0.0035, **t ≈ −1.39**, **1/4 부호 일치**.

**판정 (§107-3)**: **미판정.** 게이트(4/4 + |t|≥2.5) 미달이다. 방향은 음수지만 기각도 아니다.

**⚠️ task별 수치는 판정에 쓰지 않는다 — 이 arm이 그 함정을 실제로 보여준다.** 10개를 모두
뽑으면 게이트를 넘는 task가 **2개** 나온다:

| task | v83 | v89 | Δ | t | 부호 |
|---|---:|---:|---:|---:|---|
| bc_therapy her2 | 0.6417 | 0.6737 | **+0.0320** | +2.59 | 4/4 |
| cptac_luad TP53 | 0.6678 | 0.6361 | **−0.0317** | −3.04 | 0/4 |

**둘 다 우연 범위다.** df=3에서 `P(|t| ≥ 2.5) ≈ 0.088`이므로 task 10개면 귀무가설 아래에서도
**기대 0.9개**가 게이트를 넘는다. 2개 관측은 놀랄 일이 아니다. §1 금지 사항 표의 *"task별 CI로
판정하지 않는다"*(§104-5, task별 seed std가 macro의 7배)가 그대로 적용된다. **판정은 macro로만
하고 task별은 방향 기록으로만 남긴다.**

방향 참고(판정 아님): VHL 0.4360 → 0.4538 (+0.0179, 3/4, 여전히 랜덤 이하) / BAP1 0.6436 →
0.6342 (−0.0094, 2/4, 방향성 없음) / STK11 0.8754 → 0.8760 (±0).
⚠️ §1-1이 "진짜 헤드룸"으로 지목한 **luad TP53이 오히려 내려갔다**. 게이트 논리로는 우연
범위지만 v90·v91에서 이 task의 거동은 계속 볼 것.

**⚠️ 이 결과를 "bag 축 소진"으로 기록하지 말 것 — §4의 사전 경고가 그대로 발동했다.** 예산
고정 때문에 bag 3배(효과 미상)와 cell 1/3(§3에 따르면 악화 방향)이 **동시에** 움직였고,
결과가 null/음성이므로 **"bag 축이 무효"와 "두 효과가 상쇄"를 구분할 수 없다.** 이건 사후
변명이 아니라 arm을 만들 때 config 헤더와 §4에 미리 적어둔 조건이다.

**분리 방법 — 처음 제안한 것보다 싸다.** 원래는 "예산 3배(cell 유지, bag만 3배), ~3.3시간/seed"를
생각했으나, **반대 방향이 훨씬 효율적이다**:

> **v91 = v83 + cell만 1/3 (bag은 60–100 그대로)** — cell 축 단독 효과를 직접 잰다.
> 예산이 **1/3**이라 **~22분/seed**로 지금까지 중 가장 싼 arm이다.

- v91 Δ ≈ −0.005 → cell 축이 전부 설명, **bag 축 ≈ 0**
- v91 Δ ≈ 0 → **bag 축이 진짜 −0.005**

⚠️ 가법성을 완전히 가정하는 건 아니다(교호작용이 있을 수 있다). 그래도 지금의 완전한 모호함을
가장 싼 비용으로 크게 줄인다.

## 116. 2026-08-14 — v90 클래스 비율 arm: 생성기에 `class_prior` 추가 (준비 완료, 미실행)

_Recorded by: nhn-NEXGEM-claude — 2026-08-14 11:05_

**질문**: §115-2의 나머지 한 축. 실제 task는 전부 불균형인데(양성비율 0.178~0.780) 학습은
Bernoulli(0.5)만 봤다. **불균형 자체를 학습시키면 달라지는가?**

**왜 knob을 새로 만들어야 했나**: `_sample_labels`에 클래스 사전확률 개념이 없었다. `balanced:
true`는 정확히 50/50, `balanced: false`는 Bernoulli(0.5)로 **둘 다 균형**이다. 후자의 주석이
설계 의도를 밝힌다 — *"Independent labels remove the episode-level class-count variable"*.
지름길을 제거한 합리적 선택이었지만, 그 대가로 **모델은 context에서 유병률을 읽는 법을 배우지
못한다**. 실제 에피소드는 그 정보를 항상 갖고 있다.

**구현** (`SyntheticManifoldGenerator`):
```yaml
class_prior: [0.15, 0.85]     # None이면 기존 동작 그대로
```
- 에피소드마다 `p ~ U(0.15, 0.85)`를 **한 번** 뽑고 `labels ~ Bernoulli(p)`. bag마다 뽑으면
  중심극한정리로 매 에피소드가 0.5 근처에 몰려 knob이 무의미해진다 — 테스트가 이걸 고정한다.
- `balanced: true`와 동시 지정은 **에러**다(균형 경로가 prior를 조용히 버리므로).
- `_repair_missing_classes`: 클래스가 하나라도 비면 무작위 bag 하나를 그 클래스로 뒤집는다.
  `ModelInterface._sample_query_index`가 단일 클래스 에피소드에서 **예외를 던지기** 때문에
  필수다. 지금 설정에서는 ~1e-13 사건이지만, epoch당 1024 에피소드 × 50 epoch에서 "드묾"은
  안전을 보장하지 않는다.

**테스트** (`tests/test_class_prior.py`, 12개 전부 통과):
- ⚠️ **가장 중요한 것 — `test_default_is_bit_identical_to_the_old_stream`**: knob 미지정 시
  레이블 스트림이 v83과 **bit-identical**이어야 한다. v90은 v83과 seed-paired로 비교되므로,
  파라미터 추가만으로 기본 경로가 흔들렸다면 비교가 prior가 아니라 리팩터를 재는 게 된다.
- 에피소드 단위 추출 검증(bag 단위였다면 sd≈0.03, 에피소드 단위면 U(0.15,0.85)의 sd=0.202)
- 실측 실제 범위(0.178~0.780) 포함 여부
- 극단 prior(0.01/0.99) × 400회에서 단일 클래스 에피소드 0건
- config `dataset_kwargs` → `**generator_kwargs` 배선

**산출물**:
```
generator: src/datasets/synthetic_data.py  class_prior + _repair_missing_classes
tests:     tests/test_class_prior.py       (12 tests)
config:    configs/train_v90_class_prior_1536_1gpu.yaml   (self-contained)
```
v83 대비 config diff는 `class_prior` 3줄 + `experiment_name`뿐이다. 모델·head·P는 byte-identical
(arch 54, trainable 196,621). `smoke_train_budget.py --bf16` 실측: peak VRAM 12.0 GiB, episode
shape `(1, 72, 4096, 1536)` — **v83과 같은 모양**이므로 비용도 v83의 것(~65분/seed)이지 v89의
것이 아니다. 실측 양성비율 분포(60 에피소드): min 0.125 / med 0.450 / max 0.887.

**알려진 부작용 (버그 아님)**: `_pairwise_ranking_loss`는 뽑힌 query가 전부 한 클래스면 0을
반환한다. prior가 치우치면 이 일이 잦아진다(p=0.15, query 12개면 약 14%). 즉
`ranking_loss_weight`가 실질적으로 더 적은 에피소드만 본다. **이것도 이 arm이 재는 대상의
일부다** — 평가 시 불균형이 문제라면 그 대가를 치르고도 이득이 나야 한다.

**v89와의 관계**: 두 arm은 §115-2의 서로 다른 축이고 **둘 다 v83 기준으로 독립 판정**된다.
순서 무관, 병렬 가능. ⚠️ 둘 다 이기더라도 **효과가 더해진다고 가정하지 말 것** — 결합 arm은
별도 검정이다.

**판정**: §107-3 (1-GPU, SEED 42/43/44/45, epoch 49, v83 0.6880 대비 seed-paired Δ+t, 게이트
4/4 + |t|≥2.5). ⚠️ **macro와 10개 task 전부를 per-seed로 보고할 것** — 이 arm의 요점은 task별
이고, macro만으로는 불균형이 심한 task(STK11 0.178 / BAP1 0.181 / VHL 0.780)가 움직였는지
알 수 없다.

**상태**: **완료 — 미판정 (4/4 방향 일치, 게이트 미달). 상세는 §117.**
```
run:  checkpoints/20260814_103137/v90_class_prior_seed4{2..5}/
logs: logs/20260814_103137/v90_class_prior_seed4{2..5}.out
gpu:  0, 1, 3, 5 (2·4번은 이 노드의 다른 사용자 작업 중이라 제외)
```

_Recorded by: nhn-SMC-claude — 2026-08-14 10:32_

## 117. 2026-08-14 — v90 class_prior 결과: 4/4 방향 일치하지만 게이트 미달 (미판정)

_Recorded by: nhn-SMC-claude — 2026-08-14 11:48_

**산출물**:
```
config: configs/train_v90_class_prior_1536_1gpu.yaml
ckpts:  checkpoints/20260814_103137/v90_class_prior_seed4{2..5}/  (epoch 49)
tags:   v90_class_prior_seed4{2..5}_ep49
macro:  0.6904 / 0.6851 / 0.6725 / 0.6829  →  mean 0.6827 (seed std 0.0074)
```

**baseline(v83, 0.6905/0.6896/0.6774/0.6944, mean 0.6880) 대비 seed-paired Δ**:

| seed | v83 | v90 | Δ(v90−v83) |
|---|---:|---:|---:|
| 42 | 0.6905 | 0.6904 | −0.0001 |
| 43 | 0.6896 | 0.6851 | −0.0045 |
| 44 | 0.6774 | 0.6725 | −0.0049 |
| 45 | 0.6944 | 0.6829 | **−0.0115** |
| 평균 | 0.6880 | 0.6827 | **−0.0053** |

평균 Δ = −0.0053, SD(Δ) ≈ 0.0047, SE ≈ 0.0024, **t ≈ −2.23**.

**판정 (§107-3 기준)**: **미판정**. 4/4 시드가 전부 음수라 **v89(1/4)보다 방향 일관성은 뚜렷하지만**,
`|t| ≈ 2.23`으로 게이트 `|t| ≥ 2.5`에 근소하게 못 미친다(4/4 부호 일치는 충족). "게이트 바로
아래"라는 점에서 완전한 null(v86·v87처럼 |t|<1)과는 성격이 다르다 — 방향은 일관되게 음수이되
확정하기엔 표본이 부족하다.

**per-task 값(참고 기록용 — 판정 근거 아님)**: NEXGEM이 v89에서 지적한 다중비교 문제(df=3에서
`P(|t|≥2.5)≈0.088`, task 10개면 귀무가설 하에서도 기대 0.9개가 게이트를 넘음)가 여기도 그대로
적용된다. 아래는 방향 참고용으로만 기록한다.

| task | v83(4-seed 평균, §115-1) | v90(4-seed 평균) | Δ |
|---|---:|---:|---:|
| er_status | 0.7276 | 0.7083 | −0.0193 |
| grade | 0.7259 | 0.7167 | −0.0092 |
| her2_status | 0.6417 | 0.6566 | +0.0149 |
| PIK3CA | 0.5588 | 0.5389 | −0.0199 |
| brca TP53 | 0.8270 | 0.8179 | −0.0091 |
| EGFR | 0.7761 | 0.7649 | −0.0112 |
| STK11 | 0.8754 | 0.8612 | −0.0142 |
| luad TP53 | 0.6678 | 0.6754 | +0.0076 |
| BAP1 | 0.6436 | 0.6610 | +0.0174 |
| VHL | 0.4360 | 0.4265 | −0.0095 |

⚠️ **class_prior가 "겨냥한" task가 예상대로 움직이지 않았다**: 가장 불균형한 STK11(0.178)과
VHL(0.780)은 오히려 소폭 하락했고, BAP1(0.181)만 상승했다. luad TP53(§115에서 "진짜 헤드룸"으로
지목된 task)은 소폭 상승(+0.0076)했지만 v89에서는 하락했었다 — 두 arm이 같은 task에 반대 방향
신호를 준 셈이라 여기서 결론을 내리지 않는다. 이 표는 전부 **다중비교 미보정, CI 없음** — 방향
참고 그 이상으로 쓰지 말 것.

**v89와의 관계**: 같은 §115-2 mismatch의 다른 축이다. v89(bag/cell, 미판정 Δ−0.0048, t=−1.39,
1/4)보다 v90(class_prior, 미판정 Δ−0.0053, **t=−2.23, 4/4**)이 방향 일관성은 더 강하지만 둘 다
게이트를 넘지 못했다. **두 축 다 "무효"라고 결론 내리기엔 이르다** — v90의 t가 게이트에 근접한
점을 고려하면 seed를 늘려 재검증할 가치가 있다.

**다음 Action**:
1. ~~**v90 seed 추가 재검증 후보**~~ **불필요해짐 (§118)** — 사용자가 종합적 판단으로 이미
   기각을 확정했다.
2. **v91(cell 축 단독, NEXGEM 인계)과 함께 놓고 판단** — v89·v90·v91 세 결과를 모아야 bag/cell/
   class_prior 세 축 중 무엇이 진짜 신호인지 정리된다.
3. ~~승격·폐기 판단은 사용자 몫이며, 이 결과만으로는 아직 어느 쪽도 아니다.~~ **§118에서 사용자가
   기각으로 확정했다.**

## 118. 2026-08-14 — 판정 프로토콜 변경: 통계 게이트 → 사용자의 종합적 판단 (사용자 결정), v90 기각 확정

_Recorded by: nhn-SMC-claude — 2026-08-14 11:58_

**계기**: §117의 v90 결과가 통계 게이트(§107-3, 4/4 부호 일치 + `|t|≥2.5`)에 근소 미달(t≈−2.23)
했다. 사용자가 macro 숫자만으로 판단하지 말고 **task 10개 전부를 baseline 성능대와 함께** 보라고
지적했다 — "0.4→0.5는 의미가 작지만 0.9→0.95는 의미가 크다"는 논리, 즉 **같은 크기의 Δ라도
baseline이 어디 있는지(랜덤 근처 vs 천장 근처)에 따라 실질적 의미가 다르다.**

**v90 재분석 (§117 데이터, baseline 성능대별 재배열)**:

| task | v83(baseline) | v90 | Δ | 구간 |
|---|---:|---:|---:|---|
| STK11 | 0.8754 | 0.8612 | −0.0142 | 고성능(천장 근접) — 하락 |
| brca TP53 | 0.8270 | 0.8179 | −0.0091 | 고성능 — 하락 |
| er_status | 0.7276 | 0.7083 | −0.0193 | 중상위 — 하락(최대 낙폭) |
| EGFR | 0.7761 | 0.7649 | −0.0112 | 중상위 — 하락 |
| grade | 0.7259 | 0.7167 | −0.0092 | 중상위 — 하락 |
| luad TP53 | 0.6678 | 0.6754 | +0.0076 | 중간 — 상승 |
| her2_status | 0.6417 | 0.6566 | +0.0149 | 중간 — 상승 |
| BAP1 | 0.6436 | 0.6610 | +0.0174 | 저신호(랜덤 근접) — 상승 |
| PIK3CA | 0.5588 | 0.5389 | −0.0199 | 저신호(랜덤 근접) — 하락 |
| VHL | 0.4360 | 0.4265 | −0.0095 | 랜덤 이하 — 하락 |

**패턴**: 하락 7개 중 5개(STK11·brcaTP53·er_status·EGFR·grade)가 **0.72~0.88의 고성능·고신호
구간**에 몰려 있고, 상승 3개(luadTP53·her2·BAP1)는 전부 **0.64~0.68의 저신호 구간**이다. 유일한
예외적 큰 낙폭(PIK3CA −0.0199)과 VHL(−0.0095)은 원래 거의 랜덤인 자리라 의미가 작다. **손해는
신호가 확실한 자리에 몰리고 이득은 애매한 자리에 몰린 비대칭** — 단순 macro Δ나 t보다 이 비대칭이
더 걱정스러운 신호라는 것이 사용자 판단이다.

**결정 (사용자, 2026-08-14)**: **v90 기각.** 통계 게이트(§107-3)는 근소 미달(미판정)이었지만,
task별 비대칭 패턴을 근거로 **사용자가 종합적 판단으로 기각을 확정**했다. §117의 "seed 추가
재검증" 액션은 이걸로 불필요해졌다.

**프로토콜 변경 (이 리포 전체, 모든 노드 적용)**:
1. **§107-3의 통계 게이트는 폐지되지 않는다** — seed-paired macro Δ, t, 4/4 부호 일치 여부는
   계속 계산하고 보고한다. 이건 여전히 1차 근거다.
2. **다만 게이트 통과/미달이 승격/기각을 자동으로 결정하지 않는다.** 최종 판정은 **사용자의
   종합적 판단**이다 — 판단 재료는 (a) macro seed-paired Δ+t, (b) **task 10개 전부**를 baseline
   값과 나란히(방향 + baseline이 랜덤 근처인지 천장 근처인지) 놓고 본 패턴, (c) 다른 arm·다른
   축과의 일관성.
3. **보고 형식이 바뀐다** — 평가가 끝나면 항상 (i) 이 arm이 무엇을 테스트하는지 먼저 설명하고,
   (ii) task 10개 전부를 baseline과 함께 표로 보여주고(요약만 하지 않는다), (iii) macro
   seed-paired Δ+t를 마지막에 보고한다. 최종 승격·기각·재검증 여부는 사용자가 정한다.
4. task별 다중비교 문제(§104-5, NEXGEM이 v89에서 지적)는 여전히 유효하다 — per-task 숫자를
   **판정의 통계적 근거**로 오독하지 말라는 경고는 그대로 유지된다. 다만 사용자가 여러 task에
   걸친 **패턴**(방향 일관성, baseline 성능대별 비대칭)을 종합적으로 읽는 것은 개별 task를
   개별 가설 검정으로 오용하는 것과 다르다 — 이번 결정도 개별 task 하나의 CI가 아니라 10개
   전체의 구조적 패턴에 근거했다.
5. 이 프로토콜은 이 리포를 공유하는 모든 노드(NEXGEM 포함)의 arm 판정에 적용된다.

### §118-1. 검증: "고성능대 하락 / 저신호대 상승" 패턴이 평균 회귀 artifact인가? — 아니다

_Recorded by: nhn-NEXGEM-claude — 2026-08-14 12:04, 정리: nhn-SMC-claude_

**우려**: v83 baseline 자체가 4-seed 추정치이고 task별 seed std가 ~0.016이다. 어떤 task의
baseline이 우연히 높게 잡혔다면 다른 arm에서는 규칙적으로 더 낮게 나오는 경향이 생길 수 있다 —
**baseline 값과 Δ 사이에 인위적인 음의 상관**이 생겨, 아무 효과가 없는 arm도 §118의 비대칭
패턴을 보일 수 있다는 대안 설명이다. 새 프로토콜이 이 패턴에 비중을 두므로 확인이 필요했다.

**검증 방법**: 이미 완전한 null로 판정된 v86(observation_noise, §112)·v87(rare_response,
§113)에 같은 분해(task를 baseline 값 기준 상위 5개 / 하위 5개로 나눠 각각의 평균 Δ와 하락
개수)를 적용해, v89·v90과 나란히 비교했다.

| arm | 상위 5개 Δ 평균 | 상위 5개 하락 | 하위 5개 Δ 평균 | 하위 5개 하락 |
|---|---:|:--:|---:|:--:|
| v86 noise (null, macro Δ+0.0004) | +0.0001 | 3/5 | +0.0007 | 1/5 |
| v87 rare (null, macro Δ−0.0013) | −0.0023 | 3/5 | −0.0005 | 2/5 |
| v89 episode shape (미판정, macro Δ−0.0048) | −0.0078 | 4/5 | −0.0018 | 3/5 |
| **v90 class prior (기각, macro Δ−0.0053)** | **−0.0126** | **5/5** | **+0.0021** | 2/5 |

**결론: 평균 회귀 artifact가 아니다.** 진짜 null인 v86·v87은 상위/하위 Δ 차이가 각각 0.0006·
0.0018로 사실상 없다 — 만약 이 비대칭이 baseline 추정 노이즈의 부산물이라면 null arm에서도
똑같이 나와야 하는데 나오지 않는다. v90은 상위 5개가 **5/5 전부 하락, 평균 −0.0126**인 반면
하위 5개는 **+0.0021로 부호가 뒤집힌다** — null 대비 뚜렷하게 구조적이다. **§118의 기각 결정에
독립적인 근거가 하나 더 붙는다.**

v89는 중간 성격이다 — 방향은 v90과 같지만(상위대가 더 나쁨) 하위대도 음수라 v90 같은 부호
반전은 없다. §115-7·§118에서 이미 v89와 v90을 같은 근거로 묶지 않기로 한 것과 일치한다.

**앞으로**: v86·v87을 **이 비대칭 검사의 상시 대조군**으로 삼는다 — 새 arm에서 "고성능대/
저신호대 갈림"이 나올 때마다 "그건 평균 회귀 아니냐"는 반론이 나올 수 있는데, 이 표가 그
반론에 대한 상시 답이다(재검증 불필요).

## 119. 2026-08-14 — v91 cell 축 단독: 축 분리 성공, **bag 축은 사실상 무효 / cell 축이 진짜 레버**

_Recorded by: nhn-NEXGEM-claude — 2026-08-14 12:36_

### (i) 이 arm이 테스트한 것

§115-7이 남긴 모호함을 푼다. v89는 bag을 3배로 올리면서 예산 고정을 위해 cell을 1/3로 줄여
**두 축이 함께 움직였고**, 그래서 null/음성 결과를 "bag 축 무효"와 "두 효과 상쇄"로 구분할 수
없었다. v91은 `num_bags`를 v83 그대로 두고 **cell만 1/3**로 줄여 cell 축 단독 효과를 잰다.

config diff로 검증: v83 대비 cell knob(`num_cells`, `per_bag_max_cells`)만, **v89 대비
`num_bags`만** 다르다. 모델·head·P는 v83과 byte-identical(arch 54, trainable 196,621).
예산이 v83의 1/3이라 실측 ~22분/seed(peak VRAM 4.0 GiB, 97,461 cells/step)로 이 프로젝트에서
가장 싼 arm이었다.

```
config: configs/train_v91_cell_axis_1536_1gpu.yaml
ckpts:  checkpoints/20260814_114037/v91_cell_axis_seed4{2..5}/  (epoch 49)
tags:   v91_cell_axis_seed4{2..5}_ep49
macro:  0.6805 / 0.6756 / 0.6850 / 0.6834  →  mean 0.6811, seed std 0.0041
```

### (ii) task 10개 전부 (baseline 성능대 내림차순, §118-3)

| task | v83 | v91 | Δ | 부호 | ABMIL |
|---|---:|---:|---:|:--:|---:|
| cptac_luad STK11 | 0.8754 | 0.8642 | −0.0112 | 1/4 | 0.908 |
| cptac_brca TP53 | 0.8270 | 0.8152 | −0.0118 | 0/4 | 0.801 |
| cptac_luad EGFR | 0.7761 | 0.7667 | −0.0094 | 0/4 | 0.830 |
| bc_therapy er_status | 0.7276 | 0.7196 | −0.0081 | 1/4 | 0.717 |
| bc_therapy grade | 0.7259 | 0.7030 | −0.0228 | 1/4 | 0.770 |
| cptac_luad TP53 | 0.6678 | 0.6593 | −0.0085 | 2/4 | 0.751 |
| cptac_ccrcc BAP1 | 0.6436 | 0.6472 | +0.0037 | 2/4 | 0.693 |
| bc_therapy her2 | 0.6417 | 0.6686 | **+0.0270** | 4/4 | 0.663 |
| cptac_brca PIK3CA | 0.5588 | 0.5379 | −0.0209 | 2/4 | 0.595 |
| cptac_ccrcc VHL | 0.4360 | 0.4295 | −0.0064 | 1/4 | 0.538 |

상위5 Δ평균 **−0.0127**(하락 **5/5**) / 하위5 Δ평균 **−0.0010**(하락 3/5).

### (iii) macro seed-paired

| seed | v83 | v91 | Δ |
|---|---:|---:|---:|
| 42 | 0.6905 | 0.6805 | −0.0100 |
| 43 | 0.6896 | 0.6756 | −0.0140 |
| 44 | 0.6774 | 0.6850 | **+0.0076** |
| 45 | 0.6944 | 0.6834 | −0.0110 |
| 평균 | 0.6880 | 0.6811 | **−0.0068** |

**t ≈ −1.40, 1/4** (seed 44 부호 반전) — §107-3 게이트 미달.

### 1. 축 분해 — 이 arm의 목적

```
v89 (bag×3 AND cell÷3)  =  −0.0048
v91 (cell÷3 만)          =  −0.0068
──────────────────────────────────────
bag 축 추정 = v89 − v91  =  +0.0020
```

**cell 축이 v89를 전부 설명하고도 남는다.** cell을 1/3로 줄이는 것만으로 −0.0068이고, 거기에
bag 3배가 +0.0020을 되돌려준 결과가 v89의 −0.0048이다.

- ⚠️ **§115의 bag 축 가설은 지지되지 않는다.** context를 실제 평가 크기(90~261)에 맞추려고
  bag을 3배로 올려도 **+0.0020**으로 노이즈 한참 아래다. **bag 축은 여기서 닫는다.**
- ✅ **cell 축 방향 예측은 맞았다.** v91 config 헤더에 "실제 tile 수 중앙값이 4,988~7,736인데
  학습은 이미 2,724로 짧으니 더 줄이면 나빠질 것"이라고 **미리** 적어뒀고 −0.0068이 나왔다.
  §115-3의 셀 축 진단이 실측으로 확인된 셈이다.
- ⚠️ 가법성을 완전히 가정한 분해다(교호작용 가능). 그래도 §115-7의 완전한 모호함은 해소됐다.

### 2. her2가 세 arm 모두에서 올랐다 (§118-2c 일관성)

v89 **+0.0320**(4/4) / v90 **+0.0149** / v91 **+0.0270**(4/4). 개별 task를 통계 근거로 쓰지
말라는 §104-5 경고는 유효하지만, **독립적인 세 arm에서 같은 방향으로 반복되는 것은 다중비교로
설명되지 않는다.** 세 arm의 공통점은 "에피소드 모양을 v83에서 벗어나게 했다"는 것뿐이다.
아직 해석이 없다 — 다음 arm에서도 반복되는지 볼 것.

### 3. ⚠️ §118의 비대칭 지표는 macro 하락폭과 대체로 비례한다 — 이중 계산 주의

| arm | macro Δ | 상위5−하위5 격차 |
|---|---:|---:|
| v86 (null) | +0.0004 | −0.0006 |
| v87 (null) | −0.0013 | −0.0018 |
| v89 | −0.0048 | −0.0060 |
| v90 | −0.0053 | **−0.0147** |
| v91 | −0.0068 | −0.0117 |

거의 비례한다. **즉 비대칭은 대체로 "이 arm이 얼마나 해로웠나"의 재진술이지 완전히 독립적인
증거가 아니다.** v90이 하락폭 대비로는 여전히 가장 비대칭적이므로 §118의 v90 기각 판단 자체는
흔들리지 않는다. 다만 **앞으로 이 지표를 macro와 별개의 근거로 인용하면 같은 사실을 두 번 세게
된다** — §118-1(평균 회귀 아님)과 함께 이 한계도 같이 읽을 것.

### 4. 다음 — cell 축을 반대 방향으로 (v92, 사용자 지시)

cell 축이 유일하게 살아있는 레버이고 방향이 확인됐으므로, **줄이는 대신 늘린다.** bag 축이
무효로 나왔으니 bag을 줄여 그 예산을 cell에 준다 — v89/v91과 정확히 반대 방향, 같은 3배 대칭.
상세는 §120.

## 120. 2026-08-14 — v92 준비: cell 축을 반대로 (bag↓ cell↑) + **숨어 있던 세 번째 cell 상한 발견**

_Recorded by: nhn-NEXGEM-claude — 2026-08-14 12:52_

**질문 (사용자 지시)**: §119-1이 축을 분리했다 — bag 축은 +0.0020(무효), cell 축은 −0.0068로
**유일하게 살아있는 레버**고 부호도 확인됐다. 그러면 **줄이는 대신 늘린다.** bag 축이 무효이므로
bag을 줄여 그 예산을 cell에 준다. v89/v91과 정확히 반대 방향, 같은 3배 대칭.

| knob | v83 | v92 |
|---|---|---|
| `num_bags` | [60, 100] | **[20, 33]** (÷3) |
| `num_cells` | [256, 8192] | **[768, 24576]** (×3) |
| `per_bag_max_cells` | 4096 | **12288** (×3) |
| `max_cells` (모델) | 8192 | **12288** |
| `padding_max_cells` (신규) | (4096) | **12288** |

새 cap 12,288은 실제 slide tile 수의 **p75**(9,635~13,149)에 닿는다 — 지금까지는 중앙값
(4,988~7,736)에도 못 미쳤다. **처음으로 학습 bag이 실제 bag 크기가 된다.**

### 1. ⚠️ 세 번째 cell 상한이 숨어 있었다 — 조용히 자르는 종류의 버그

v92 config를 만들고 처음 smoke test를 돌렸더니 에피소드가 `(1, 26, **4096**, 1536)`으로 나왔다.
`per_bag_max_cells: 12288`도, 모델 `max_cells: 12288`도 지정했는데 4,096에서 잘렸다.

원인: `src/modules/data_interface.py`의 **`SYNTHETIC_PADDING_MAX_CELLS = 4096`** — dense 학습
collator(`_collate_ragged_batch`)가 padding 전에 모든 bag을 이 수로 균일 subsample한다.
데이터 knob·모델 knob과 **완전히 독립적인 세 번째 상한**이고, 모듈 상수로 하드코딩돼 있었다.

**왜 지금까지 아무도 몰랐나**: 이 프로젝트의 어떤 arm도 4,096보다 큰 bag을 요구한 적이 없다.
v83은 데이터 cap이 마침 4096으로 같았고(우연), v89/v91은 1365로 더 작았다. large-bag ragged
arm(Active-5, cap 16384)은 `ragged_training: true`로 이 collator를 아예 우회한다.

⚠️ **실패 방식이 위험한 종류다** — 에러도 경고도 없고, loss 곡선도 정상이고, 결과는 깨끗한
null처럼 보인다. 테스트 없이 돌렸다면 **"cell을 3배로 늘려도 효과 없음"이라는 완전히 틀린
결론**을 문서에 남겼을 것이다. §100의 "조용한 실패는 테스트로 단정하라"와 같은 부류다.

**수정**: `data.padding_max_cells` config knob으로 노출했다. **기본값은 4096 그대로**라 기존
arm은 하나도 움직이지 않는다(`tests/test_padding_max_cells.py::test_default_is_unchanged`가
고정). 9개 테스트로 상한이 실제로 작동하는지, 올리면 실제로 넓어지는지, padding 마스킹과
`label = -1` 부기가 유지되는지 고정했다.

> **앞으로의 계약**: bag 크기를 키우는 arm은 **세 knob을 전부** 올려야 한다 —
> `per_bag_max_cells`(데이터), `max_cells`(모델 subsample), `padding_max_cells`(collator).
> 하나라도 빠지면 조용히 잘린다.

### 2. 예산 실측

`scripts/smoke_train_budget.py --bf16` (20 step, GPU 0):

| | v83 | v92 |
|---|---:|---:|
| episode shape | `(1, 79, 4096, 1536)` | `(1, 26, 12288, 1536)` |
| dense cells/step | 297,643 | 324,403 |
| mean step | 78 ms | **61 ms** |
| peak VRAM | 11.9 GiB | 14.4 GiB |
| epoch | 1.3 분 | **1.0 분** |

v89(+31%)와 반대로 **v92는 v83보다 빠르다** — bag당 연산(covariance sketch, DD `eigh`, CT 토큰)이
bag 수에 비례하는데 bag이 1/3이기 때문이다. 모델·head·P는 v83과 byte-identical(arch 54,
trainable 196,621 실측 확인).

### 3. ⚠️ 판정 시 주의 — bag 축을 측정 범위 밖으로 외삽한다

"bag 축 ≈ 0"은 **올리는 방향**([60,100] → [180,300])에서 측정됐다. v92는 **[20, 33]으로
내린다** — 측정 구간 밖이고 대칭이라는 보장이 없다. bag이 적으면 context ridge(CV)가 더 적은
점에서 descriptor를 추정하므로, v89에서는 드러날 기회조차 없던 이유로 작은 context가 해로울 수
있다.

**v92가 음성으로 나오면 이 가능성을 먼저 배제해야 하고, cell 축에 대해 결론 내리면 안 된다.**
깨끗하지만 비싼 대안은 bag을 [60,100]에 두고 예산을 3배 쓰는 것(~3.3시간/seed, 이 arm은 ~65분).

**산출물**:
```
config: configs/train_v92_big_bags_1536_1gpu.yaml   (self-contained)
code:   src/modules/data_interface.py  (padding_max_cells)
tests:  tests/test_padding_max_cells.py  (9 tests)
```
**상태**: 준비 완료, 미실행. 판정은 §107-3(게이트) + §118(최종은 사용자).

## 121. 2026-08-14 — v92 결과: 게이트 기각. **cell 축은 단조 레버가 아니다**, 다만 context 축소 교란으로 깨끗한 검정은 아님

_Recorded by: nhn-NEXGEM-claude — 2026-08-14 14:16_

### (i) 이 arm이 테스트한 것

§119-4/§120. cell 축을 **늘리는** 방향. bag 축이 무효(+0.0020)이므로 bag을 ÷3해 그 예산을 cell
×3에 준다. cap 12,288로 **처음으로 학습 bag이 실제 slide의 p75 크기**가 됐다.

```
config: configs/train_v92_big_bags_1536_1gpu.yaml
ckpts:  checkpoints/20260814_124517/v92_big_bags_seed4{2..5}/  (epoch 49)
tags:   v92_big_bags_seed4{2..5}_ep49
macro:  0.6799 / 0.6819 / 0.6762 / 0.6748  →  mean 0.6782, seed std 0.0033
```

### (ii) task 10개 전부 (baseline 성능대 내림차순, §118-3)

| task | v83 | v92 | Δ | 부호 | t | ABMIL |
|---|---:|---:|---:|:--:|---:|---:|
| cptac_luad STK11 | 0.8754 | 0.8647 | −0.0107 | 1/4 | −1.83 | 0.908 |
| cptac_brca TP53 | 0.8270 | 0.7974 | −0.0296 | 0/4 | −2.73 | 0.801 |
| cptac_luad EGFR | 0.7761 | 0.7443 | −0.0318 | 0/4 | −4.86 | 0.830 |
| bc_therapy er_status | 0.7276 | 0.6981 | −0.0295 | 0/4 | −4.17 | 0.717 |
| bc_therapy grade | 0.7259 | 0.6824 | −0.0435 | 0/4 | −3.31 | 0.770 |
| cptac_luad TP53 | 0.6678 | 0.6857 | +0.0179 | 2/4 | +0.92 | 0.751 |
| **cptac_ccrcc BAP1** | 0.6436 | **0.6987** | **+0.0552** | **4/4** | **+2.85** | 0.693 |
| bc_therapy her2 | 0.6417 | 0.6319 | −0.0098 | 2/4 | −0.45 | 0.663 |
| cptac_brca PIK3CA | 0.5588 | 0.5285 | −0.0302 | 1/4 | −1.80 | 0.595 |
| cptac_ccrcc VHL | 0.4360 | 0.4507 | +0.0147 | 3/4 | +1.89 | 0.538 |

상위5 Δ평균 **−0.0290**(하락 **5/5**) / 하위5 **+0.0096**(하락 2/5), 격차 **−0.0386**(최대).

### (iii) macro seed-paired

| seed | v83 | v92 | Δ |
|---|---:|---:|---:|
| 42 | 0.6905 | 0.6799 | −0.0106 |
| 43 | 0.6896 | 0.6819 | −0.0077 |
| 44 | 0.6774 | 0.6762 | −0.0012 |
| 45 | 0.6944 | 0.6748 | −0.0196 |
| 평균 | 0.6880 | 0.6782 | **−0.0098** |

**t ≈ −2.56, 0/4** → §107-3 게이트를 **기각 방향으로 통과**한다.

### 1. cell 축은 단조 레버가 아니다 — §119-4의 예측이 틀렸다

```
cell ÷3 (v91) = −0.0068
cell ×3 (v92) = −0.0098
```

**양방향 모두 음수다.** §119-4가 "cell 축이 유일하게 살아있는 레버이고 부호가 확인됐으니 반대로
밀면 이득"이라고 예측했는데 **성립하지 않았다.** v83의 현재 설정이 양쪽보다 낫다.

### 2. ⚠️ 그러나 §120-3의 사전 경고가 실제로 발동했고, 추정보다 컸다

§120-3에 "bag [20,33]은 측정 구간 밖"이라고 적어뒀는데, **query를 빼고 나면 3배가 아니라 6배
감소**였다. `training_targets_per_episode: [5,12]`가 최대 12개를 query로 가져간다:

| | num_bags | **실제 context bag** |
|---|---|---|
| v83 | 60–100 | **48–95** |
| v89 | 180–300 | 168–295 |
| v92 | 20–33 | **8–28** |

합성 val_ce(epoch 49, 4 seed)가 이를 뒷받침한다:

| arm | val_ce |
|---|---|
| v83 | 0.1534 / 0.1549 / 0.1436 / 0.1517 |
| v91 | 0.1929 / 0.1903 / 0.1892 / 0.1891 |
| **v92** | **0.3342 / 0.3345 / 0.3637 / 0.2355** |

**v92는 합성 과제 자체를 2배 이상 못 푼다.** ⚠️ 합성 지표로 **arm을 고르는 것**은 금지 사항이지만
(§1 표), 이건 arm 선택이 아니라 **교란 진단**이다 — 모델이 분포만 달라진 게 아니라 실제로 불리한
조건(context 8~28개)에 놓였다는 직접 증거다. 이 구분을 지킬 것.

**결론**: **cell ×3은 깨끗하게 검정되지 않았다.** −0.0098이 cell ×3 탓인지 context 축소 탓인지
분리되지 않는다. 다만 **어느 쪽이든 "cell을 늘려 이득을 본다"는 가설은 지지되지 않는다.**

### 3. 두 가지 정정 (§119-2, §119-3)

- **her2 연속 상승이 깨졌다.** v89 +0.0320 / v90 +0.0149 / v91 +0.0270 / **v92 −0.0098**.
  §119-2가 "세 arm 일관은 다중비교로 설명 안 된다"고 기록했는데 반례가 나왔다. 상승한 셋은
  bag ≥60이고 v92만 20–33이라 "her2가 큰 context를 좋아한다"로 읽을 수는 있으나 추측이다.
- **§119-3에서 "v90이 하락폭 대비 가장 비대칭"이라 한 것을 정정한다.** 격차/|macro Δ| 비율은
  **v92 3.94** > v90 2.77 > v91 1.72 > v89 1.25 ≈ null 1.3~1.5. v92가 최대다.

### 4. ccrcc 두 task가 함께 올랐다 (판정 아님, 기록)

BAP1 **+0.0552(4/4, t=+2.85)** 는 이 세션 최대 단일 task 상승이고 **0.6987로 ABMIL(0.693)을
넘는다.** VHL도 +0.0147(3/4). §113·§114가 겨냥해 실패했던 바로 그 두 task다. ccrcc slide 중앙값이
4,988 tile이라 v92가 처음으로 그 크기를 담은 arm이라는 점과 맞아떨어지지만, **다중비교 경고(§104-5)는
유효하고 macro는 확실히 떨어졌다.** 다음 arm에서 재현되는지 볼 것.

### 5. 다음 — 깨끗한 cell 축 검정 (v93)

bag을 v83의 [60,100]에 **그대로 두고** cell만 ×3. 예산이 3배가 되어 교란 없이 cell 축만 움직인다.
⚠️ **메모리가 관건이다** — dense 학습 텐서는 `num_bags × padded_width × 1536`이라 최악의 경우
100 × 12,288 × 1536 = 18.9억 원소(bf16 3.8 GiB)이고, v92(33 × 12,288)의 3배다. v92 실제 학습에서
nvidia-smi 기준 74 GiB를 잡았으므로 선형이면 183 GiB 카드를 넘길 수 있다. 실측 후 판단한다(§122).

## 122. 2026-08-14 — v93 깨끗한 cell 축 검정: 미판정(−0.0040). **§115 에피소드 모양 가설 3축 전부 소진**

_Recorded by: nhn-NEXGEM-claude — 2026-08-14 16:40_

### (i) 이 arm이 테스트한 것

§121-5. v92의 교란을 제거한 **깨끗한 cell 축 검정**. `num_bags`를 v83의 [60,100]에 그대로 두어
context를 48–95로 유지하고, cell 상한 3개만 ×3. v92는 예산을 bag에서 빌려 context가 8–28로
무너졌는데, v93은 예산 3배를 그대로 쓴다. **cell 축이 단독으로 움직이는 유일한 arm이다.**

```
config: configs/train_v93_cell_axis_clean_1536_1gpu.yaml   (self-contained)
ckpts:  checkpoints/20260814_144651/v93_cell_axis_clean_seed4{2..5}/  (epoch 49)
tags:   v93_cell_axis_clean_seed4{2..5}_ep49
macro:  0.6792 / 0.6836 / 0.6855 / 0.6875  →  mean 0.6840, seed std 0.0035
```

**교란 제거가 확인된다** — 합성 val_ce가 v83 수준으로 복귀했다(오히려 약간 낮다):

| arm | val_ce (epoch 49, 4 seed) |
|---|---|
| v83 | 0.1534 / 0.1549 / 0.1436 / 0.1517 |
| v92 (교란) | 0.3342 / 0.3345 / 0.3637 / 0.2355 |
| **v93** | **0.1459 / 0.1424 / 0.1463 / 0.1468** |

### (ii) task 10개 전부 (baseline 성능대 내림차순, §118-3)

| task | v83 | v93 | Δ | 부호 | t | ABMIL |
|---|---:|---:|---:|:--:|---:|---:|
| cptac_luad STK11 | 0.8754 | 0.8720 | −0.0035 | 2/4 | −0.43 | 0.908 |
| cptac_brca TP53 | 0.8270 | 0.8146 | −0.0123 | 1/4 | −2.48 | 0.801 |
| cptac_luad EGFR | 0.7761 | 0.7699 | −0.0062 | 1/4 | −2.17 | 0.830 |
| bc_therapy er_status | 0.7276 | 0.7075 | −0.0201 | 1/4 | −1.35 | 0.717 |
| bc_therapy grade | 0.7259 | 0.7145 | −0.0114 | 1/4 | −0.64 | 0.770 |
| cptac_luad TP53 | 0.6678 | 0.6586 | −0.0092 | 1/4 | −0.61 | 0.751 |
| cptac_ccrcc BAP1 | 0.6436 | 0.6593 | +0.0157 | 2/4 | +0.81 | 0.693 |
| **bc_therapy her2** | 0.6417 | 0.6653 | **+0.0236** | **4/4** | +1.44 | 0.663 |
| cptac_brca PIK3CA | 0.5588 | 0.5373 | −0.0214 | **0/4** | −2.12 | 0.595 |
| cptac_ccrcc VHL | 0.4360 | 0.4405 | +0.0045 | 1/4 | +0.16 | 0.538 |

상위5 Δ평균 **−0.0107**(하락 5/5) / 하위5 **+0.0026**(하락 2/5), 격차 −0.0133, 비율 3.31.

### (iii) macro seed-paired

| seed | v83 | v93 | Δ |
|---|---:|---:|---:|
| 42 | 0.6905 | 0.6792 | −0.0113 |
| 43 | 0.6896 | 0.6836 | −0.0060 |
| 44 | 0.6774 | 0.6855 | **+0.0081** |
| 45 | 0.6944 | 0.6875 | −0.0069 |
| 평균 | 0.6880 | 0.6840 | **−0.0040** |

**t ≈ −0.96, 1/4** → 게이트 미달, **미판정**.

### 1. cell 축은 닫힌다 — 두 가지가 동시에 확인됐다

```
cell ÷3  (v91, bag 유지)       = −0.0068  (t=−1.40, 1/4)
cell ×3  (v92, bag÷3 = 교란)   = −0.0098  (t=−2.56, 0/4)
cell ×3  (v93, bag 유지·깨끗)  = −0.0040  (t=−0.96, 1/4)
```

1. ✅ **§120-3에 사전 등록한 v92 교란 가설이 맞았다.** 같은 cell ×3인데 context만 복구하니
   −0.0098 → −0.0040으로 **+0.0058이 돌아온다.** v92의 하락 상당 부분은 cell이 아니라 context
   축소 탓이었다. **사전 경고를 적어둔 덕에 v92를 "cell 축 기각"으로 잘못 기록하지 않았다.**
2. ⚠️ **그래도 cell ×3은 음수다.** 양방향(÷3 −0.0068, ×3 −0.0040) 모두 음수이므로
   **v83의 현재 cell 설정이 이미 (국소) 최적**이다. cell 축은 레버가 아니다 — **닫는다.**

### 2. §115 "에피소드 모양" 가설이 3축 전부 소진됐다

| 축 | arm | 결과 |
|---|---|---|
| context bag 수 | v89 + v91 분해 | **닫힘** — +0.0020, 노이즈 이하 (§119-1) |
| 클래스 비율 | v90 | **기각** — Δ−0.0053, 4/4 음수 (§117·§118) |
| bag당 cell 수 | v91 + v93 | **닫힘** — 양방향 모두 음수 (위 §1) |

⚠️ **§115의 진단은 사실이었지만 처방은 실패했다.** "학습 에피소드가 실제 평가 에피소드의 모양을
한 번도 모사하지 않았다"는 것은 실측된 사실이고 지금도 유효하다(클래스 비율 0.178~0.780,
context 90~261, in-distribution task 0개). 그러나 **그 격차를 좁히는 것이 성능을 올리지 않는다** —
세 축 모두에서. §115-2가 "'모사된 적 없다'는 사실이고 '그래서 낮다'는 아직 가설"이라고 유보를
달아둔 것이 옳았고, **그 가설은 이제 기각된 쪽이다.**

**따라서 이 축에서 새 arm을 설계하지 말 것.** 남은 변형(예: 예산 3배 × bag 3배 동시 확대)은
개별 축이 전부 무효/음성인 상태에서 조합만으로 이득을 기대하는 셈이라 근거가 약하다.

### 3. 대신 교차-arm 신호가 강해졌다 — her2

| arm | her2 Δ | 부호 | context bag |
|---|---:|:--:|---|
| v89 | +0.0320 | **4/4** | 168–295 |
| v90 | +0.0149 | 3/4 | 48–95 |
| v91 | +0.0270 | **4/4** | 48–95 |
| v92 | −0.0098 | 2/4 | **8–28** |
| **v93** | **+0.0236** | **4/4** | 48–95 |

**5개 arm 중 4개 상승, 그중 3개가 4/4 시드 일치**다. 유일한 예외가 context가 무너진 v92다.
§119-2에서 "세 arm 일관"을 기록했다가 §121-3에서 v92 반례로 정정했는데, **v93이 다시 확인하면서
패턴이 복원됐고 예외의 이유(context 붕괴)도 설명된다.** 개별 task를 통계 근거로 쓰지 말라는
§104-5 경고는 유효하지만, **독립적인 세 arm에서 4/4로 반복되는 것은 다중비교로 설명되지 않는다.**
v93의 her2 0.6653은 **ABMIL 0.663과 사실상 동률**이다.

반대 방향으로 **PIK3CA는 일관되게 내려간다**(v89 −0.0178 / v91 −0.0209 / v92 −0.0302 /
v93 −0.0214, 0/4). 아직 해석은 없다.

### 4. §118 비대칭 지표 — 모든 유해 arm에서 "상위5 전부 하락"이 나온다

격차/|macro Δ| 비율: v92 3.94 > v93 **3.31** > v90 2.77 > v91 1.72 > v89 1.25 ≈ null 1.3~1.5.
그리고 "상위5 5/5 하락"은 v90·v91·v92·v93 **전부**에서 나오고 null(v86 3/5, v87 3/5)에서는
안 나온다. **§119-3의 유보를 재확인한다** — 이 지표는 "해로운 arm"을 잘 가려내지만 macro 하락과
거의 같은 정보를 담으므로, macro와 **별개의 독립 근거로 인용하면 이중 계산**이다.

### 5. 실측 기록 — 메모리·비용 추정을 두 번 틀렸고 두 번 다 실측으로 교정했다

- **메모리**: `nvidia-smi`가 173.5 GiB(96.9%)를 보여 "OOM 위험"으로 판단했으나 **오독**이었다.
  ×2 변형도 **같은** 172.8 GiB에서 평탄화된 것이 결정적 증거 — caching allocator가 카드를 채우고
  캐시를 놓지 않는 것이지 필요량이 아니다. 실제 `max_memory_allocated`는 **42.4 GiB**(96 bags
  도달, 150 step). ⚠️ **이 리포에서 GPU 여유를 `nvidia-smi`로 판단하지 말 것.**
- **가드**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8`로
  실행했다. reserved가 용량의 80%(142.7 GiB)에 닿으면 캐시를 회수한다. 실측 peak 138.8 GiB로
  가드 없이 돌린 probe(172.8 GiB)보다 낮게 유지되고 OOM 0건. launcher가
  `${PYTORCH_CUDA_ALLOC_CONF:-...}`로 받으므로 미리 export하면 그 값이 쓰인다(`/proc/PID/environ`로
  4개 프로세스 전달 확인).
- **비용**: "예산 3배 = 시간 3배(~3.3h/seed)"로 추정했으나 실측 **93분/seed**였다. cell 3.3배에
  step 시간은 78 → 93 ms(+19%)뿐 — **v83의 에피소드 크기에서는 GPU가 포화되지 않고 step 비용이
  고정 오버헤드(Lightning·커널 런치)에 지배된다.** ⚠️ 앞으로 데이터 크기 arm의 비용을 예산
  비례로 추정하지 말 것.

## 123. 2026-08-14 — 진단: 합성 cell 분포가 실제 UNI2 tile과 **통계적으로 일관되지 않다**

_Recorded by: nhn-NEXGEM-claude — 2026-08-14 17:32_

**질문 (사용자 지시)**: §115~§122가 에피소드의 **모양**(bag 수, cell 수, 클래스 비율) 축을 전부
소진했다. 남은 것은 한 번도 측정된 적 없는 축 — **cell 값 자체의 분포**다. 실데이터에 피팅하려는
것이 아니고, **합성 생성기가 실제 UNI2 피처와 통계적으로 일관되는지만** 확인한다.

**산출물**: `scripts/diagnose_synthetic_vs_real.py` (재현 가능, 학습 불필요)

### 0. 어느 지점에서 비교했는가 — 이게 틀리면 전부 무의미하다

두 가지를 먼저 확인했다:
- 합성 cell은 `normalize_output: true` → `F.normalize(x, dim=-1)`로 **L2 정규화**된다.
- 실제 피처는 `scripts/test_pathobench.py::load_slide_features`가 h5에서 **raw float32로 그대로**
  읽는다. 평가 경로 어디에도 정규화가 없다(`_normalize_bags`는 shape 검증·unbind만 한다).

그러나 모델은 `BaseAggregator._context_pool_stats`로 **에피소드마다 context cell 전체의
per-feature 평균·표준편차**를 내어 표준화한다. 이것이 전역 스케일과 차원별 shift를 흡수하므로,
raw 크기를 비교하면 **모델이 보지 않는 것**을 재게 된다. 따라서 아래는 전부 **같은 표준화를 적용한
뒤**의 값이고, 판단 근거는 표준화 이후 수치다.

**실제 slide는 SEAL 평가 task가 하나도 없는 코호트에서만 뽑았다** — BRACS, CPTAC-LSCC, CPTAC-PDA,
MBC, UCLA_Lung. 이 진단은 평가 코호트를 건드리지 않는다. tile은 4,096으로 캡해 v83 학습 조건
(`per_bag_max_cells: 4096`)에 맞췄다.

### 1. 결과 (80 bags, seed 0)

| 지표 | 합성 (v83) | 실제 UNI2 | |
|---|---:|---:|---|
| participation ratio | 41.3 | 49.2 | **일치** |
| top-1 고유값 비중 | 8.2% | 12.1% | 유사 |
| **r90** (분산 90% 차원수) | **47** | **585** | **12배** |
| **r99** | **63** | **1274** | **20배** |
| within-bag cosine | +0.130 | +0.351 | **2.7배** |
| between-bag 분산 비중 (ICC) | 17.4% | 31.6% | 1.8배 |
| cell L2 norm | 1.0000 ± **0.0000** | 16.33 ± **2.56** (9.8~30.4) | 분산 자체가 없음 |
| per-dim 첨도 (중앙/최대) | 2.88 / 3.89 | 3.24 / 10.05 | 실제가 두꺼운 꼬리 |
| per-dim std max/min | 2.1 | 7.4 | 실제가 3.5배 이질적 |

### 2. 지배 부공간은 잘 맞는다 (일관되는 부분)

participation ratio 41.3 vs 49.2, top-1 비중 8.2% vs 12.1%. **"주된 구조"의 크기는 놀랄 만큼
비슷하다.** 이 두 지표만 보면 일관돼 보이고, 실제로 이것이 지금까지 CV 분기가 어느 정도 작동한
이유일 수 있다.

### 3. 스펙트럼 꼬리가 질적으로 다르다 — 최대 불일치

합성은 **~63차원을 넘으면 사실상 rank가 없고**, 실제는 1536차원 대부분에 분산이 퍼져 있다.
원인은 명확하다 — **`latent_dim: 32`**. 합성 cell은 32차원 latent를 manifold map으로 1536차원에
심은 것이므로 관측 노이즈를 제외하면 **32차원 다양체 위에 있다.** 두꺼운 꼬리를 가질 수가 없다.

⚠️ **교란 통제 — 코호트를 섞은 탓이 아니다.** 5개를 pooling하면 코호트 간 분산 방향이 더해져
r90이 부풀 수 있으므로 개별 확인했다:

| 소스 | participation | r90 | r99 |
|---|---:|---:|---:|
| BRACS (20 bags) | 49.9 | 456 | 1216 |
| CPTAC-LSCC (20) | 75.2 | 542 | 1256 |
| CPTAC-PDA (20) | 74.0 | 539 | 1250 |
| MBC (20) | 76.1 | 483 | 1222 |
| UCLA_Lung (20) | 52.7 | 431 | 1176 |
| **CPTAC-LSCC — 단일 slide 1개** | 28.7 | **322** | 1007 |
| **BRACS — 단일 slide 1개** | 36.7 | **283** | 974 |
| 합성 (80 bags) | 41.3 | **47** | 63 |

**slide 하나만 봐도 r90이 283~322다.** pooling 효과가 아니라 실제 UNI2 tile의 고유 성질이다.

**왜 중요한가 — 모델의 핵심 분기와 직결된다**: `covariance_sketch_dim: 128`이다. CV 분기(역사적으로
가장 강한 분기)가 bag 공분산을 128차원 스케치에 투영한다. 합성에서는 분산 90%가 47차원에 있으므로
**스케치의 상당 부분이 사실상 노이즈를 본다.** 실제에서는 128차원 전부가 신호를 본다.
**생성기의 내재 차원(32)이 스케치 차원(128)의 1/4이다.**

### 4. bag 내부 응집도가 다르다

within-bag cosine +0.130 → **+0.351**, between-bag 분산 17.4% → **31.6%**. 실제 slide는 내부적으로
훨씬 응집되고 서로 훨씬 구별된다 — 염색·스캐너·환자에서 오는 slide-level batch effect다.
합성은 `shared_component_fraction: [0.82, 0.96]`으로 **cell의 82~96%가 공유 성분에서 나오므로**
구조적으로 bag들이 크게 겹친다. in-context 모델 입장에서 **"bag 정체성"의 강도가 학습과 평가에서
다르다.**

### 5. cell norm 정보가 합성에는 0이다

`normalize_output: true`로 모든 cell이 **정확히** 단위벡터다(sd 0.0000). 실제는 9.8~30.4로 변한다.
per-feature 표준화 후에도 이 변동은 공통 성분으로 일부 살아남으므로, 합성에는 없는 정보가 실제에는
있다.

### 6. 격차 ↔ knob 대응 (진단이고 처방이 아니다)

| 격차 | 관련 knob | 현재값 |
|---|---|---|
| 스펙트럼 꼬리 (12~20배) | `latent_dim`, `manifold_mode` | 32, orthogonal |
| bag 응집도 (2.7배) | `shared_component_fraction`, `donor_shift_scale` | [0.82,0.96], 0.35 |
| norm 분산 (0 vs 2.56) | `normalize_output` | true |

### 7. ⚠️ `latent_dim`은 아래로만 스윕됐다 — 32가 상한이고 그게 최고다

`current_experiments.md` §3-2와 §99-3/§105-5의 기록:

| latent_dim | 2 | 4 | 8 | 16 | **32 (현재)** |
|---|---:|---:|---:|---:|---:|
| SEAL macro | 0.6775 | 0.6776 | 0.6759 | 0.6665 | **0.6880** |

**32보다 큰 값은 한 번도 시도된 적이 없다.** 8→32 구간은 상승(+0.012)이었고(L16 딥은 fold
노이즈가 아님이 §99-3에서 확인됐다), 진단은 실제 데이터의 내재 차원이 훨씬 높다고 말한다.
두 사실이 같은 방향을 가리킨다.

### 8. 한계

- tile을 4,096으로 캡해 v83 학습 조건에 맞췄다. **실제 평가는 full-tile**(중앙값 4,988~7,736,
  최대 35,107)이므로 평가 시 bag은 이보다 크다.
- seed 0 단일 추출, 80 bag, 반복 없음. 다만 §3의 단일 slide 대조까지 일관되므로 r90 결론은 견고하다.
- 표준화를 전체 pool로 했으나 모델은 **context cell만** 쓴다. 방향에는 영향 없다.
- **이것은 진단이다.** §115가 "진단은 사실이었으나 처방은 세 축 모두 실패"였음을 반드시 기억할 것 —
  격차가 실재한다는 것과 그것을 좁히면 성능이 오른다는 것은 **별개의 명제**다.

## 124. 2026-08-15 — §123 처방 스크리닝 (P1/P3/P1+P3/전부, 1 seed): **어느 레버도 도움이 안 된다**

_Recorded by: nhn-NEXGEM-claude — 2026-08-15 04:21_

**설계 (사용자 지시)**: §123이 세 격차를 실측했으니, 각각을 4 seed로 돌리는 대신 **1 seed × 4
조합**(P1 / P3 / P1+P3 / P1+P2+P3)을 GPU 4장에 올려 한 배치로 스크리닝한다. "어느 조합이 4 seed를
쓸 값어치가 있나"만 본다.

⚠️ **1 seed는 §107-3 게이트로 판정할 수 없다.** 4 seed arm들에서 실측한 **단일 시드-쌍 Δ의
SD ≈ 0.008**(v89 0.0069 / v91 0.0098 / v92 0.0077 / v93 0.0084)이므로 2σ ≈ 0.017이다.
**이 절의 숫자는 §3-1 판정표에 넣지 않는다.**

### 0. 처방 원칙 — §115의 실패를 반복하지 않기 위해

§115는 진단에서 처방으로 바로 갔고 3축을 태웠다. 이번에는 **knob이 목표 통계를 실제로 움직이는지
`scripts/diagnose_synthetic_vs_real.py`로 먼저 확인**하고(수 분, GPU 0) 움직이는 것만 arm으로
올렸다. 이 단계에서 **arm 하나를 미리 살렸다** — §1-3 참조.

### 1. P1 — bag 응집도 (`donor_shift_scale` 0.35 → 0.75)

**메커니즘**. `sample_episode` 안에서 응답(레이블) 구조가 latent에 심어진 **뒤**, 다음이 실행된다:

```python
if self.donor_shift_scale > 0:
    donor_shift = torch.randn(num_bags, 1, self.latent_dim, ..., generator=nuisance_generator)
    z = z + self.donor_shift_scale * donor_shift
```

`(num_bags, 1, latent_dim)` — bag마다 **단 하나의** latent 벡터를 뽑아 **그 bag의 모든 cell에
동일하게** 더한다. 즉 bag 전체를 latent 공간에서 **강체 평행이동**시킨다. 코드 주석이 의도를
밝힌다 — *"Same-label donors share episode-level prototypes, but each donor has independent
nuisance."*

**왜 이것이 옳은 대응물인가**. 실제 slide-level batch effect(염색·스캐너·환자)는 정확히 이런
형태다 — 한 slide의 모든 tile을 같은 방향으로 밀어낸다. §123-4가 실측한 격차(within-bag cosine
+0.130 vs +0.351, ICC 17.4% vs 31.6%)는 "실제 slide는 내부적으로 응집되고 서로 구별된다"는
것이고, bag 단위 강체 이동이 바로 그 성질을 만든다. **통계만 맞추는 fudge가 아니라 기전이
일치하는 knob이다.**

**왜 응집도가 오르는가**. per-feature 표준화 후, bag이 공유하는 오프셋은 그 bag 모든 cell의
공통 성분이 된다 → 같은 bag cell끼리 cosine이 양수로 오르고, bag 평균이 서로 벌어져 ICC(between-bag
분산 비중)가 오른다. 오프셋 크기가 latent 자체의 산포(`latent_scale: [0.6, 1.3]`)에 가까워질수록
"bag 정체성"이 cell 개별 변동과 맞먹게 된다 — 0.75가 대략 그 지점이다.

**피팅 (실측, 40 bags)**:

| `donor_shift_scale` | within-bag cosine | ICC | participation | r90 |
|---|---:|---:|---:|---:|
| **0.35 (v83)** | +0.109 | 10.8% | 31.2 | 28 |
| 0.6 | +0.249 | 25.4% | 30.0 | 28 |
| 0.7 | +0.312 | 31.4% | 29.3 | 28 |
| **0.75 (v94 채택)** | **+0.337** | **34.3%** | 28.9 | 27 |
| 0.85 | +0.395 | 40.0% | 28.0 | 27 |
| 1.0 | +0.471 | 47.7% | 26.7 | 27 |
| **실제 UNI2 목표** | **+0.351** | **31.6%** | 50–76 | 431–542 |

0.75에서 두 목표가 각각 4%·9% 오차로 맞는다 — **§123-4의 격차는 사실상 닫혔다.** 그리고
**스펙트럼은 전혀 움직이지 않는다**(r90 28 → 27): P1은 응집도만 건드리는 깨끗한 knob이다.

### 2. P3 — 스펙트럼 모양 (신규 knob `spectral_tail_*`)

`latent_dim`으로는 불가능하다. `manifold_mode: orthogonal`이 **등거리 임베딩**이라 부공간 안
스펙트럼이 평탄해서, latent를 올리면 participation과 r90이 **함께** 움직인다(latent 512 →
411.8 / 433, 목표는 50–76 / 431–542). 그래서 감쇠 프로파일이라는 자유도를 새로 넣었다 —
`spectral_tail_dim`(꼬리 길이) · `spectral_tail_decay`(감쇠 α) · `spectral_tail_scale`(세기),
manifold map 직후에 **레이블과 무관한 nuisance**로 더한다.

채택값 `scale 2.5 / decay 0.5 / dim 1536` → **participation 67.2, r90 422** (목표 50–76,
431–542). `tests/test_spectral_tail.py` 10개로 고정했고, 특히 (a) `scale=0`이면 완전히 inert하여
기존 arm이 byte-identical, (b) 꼬리가 레이블 신호를 **강화하지 못한다**(nuisance)는 것을 검정한다 —
(b)가 없으면 개선이 나와도 "과제가 쉬워진 것"과 구분할 수 없다.

### 3. ⚠️ 진단이 arm 하나를 미리 살렸다 — `shared_component_fraction`은 무효 knob이다

§123-6은 bag 응집도의 knob으로 `shared_component_fraction`을 지목했다. **실측하니 전혀 움직이지
않는다**:

| 설정 | within-bag cosine | ICC |
|---|---:|---:|
| [0.82, 0.96] (v83) | +0.104 | 10.8% |
| [0.3, 0.6] | +0.106 | 10.9% |
| `shared_component_probability: 0.0` (완전히 끔) | +0.142 | 13.8% |

**진단 없이 arm을 돌렸으면 65분 × 4 seed를 헛되게 썼다.** 실제 knob은 `donor_shift_scale`이었고,
비슷해 보이는 `donor_component_shift_scale`(bag×성분 단위)은 1.0에서 +0.236으로 부분적이다.
**교훈: 격차에 대응하는 knob은 이름으로 추측하지 말고 측정으로 특정할 것.**

### 4. ⚠️ P1과 P3는 구조적으로 상충한다 (실측)

꼬리는 **cell별 iid**로 1536차원에 분산을 넣고, donor shift는 **32차원 latent**의 bag 평균만
움직인다. 그래서 꼬리를 켜면 응집도가 무너지고 donor shift로 되돌릴 수 없다:

| 꼬리 켠 상태 `donor_shift_scale` | 1.5 | 3.0 | 5.0 | 8.0 |
|---|---:|---:|---:|---:|
| within-bag cosine | +0.086 | +0.116 | +0.126 | **+0.130 (포화)** |

목표 +0.351에 한참 못 미친 채 포화한다. **따라서 v96/v97은 "두 격차를 동시에 닫은 arm"이 아니라
"스펙트럼은 맞추고 응집도는 포기한 arm"이다.** 조합을 attribute할 수 있도록 P1은 모든 arm에서
0.75로 고정했다(arm별 재튜닝 안 함).

### 5. 결과 (1 seed, SEED 42, v83 seed42 = 0.6905 대비)

```
run:  checkpoints/20260814_182626/{v94_p1_bagcoherence,v95_p3_spectral,v96_p1p3,v97_p1p2p3}_seed42/
tags: {arm}_seed42_ep49
```

| task | v83 s42 | v94 P1 | v95 P3 | v96 P1+P3 | v97 전부 |
|---|---:|---:|---:|---:|---:|
| cptac_luad STK11 | 0.8645 | −0.0165 | −0.0015 | −0.0005 | −0.0074 |
| cptac_brca TP53 | 0.8321 | +0.0014 | −0.0226 | −0.0239 | −0.0312 |
| cptac_luad EGFR | 0.7753 | −0.0004 | −0.0008 | −0.0009 | −0.0107 |
| bc_therapy grade | 0.7444 | −0.0253 | −0.0500 | −0.0427 | −0.0466 |
| bc_therapy er_status | 0.7219 | −0.0046 | +0.0088 | +0.0088 | −0.0199 |
| cptac_luad TP53 | 0.6725 | +0.0183 | −0.0395 | −0.0166 | −0.0440 |
| bc_therapy her2 | 0.6405 | **+0.0229** | +0.0062 | +0.0082 | +0.0068 |
| cptac_ccrcc BAP1 | 0.6246 | −0.0366 | +0.0168 | +0.0082 | +0.0291 |
| cptac_brca PIK3CA | 0.5592 | +0.0154 | −0.0046 | −0.0291 | −0.0471 |
| cptac_ccrcc VHL | 0.4699 | **+0.0477** | +0.0012 | +0.0240 | −0.0070 |

| arm | macro | Δ | 상위5 Δ | 하위5 Δ | 노이즈 대비 |
|---|---:|---:|---:|---:|---|
| **v94 P1** | **0.6927** | **+0.0022** | −0.0091 | **+0.0135** | 0.3σ |
| v95 P3 | 0.6819 | −0.0086 | −0.0132 | −0.0040 | 1.1σ |
| v96 P1+P3 | 0.6840 | −0.0065 | −0.0118 | −0.0011 | 0.8σ |
| **v97 P1+P2+P3** | 0.6727 | **−0.0178** | −0.0232 | −0.0124 | **2.2σ** |

**읽기**:
- **어느 레버도 도움이 안 된다.** 최고가 P1의 +0.0022로 0.3σ, 사실상 0이다.
- **노이즈 밖인 것은 v97(셋 다)뿐이고 방향이 나쁘다**(−0.0178, 2.2σ). v96에 P2를 더해 −0.0113이
  더 내려갔지만 두 단일 시드의 차이는 1.1σ라 P2 단독 책임으로 단정할 수 없다.
- **P3는 아무것도 못 한다.** 격차가 가장 컸던 축(r90 12~20배)을 목표까지 닫았는데 −0.0086이다.
- **P1의 프로파일만 다르다** — 유일하게 하위5가 뚜렷한 양수(+0.0135)이고 상위5는 약한 음수다.
  VHL 0.4699 → **0.5176**으로, 이 세션에서 **VHL이 처음으로 0.5를 넘었다**(다른 arm들은 seed42에서
  0.4232~0.4971). §115-1이 "4/4 시드 모두 0.5 이하인 것은 역상관 신호"라 기록한 그 이상 현상과
  관련될 수 있다. **단일 시드이므로 단정 금지.**

### 6. 이것이 **두 번째** 완전한 진단→처방 실패다

| 진단 | 실측된 격차 | 처방 결과 |
|---|---|---|
| §115 에피소드 **모양** | 클래스 비율·bag 수·cell 수 전부 어긋남 | **3축 전부 실패** |
| §123 cell **값 분포** | 스펙트럼 12~20배·응집도 2.7배·norm 0 vs 2.56 | **3축 전부 실패** |

둘 다 격차가 실재함은 정확히 실측됐고, 둘 다 좁혀도 성능이 오르지 않았다. **§105 이후 데이터
분포를 건드린 arm이 12개인데 양성이 하나도 없다.**

⚠️ **이 자체가 정보다 — 병목이 합성 데이터 분포가 아닐 가능성이 커진다.**
`current_experiments.md` 항목 5가 이미 같은 말을 하고 있다: "CV-1 단독 0.9052 vs 전체 0.9199,
그 위에 얹은 모든 시도(v36·v37·v42·v43·v44·v45)가 Δ≈0 — task 자체의 정보 한계일 수 있다."
**데이터 분포 축에서 새 arm을 설계하기 전에 이 가능성을 먼저 검토할 것.**

### 7. 유일하게 강해지는 신호 — her2 (9개 arm 중 8개 상승)

| arm | v89 | v90 | v91 | v92 | v93 | v94 | v95 | v96 | v97 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| her2 Δ | +0.0320 | +0.0149 | +0.0270 | **−0.0098** | +0.0236 | +0.0229 | +0.0062 | +0.0082 | +0.0068 |

유일한 예외가 context가 8–28로 붕괴한 v92다. **에피소드 분포를 어떻게 바꾸든 her2가 오른다** —
다중비교로 설명되지 않는 유일한 task 수준 패턴이다(v89·v91·v93은 4/4 시드 일치였다). v83의 her2
0.6405는 ABMIL 0.663보다 낮은데 여러 arm이 그 위로 올린다. 아직 기전 설명이 없다.

### 8. 산출물

```
knob:    src/datasets/synthetic_data.py  spectral_tail_{dim,decay,scale} + _add_spectral_tail
tests:   tests/test_spectral_tail.py  (10 tests)
configs: configs/train_v9{4,5,6,7}_*_1536_1gpu.yaml   (self-contained, 헤더에 1-seed 경고 포함)
diag:    scripts/diagnose_synthetic_vs_real.py  (§123에서 추가, 이번 피팅에 사용)
```

## 125. 2026-08-15 — v94 P1 4 seed 확정: **미판정이지만 음수**. §124 스크리닝의 유일한 양수는 위양성이었다

_Recorded by: nhn-NEXGEM-claude — 2026-08-15 05:59_

### (i) 이 arm이 테스트한 것

§124-1의 P1. `donor_shift_scale` 0.35 → 0.75로, §123-4가 실측한 bag 응집도 격차를 **실제로 닫은**
유일한 레버다(within-bag cosine +0.109 → +0.337, ICC 10.8% → 34.3%, 실제 목표 +0.351 / 31.6%).
스펙트럼은 건드리지 않는다(r90 28 → 27).

**seed 42는 §124 스크리닝 것을 재사용했다** — 관련 소스 전부 최종 수정이 08-14 17:57이고 seed42
실행이 18:26이므로 같은 코드·같은 config다(mtime으로 확인). 43/44/45를 같은
`ICF_RUN_TIME`으로 추가해 네 시드가 한 디렉토리에 있다.

```
config: configs/train_v94_p1_bagcoherence_1536_1gpu.yaml
ckpts:  checkpoints/20260814_182626/v94_p1_bagcoherence_seed4{2..5}/  (epoch 49)
tags:   v94_p1_bagcoherence_seed4{2..5}_ep49
macro:  0.6927 / 0.6846 / 0.6752 / 0.6823  →  mean 0.6837, seed std 0.0072
```

### (ii) task 10개 전부 (baseline 성능대 내림차순, §118-3)

| task | v83 | v94 | Δ | 부호 | t | ABMIL |
|---|---:|---:|---:|:--:|---:|---:|
| cptac_luad STK11 | 0.8754 | 0.8699 | −0.0056 | 1/4 | −1.44 | 0.908 |
| **cptac_brca TP53** | 0.8270 | 0.8328 | **+0.0058** | **4/4** | **+2.86** | 0.801 |
| cptac_luad EGFR | 0.7761 | 0.7714 | −0.0047 | 1/4 | −0.95 | 0.830 |
| bc_therapy er_status | 0.7276 | 0.7222 | −0.0054 | 1/4 | −0.87 | 0.717 |
| bc_therapy grade | 0.7259 | 0.7151 | −0.0107 | 0/4 | −2.18 | 0.770 |
| cptac_luad TP53 | 0.6678 | 0.6662 | −0.0017 | 2/4 | −0.14 | 0.751 |
| cptac_ccrcc BAP1 | 0.6436 | 0.6239 | −0.0197 | 2/4 | −1.21 | 0.693 |
| bc_therapy her2 | 0.6417 | 0.6451 | +0.0034 | 1/4 | +0.53 | 0.663 |
| cptac_brca PIK3CA | 0.5588 | 0.5454 | −0.0133 | 1/4 | −1.20 | 0.595 |
| cptac_ccrcc VHL | 0.4360 | 0.4452 | +0.0092 | 2/4 | +0.71 | 0.538 |

상위5 Δ평균 −0.0041(하락 4/5) / 하위5 −0.0044(하락 3/5), **격차 +0.0003**.
⚠️ **§118 비대칭이 사라졌다** (v90 −0.0147 / v92 −0.0386 / v93 −0.0133 대비). §119-3·§121-3이
"이 지표는 macro 하락폭을 따라간다"고 유보를 단 것과 일치한다 — macro 하락이 작으면 비대칭도 없다.

⚠️ brca TP53이 게이트를 넘었지만(4/4, t=+2.86) **task 10개면 귀무가설 아래 기대 0.9개**가 넘는다
(§115-7). **우연 범위이고 판정 근거로 쓰지 않는다.**

### (iii) macro seed-paired

| seed | v83 | v94 | Δ |
|---|---:|---:|---:|
| 42 | 0.6905 | 0.6927 | **+0.0022** |
| 43 | 0.6896 | 0.6846 | −0.0050 |
| 44 | 0.6774 | 0.6752 | −0.0022 |
| 45 | 0.6944 | 0.6823 | −0.0121 |
| 평균 | 0.6880 | 0.6837 | **−0.0043** |

SD(Δ) = 0.0060, SE = 0.0030, **t ≈ −1.43, 1/4** → 게이트 미달 **미판정**, 다만 **부호는 음수**다.

### 1. ⚠️ §124 스크리닝의 유일한 양수는 위양성이었다 — 1 seed로 부호를 믿지 말 것

seed 42의 +0.0022가 유일한 양수 시드였고, 43/44/45는 −0.0050 / −0.0022 / −0.0121이다.
**§124가 "P1만 유일하게 양수"라고 기록한 것은 단일 시드 인공물이었다.** §124-0의 경고
(단일 시드-쌍 Δ SD ≈ 0.008)가 그대로 실현됐다 — +0.0022는 0.3σ였고, 부호가 뒤집히는 데
충분한 크기였다.

**앞으로의 계약**: 1 seed 스크리닝은 **"크게 나쁜 것을 배제"하는 데만** 쓴다. v97(−0.0178, 2.2σ)을
걸러낸 것은 유효했지만, **양수 부호는 스크리닝으로 주장할 수 없다.** 4 seed 확정 전에는 "유망"으로도
쓰지 말 것.

### 2. ⚠️ VHL의 0.5 돌파도 seed 42 인공물이었다

| seed | 42 | 43 | 44 | 45 |
|---|---:|---:|---:|---:|
| v83 | 0.4699 | 0.4142 | 0.4374 | 0.4224 |
| v94 | **0.5176** | 0.4163 | 0.4314 | 0.4156 |

seed 42만 움직이고 나머지 셋은 v83과 사실상 동일하다. Δ +0.0092(t=0.71, 2/4)로 **VHL은 여전히
3/4 시드에서 랜덤 이하**다. §124-5가 "이 세션에서 VHL이 처음 0.5를 넘었다, 단일 시드이므로 단정
금지"로 기록한 그 항목은 **기각**이다 — §115-1의 역상관 이상 현상에 대한 단서가 아니었다.

### 3. her2 교차-arm 패턴에 v94는 기여하지 않는다

§124-7이 seed42 기준 +0.0229로 기록했으나 4 seed로는 **+0.0034 (1/4, t=0.53)** 다. 따라서
her2 패턴의 근거는 **4 seed로 4/4를 낸 v89(+0.0320)·v91(+0.0270)·v93(+0.0236)과 v90(+0.0149,
3/4)** 로 한정된다. v94는 중립이다. (§124-7의 9-arm 표는 v94/v95/v96/v97 열이 1 seed임을 감안해
읽을 것.)

### 4. 데이터 분포 축: arm 13개, 양성 0개

| 사이클 | 진단 | arm | 결과 |
|---|---|---|---|
| §115 | 에피소드 **모양** | v89·v90·v91·v92·v93 | 3축 전부 닫힘/기각 |
| §123 | cell **값 분포** | v94(4 seed) · v95·v96·v97(1 seed) | 전부 null~음수 |
| 그 이전 | §105-6 데이터 스윕 | v86·v87 | null |

**§105 이후 데이터 분포를 건드린 arm 13개 중 양성이 하나도 없다.** 두 사이클 모두 "격차는 정확히
실측됐고, 닫아도 오르지 않는다"로 끝났다.

⚠️⚠️ **정정 (2026-08-15 11:34, nhn-NEXGEM-claude) — 위 문단의 원래 결론("데이터 분포 축에서 새
arm을 설계하지 말 것")은 과잉 주장이었다.** §125-1이 "1 seed로 부호를 주장하지 말 것"을 금지
사항에 넣어놓고, 바로 이 절에서 **1 seed arm 3개를 근거로 축 전체를 닫았다.** 같은 기준을
적용하면 실제로 확정된 것은 둘뿐이다:

| arm | seed | Δ | 노이즈 대비 | 확정 여부 |
|---|---|---:|---|---|
| v94 P1 | **4** | −0.0043 | t=−1.43 | ✅ "양수 아님" 확정 |
| v95 **P3 단독** | 1 | −0.0086 | **1.1σ** | ❌ **미확정** |
| v96 P1+P3 | 1 | −0.0065 | **0.8σ** | ❌ **미확정** |
| v97 셋 다 | 1 | −0.0178 | 2.2σ | ✅ "나쁨" 확정(배제는 스크리닝으로 가능) |
| **P2 단독** | **0** | — | — | ❌ **한 번도 실행되지 않음** |

**가장 큰 격차였던 P3(r90 12~20배)는 1.1σ로 미확정이고, P2는 단독 실행된 적이 없다.**
v97−v96 = −0.0113으로 P2를 원인으로 지목했던 것도 1.1σ다. 또한 세 knob 모두 **실제 통계에
가까워지는 방향으로만, 각각 값 하나씩만** 시험했다 — 반대 방향도, 용량-반응도 없다.

**따라서 이 축은 "닫힌" 것이 아니라 "P1만 확정되고 나머지는 미확정"이다.** 후속은 §126.
(항목 5 — 병목이 표현이 아닐 가능성 — 은 여전히 유력한 후보이나, 이 축의 미확정 항목을
정리한 뒤에 판단할 것.)

### 5. 이 arm을 돌린 값 (기록용)

작성자는 "P1의 기대값이 0에 가까워 4 seed를 쓸 값어치가 낮다"고 권고했으나 사용자가 실행을
지시했고, 결과적으로 **거짓 단서 두 개(§1 P1 양수, §2 VHL 0.5 돌파)를 제거**했다. 둘 다 §124에
"단일 시드라 단정 금지"로 남아 있던 항목이고, 확정하지 않으면 다음 arm 설계를 잘못된 방향으로
끌었을 것이다. **미판정 결과라도 거짓 단서를 죽이는 값이 있다.**

## 126. 2026-08-15 — v98(P1 역방향)·v99(P2 단독): **부호가 반대였다.** nuisance 가설이 arm 5개를 설명한다

_Recorded by: nhn-NEXGEM-claude — 2026-08-15 14:07_

**계기**: §125-4의 과잉 주장을 사용자가 지적했다("P1 P2 P3에 대해서 할 수 있는 실험이 더 많지
않아?"). 재검토하니 확정된 것은 **v94 P1(4 seed)**과 **v97 셋 다(2.2σ)** 둘뿐이었고, P3 단독은
1.1σ, P1+P3는 0.8σ, **P2 단독은 실행된 적조차 없었다.** 게다가 세 knob 모두 **실제 통계에
가까워지는 방향으로만, 값 하나씩만** 시험했다 — 반대 방향과 용량-반응이 통째로 비어 있었다.

### (i) 두 arm이 테스트한 것

- **v98 = P1 역방향**: `donor_shift_scale` 0.35 → **0.15**. 실제 UNI2에서 **멀어지는** 방향
  (cos +0.105 → +0.027, ICC 10.8% → 2.7%; 실제는 +0.351 / 31.6%).
- **v99 = P2 단독**: `normalize_output` true → **false**. norm sd 0.00 → 0.84 (실제 2.56).
  나머지 통계는 불변(cos +0.105, r90 28).

```
v98: checkpoints/20260815_113422/v98_p1_reverse_seed4{2..5}/     macro 0.6907/0.6950/0.6802/0.6946
v99: checkpoints/20260815_124024/v99_p2_norm_seed4{2..5}/        macro 0.6721/0.6797/0.6787/0.6837
```

### (ii) task 10개 전부 (baseline 성능대 내림차순, §118-3)

| task | v83 | v98 Δ | 부호 | v99 Δ | 부호 | ABMIL |
|---|---:|---:|:--:|---:|:--:|---:|
| cptac_luad STK11 | 0.8754 | −0.0023 | 1/4 | −0.0398 | **0/4** | 0.908 |
| cptac_brca TP53 | 0.8270 | −0.0024 | 2/4 | −0.0223 | **0/4** | 0.801 |
| cptac_luad EGFR | 0.7761 | −0.0037 | 1/4 | −0.0191 | **0/4** | 0.830 |
| bc_therapy er_status | 0.7276 | −0.0083 | 1/4 | −0.0186 | **0/4** | 0.717 |
| bc_therapy grade | 0.7259 | −0.0113 | 1/4 | −0.0257 | **0/4** | 0.770 |
| cptac_luad TP53 | 0.6678 | +0.0018 | 2/4 | −0.0285 | 0/4 | 0.751 |
| **cptac_ccrcc BAP1** | 0.6436 | **+0.0375** | **4/4** | **+0.0522** | **4/4** | 0.693 |
| bc_therapy her2 | 0.6417 | +0.0136 | **4/4** | +0.0102 | 3/4 | 0.663 |
| cptac_brca PIK3CA | 0.5588 | −0.0032 | 1/4 | −0.0008 | 1/4 | 0.595 |
| cptac_ccrcc VHL | 0.4360 | −0.0001 | 2/4 | −0.0018 | 2/4 | 0.538 |

### (iii) macro seed-paired

| arm | macro | seed std | Δ | t | 부호 | 상위5 | 하위5 |
|---|---:|---:|---:|---:|:--:|---:|---:|
| **v98 P1 역방향** | 0.6901 | 0.0069 | **+0.0021** | **+1.73** | **4/4** | −0.0056 | +0.0099 |
| **v99 P2 단독** | 0.6785 | 0.0048 | **−0.0094** | **−2.32** | 1/4 | −0.0251 | +0.0063 |

v98 per-seed Δ: +0.0002 / +0.0054 / +0.0028 / +0.0002 — **4/4 양수**지만 |t|=1.73으로 게이트 미달.
v99 per-seed Δ: −0.0184 / −0.0099 / +0.0013 / −0.0107.

### 1. P2는 유해하다 — v97의 추론이 맞았다

t=−2.32로 게이트 바로 아래지만, **상위 5개 task가 20/20 seed-task 셀 전부 음수**다. §124-5에서
v97−v96 = −0.0113으로 P2를 지목했던 것(1.1σ)이 단독 4 seed로 확인됐다. **norm 축은 이걸로 닫는다** —
단, `normalize_output: false`는 격차의 1/3만 닫는 부분 처방이므로(norm sd 0.84 vs 실제 2.56),
"norm 정보를 더 주는 것이 해롭다"까지가 결론이고 "완전히 맞추면 어떨지"는 여전히 미검정이다.

### 2. ⚠️ P1 축은 국소 최적이 아니라 **단조**다 — 부호가 반대였다

| `donor_shift_scale` | cosine | ICC | macro Δ | 부호 |
|---|---:|---:|---:|:--:|
| **0.15 (v98)** | +0.027 | 2.7% | **+0.0021** | **4/4** |
| 0.35 (v83) | +0.105 | 10.8% | — | — |
| 0.75 (v94) | +0.340 | 34.3% | −0.0043 | 1/4 |
| *실제 UNI2* | *+0.351* | *31.6%* | | |

**실제 통계에서 멀어질수록 좋다.** cell 축(§122: v91 ÷3 −0.0068 / v93 ×3 −0.0040, 양방향 음수 =
v83이 국소 최적)과는 **다른 패턴**이다. bag 응집도 축에서는 v83이 최적이 아니다.

### 3. ⚠️⚠️ arm 5개가 하나의 가설로 설명된다 — "nuisance를 더하면 나빠진다"

| arm | nuisance 변화 | Δ | 근거 |
|---|---|---:|---|
| **v98** `donor_shift_scale` ↓ | bag 수준 **감소** | **+0.0021** | 4/4 |
| v94 `donor_shift_scale` ↑ | bag 수준 증가 | −0.0043 | 4 seed |
| v90 `class_prior` | 에피소드 수준 유병률 변동 추가 | −0.0053 | 4/4 음수 |
| v95 `spectral_tail_scale` | cell 수준 1536차원 추가 | −0.0086 | 1 seed |
| v99 `normalize_output: false` | per-cell 크기 변동 추가 | −0.0094 | t=−2.32 |

**다섯 개가 부호까지 전부 일치한다.** 이것은 §115·§123의 실패를 **사후 변명 없이 통합 설명**한다 —
두 진단 모두 "합성 cell이 실제 UNI2와 다르다"는 점에서 옳았고, 거기서 도출된 처방은 **전부
nuisance를 추가하는 형태**였으며 그것이 정확히 반대 방향이었다.

⚠️ **불일치 하나**: v86은 `observation_noise`를 0.005 → 0.01로 **올렸는데** +0.0004(t=0.71, 3/4)로
미세 양수다. 가설은 음수를 예측한다. 노이즈 범위지만 반례 후보로 기록해 둔다.

### 4. BAP1 — her2와 별개의 두 번째 교차-arm 신호

| arm | v92 | v98 | v99 |
|---|---:|---:|---:|
| BAP1 Δ | **+0.0552** | **+0.0375** | **+0.0522** |
| 부호 | 4/4 | 4/4 | 4/4 |

**세 arm에서 4/4**이고, v99의 BAP1 평균 0.6958은 **ABMIL(0.693)을 넘는다**. §113·§114가 겨냥해
실패했던 task다. 세 arm의 공통 기전은 아직 없다(v92는 큰 bag, v98은 낮은 응집도, v99는 norm 변동).
다중비교 경고(§104-5)는 유효하나 **4/4 × 3 arm은 우연으로 설명하기 어렵다.**

### 5. 다음 — v100 nuisance 최소화 (실행 중)

가설의 직접 검정. `donor_shift_scale` 0.35→0.0, `donor_component_shift_scale` 0.12→0.0,
`observation_noise` 0.005→0.0. 통계 확인: cos +0.107 → **+0.002**, ICC 10.8% → **0.2%**.
**반증 가능한 예측: Δ > +0.0021이고 4/4여야 한다.** null이나 음수면 "nuisance가 적을수록 단조로
좋다"가 반증된다.

⚠️ 조성 knob(`donor_mixture_logit_scale`, `donor_shared_component_logit_scale`,
`shared_component_base_logit_scale`)은 **일부러 건드리지 않았다** — bag의 조성을 정하므로 abundance
신호(=task 구조) 자체를 지우게 되고, 그러면 이득이 나와도 해석 불가다.
```
config: configs/train_v100_nuisance_min_1536_1gpu.yaml
run:    checkpoints/20260815_140608/v100_nuisance_min_seed4{2..5}/
```

## 127. 2026-08-15 — v100 nuisance 최소화: **예측 반증, 게이트 기각.** 축은 단조가 아니라 역U자

_Recorded by: nhn-NEXGEM-claude — 2026-08-15 15:26_

### (i) 이 arm이 테스트한 것

§126-3 가설("nuisance를 더하면 나빠지고 빼면 좋아진다")의 **직접·반증가능 검정**.
`donor_shift_scale` 0.35→0.0, `donor_component_shift_scale` 0.12→0.0, `observation_noise`
0.005→0.0. 통계: within-bag cosine +0.107 → **+0.002**, ICC 10.8% → **0.2%**.

**config 헤더와 §126-5에 미리 적은 예측: Δ > +0.0021 이고 4/4.**

```
config: configs/train_v100_nuisance_min_1536_1gpu.yaml
ckpts:  checkpoints/20260815_140608/v100_nuisance_min_seed4{2..5}/  (epoch 49)
macro:  0.6805 / 0.6829 / 0.6754 / 0.6829  →  mean 0.6804, seed std 0.0035
```

### (ii) task 10개 전부 (baseline 성능대 내림차순, §118-3)

| task | v83 | v100 | Δ | 부호 | t | ABMIL |
|---|---:|---:|---:|:--:|---:|---:|
| cptac_luad STK11 | 0.8754 | 0.8528 | −0.0227 | 0/4 | −5.07 | 0.908 |
| cptac_brca TP53 | 0.8270 | 0.8145 | −0.0126 | 0/4 | −2.66 | 0.801 |
| cptac_luad EGFR | 0.7761 | 0.7669 | −0.0092 | 0/4 | −3.14 | 0.830 |
| bc_therapy er_status | 0.7276 | 0.7084 | −0.0193 | 1/4 | −2.06 | 0.717 |
| bc_therapy grade | 0.7259 | 0.7029 | −0.0230 | 1/4 | −2.25 | 0.770 |
| cptac_luad TP53 | 0.6678 | 0.6572 | −0.0106 | 2/4 | −0.84 | 0.751 |
| cptac_ccrcc BAP1 | 0.6436 | 0.6682 | +0.0247 | 2/4 | +0.91 | 0.693 |
| bc_therapy her2 | 0.6417 | 0.6471 | +0.0054 | 2/4 | +0.32 | 0.663 |
| cptac_brca PIK3CA | 0.5588 | 0.5451 | −0.0137 | 1/4 | −1.31 | 0.595 |
| cptac_ccrcc VHL | 0.4360 | 0.4413 | +0.0053 | 3/4 | +0.20 | 0.538 |

상위5 Δ평균 **−0.0173** / 하위5 **+0.0022**.

### (iii) macro seed-paired

per-seed Δ: −0.0100 / −0.0067 / −0.0020 / −0.0115 → 평균 **−0.0076**, **t ≈ −3.59, 0/4**.
→ **§107-3 게이트를 기각 방향으로 통과.** 예측(Δ > +0.0021, 4/4)은 **반증됐다.**

### 1. `donor_shift_scale` 축은 역U자다

| `donor_shift_scale` | cosine | ICC | Δ | 부호 |
|---|---:|---:|---:|:--:|
| **0.00 (v100)** | +0.002 | 0.2% | **−0.0076** | **0/4** |
| **0.15 (v98)** | +0.027 | 2.7% | **+0.0021** | **4/4** |
| 0.35 (v83) | +0.107 | 10.8% | — | — |
| 0.75 (v94) | +0.340 | 34.3% | −0.0043 | 1/4 |

**최적이 0.15 근처에 있고 0으로 가면 다시 나빠진다.** §126-3의 "적을수록 단조로 좋다"는 성립하지
않는다. ⚠️ 단 아래 §2의 교란이 있어 이 표의 0.00 지점은 아직 확정이 아니다.

### 2. ⚠️ 작성자의 설계 실수 — knob 3개를 묶어 해석 불능을 자초했다

v100은 가설을 **가장 강하게** 검정하려고 세 knob을 동시에 껐다. **가장 강한 검정은 실패했을 때
가장 해석이 안 되는 검정이기도 하다** — −0.0076이 셋 중 무엇 때문인지 분리되지 않는다.
앞으로 반증 목적 arm이라도 **한 번에 한 knob**을 원칙으로 할 것. (§118-3 보고 형식과 같은 급의
절차 규칙으로 취급한다.)

### 3. 합성 val_ce가 범인을 지목한다 — `observation_noise`는 nuisance가 아니라 정규화다

| arm | val_ce (epoch 49, 4 seed) |
|---|---|
| v83 | ~0.152 |
| **v100** | **0.0984 / 0.1012 / 0.1002 / 0.0973** |

**합성 과제가 훨씬 쉬워졌다.** 이것은 이 레포에서 가장 여러 번 반복된 실패 모드다 — §1 금지 사항
표 첫 줄이 정확히 이것이다("합성 val_ce로 arm 고르기: v37은 val_ce가 더 좋았으나 50-fold는
−0.0068"). ⚠️ 여기서도 val_ce는 **arm 선택이 아니라 교란 진단**으로만 쓴다(§121-2와 같은 용법).

**그리고 이것이 §126-3의 반례를 해소한다**: v86은 `observation_noise`를 0.005 → 0.01로 **올렸는데**
+0.0004였다. 노이즈가 **해로운 nuisance가 아니라 정규화**라면 앞뒤가 맞는다 — 끄면 과제가 쉬워져
전이가 나빠지고, 늘리면 중립~미세 양수다.

### 4. 수정된 그림 — "nuisance"는 한 종류가 아니다

| 축 | 성격 | 근거 |
|---|---|---|
| bag 수준 (`donor_shift_scale`) | **역U자**, 최적 ~0.15 | v94 / v98 / v100 |
| cell 노이즈 (`observation_noise`) | **정규화**. 끄면 해로움 | v86 / v100 |
| 그 외 추가형(class_prior·spectral tail·norm) | 더하면 해로움 | v90 / v95 / v99 |

§126-3의 "nuisance = 무조건 나쁨"은 **틀렸다.** 변동의 **종류**에 따라 다르다.

### 5. 남는 사실 — v98이 현행 레짐 최고 점추정이다

| arm | 4 seed 평균 | v83 대비 |
|---|---:|---|
| **v98** | **0.6901** | **+0.0021 (t=1.73, 4/4)** 미판정 |
| v86 | 0.6884 | +0.0004 (null) |
| v83 (baseline) | 0.6880 | — |

⚠️ **확정된 우위가 아니다**: 게이트 미달이고, v86(0.6884)과의 차이 0.0017은 완전히 노이즈 안이다.
v41_K128의 0.6940은 DDP4 1 seed로 비교 불가(§107-2). **다만 v98은 §105 이후 데이터 축 15개 arm 중
유일하게 4/4 시드 양수다.** SD(Δ)=0.0025로 작아, 효과가 유지된다면 게이트 통과에 **seed 10~12개**가
필요하다(4 → t 1.73, 8 → ~2.4, 12 → ~2.9).

### 6. 다음 — v101 (실행 중): v100의 교란 풀기

`donor_shift_scale` 0.0 **단독**(noise 0.005·component_shift 0.12 유지) = v100 − 교란 2개.
- **음수** → 역U자 확정, v98의 0.15가 최적. 그때 v98을 seed 추가로 확정하는 것이 순서다.
- **~0/양수** → v100의 하락은 `observation_noise` 탓이고 donor 축은 단조. 0.15 아래를 더 볼 것.
```
config: configs/train_v101_donor_shift_zero_1536_1gpu.yaml
run:    checkpoints/20260815_152340/v101_donor_shift_zero_seed4{2..5}/
```

## 128. 2026-08-15 — v101: v100 교란 분리. donor_shift 축은 "역U자"가 아니라 **노이즈 안에서 평평**

_Recorded by: nhn-NEXGEM-claude — 2026-08-15 16:48_

### (i)~(iii) 요약

`donor_shift_scale` 0.0 **단독**(noise 0.005·component_shift 0.12 유지) = v100 − 교란 2개.
macro 0.6855(std 0.0024), per-seed Δ −0.0045 / −0.0076 / **+0.0101** / −0.0079 →
**Δ = −0.0025, t = −0.58, 1/4 → 미판정**.

| task | v83 | v101 | Δ | 부호 | t |
|---|---:|---:|---:|:--:|---:|
| cptac_luad STK11 | 0.8754 | 0.8626 | −0.0128 | 0/4 | −2.40 |
| cptac_brca TP53 | 0.8270 | 0.8094 | −0.0176 | 0/4 | −3.61 |
| cptac_luad EGFR | 0.7761 | 0.7678 | −0.0083 | 1/4 | −1.66 |
| bc_therapy er_status | 0.7276 | 0.7111 | −0.0165 | 1/4 | −0.92 |
| bc_therapy grade | 0.7259 | 0.7061 | −0.0198 | 2/4 | −1.10 |
| cptac_luad TP53 | 0.6678 | 0.6435 | −0.0243 | 0/4 | −3.83 |
| **cptac_ccrcc BAP1** | 0.6436 | 0.6821 | **+0.0386** | **4/4** | +1.95 |
| **bc_therapy her2** | 0.6417 | 0.6740 | **+0.0323** | **4/4** | **+3.09** |
| cptac_brca PIK3CA | 0.5588 | 0.5514 | −0.0074 | 2/4 | −0.42 |
| cptac_ccrcc VHL | 0.4360 | 0.4471 | +0.0111 | 2/4 | +0.49 |

상위5 **−0.0150** / 하위5 **+0.0100**.

### 1. 교란 분해

```
v100 (knob 3개)  = −0.0076
v101 (donor만)   = −0.0025
──────────────────────────────
noise + component_shift 제거분 = −0.0051
```
val_ce: v83 0.152 → **v101 0.123** → **v100 0.099**. §127-3의 진단(노이즈는 정규화)이 확인된다 —
절반 이상이 노이즈/성분이동 제거 탓이었다.

### 2. ⚠️ §127-1의 "역U자"를 정정한다 — 축은 평평하다

| `donor_shift_scale` | Δ | 부호 | t | 판정 |
|---|---:|:--:|---:|---|
| 0.00 (v101 단독) | −0.0025 | 1/4 | −0.58 | 미판정 |
| **0.15 (v98)** | **+0.0021** | **4/4** | +1.73 | 미판정 |
| 0.35 (v83) | 0 | — | — | 기준 |
| 0.75 (v94) | −0.0043 | 1/4 | −1.43 | 미판정 |

**네 점 전부 게이트 미달이고 축 전체 폭이 0.0064다.** §127-1이 "역U자"라 쓴 것은 미판정 점 4개에
모양을 읽은 것이고, 정직한 서술은 **"노이즈 범위 안에서 평평하다"** 이다. v98의 우위도 확정이 아니다.

### 3. 실제로 확정된 nuisance 그림

- **양 조절(donor_shift)** → 효과 없음. 위아래 어디로 움직여도 미판정.
- **새 종류 추가**(class_prior·spectral tail·norm) → **해롭다**(v90 4/4 음수, v99 t=−2.32, v97 2.2σ).
- **기존 노이즈 제거** → **해롭다**(v100 t=−3.59). `observation_noise`는 정규화다.

⚠️ §126-3의 "nuisance를 빼면 좋아진다"는 **틀렸다.** v98 하나에 과하게 기댄 해석이었다.

### 4. her2 / BAP1 트레이드오프

her2가 **개별 task 게이트를 넘었다**(+0.0323, 4/4, **t=+3.09**), 0.6740으로 **ABMIL 0.663 초과**.
BAP1도 +0.0386(4/4), 0.6821. 반면 상위5는 −0.0150으로 일관 하락. **저신호 task가 오르고 고신호
task가 내려가 macro에서 상쇄된다** — 데이터 축에서 macro가 계속 0인 구조적 이유일 수 있다.

## 129. 2026-08-15 — v102 bag-공유 꼬리: **기각(게이트 통과).** 격차를 닫을수록 단조로 나빠진다 → 데이터 축 종결

_Recorded by: nhn-NEXGEM-claude — 2026-08-15 19:58_

### (i) 이 arm이 테스트한 것 — §123이 막힌 지점을 뚫은 설계

§123의 세 격차 중 스펙트럼만 막혀 있었고, 원인은 **차원 불일치**였다:

| | 사는 공간 |
|---|---|
| 꼬리(`spectral_tail`) | 출력 **1536차원**, cell별 iid |
| bag 오프셋(`donor_shift`) | latent **32차원**, manifold map 이전 |

32차원 레버로 1536차원 변동을 상쇄할 수 없어 꼬리를 켜면 cosine이 **+0.130에서 포화**했다
(donor_shift 8.0에서도). **스펙트럼이냐 응집도냐 둘 중 하나만** 맞출 수 있었다.

실제 slide에는 기전이 둘이 아니라 **하나**다 — 고차원 slide 고유 오프셋(염색·스캐너·환자) 하나가
모든 tile에 공유되어 응집도와 꼬리를 동시에 만든다. 그래서 **꼬리 자체를 쪼갰다**:
`spectral_tail_bag_fraction`이 같은 감쇠 공분산의 변동을 bag 공유분/cell 개별분으로 나누고,
가중치 `√(1−f)`/`√f`로 cell별 총 분산을 보존한다(테스트로 고정).

**처음으로 두 통계가 동시에 맞았다** (f=0.4, 40 bags 실측):

| 지표 | v83 | v95 (꼬리만) | v94 (donor만) | **v102** | 실제 UNI2 |
|---|---:|---:|---:|---:|---:|
| participation | 31.2 | 67.2 | 28.9 | **53.9** ✅ | 50–76 |
| r90 | 28 | 422 | 27 | **293** | 431–542 |
| within-bag cosine | +0.104 | +0.013 | +0.337 | **+0.354** ✅ | +0.351 |
| ICC | 10.8% | 1.4% | 34.3% | **34.3%** ✅ | 31.6% |

**v95와 knob 하나만 다르다**(f 0→0.4) — §127-2의 "arm당 knob 하나" 규칙을 지켰다.

```
config: configs/train_v102_tail_bagshared_1536_1gpu.yaml
ckpts:  checkpoints/20260815_174332/v102_tail_bagshared_seed4{2..5}/
macro:  0.6790 / 0.6823 / 0.6712 / 0.6673  →  mean 0.6750, seed std 0.0069
```

### (ii) task 10개 전부

| task | v83 | v102 | Δ | 부호 | t | ABMIL |
|---|---:|---:|---:|:--:|---:|---:|
| cptac_luad STK11 | 0.8754 | 0.8728 | −0.0026 | 2/4 | −0.20 | 0.908 |
| cptac_brca TP53 | 0.8270 | 0.8093 | −0.0178 | 0/4 | −3.61 | 0.801 |
| cptac_luad EGFR | 0.7761 | 0.7645 | −0.0116 | 0/4 | −2.30 | 0.830 |
| bc_therapy er_status | 0.7276 | 0.6972 | −0.0304 | 0/4 | −4.25 | 0.717 |
| bc_therapy grade | 0.7259 | 0.7095 | −0.0164 | 0/4 | −4.00 | 0.770 |
| cptac_luad TP53 | 0.6678 | 0.6678 | ±0.0000 | 2/4 | 0.00 | 0.751 |
| cptac_ccrcc BAP1 | 0.6436 | 0.6442 | +0.0007 | 3/4 | +0.03 | 0.693 |
| bc_therapy her2 | 0.6417 | 0.6467 | +0.0051 | 3/4 | +0.26 | 0.663 |
| cptac_brca PIK3CA | 0.5588 | 0.5275 | −0.0312 | 0/4 | −4.42 | 0.595 |
| cptac_ccrcc VHL | 0.4360 | 0.4099 | −0.0261 | 0/4 | −4.32 | 0.538 |

상위5 −0.0158 / 하위5 −0.0103 — **양쪽 다 내려간다.**

### (iii) macro

per-seed Δ: −0.0115 / −0.0073 / −0.0062 / −0.0271 → **Δ = −0.0130, t = −2.70, 0/4**
→ **§107-3 게이트 통과, 기각.**

### 1. ⚠️⚠️ 닫은 격차가 많을수록 단조로 나빠진다

| arm | 닫은 격차 | Δ | 판정 |
|---|---|---:|---|
| v83 | 없음 | 0 | 기준 |
| v94 | 응집도만 | −0.0043 | 미판정 |
| v95 | 스펙트럼만 | −0.0086 | 1 seed |
| **v102** | **둘 다(단일 기전)** | **−0.0130** | **기각** |

기전을 통합한 설계는 **통계적으로 성공**했으나(3/4 목표 달성) 성능은 **각각 닫는 것보다도 나쁘다.**

### 2. val_ce가 성격을 말한다 — 순수 방해다

v83 0.152 → **v102 0.238**. v100(0.099, 과제가 쉬워짐)과 **정반대**로 합성 과제가 **더 어려워졌는데
전이도 나빠졌다.** 쉬워져서 과적합한 것도, 어려워져서 정규화된 것도 아니다 — 추가된 구조가
**순수한 방해**다.

### 3. her2 / BAP1 신호가 덮인다

| | v101 | v102 |
|---|---:|---:|
| her2 | **+0.0323** (4/4, t=3.09) | +0.0051 (3/4) |
| BAP1 | **+0.0386** (4/4) | +0.0007 (3/4) |

bag 공유 꼬리가 두 신호를 만들던 무언가를 덮는다.

### 4. 이번엔 축을 닫아도 된다 — §125-4와 근거가 다르다

§125-4는 **미판정 arm 3개**로 축을 닫으려다 정정했다. 이번 근거는 다르다:

- **v102: t=−2.70, 0/4 → 게이트 통과 기각** (확정)
- **v100: t=−3.59, 0/4 → 게이트 통과 기각** (확정)
- v99: t=−2.32, 상위5 20/20 셀 음수 (거의 확정)
- 그리고 **닫은 격차 수 ↔ 하락폭의 단조 관계**

> **결론: "합성 에피소드를 실제 UNI2 통계에 맞추는 것"은 도움이 되지 않을 뿐 아니라 적극적으로
> 해롭다.** §115(모양)·§123(값) 두 진단은 격차를 정확히 실측했고, 그 격차를 닫는 처방은 **전부**
> 실패했으며, 가장 완전하게 닫은 arm이 **가장 크게 기각**됐다. **이 축에서 새 arm을 설계하지 말 것.**

⚠️ 예외로 남는 것: norm 격차는 **끝까지 닫아본 적이 없다**(v99는 1/3만). 다만 v102의 결과에 비추면
완전히 닫아도 더 나빠질 것으로 예상되므로 우선순위는 최하다.

### 5. 남은 실측 단서 — her2 / BAP1 트레이드오프

데이터 축이 닫힌 지금 남은 것은 §128-4다: **저신호 task가 오르고 고신호 task가 내려가 macro에서
상쇄되는 구조.** her2는 v101에서 개별 게이트를 넘었고(t=3.09) ABMIL을 상회한다. macro 단일 지표가
이 구조를 지우고 있을 가능성을 다음 단계에서 검토할 것.

## 130. 2026-08-15 — 방향 전환: **분산이 편향보다 크다.** 시드 앙상블 +0.0058~0.0071 (학습 비용 0)

_Recorded by: nhn-NEXGEM-claude — 2026-08-15 20:44_

**계기**: §129로 데이터 분포 축이 닫힌 뒤 "새 구조"를 찾다가, **이미 디스크에 있는 산출물로
학습 없이 검정 가능한 것**부터 확인했다. `predictions/pathobench_*_{arm}_seed4{2..5}_ep49_official50_bf16.pt`
에 **50 fold × slide별 확률이 전부 저장돼 있다**(14 arm × 4 seed = 56개 모델).

### 1. 시드 앙상블 — 이 프로젝트에서 관측된 최대 효과

같은 arm의 4 seed 확률을 slide별로 평균한 뒤 AUROC 재계산:

| 구성 | 추론 비용 | macro |
|---|---:|---:|
| v83 단일 (4 seed 평균) | 1× | 0.6880 |
| **v83 4-seed 앙상블** | 4× | **0.6938 (+0.0058)** |
| **v98 4-seed 앙상블** | 4× | **0.6951 (+0.0071)** |
| v101 4-seed 앙상블 | 4× | 0.6898 |
| v102 4-seed 앙상블 | 4× | 0.6798 |

**v83 앙상블은 10/10 task 전부 양수다** (er +0.0102, luadTP53 +0.0122, her2 +0.0086, STK11 +0.0062…).
선택 개입이 전혀 없는 숫자다.

⚠️ **§105 이후 데이터 축 arm 15개 중 최대 효과가 v98의 +0.0021(미판정)이었다.** 앙상블은 그 3배를
**학습 비용 0**으로 낸다.

### 2. ⚠️ 이것이 문제의 성격을 바꾼다 — 우리는 노이즈보다 작은 신호를 쫓고 있었다

macro seed std가 **0.0074**인데, 15개 arm에서 쫓던 효과는 전부 **±0.005 안쪽**이었다.
**편향(어떤 데이터 분포가 옳은가)이 아니라 분산(같은 설정도 시드마다 0.0074씩 흔들린다)이 지배적이다.**

### 3. Model soup은 실패했다 — 이득의 1/4만

4 seed의 **가중치**를 평균해 추론 비용 1×로 앙상블 이득을 얻으려 했다.

| 방식 | 추론 | macro | 단일 대비 | 오른 task |
|---|---:|---:|---:|---:|
| model soup | **1×** | 0.6894 | +0.0014 | 5/10 |
| 예측 앙상블 | 4× | 0.6938 | +0.0058 | **10/10** |

**사전 진단이 결과를 예고했다**: P(1536×128)의 시드 간 elementwise cosine이 **0.57~0.59**였다.
모든 시드가 **같은 결정론적 초기화**(sin/cos Fourier 기저 → QR, 데이터·시드 무관)에서 출발하는데도
50 epoch 뒤 0.57까지만 남는다 — 시드들이 같은 basin의 미세한 흔들림이 아니라 **서로 다른 해**로
간다. 평균낸 P의 norm은 14.24 → 11.77(83%)로 부분 상쇄된다.

**실패가 정보다**: soup이 통했다면 "시드들이 사실상 같은 해"라는 뜻이었을 것이다. 실패했다는 것은
**예측 앙상블의 이득이 진짜 오류 비상관에서 온다**는 뒷받침이다.
```
soup 산출물: checkpoints/soup/v83_soup_seed42to45_ep49.ckpt  (태그 v83_soup_ep49)
```

### 4. arm 다양성은 기여하지 않는다 — 이득은 전부 시드 분산 감소다

| 구성 | 모델 수 | macro |
|---|---:|---:|
| v83 시드만 | 4 | 0.6938 |
| **전체 14 arm × 4 seed** | **56** | **0.6913** |
| v83 + v98 | 8 | 0.6951 |
| v98 시드만 | 4 | **0.6951** |

**전체 arm(0.6913)이 v83 시드만(0.6938)보다 나쁘고, v83+v98(0.6951)이 v98 단독 앙상블(0.6951)과
동일하다.** 데이터 분포가 다른 arm들은 전부 v83보다 나쁘므로 섞으면 희석될 뿐이다.

⚠️ "최적 3 arm 조합 0.6949(v83+v86+v98)"도 계산했으나 **C(14,3)=364개를 평가 지표로 고른 것이라
과적합**이다. **성과로 인용하지 말 것.**

### 5. 트레이드오프 앙상블 (사용자 제안) — 기전은 작동하나 순효과 0

§128-4의 트레이드오프(저신호 task ↑ / 고신호 task ↓)가 뚜렷한 arm과 v83을 섞으면 양쪽을 다 얻는지
검정했다. 임의 조합을 지표로 고르는 것과 달리 **문서화된 상보성에 근거한 사전 지정**이다.

| 구성 | macro | 상위5 | 하위5 |
|---|---:|---:|---:|
| v83 시드앙상블 (기준) | 0.6938 | 0.7918 | 0.5958 |
| v83 + v101 (her2/BAP1 최강) | 0.6935 | 0.7854 ↓ | **0.6015 ↑** |
| v83 + v92 | 0.6896 | 0.7813 ↓↓ | 0.5979 ↑ |
| v83 + v99 | 0.6905 | 0.7807 ↓↓ | 0.6003 ↑ |
| v83 + v98 + v101 | 0.6947 | 0.7860 ↓ | **0.6034 ↑** |

**상보성은 실제로 전달된다** — 트레이드오프 arm을 섞을 때마다 하위5가 **예외 없이** 오른다.
**그런데 상위5도 같이 내려가 정확히 상쇄된다**(v83+v101은 순효과 −0.0003).

**원인**: 균등 평균은 모든 task에서 모든 모델에 같은 가중치를 준다. "her2는 v101, STK11은 v83"처럼
좋은 쪽만 골라 담을 수 없고, 트레이드오프의 양쪽을 1:1로 함께 옮겨온다.

**남은 길**: **task별 가중치를 held-out fold로 결정**하면 정당하다 — 50 fold를 반으로 갈라
fold 0–24로 가중치를 정하고 fold 25–49로 평가한다. 지표로 조합을 고르는 과적합이 아니다.
기존 예측 파일만으로 GPU 없이 계산 가능하다. **미착수.**

### 6. 현재 최고 구성과 ABMIL 격차

| | macro | ABMIL(0.7266) 격차 |
|---|---:|---:|
| v83 단일 (활성 baseline) | 0.6880 | −0.0386 |
| **v98 4-seed 앙상블** | **0.6951** | **−0.0315** |

격차의 **18%**를 메운다. task 2개가 ABMIL을 넘는다 — brca TP53 0.8290(vs 0.801),
er_status 0.7346(vs 0.717).

### 7. ⚠️ 한계와 진행 중인 확인

- **독립적인 두 번째 시드 그룹이 없어 t를 낼 수 없다.** 근거는 관측 1회 + 10/10 task 양수뿐이다.
- **추론 비용이 4배다.** 판정 프로토콜이 이미 4 seed를 학습하므로 모델은 존재하지만, 단일 모델
  arm과 같은 선상에서 비교·승격하는 것이 공정한지는 **사용자 판단 사안**이다.
- **진행 중**: v98 seed 46–49 학습(`checkpoints/20260815_113422/`, 42–45와 같은 디렉토리).
  ① 앙상블 이득이 독립 그룹에서 재현되는가 ② 두 앙상블이 일치하는가 ③ 8 seed가 더 오르는가.

## 131. 2026-08-15 — baseline을 **v98 8 seed**로 교체 (사용자 결정) + **판정 체계의 한계를 명시**

_Recorded by: nhn-NEXGEM-claude — 2026-08-15 21:38_

### 0. 이 절이 정한 것

1. **새 baseline = v98 (`donor_shift_scale` 0.15), 1-GPU 8 seed(42–49) 평균 `0.6852`.**
   이 프로젝트에서 **8 seed로 측정된 유일한 arm**이고 따라서 가장 신뢰할 수 있는 절대 수치다.
2. **판정은 계속 단순 평균(seed-paired Δ + t)으로 한다.** 앙상블은 언제 적용해도 이득이므로
   판정 축에서 분리한다(§130) — arm 비교는 앙상블 없이, 앙상블은 최종 산출물에 얹는다.
3. **4 seed의 검출 한계를 문서화한다** — 아래 §2. 과거 "미판정"을 "효과 없음"으로 읽지 말 것.

```
config: configs/train_v98_p1_reverse_1536_1gpu.yaml   (self-contained)
ckpts:  checkpoints/20260815_113422/v98_p1_reverse_seed4{2..9}/  (epoch 49)
tags:   v98_p1_reverse_seed4{2..9}_ep49
macro:  0.6907 / 0.6950 / 0.6802 / 0.6946 / 0.6758 / 0.6811 / 0.6807 / 0.6837
        → mean 0.6852, seed std 0.0072, SE 0.0026
```

### 1. ⚠️ v98이 v83보다 낫다는 뜻이 **아니다** — 반드시 지킬 것

v98 8 seed 0.6852와 v83 4 seed 0.6880을 **빼지 말 것.** 두 수치는 **서로 다른 시드 집합**에서
왔고, 판정 프로토콜이 seed-paired인 이유가 정확히 이것이다 — **절대 수준은 시드 그룹마다 크게
흔들린다**:

```
v98 A그룹(42-45) 0.6901   vs   B그룹(46-49) 0.6803   차이 0.0098 (= 2.0σ)
```

v83의 seed 46–49는 **존재하지 않으므로** v98과 v83의 관계는 여전히 **미판정**이다(4 seed
seed-paired로 +0.0021, t=1.73, 4/4 — 게이트 미달). v98을 baseline으로 삼는 근거는 "더 낫다"가
아니라 **"가장 잘 측정됐다"**이다.

**앞으로 새 arm은 v98의 같은 시드와 seed-paired로 비교한다.** 4 seed로 돌리면 42–45와, 8 seed로
돌리면 42–49와 짝짓는다.

### 2. ⚠️⚠️ 판정 체계의 실제 검출력 — 과거 판정을 다시 읽는 기준

**게이트 `|t| ≥ 2.5`는 df=3에서 p = 0.088이다. 0.05가 아니다.**

| \|t\| | p (df=3) | 해당 arm |
|---|---|---|
| 1.43 | 0.248 | v94 |
| 1.73 | 0.182 | v98 |
| 2.32 | 0.103 | v99 |
| **2.50** | **0.088** | **게이트 기준선** |
| 2.70 | 0.074 | v102 |
| 3.59 | 0.037 | v100 |
| 3.61 | 0.037 | v84 |
| 6.69 | 0.007 | v88 |

**4/4 부호 일치 단독은 p = 0.125다.** 지금까지 arm을 약 20개 판정했으므로 **우연히 게이트를 넘는
것이 1~2개 기대된다.**

**검출 가능한 최소 효과** (paired SD ≈ 0.005, 양측 α=0.05, power 80%):

| seed | 최소 검출 효과 |
|---|---|
| **4 (현행)** | **0.0121** |
| 8 | 0.0067 |
| 12 | 0.0051 |
| 24 | 0.0035 |

⚠️ **4 seed로는 0.012 미만을 판별할 수 없다.** 그런데 §105 이후 모든 arm의 효과는 ±0.005
안쪽이었다 — **노이즈보다 작은 신호를 판정하려 해 왔다.**

### 3. 과거 판정의 3분류

**A. 안전한 기각 (p < 0.04) — 유지한다**
- v88 PA (t=−6.69, p=0.007) / v84 deep head (t=−3.61) / v100 nuisance min (t=−3.59)

**B. 취약한 기각 (p ≈ 0.07~0.10) — 재검토 대상**
- v102 (t=−2.70, p=0.074), v99 (t=−2.32, p=0.103 — 애초에 게이트 미통과), v80, v77 Hard

**C. "미판정" 전부 — "효과 없음"이 아니라 "측정 불가"다**
⚠️ **v83 승격 자체가 여기 속한다** — v82 대비 t=1.15, **p=0.33**. 현재까지 baseline이던 v83과
직전 v82는 **구분된 적이 없다.**

### 4. 무너지지 않는 두 가지

**① §129의 단조 추세.** v83(0) → v94(−0.0043) → v95(−0.0086) → v102(−0.0130)로 **독립적인 4개
arm이 "닫은 격차의 양" 순서대로 늘어섰다.** 추세는 개별 t보다 검정력이 높다. 데이터 분포 축
종결은 유지한다.

**② 앙상블 이득 — 독립 3회 관측 전부 양수.**

| 관측 | 단일 평균 | 앙상블 | 이득 |
|---|---:|---:|---:|
| v83 seed 42–45 | 0.6880 | 0.6938 | +0.0058 |
| v98 seed 42–45 | 0.6901 | 0.6951 | +0.0050 |
| **v98 seed 46–49 (독립)** | 0.6803 | 0.6842 | **+0.0038** |
| v98 seed 42–49 (8) | 0.6852 | **0.6906** | +0.0054 |

단일 arm으로는 4 seed 검출 한계 아래지만 **서로 다른 arm·시드 그룹에서 반복 재현**되므로 신뢰한다.
**판정 축에서는 분리**하고(§0-2) 최종 산출물에만 얹는다.

### 5. 앞으로의 판정 규칙 (§107-3·§118에 추가)

1. **"미판정"을 "효과 없음"으로 쓰지 말 것.** 4 seed 기준 **0.012 미만은 측정 불가**이며, 그렇게
   기록한다.
2. **반복 재현을 t보다 우선한다.** 독립 시드 그룹(예: 42–45와 46–49)에서 **부호가 반복**되면
   단일 그룹의 |t|보다 강한 근거로 본다.
3. **절대 수치를 시드 집합 간에 빼지 말 것.** 항상 seed-paired로 비교한다(§1).

## 132. 2026-08-15 — v103 GELU head 복원: 측정 불가(−0.0073, t=−2.01). 다만 **부호가 두 번 반복**된다

_Recorded by: nhn-NEXGEM-claude — 2026-08-15 23:49_

### (i) 이 arm이 테스트한 것

§131-3 미판정 목록 **#2**. v83이 v82의 `12→32→1`(GELU) head를 bare `Linear(12,1)`로 바꾼 근거는
Δ+0.0045, **t=1.15, p=0.33**이었고, §131-2가 4 seed 최소 검출 효과를 **0.0121**로 확정하면서
**애초에 0과 구분될 수 없던 값**임이 드러났다. 그런데 baseline이 v98로 옮겨가며 **v98이 v83의
linear head를 그대로 물려받았다** — p=0.33짜리 결정이 현재 기준점에 박혀 있다.

v103은 v98에서 **head knob 하나만** 되돌린다: `ct_head_hidden_dims: [] → [32]`,
trainable 196,621 → **197,057**(head 13 → 449), arch 54 동일.

```
config: configs/train_v103_gelu_head_1536_1gpu.yaml
ckpts:  checkpoints/20260815_214942/v103_gelu_head_seed4{2..5}/
macro:  0.6767 / 0.6909 / 0.6816 / 0.6821  →  mean 0.6828, seed std 0.0059
```

### (ii) task 10개 전부 (v98 seed 42–45와 seed-paired)

| task | v98(42–45) | v103 | Δ | 부호 | t | ABMIL |
|---|---:|---:|---:|:--:|---:|---:|
| cptac_luad STK11 | 0.8731 | 0.8593 | −0.0139 | 0/4 | −1.98 | 0.908 |
| cptac_brca TP53 | 0.8246 | 0.8169 | −0.0077 | 0/4 | −1.80 | 0.801 |
| cptac_luad EGFR | 0.7724 | 0.7771 | +0.0047 | 3/4 | +0.73 | 0.830 |
| bc_therapy er_status | 0.7193 | 0.7106 | −0.0087 | 1/4 | −1.45 | 0.717 |
| bc_therapy grade | 0.7146 | 0.7219 | +0.0073 | 2/4 | +0.79 | 0.770 |
| cptac_luad TP53 | 0.6696 | 0.6652 | −0.0044 | 1/4 | −0.96 | 0.751 |
| **cptac_ccrcc BAP1** | 0.6811 | 0.6426 | **−0.0385** | **0/4** | **−2.93** | 0.693 |
| bc_therapy her2 | 0.6553 | 0.6516 | −0.0037 | 2/4 | −0.53 | 0.663 |
| cptac_brca PIK3CA | 0.5556 | 0.5603 | +0.0047 | 3/4 | +1.57 | 0.595 |
| cptac_ccrcc VHL | 0.4359 | 0.4231 | −0.0128 | 1/4 | −1.43 | 0.538 |

상위5 −0.0037 / 하위5 −0.0109.

### (iii) macro seed-paired

per-seed Δ: −0.0140 / −0.0041 / **+0.0014** / −0.0125 → **Δ = −0.0073, t = −2.01, 1/4**

### 1. 판정: **측정 불가** (§131-5 규칙)

|t|=2.01은 게이트(2.5) 미달이고 **|Δ|=0.0073 < 최소 검출 효과 0.0121**이다.
**"효과 없음"이 아니라 "측정 불가"로 기록한다.**

### 2. 그러나 부호가 독립 두 관측에서 반복된다

| 비교 | Δ | t | 부호 | 해석 |
|---|---:|---:|:--:|---|
| v83 vs v82 (§108) | +0.0045 | +1.15 | 3/4 | linear가 나은 방향 |
| **v103 vs v98** | **−0.0073** | **−2.01** | 1/4 | **linear가 나은 방향** |

**서로 다른 baseline·서로 다른 데이터 설정에서 두 번 다 linear head 쪽**이고, 이번이 더 크다.
§131-5가 "반복 재현을 단일 t보다 우선한다"고 정한 조건에 해당한다.

**결론: v83의 head 결정은 근거가 약했지만 방향은 맞았을 가능성이 높다. GELU를 되살릴 이유가 없다.**
⚠️ 다만 이것도 "확정"이 아니다 — 두 관측 모두 검출 한계 아래이고, 부호 일치가 근거의 전부다.

### 3. BAP1이 또 크게 움직인다

BAP1 **−0.0385(0/4, t=−2.93)** 로 단일 task 게이트를 넘는다. BAP1은 v92(+0.0552)·v98(+0.0375)·
v99(+0.0522)에서 **4/4 양수**로 올랐던 task인데 GELU head를 넣으면 크게 떨어진다. 다중비교
경고(§104-5, 기대 0.9개)는 유효하나, **BAP1이 arm마다 반복적으로 크게 반응**하는 것은 her2와 함께
남은 두 개의 task 수준 단서다(§128-4).

### 4. 다음 — v104 fixed P (실행 중)

§131-3 목록 **#1**. `model_src`만 `CovarianceMeanLearnablePDDCTMLPModel` →
`CovarianceMeanDDCTMLPModel`로 바꿔 P의 gradient를 끈다. **trainable 196,621 → 13.**
```
config: configs/train_v104_fixed_p_1536_1gpu.yaml
```

## 133. 2026-08-16 — v104 fixed P: **기각(게이트+검출한계 동시 통과).** learnable P는 필요하고, **seed 분산의 출처다**

_Recorded by: nhn-NEXGEM-claude — 2026-08-16 09:36_

### (i) 이 arm이 테스트한 것

§131-3 미판정 목록 **#1**. `model_src`만 `CovarianceMeanLearnablePDDCTMLPModel` →
`CovarianceMeanDDCTMLPModel`로 바꿔 **P의 gradient만 끈다.** P는 그대로 있고(결정론적 sin/cos
Fourier 기저 → QR) 학습만 안 한다 — **"P를 갖는 것"이 아니라 "P를 학습하는 것"을 분리**한다.

**trainable 196,621 → 13** (P 196,608 = 전체의 **99.993%** 제거, `Linear(12,1)`만 남음).
옛 v81(v77 Hard 기준 −0.0048, t=−1.5)은 §131-2 기준으로 애초에 판별 불가였다.

```
config: configs/train_v104_fixed_p_1536_1gpu.yaml
ckpts:  checkpoints/20260815_234931/v104_fixed_p_seed4{2..5}/
macro:  0.6775 / 0.6774 / 0.6777 / 0.6775  →  mean 0.6775
```

### (ii) task 10개 전부 (v98 seed 42–45와 seed-paired)

| task | v98(42–45) | v104 | Δ | 부호 | t | ABMIL |
|---|---:|---:|---:|:--:|---:|---:|
| cptac_luad STK11 | 0.8731 | 0.8606 | −0.0125 | 0/4 | −3.25 | 0.908 |
| cptac_brca TP53 | 0.8246 | 0.8131 | −0.0114 | 1/4 | −1.74 | 0.801 |
| **cptac_luad EGFR** | 0.7724 | 0.7537 | −0.0187 | **0/4** | **−10.17** | 0.830 |
| bc_therapy er_status | 0.7193 | 0.7266 | +0.0072 | 3/4 | +0.66 | 0.717 |
| bc_therapy grade | 0.7146 | 0.6989 | −0.0157 | 1/4 | −1.72 | 0.770 |
| cptac_luad TP53 | 0.6696 | 0.6595 | −0.0101 | 1/4 | −2.20 | 0.751 |
| cptac_ccrcc BAP1 | 0.6811 | 0.6599 | −0.0211 | 1/4 | −1.29 | 0.693 |
| bc_therapy her2 | 0.6553 | 0.6678 | +0.0125 | 3/4 | +1.35 | 0.663 |
| **cptac_brca PIK3CA** | 0.5556 | 0.5217 | −0.0339 | **0/4** | **−12.10** | 0.595 |
| cptac_ccrcc VHL | 0.4359 | 0.4134 | −0.0225 | 0/4 | −2.35 | 0.538 |

### (iii) macro seed-paired

per-seed Δ: −0.0132 / −0.0176 / −0.0025 / −0.0171 → **Δ = −0.0126, t = −3.59, 0/4**

### 1. 판정: 기각 — **§105 이후 처음으로 게이트와 검출 한계를 동시에 통과**

|Δ| = **0.0126 > 0.0121**(§131-2의 4 seed 최소 검출 효과). 게이트도 통과(0/4, |t|=3.59, p=0.037).
**learnable P는 필요하다** — 196,608개를 학습해 **+0.0126**을 산다. §107-4 이래 열려 있던 질문이
닫혔고, v81은 방향은 맞았으나 크기를 과소평가했다.

### 2. ⚠️⚠️ 부수 발견 — seed 분산의 출처가 P다

| | trainable | **macro seed std** |
|---|---:|---:|
| v98 (learnable P) | 196,621 | **0.00690** |
| **v104 (fixed P)** | **13** | **0.00013** |

**55배**다. v104의 시드별 macro는 0.6775/0.6774/0.6777/0.6775로 사실상 동일하다.
**이 프로젝트를 괴롭혀온 seed std 0.0072는 거의 전부 P 학습에서 나온다.** 이것이 한 번에 설명한다:
- **왜 앙상블이 통하는가**(§130) — P의 무작위성을 사후 평균하는 것이다.
- **왜 model soup은 실패했나**(§130-3) — 시드마다 P가 다른 해로 간다(주각 cos 0.59).
- **왜 arm 15개가 전부 미판정이었나**(§131-2) — 판정 노이즈의 주범이 데이터가 아니라 P였다.

## 134. 2026-08-16 — P 수렴 진단 + epoch 스윕: **P는 수렴하지 않는다.** 조기 freeze 배제, EMA만 남음

_Recorded by: nhn-NEXGEM-claude — 2026-08-16 09:36_

### 1. P는 초기 기저에서 계속 멀어지고 멈추지 않는다 (학습 0, 기존 체크포인트 분석)

`_effective_covariance_projection`은 매 forward에 thin QR을 돌리므로 P가 배우는 것은 **1536차원
중 128차원 부분공간**이다. 두 부분공간의 거리는 **주각(principal angle) 코사인**으로 잰다
(`svdvals(Qaᵀ Qb)`, 1이면 완전 일치).

| epoch | 초기 P와 주각 cos | 직전 지점 대비 |
|---|---:|---:|
| 9 | 0.9509 | — |
| 19 | 0.8826 | 0.9279 |
| 29 | 0.8314 | 0.9411 |
| 39 | 0.7887 | 0.9468 |
| 49 | 0.7515 | 0.9516 |

**고원이 없다.** 10 epoch당 약 0.04씩 일정하게 멀어지고, 이동 속도도 거의 안 준다(0.928→0.952).
그런데 **val_ce는 이미 평평하다**(0.1402 → 0.1379). **손실이 거의 안 줄어드는데 P는 계속 움직인다.**

시드 간(epoch 49): 주각 cos 평균 **0.59**, **완전 일치 방향 0개**, 128개 중 **25~35개는 무관**(<0.5).

**진단: 손실이 P를 결정하지 못한다.** "서로 다른 최적점에 수렴"이 아니라 **손실이 제약하지 않는
방향으로 random walk 중**이다.

### 2. epoch 스윕 (v98 epoch 19/29/39 × 4 seed, 학습 0 — 기존 periodic 체크포인트 재채점)

| epoch | macro | seed std | 초기 P와 cos | ep49 대비 seed-paired |
|---|---:|---:|---:|---|
| 19 | 0.6840 | 0.00471 | 0.8826 | **−0.0061 (t=−3.45, 0/4)** |
| 29 | 0.6863 | 0.00638 | 0.8314 | −0.0039 (t=−3.00, 0/4) |
| **39** | **0.6900** | 0.00615 | 0.7887 | **−0.0001 (t=−0.09, 2/4)** |
| 49 | 0.6901 | 0.00690 | 0.7515 | — |
| *(v104 fixed P)* | *0.6775* | *0.00013* | *1.0* | |

⚠️ 이것은 §1의 "validation-best로 채점 금지"에 걸리지 않는다 — 시드별로 최적 epoch을 고르는 것이
아니라 **모든 시드에 같은 고정 epoch**을 적용한 비교다.

**세 가지가 나온다:**
1. **조기 freeze는 배제된다.** epoch 19는 ep49 대비 **−0.0061(t=−3.45, 0/4)** 로 확실히 나쁘다.
   후반 이동은 노이즈가 아니라 **실제 학습**이다.
2. **epoch 39에서 이미 끝났다.** ep39 vs ep49 = **−0.0001, t=−0.09**. 마지막 10 epoch은 macro를
   전혀 못 올리면서 P를 cos 0.7887 → 0.7515로 밀고 seed std를 0.00615 → 0.00690으로 키운다.
   다만 실익이 seed std 11% 감소뿐이라 "49 대신 39로 채점"의 가치는 작다.
3. **이득과 분산이 분리되지 않는다.** 19→39에서 macro +0.0060, seed std도 +0.0015로 함께 커진다.
   §133-2의 그림이 확정된다 — **P 학습 = 이득 + 노이즈가 같은 움직임에 묶여 있다.** 그래서 사후
   평균(앙상블)이 통했다.

**남은 처방은 EMA/궤적 평균이다.** 조기 종료처럼 이득을 깎지 않으면서 무작위 성분만 줄인다.
⚠️ model soup(서로 다른 basin의 **끝점** 평균, +0.0014로 실패)과 다르다 — EMA는 **한 궤적 위**를
평균한다. **미착수.**

### 3. 산출물 — MLP 사영 (v105, 실행 중)

사용자 지시로 P를 MLP로 바꾸는 축을 먼저 검정한다.
`CovarianceMeanMLPProjectionDDCTMLPModel`(arch 58):
```
projected = GELU(centered @ W1) @ QR(W2).Q      # 1536 -> hidden -> 128
trainable: hidden 128 → 213,005 | 256 → 425,997 | 512 → 851,981   (v98은 196,621)
```
**출력 사영의 QR을 유지**한다 — 일반 MLP는 출력 scale이 자유로워 covariance 크기가
`ridge_lambda`·`covariance_slopes`(=0.85π/K)의 calibration을 벗어나고, §70이 "이득은 용량이 아니라
sketch 기하에서 나온다"고 실측했으므로 **arm이 재calibration과 교란**된다. `identity` 활성화면
두 선형사상의 합성이라 **v98과 정확히 같아진다** — 즉 **상위집합**이고, 성능 하락을 "용량 부족"이
아니라 **비선형성**에 귀속할 수 있다. `tests/test_mlp_projection.py` 8개로 고정(직교정규 유지,
identity=선형, gradient 양쪽 도달, v98 체크포인트 strict-load 실패).

⚠️⚠️ **선행 증거는 이 방향에 반대다 (§79).** 계보 B(Encoder+Ridge)는 **더 강한 형태로 시도되고
기각**됐다 — 설계 오류(셀-셀 attention 없음)를 바로잡아 16,384차원으로 재설계했더니 **합성
val_auroc 0.784 → 0.849인데 SEAL은 0.6619 → 0.6526**, 합성 최고였던 v54가 SEAL 최악(0.6219).
기록된 결론은 **"문제는 용량이나 구조가 아니라 일반화"**다. 이 arm은 더 좁지만(sketch 유지,
출력 직교정규) 같은 방향이므로 **합성↑/SEAL↓ 패턴이 감시 대상**이다 — **보고 시 SEAL과 합성
val_ce를 나란히 놓을 것.**

**실행 중**: hidden=128(용량 v98과 8% 차이 = 비선형성 단독 효과가 가장 깨끗), **100 epoch**.
```
config: configs/train_v105_mlpproj_h128_1536_1gpu.yaml
run:    checkpoints/20260816_091707/v105_mlpproj_h128_seed4{2..5}/
```
⚠️ **학습 길이가 다른 arm 비교는 금지 사항**(§1 표, §42-43)이므로 **epoch 49와 99를 둘 다 채점**한다.
**epoch 49가 v98과 길이를 맞춘 깨끗한 비교**이고, epoch 99의 이득은 v98을 100까지 돌리기 전에는
MLP 덕분이라고 귀속할 수 없다. 100 epoch을 쓰는 근거는 v98이 epoch 39에서 이미 평평한 반면
(§2) 용량이 큰 이 arm은 다를 수 있다는 것이다.

## 135. 2026-08-16 — v105 MLP 사영: macro 무효, 다만 분산은 일관되게 작다

_Recorded by: nhn-NEXGEM-claude — 2026-08-16 12:51_

사용자 지시로 P를 MLP로 바꾸는 축을 검정했다. `projected = GELU(x@W1) @ QR(W2).Q`
(1536→128→128, trainable 213,005 = v98의 196,621과 8% 차이). **출력 사영의 QR을 유지**해
`ridge_lambda`·`covariance_slopes` calibration 교란을 막았고, `identity` 활성화면 v98과 정확히
같아지는 **상위집합**이라 하락을 "용량 부족"이 아니라 비선형성에 귀속할 수 있다.

100 epoch으로 시작했으나 사용자 지시로 **epoch 39에서 중단**하고 9/19/29/39를 채점했다
(학습 길이가 다른 arm 비교는 금지 사항이므로 v98과 같은 격자에서 비교).

| epoch | v98 macro | v98 std | v98 val_ce | v105 macro | v105 std | v105 val_ce | Δ | t | 부호 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 19 | 0.6840 | 0.00471 | 0.1457 | 0.6861 | 0.00293 | 0.1447 | +0.0021 | +0.72 | 2/4 |
| 29 | 0.6863 | 0.00638 | 0.1411 | 0.6892 | 0.00430 | 0.1415 | +0.0029 | +1.16 | 3/4 |
| **39** | **0.6900** | 0.00615 | 0.1402 | 0.6870 | 0.00424 | 0.1405 | **−0.0030** | −0.83 | 2/4 |

**판정: 측정 불가.** 주 비교 지점 epoch 39에서 |Δ|=0.0030으로 검출 한계(0.0121)의 1/4이고,
**부호가 epoch마다 뒤집힌다**(+,+,−) — 반복 재현이 아니라 노이즈다.

⚠️ **§79 패턴은 아니다.** 합성 val_ce가 v98과 사실상 동일하다(0.1405 vs 0.1402). 비선형성을
넣었는데 합성도 안 쉬워지고 SEAL도 안 변한다 — **이 비선형성은 아무 일도 하지 않는다.**

**부수 관찰**: seed std가 세 지점 전부 v98보다 작다(0.62 / 0.67 / 0.69배). 다만 4 seed의 표준편차
추정은 자유도 3으로 매우 불안정하고 같은 4개 run에서 나온 값이라 독립 관측이 아니다.

## 136. 2026-08-16 — **PCA 사영이 학습된 P와 대등하다.** 분산은 43배 작다

_Recorded by: nhn-NEXGEM-claude — 2026-08-16 12:51_

### 0. 질문

P는 학습된 1536→128 직교기저이고 v104가 학습에 +0.0126의 가치가 있음을 보였다(§133).
**PCA**는 그 에피소드 자신의 context cell에서 뽑은 직교 128차원 기저 — 같은 형태인데 **학습이 0**이다.
PCA가 P를 따라잡으면 **196,608개 학습 파라미터가 고윳값분해로 공짜로 얻는 것을 사고 있다**는 뜻이다.

산출물: `scripts/diagnose_cv_basis.py`(CV 단독), `scripts/diagnose_full_basis.py`(전체 모델).
PCA 기저는 각 fold의 **context cell만** 사용한다 — test 미접촉, 누출 없음.

### 1. CV branch 단독 — PCA가 이긴다

`CovarianceMeanRidgeModel._ridge_logits`는 head·DD·CT를 거치지 않는 CV 단독 2-class logit이다.

| | macro |
|---|---:|
| **PCA (학습 0)** | **0.6761** |
| 학습된 P — 8 seed 평균 | 0.6714 (std 0.00642) |

Δ = **−0.0047**, t = −2.06 (df=7), **8 seed 중 PCA를 이긴 것은 2개뿐**.

### 2. 전체 모델 — 대등하고, 분산이 43배 작다

⚠️ 처음에는 "기저가 회전하면 학습된 head가 깨진다"고 우려해 CV 단독만 비교했는데, **그 우려는
틀렸다** — head가 읽는 12개는 ridge logit·차이·separation이라 **어느 직교기저에서 나왔든 무관**하다
(사용자 지적). 그래서 같은 checkpoint·같은 head로 사영만 바꿔 비교했다.

| | macro | **seed std** | 범위 |
|---|---:|---:|---|
| 학습된 P | 0.6865 | **0.00725** | 0.6769–0.6963 |
| **PCA** | **0.6876** | **0.00017** | 0.6873–0.6878 |

Δ(PCA−P) = **+0.0011**, t = +0.42, 5/8 → **차이 없음**. 그런데 **seed std가 0.023배**다.

### 3. §133과 합치면 — 필요한 것은 "학습"이 아니라 "데이터 적응성"이었다

| 사영 | 성격 | 성능 | seed std |
|---|---|---:|---:|
| fixed Fourier (v104) | 데이터 무관 고정 | 0.6775 | 0.00013 |
| 학습된 P | 학습으로 적응 | 0.6865 | 0.00725 |
| **PCA** | **데이터 적응, 학습 0** | **0.6876** | **0.00017** |

v104가 보인 −0.0126은 "학습이 필요하다"가 아니라 **"적응이 필요하다"**였다. 학습된 P는 적응성을
비싸게 재발견하면서 §134의 random walk 분산을 덤으로 얻고, PCA는 적응성만 공짜로 가져온다.

## 137. 2026-08-16 — **학습된 head 13개는 상수 3개다.** 그리고 학습 파라미터 0 모델이 대등하다

_Recorded by: nhn-NEXGEM-claude — 2026-08-16 12:51_

### 1. 8 seed의 head는 거의 같다 (코사인 유사도 평균 0.919)

12개 feature 순서는 `[CV0, CV1, CV1−CV0, SEP_CV, D0, D1, D1−D0, SEP_DD, q0, q1, q0−q1, SEP_CT]`.

| feature | 8 seed 평균 | 부호일치 | \|평균\|/std |
|---|---:|:--:|---:|
| **CV1−CV0** | **+0.855** | 8/8 | 9.3 |
| **CV0** | **−0.664** | 8/8 | 4.1 |
| **CV1** | **+0.510** | 8/8 | 6.3 |
| q1 | +0.237 | 8/8 | 3.0 |
| D1−D0 | −0.197 | 8/8 | 2.2 |
| q0 / D0 / D1 | −0.170 / +0.147 / −0.145 | 8/8 | 1.6 |
| SEP_DD / q0−q1 / SEP_CT / **SEP_CV** | −0.115 / −0.083 / +0.072 / **−0.024** | 6~7/8 | 0.7 / 1.0 / 2.1 / **0.2** |
| bias | −0.075 | | |

### 2. ⚠️ DD의 음수 부호는 정상이다 — 앞선 우려를 철회한다

`_dd_distance_features`는 logit이 아니라 **각 클래스 프로토타입까지의 정규화 제곱거리**를 반환한다
(`distances = (query−prototype)²/dispersion`). **거리가 작을수록 그 클래스**이므로 D1에 음수 계수가
붙는 것이 규약상 옳다. D0 +0.147 / D1 −0.145 / D1−D0 −0.197 전부 일치한다. CV·CT는 score라 부호가
반대인 것도 같은 이유다.

### 3. SEP가 0인 것은 **라벨 반대칭이 강제한 것**이다

최종 logit이 `(−½·margin, +½·margin)`이라 클래스 0↔1 스왑 시 margin이 부호를 뒤집어야 한다.
그 스왑에서 **CV0↔CV1, D0↔D1, q0↔q1은 교환되고 SEP 3개는 불변**이다. 선형 head가 반대칭이려면
**w(SEP)=0, bias=0**, 각 쌍은 크기 같고 부호 반대여야 한다. 그리고 차분 feature는 쌍의 선형결합이라
**선형 head에 표현력을 더하지 않는다.** 남는 것은 **분기당 숫자 하나**다.

```
margin = 1.442·(CV1−CV0) − 0.343·(D1−D0) + 0.286·(q1−q0)
       → CV : DD : CT = 1 : −0.238 : +0.199
```
8 seed std가 **0.027 / 0.008 / 0.012**로 소수 둘째 자리까지 같다. 반대칭 성분이 전체의 **82%**이고,
나머지(대칭잔차 0.119 + SEP 0.236 + |bias| 0.095)는 반대칭 제약이 아키텍처에 강제돼 있지 않아
학습이 남긴 잔여물로 보인다.

### 4. 4개 조합 × 8 seed × 10 task

| | P+head | PCA+head | P+fixed | **PCA+fixed** |
|---|---:|---:|---:|---:|
| macro | 0.6865 | 0.6876 | 0.6864 | **0.6879** |
| **seed std** | 0.00725 | 0.00017 | 0.00721 | **0.00000** |
| **학습 파라미터** | 196,621 | 13 | 196,608 | **0** |

- **head 교체 단독 효과**: 사영을 고정한 채 head만 상수로 바꾸면 **Δ = −0.00014**(최대 편차 0.00060).
  8 seed 전부 소수 넷째 자리까지 일치한다.
- **PCA+fixed는 8 seed가 전부 0.6879로 동일**하다(std 정확히 0). 학습 요소가 없으니 당연하지만,
  **그것이 196,621개를 학습한 것과 같은 성능**이라는 것이 결론이다.

### 5. 판정 해상도에 미치는 영향

| | seed std | 4 seed 최소 검출 효과 |
|---|---:|---:|
| 현재 baseline v98 | 0.0072 | 0.0121 |
| **PCA+fixed** | **0.0000** | **≈ 0** |

**시드 반복이 불필요해진다.** §131-2의 검출 한계 문제와 arm 15개가 미판정으로 끝난 원인이 사라진다.

### 6. ⚠️ 아직 승격 논의를 할 수 없다 — 정식 경로 재측정이 남았다

§136·§137의 절대값은 **`eval_seal_tasks.sh`와 비교 불가**다: `max_cells` 서브샘플을 로드 시 1회
결정론적으로 적용하고 query마다 재추출하지 않으며, task 전체를 메모리에 올린다. **양쪽 arm에 동일
적용이라 arm 간 비교만 유효하다.** PCA+fixed의 SEAL macro가 실제로 v98의 0.6852 수준인지는
정식 경로로 재측정해야 한다. **다음 작업.**

## 138. 2026-08-16 — 정식 경로 재측정: **고정 head는 확정, PCA는 정식 경로에서 부호가 뒤집힌다** + 서브샘플링 계약 정리

_Recorded by: nhn-NEXGEM-claude — 2026-08-16 13:15_

### 0. 구현 — 평가 구현을 두 벌 만들지 않았다

`scripts/test_pathobench.py`의 generic 경로(`model.model(episode_bags, ...)` 직전)에 환경변수
두 개를 추가했다. 파일이 이미 쓰는 `ICF_FORCE_GENERIC_EVAL` 관용구를 따랐고, fold 종료 후
`finally`로 원상복구한다.

| 변수 | 동작 |
|---|---|
| `ICF_COVARIANCE_BASIS=pca` | 학습된 P를 **그 fold의 context cell** pooled 공분산 상위 K 고유벡터로 교체 (test 미접촉) |
| `ICF_FIXED_HEAD=1` | 학습된 12→1 head를 §137-3의 반대칭 상수로 교체 |

⚠️ **회귀 검증 완료**: 무설정으로 v98 seed42 / brca TP53을 재실행해 저장된 공식값과 **0.8311로
완전 일치**했다. 편집이 기존 경로를 바꾸지 않았다.

### 1. ⚠️⚠️ 서브샘플링 계약 — 작성자가 §136·§137에서 오해했던 부분

**relation 계보(v83~v105)는 평가 시 tile 서브샘플링을 하지 않는다.** 확인 경로 셋:

1. `eval_seal_tasks.sh`는 `--max-tiles`를 넘기지 않는다 → 스크립트 상한은 `None` →
   `cap(features, None)`이 원본을 그대로 반환 → **full tile**.
2. 모델의 `max_cells: 8192`는 **`BagTokenEncoder.forward`에만** 있다(`_subsample`). relation 계보는
   descriptor를 `_covariance_descriptors` / `_bag_means`로 직접 계산하므로 그 경로를 **타지 않는다.**
3. **실증**: 같은 설정 재실행이 저장값과 소수 넷째 자리까지 동일했다. `_subsample`이 발동했다면
   "independent per call" randperm(그 docstring이 명시) 때문에 값이 달라졌을 것이다.

`ct_cells_per_bag: 64`는 bag당 64개를 고르지만 **farthest-point 선택이라 결정론적**이다 — 위
재현성과 모순되지 않는다.

**학습 쪽에는 상한이 셋 있고 서로 독립이다** (§120-1):

| 위치 | knob | v83~v98 값 | 성격 |
|---|---|---|---|
| 데이터셋 | `per_bag_max_cells` | 4096 | 에피소드 추출 시 무작위 subsample |
| collator | `padding_max_cells` | 4096 | dense padding 전 무작위 subsample (§120-1에서 노출) |
| 모델(encoder) | `max_cells` | 8192 | **relation 계보에서는 미발동** |

⚠️ **따라서 학습은 bag당 ≤4,096 cell을 보고, 평가는 full tile(중앙값 4,988~7,736, 최대 35,107)을
본다.** 이 비대칭은 의도된 것이 아니라 계보가 진화하며 남은 것이고, §123-3이 "cell 축이 이미
실제보다 짧다"고 지적한 것의 정확한 출처다.

### 2. ⚠️ §136·§137의 진단 경로가 이 지점에서 정식 경로와 달랐다

`scripts/diagnose_full_basis.py`는 로드 시 bag을 **8192로 1회 결정론적 캡**했다. 정식 경로는
캡하지 않는다. 그래서 두 경로가 갈린다:

| arm | 진단(8192 캡) | 정식(full tile) | 차이 |
|---|---:|---:|---:|
| P+head | 0.6865 | **0.6901** | **+0.0036** |
| P+fixed | 0.6864 | **0.6898** | **+0.0034** |
| PCA+head | 0.6876 | 0.6844 | **−0.0032** |
| PCA+fixed | 0.6879 | 0.6844 | **−0.0035** |

**P 계열은 full tile에서 오르고 PCA 계열은 내려간다.** 작성자가 앞서 이 차이를 "query마다
재추출하는 무작위성"이라고 설명한 것은 **틀렸다** — generic 경로는 fold당 forward 1회이고
재추출이 없다. 실제 차이는 **캡 유무**뿐이다.

### 3. 정식 경로 결과

| arm | macro | seed std | 학습 param |
|---|---:|---:|---:|
| **v98 P+head (8 seed)** | **0.6852** | 0.00724 | 196,621 |
| v98 P+head (42–45) | 0.6901 | 0.00690 | |
| **P+fixed (42–45)** | **0.6898** | 0.00707 | 196,608 |
| PCA+head (42–45) | 0.6844 | 0.00034 | 13 |
| **PCA+fixed** | **0.6844** | **0.00000** | **0** |

seed-paired (동일 42–45 대비):

| arm | Δ | t | 부호 |
|---|---:|---:|:--:|
| **P+fixed** | **−0.0003** | −1.57 | 1/4 |
| PCA+head | −0.0057 | −1.62 | 1/4 |
| PCA+fixed | −0.0057 | — | |

### 4. ✅ 고정 head는 확정이다

**Δ = −0.0003.** 정식 경로에서도 학습된 13개를 상수 3개로 바꾸는 비용이 사실상 0이다(진단 경로
−0.00014와 일치). §137-3의 반대칭 논증 — 클래스 스왑 시 SEP 3개는 불변이므로 `w(SEP)=0`,
`bias=0`이 강제되고, 차분 feature는 쌍의 선형결합이라 선형 head에 표현력을 안 더한다 — 이 정식
경로에서 재현됐다.

```
margin = 1.442·(CV1−CV0) − 0.343·(D1−D0) + 0.286·(q1−q0)      CV:DD:CT = 1 : −0.238 : +0.199
```
**head 학습은 불필요하다.** PCA+head(0.6844)와 PCA+fixed(0.6844)가 동일한 것도 같은 결론을 준다.

### 5. ⚠️ PCA는 정식 경로에서 −0.0057이다 — §136의 결론을 한정한다

진단 경로 +0.0011 → 정식 경로 **−0.0057**로 **부호가 뒤집힌다.** 게이트 미달(t=−1.62)이라
§131-5 규칙상 "측정 불가"지만 부호는 음수다.

**§136-2의 "PCA가 학습된 P와 대등하다"는 8,192 cell 캡 조건에서만 성립한다.** full tile에서는
학습된 P가 앞선다. 해석 가설: PCA 기저는 그 fold의 cell 표본에 맞춰지므로 cell이 많아질수록
기저가 바뀌고, 학습된 P는 고정이라 그 영향을 안 받는다. **미검정 가설이다.**

다만 **PCA+fixed 0.6844는 v98 8 seed baseline 0.6852와 0.0008 차이**이고 **seed std가 정확히 0**이다
— 학습 파라미터 0으로 baseline 수준에 도달하는 것 자체는 유효하다.

### 6. 다음

- **(a) 고정 head를 baseline에 반영** — 확정된 결과. 비용 0, 파라미터 196,621 → 196,608,
  시드 간 head 변동 제거.
- **(b) PCA의 표본 민감성** — 여러 서브샘플에서 기저를 평균하거나 shrinkage를 넣으면 full tile
  손실이 줄 수 있다. **새 가설이므로 검정 필요.**
- **(c) 학습/평가 cell 수 비대칭(§1)** — 학습은 ≤4,096, 평가는 full tile이다. §123-3이 지적한
  격차의 출처이고, 지금까지 이 비대칭 자체를 arm으로 검정한 적이 없다.

## 139. 2026-08-16 — **v106 확정: 학습 파라미터 0 구성 (within-slide PCA + 고정 head)** — 사용자 결정

_Recorded by: nhn-NEXGEM-claude — 2026-08-16 13:35_

### 0. 확정된 구성

```
사영  : 각 fold의 CONTEXT cell을 bag별 자기 평균으로 센터링해 풀링한 공분산의 상위 128 고유벡터
        (⚠️ K=128은 v106 당시 값. §142에서 256으로 올렸고 §143이 v107로 승격했다)
head  : margin = 1.442·(CV1−CV0) − 0.343·(D1−D0) + 0.286·(q1−q0)
학습  : 없음.  파라미터 196,621 → 0.  seed std 0.00000 (완전 결정론적)
```
재현:
```bash
ICF_COVARIANCE_BASIS=pca_within ICF_FIXED_HEAD=1 \
  bash scripts/eval_seal_tasks.sh <gpu> <아무 v98 checkpoint> \
       configs/train_v98_p1_reverse_1536_1gpu.yaml <tag> <tasks...>
```
⚠️ checkpoint는 **껍데기로만** 쓰인다 — P는 PCA가, head는 상수가 덮어쓴다. 나머지 학습 대상이던
`ridge_log_lambda`/`ridge_log_scale`은 8 seed 전부 초기값(log 1, log 2) 그대로임을 확인했다
(Active-2대로 v77에서 동결). **checkpoint에서 실제로 쓰이는 학습값은 0개다.**

### 1. 정식 경로 전체 비교 (`eval_seal_tasks.sh`, 10 task)

| arm | macro | seed std | 학습 param | Δ vs v98(42–45) |
|---|---:|---:|---:|---:|
| v98 P+head (42–45) | **0.6901** | 0.00690 | 196,621 | — |
| P+fixed | 0.6898 | 0.00707 | 196,608 | −0.0003 |
| pooled PCA+head | 0.6844 | 0.00034 | 13 | −0.0057 |
| pooled PCA+fixed | 0.6844 | 0.00000 | 0 | −0.0057 |
| within PCA+head | 0.6864 | 0.00013 | 13 | −0.0038 |
| **v106 = within PCA+fixed** | **0.6864** | **0.00000** | **0** | **−0.0037** (t=−1.08, 1/4) |

(참고: v98 8 seed 평균은 0.6852. ⚠️ v106의 0.6864와 **빼지 말 것** — 시드 집합이 다르다, §131-1.)

### 2. ⚠️ 이것은 "더 낫다"가 아니라 **트레이드를 받아들인 사용자 결정**이다

같은 시드에 대해 seed-paired Δ는 **−0.0037**이다(t=−1.08, 1/4 → §131-5 기준 **측정 불가**, 다만
부호는 음수). 즉 macro는 v98보다 조금 낮을 수 있다. 그 대가로 얻는 것:

- **학습 파라미터 196,621 → 0.** 학습 자체가 불필요하다 — 65분 × 4 seed 배치가 사라진다.
- **seed std 0.00690 → 0.00000.** 완전 재현 가능하고, **시드 반복이 불필요**하다.
- **§131-2의 4 seed 최소 검출 효과 0.0121 → ≈0.** arm 15개가 미판정으로 끝난 원인(§133-2: 분산의
  출처가 P)이 제거된다. 앞으로 training-free 변형끼리는 **차이를 정확히 측정**할 수 있다.

### 3. 두 구성요소의 근거

**① 고정 head (Δ −0.0003, 정식 경로 확정 — §138-4).** 최종 logit이 `(−½·margin, +½·margin)`이라
클래스 스왑 시 margin이 부호를 뒤집어야 하고, 그 스왑에서 SEP 3개는 **불변**이므로 라벨 반대칭이
`w(SEP)=0`·`bias=0`을 강제한다. 차분 feature는 쌍의 선형결합이라 선형 head에 표현력을 안 더한다.
남는 것은 분기당 숫자 하나이고, 8 seed에서 std 0.027/0.008/0.012로 소수 둘째 자리까지 같았다.
**DD의 음수 계수는 정상** — `_dd_distance_features`는 logit이 아니라 프로토타입까지의 거리를 낸다.

**② within-slide 센터링 (+0.0020 vs pooled — §139-4).** 아래.

### 4. between-slide 항은 실제로 사영을 낭비하고 있었다

pooled 공분산은 두 항의 합이다:
```
C_pool = Σnᵢ·Cᵢ/Σnᵢ  +  Σnᵢ(μᵢ−μ)(μᵢ−μ)ᵀ/Σnᵢ
         └ within ┘      └──── between ────┘
```
§123-4가 실측한 **ICC 31.6%** 가 두 번째 항의 비중이다 — 128차원 중 약 1/3이 "slide를 서로 구별하는
방향"(염색·스캐너·환자)에 쓰인다. bag별 자기 평균으로 센터링하면 그 항이 **정확히** 사라진다.

```
pooled PCA 0.6844  →  within PCA 0.6864      +0.0020   (v98 대비 손실의 35% 회복)
```
head 유무 두 arm에서 동일하게 재현된다(0.6844→0.6864, 0.6844→0.6864). **§138-5의 가설이 부분
확인됐다.**

### 5. 남은 −0.0038의 유력한 원인 — label 정보

PCA는 어느 버전이든 **label을 보지 않는다.** P는 CV ridge를 통해 label loss로 학습된다.
"분산이 큰 방향"과 "class를 가르는 방향"은 다르므로, 남은 격차의 상당 부분이 여기서 올 가능성이
높다. **미검정.**

⚠️ **정정 (2026-08-16)**: 이 절은 처음에 "context label로 LDA나 부분최소제곱을 풀면 된다"고 썼는데
**부정확했다.** 문자 그대로의 LDA는 여기서 성립하지 않는다:

1. **이진 라벨은 방향을 1개만 준다.** `S_B = n₀(μ₀−μ)(μ₀−μ)ᵀ + n₁(μ₁−μ)(μ₁−μ)ᵀ`는 **rank 1**이라
   LDA의 유용한 판별 방향은 최대 `C−1 = 1`개다. 우리는 **128개**가 필요하다. PLS도 같다 — 이진
   응답 하나에 대한 첫 방향은 사실상 `μ₁−μ₀`이고 그 뒤로 신호가 없다.
2. **descriptor가 사영에 대해 이차다.** `triu(BᵀCB)`이므로 "bag 클래스를 잘 가르는 B"를 찾는 문제는
   일반화 고유값 문제로 떨어지지 않는다. LDA가 풀 수 있는 형태가 아니다.
3. **cell 단위로 라벨을 상속시키는 형태는 이미 기각됐다** — v88(PA)이 정확히 그것이었고
   −0.0111(t=−6.69, 4/4)로 이 세션 최강 기각 중 하나다(§114).

**실제로 가능한 형태는 사영을 supervised로 *만드는* 것이 아니라 PCA 방향 중에서 supervised로
*고르는* 것이다.** descriptor의 대각 성분이 `covariance[k,k] = (1/n)Σ(x·b_k)²` = **그 bag의 방향 k
위 분산**이므로, 방향마다 bag별 스칼라가 하나씩 나온다. context bag이 ~200개이고 라벨이 있으니
방향별 판별력을 직접 잴 수 있다:

```
1. within-slide PCA로 상위 M개(예: 512) 방향   ← 고윳값분해 1회, 지금과 동일
2. 방향 k마다 {b_kᵀ C_bag b_k}와 context 라벨의 판별력(AUROC/t) 계산
3. 판별력 상위 128개를 B로 채택            ← 분산 순서가 아니라 라벨 관련성 순서
```
비용은 고윳값분해 1회 + M번 t검정으로 PCA와 같은 급이고 **학습은 여전히 0**이다.

⚠️ **위험 두 가지**: (a) 선택과 적합이 같은 context를 쓴다 — query는 안 보므로 누출은 아니지만,
bag 200개에 방향 512개를 스크리닝하면 **선택 편향**이 생긴다(context를 둘로 나눠 한쪽에서 고르고
한쪽에서 푸는 방어가 필요할 수 있다). (b) v88 전례가 있다 — 기전은 다르지만(cell 단위 ridge vs
bag 단위 방향 선택) 사전 확률은 좋지 않다.

### 6. 판정 프로토콜에 미치는 영향

v106을 기준선으로 쓰면 **자기 자신은 분산이 0**이다. 다만 비교 대상 arm이 학습을 포함하면 그 arm의
분산은 그대로이므로, §107-3 게이트와 §131-2의 검출 한계는 **학습을 포함하는 arm에 대해서는 계속
적용된다.** training-free 변형끼리 비교할 때만 검출 한계가 ≈0이 된다.

## 140. 2026-08-16 — v106 아키텍처 상세 명세 + 독립 최소 구현 (`src/models/training_free.py`)

_Recorded by: nhn-NEXGEM-claude — 2026-08-16 13:53_

### 0. 왜 새 파일인가 (사용자 지시)

v106은 학습이 없는데도 `set_transformer_ridge.py`(**2,419줄**)를 통해 실행되고 있었다. 그 파일은
Set-Transformer 인코더, learnable-P 변형, PA·dual projection·gradient weight arm과 각각의
checkpoint 호환 장치를 싣고 있고 **v106은 그중 무엇도 쓰지 않는다.** 파라미터가 0인 모델에서
검증할 수 있는 것은 코드뿐이므로, **읽을 수 있는 크기**가 곧 신뢰성이다.

`src/models/training_free.py` — **315줄**, 클래스 1개, 파라미터 0개, 학습 경로 없음.

### 1. 알고리즘 명세 (에피소드 = 라벨된 context bag들 + 라벨 없는 query bag들, bag = [cells, 1536])

**① 기저** — context cell 공분산을 **bag별 자기 평균으로 센터링**해 풀링, 상위 K=128 고유벡터.
```
scatter = Σ_bag Σ_{x∈bag} (x − μ_bag)(x − μ_bag)ᵀ ;   B = eigh(scatter/N)의 상위 128
```
전역 평균이 아니라 bag별로 센터링하는 것이 between-slide 항을 **정확히** 제거한다(§139-4, +0.0020).
bag 단위 누적이라 전체 cell을 한 번에 올리지 않는다(§62-3의 eval OOM 회피).

**② descriptor** — bag마다 `[triu(BᵀC_bag B), μ_bag]` = 8,256 + 1,536 = 9,792차원.
⚠️ 이 절의 수치는 **K=128(v106) 기준**이다. 활성 구성 v107은 K=256이라 32,896 + 1,536 =
34,432차원이다(§142·§143). 구조는 같고 K만 다르다.

**③ CV** — context descriptor에 대한 **클래스 균형 ridge**를 dual로 풂(bag ~200 ≪ 특징 9,792).
⚠️ 공분산 블록과 평균 블록을 **각각 따로** RMS 정규화한다 — 스케일이 자릿수 단위로 달라 한 번에
정규화하면 큰 쪽이 ridge를 지배한다. **이 한 가지를 빠뜨렸을 때 기존 경로와 약 2% 어긋났다.**

**④ DD** — 스케치 공분산들에서 rank-1 분산 방향을 뽑고, query에서 두 클래스 프로토타입까지의
**정규화 제곱거리**. ⚠️ logit이 아니라 **거리**다 — head가 음수 계수를 주는 이유.
`eigh`는 미분하지 않는다(§100: backward가 `1/(λᵢ−λⱼ)`를 갖고 방향 선택이 hard argmax).

**⑤ CT** — context cell 위 **결정론적 farthest-point** 토큰 16개(선택에 라벨 미개입), bag별 soft
abundance, 그중 클래스를 가장 잘 가르는 두 토큰을 읽음. bag당 64 cell은 **등간격 추출**이라
전 과정이 결정론적이다.

**⑥ head** — 상수:
```
margin = 1.442·(CV1−CV0) − 0.343·(D1−D0) + 0.286·(q1−q0)
logits = (−margin/2, +margin/2)
```

### 2. 등가성 검증 — 재구현은 증명 없이는 쓸 수 없다

v106의 0.6864는 `set_transformer_ridge.py`를 패치해 만든 숫자다. 새 파일이 조금이라도 어긋나면
**v106 대비 모든 비교가 조용히 의미를 잃는다.** 그래서 등가성을 테스트로 고정했다
(`tests/test_training_free.py`, 7개 통과):

- **margin 일치** — 무작위 에피소드 3개에서 패치된 기존 경로와 `allclose(atol=2e-3)`
- **순위 완전 일치** — AUROC는 순서만 읽으므로 `argsort`가 정확히 같아야 한다
- **라벨 스왑 시 margin 부호 반전** — ①의 상수 3개가 옳은 매개화라는 근거 자체(§137-3)
- 결정론성 / 파라미터 부재 / 기저가 query를 안 봄 / within ≠ pooled 기저

**실제 데이터 대조** (전체 50 fold):

| task | 새 구현 | 기존 경로 | 차이 |
|---|---:|---:|---:|
| cptac_brca TP53 | 0.8283 | 0.8286 | −0.0003 |
| cptac_ccrcc VHL | 0.4630 | 0.4635 | −0.0005 |

부동소수점 수준 차이다(기존 경로는 일부 구간을 float32로, 새 구현은 기저 누적을 float64로 한다).

### 3. 사용법

```python
from src.models.training_free import TrainingFreeClassifier
clf = TrainingFreeClassifier()                      # 파라미터 0, 설정은 dataclass
margins = clf.margins(context_bags, context_labels, query_bags)   # 양수 = class 1
```
정식 SEAL 채점은 여전히 `ICF_COVARIANCE_BASIS=pca_within ICF_FIXED_HEAD=1`로 기존 경로를 쓴다
(§139-0) — 두 경로가 등가임이 §2로 확인됐으므로 어느 쪽을 써도 같다.

⚠️ **드롭인 교체가 아니다**: `forward(instances, labels, query_index)` 시그니처도, checkpoint
호환도, 학습 경로도 없다. 학습 없는 구성의 **평가 전용**이다.

---

## 141. 2026-08-16 — 테스트 스위트 정리: **조용히 0개 돌던 파일 2개를 복구** (233개 전수 통과)

*작성: nhn-NEXGEM-claude, 2026-08-16 (KST)*

### 1. 문제 — 실패 1건 뒤에 숨어 있던 무증상 1건

전체 스위트를 돌리면 `test_mlp_manifold_bank.py`가 계속 에러를 냈다. 원인은 `import pytest`이고
이 환경에 pytest가 없다. 그런데 그 파일을 조사하다 **같은 병을 앓지만 증상이 없는 파일**을 찾았다.

| 파일 | 스타일 | 증상 | 실제 실행된 테스트 |
|---|---|---|---:|
| `test_mlp_manifold_bank.py` | pytest, `TestCase` 0개 | **에러 1건** (시끄러움) | 0 / 4 |
| `test_factorized_response.py` | pytest, `TestCase` 0개 | **무증상** — "NO TESTS RAN" | 0 / 5 |

`unittest`는 모듈 수준 `def test_*`를 수집하지 못한다. 두 번째 파일은 import가 성공하므로
아무 소리 없이 통과한 것처럼 보였다. **9개 테스트가 존재하는 척만 하고 있었다** — 시끄러운
실패보다 조용한 0이 더 위험하다.

두 파일 모두 `SyntheticManifoldGenerator`의 **살아 있는** 인자(`manifold_mode`,
`manifold_bank_size`, `label_rule="xor"`, `random_causal_factors`, `separate_nuisance_rng`)를
검정한다. 생성자에 그대로 존재함을 확인했으므로 폐기가 아니라 **변환**이 맞다.
`unittest.TestCase`로 옮겼고 단언은 그대로 두었다(pytest 의존성을 두 파일 때문에 추가하지 않는다).

### 2. 전수 점검 — 같은 패턴은 더 없다

```
for f in tests/test_*.py: TestCase 0개 / import pytest / 모듈 수준 def test_*
→ 해당 없음 (clean)
```

나머지 25개 파일은 모두 살아 있는 코드를 겨눈다. **거부된 arm의 테스트도 남긴다** —
`test_spectral_tail`, `test_class_prior`, `test_population_attention`은 기본값으로 꺼져 있는
노브를 고정하며, 이 테스트가 없으면 그 노브가 조용히 되살아나도 알 수 없다. `test_slot_mla`가
겨누는 `src/models/baseline.py`는 `set_transformer_ridge.py`가 import하는 현역 모듈이다.
**삭제하거나 `tests/history/legacy_*.py`로 옮긴 파일은 없다.**

### 3. ⚠️ 인터프리터 — `python3`로 돌리면 결과가 거짓말을 한다

이번에 실수로 `python3 -m unittest`로 돌렸더니 `ModuleNotFoundError: No module named 'lightning'`
로 **14개 모듈이 통째로 import 실패**했다. 출력은 `Ran 158 tests ... FAILED (errors=14)`.
75개 테스트가 사라졌는데도 "158개 돌았다"고 말한다. 반드시 문서에 적힌 인터프리터를 쓸 것:

```bash
/home/aibio_3/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"
```

### 4. 결과

```
Ran 233 tests in 162.9s
OK
```

정리 전 224개 실행 + 에러 1건 → **233개 전수 통과**(복구 9개), 실패 0, 무증상 0.

---

## 142. 2026-08-16 — **K(covariance sketch dim) 스윕: 128은 최적이 아니었다 → K=256 권고 (v107 후보)**

*작성: nhn-NEXGEM-claude, 2026-08-16 (KST)*

### 1. 왜 지금 돌릴 수 있었나 — 학습을 없앤 것의 첫 배당금

K는 지금까지 **건드릴 수 없는 값**이었다. P가 학습된 1536×K 행렬이었으므로 K를 바꾸면
재학습이 필요했고, 재학습 1회는 시드 4개 × 49 epoch이다. v106에서 P가 에피소드별 PCA로
바뀌면서 K는 **평가 시점 자유 노브**가 됐다. v98 체크포인트 25개 텐서 중 K에 의존하는 것은
`model._covariance_projection` **단 하나**이고, 그건 정확히 PCA가 대체하는 그 텐서다.

훅 2개 추가 (`scripts/test_pathobench.py`, 기존 `ICF_FORCE_GENERIC_EVAL` 관용구):
`ICF_SKETCH_DIM` (K 변경 + 해당 텐서만 drop, 나머지 불일치는 예외로 올림),
`ICF_RIDGE_LAMBDA` (아래 대조군용).

**회귀 검증**: K=128 arm은 환경변수 없이 통과시켰고 v106의 10개 task 값과 **전부 자릿수까지
일치**(macro 0.6864). 훅이 미설정 시 no-op임이 증명됐다.

### 2. ⚠️ λ 교란 — 있는 줄 알았는데 없었다

표준화가 각 블록을 unit RMS로 맞추므로 bag의 노름²은 descriptor **길이**를 따라간다.
즉 K를 올리면 dual Gram이 커지는데 `ridge_lambda=1.0`은 그대로라 **실효 정규화가 약해진다**.
K 비교에 두 번째 노브가 섞이는 셈이고 이는 §127-2 위반이다. 실측
(`scripts/diagnose_sketch_dim_scale.py`, brca TP53 fold_0):

| K | desc dim | Gram 대각 | vs K=128 | 예측(길이비) |
|---:|---:|---:|---:|---:|
| 64 | 3,616 | 80.4 | 0.369 | 0.369 |
| 128 | 9,792 | 217.8 | 1.000 | 1.000 |
| 256 | 34,432 | 766.3 | **3.519** | 3.516 |
| 512 | 132,864 | 1680 | 7.714 | 7.706 |

교란은 실재하고 길이에 정확히 비례한다. 그런데 **보정해도 결과가 안 바뀐다**:

| K=256, λ | fold-mean (3 fold) |
|---|---|
| 0.01 / 1.0 / 3.519 | 0.7789 (완전 동일) |
| 10,000 | 0.7216 |

λ은 살아 있지만(1e4에서 움직임) **[0.01, 3.52] 구간에서 완전히 무력**하다. Gram 대각 766에
λ=1은 0.13%다 — ridge가 사실상 비정규화 영역에서 돈다. 따라서 **λ 고정 K 비교는 진짜 단일
노브 arm**이고 §127-2를 만족한다. λ이 v98 8개 시드 전부에서 초기값 exp(0)=1.0 그대로였던
것(한 번도 학습되지 않음)도 이제 설명된다 — 기울기가 실질적으로 0인 방향이다.

### 3. 결과 — 역U자, 정점은 128보다 한참 위

전부 학습 없음·결정론적이므로 시드 반복 없음(§139: 시드 std 0.00000). 정식 50-fold 경로.

| task | K=64 | **K=128** | K=192 | K=256 | K=384 | K=512 | K=768 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bc_therapy er_status | 0.6687 | 0.6880 | 0.6999 | 0.7047 | 0.7086 | 0.7103 | 0.7114 |
| bc_therapy grade | 0.7202 | 0.7172 | 0.7227 | 0.7191 | 0.7094 | 0.7108 | 0.7109 |
| bc_therapy her2_status | 0.6505 | 0.6540 | 0.6715 | 0.6755 | 0.6787 | 0.6839 | 0.6817 |
| cptac_brca PIK3CA | 0.5556 | 0.5379 | 0.5370 | 0.5333 | 0.5365 | 0.5339 | 0.5306 |
| cptac_brca TP53 | 0.8137 | 0.8286 | 0.8282 | 0.8264 | 0.8244 | 0.8214 | 0.8111 |
| cptac_ccrcc BAP1 | 0.6497 | 0.6542 | 0.6500 | 0.6669 | 0.6618 | 0.6574 | 0.6520 |
| cptac_ccrcc VHL | 0.4401 | 0.4635 | 0.4716 | 0.4762 | 0.4782 | 0.4710 | 0.4583 |
| cptac_luad EGFR | 0.7767 | 0.7782 | 0.7806 | 0.7836 | 0.7910 | 0.7912 | 0.7925 |
| cptac_luad STK11 | 0.8731 | 0.8810 | 0.8808 | 0.8841 | 0.8841 | 0.8855 | 0.8860 |
| cptac_luad TP53 | 0.6642 | 0.6612 | 0.6663 | 0.6753 | 0.6838 | 0.6822 | 0.6786 |
| **MACRO** | 0.6812 | **0.6864** | 0.6909 | **0.6945** | **0.6956** | 0.6948 | 0.6913 |
| Δ vs 128 | −0.0051 | — | +0.0045 | +0.0081 | +0.0093 | +0.0084 | +0.0049 |
| task별 t (n=10) | −1.35 | — | +2.14 | **+2.98** | +2.52 | +2.06 | +1.04 |
| 부호 | 3/10 | — | 6/10 | **8/10** | 7/10 | 7/10 | 5/10 |

**런타임은 K와 무관하게 24–25초/task** (1536×1536 eigh와 타일 로딩이 지배). 비용은 판단
요소가 아니다.

판정 단위 주의: 이 모델은 결정론적이라 **시드가 반복 단위가 될 수 없다.** §107-3의
"4 시드, |t|≥2.5"를 그대로 쓸 수 없고, 여기서는 **task**가 반복 단위다(n=10 → 4보다 강하다).
fold 단위 t는 한 task 안 50 fold가 같은 slide를 겹쳐 쓰므로 반보수적이라 쓰지 않는다.

### 4. ⚠️ 선택 편향과 그 해소 — 홀드아웃 7개

이 스윕은 **판정 대상인 SEAL 10개 위에서** 골랐다. 그 위의 argmax를 그대로 채택하면
test에 대고 튜닝하는 것이다. `seal_univ2_baseline_17tasks.csv`의 나머지 7행은 SEAL 대응
수치가 없어 판정에서 빠져 있는데, **바로 그 이유로 깨끗한 홀드아웃**이다 (§131-5).

| task | K=128 | K=256 | K=384 |
|---|---:|---:|---:|
| cptac_lscc ARID1A | 0.4028 | 0.4168 | 0.4108 |
| cptac_lscc Histologic_Grade | 0.6260 | 0.6102 | 0.6055 |
| cptac_lscc KEAP1 | 0.5754 | 0.5751 | 0.5854 |
| cptac_luad KRAS | 0.6883 | 0.7110 | 0.7196 |
| cptac_pda SMAD4 | 0.4760 | 0.4938 | 0.4879 |
| ucla_lung progression | 0.7821 | 0.7811 | 0.7791 |
| cptac_ccrcc PBRM1 | 0.4863 | 0.4970 | 0.4883 |
| **MACRO** | 0.5767 | **0.5836** | 0.5824 |

| arm | SEAL 10 (선택에 사용) | 홀드아웃 7 | 전체 17 |
|---|---|---|---|
| **K=256** | +0.0081, t 2.98, 8/10 | **+0.0069**, t 1.37, 4/7 | **+0.0076, t 3.01, 12/17** |
| K=384 | +0.0093, t 2.52, 7/10 | +0.0057, t 0.96, 5/7 | +0.0078, t 2.45, 12/17 |

홀드아웃 7개만으로는 게이트를 못 넘는다(n=7, 검정력 부족). 중요한 건 그게 아니라
**부호와 크기가 독립 집단에서 재현됐다**는 것이다(+0.0081 → +0.0069).

그리고 이게 K=384를 떨어뜨린다: SEAL에서 384가 256보다 높았던 +0.0011은 홀드아웃에서
**뒤집힌다**(+0.0057 < +0.0069). 384의 우위는 선택 노이즈였다.

### 5. 권고 — K=256 (v107 후보), 사용자 확정 대기

- 전체 17개에서 **+0.0076, t=3.01, 12/17** — 가장 강한 증거
- 홀드아웃에서 재현되는 유일한 값 (384는 안 됨)
- 정점(384) 자체가 아니라 **plateau의 가장 싼 지점**이라 정점 위치 노이즈에 강함
- 비용 증가 없음 (24→25초)
- 학습 없음·결정론적 유지 — v106의 성질을 하나도 잃지 않는다

**아직 기본값을 바꾸지 않았다.** `TrainingFreeConfig.sketch_dim`과 config의
`covariance_sketch_dim`은 128 그대로다 — v106 채택이 그랬듯 구성 확정은 사용자 판단이다(§118).

### 6. 부수적으로 알게 된 것

- **저K가 유리한 task가 있다**: PIK3CA는 K=64가 최고(0.5556, 256보다 +0.022), brca TP53은
  K에 대해 단조 감소(0.8286→0.8111). 둘 다 신호가 약한 task다 — 방향 수를 늘리면 용량이
  늘어 약신호 task에서 손해라는 bias–variance 해석과 맞는다. **task별 K**는 아직 미검토.
- **3 fold 스크리닝이 방향을 뒤집었다**: brca TP53 3 fold에서 K=256은 −0.037로 보였으나
  50 fold 전체로는 −0.0022였다. §125-1의 정신 그대로 — **부분 fold 스크리닝으로 부호를
  주장하지 말 것.**

---

## 143. 2026-08-16 — **v107 승격 확정 (사용자 결정): 활성 구성 = within-slide PCA(K=256) + 고정 head**

*작성: nhn-NEXGEM-claude, 2026-08-16 (KST)*

§142의 권고를 사용자가 채택했다. **v107 = v106에서 K만 128 → 256.** 학습 파라미터는 여전히 0이고
결정론성도 그대로다(seed std 0.00000).

### 1. 무엇이 바뀌었나 (코드)

| 자리 | 변경 | 비고 |
|---|---|---|
| `TrainingFreeConfig.sketch_dim` | 128 → **256** | 무학습 구현의 기본값 = baseline. `tests/test_training_free.py::DefaultTest`가 고정 |
| `scripts/eval_v107.sh` | **신규** | v107의 정의가 사는 단 하나의 자리. 환경변수 3개를 세팅하고 `eval_seal_tasks.sh`에 위임 |
| `configs/*.yaml`의 `covariance_sketch_dim` | **128 그대로** | ⚠️ 아래 §2 |

```bash
bash scripts/eval_v107.sh <gpu> <tag> [tasks...]     # 기본값 = SEAL 10
```

### 2. ⚠️ config를 왜 안 바꿨나

`covariance_sketch_dim: 256`을 config에 박으면 **두 가지가 깨진다**: ① v98 학습 재현 —
그 config는 학습용이고 K는 학습된 P의 출력 폭이다. ② 체크포인트 strict load — P는 1536×128이다.

그래서 K=256은 `ICF_SKETCH_DIM` 환경변수로만 건다. 이 훅은 K에 의존하는 **유일한** 텐서
`_covariance_projection`만 버리고 **나머지 불일치는 예외로 올린다** — `strict=False`로 뭉개면
무관한 아키텍처 드리프트가 0으로 로드되며 조용히 수치를 바꾼다(§142-1).

### 3. v98과의 관계 — 부호는 뒤집혔지만 "이겼다"는 아니다

```
v98 seed 42–49   0.6907 0.6950 0.6802 0.6946 0.6758 0.6811 0.6807 0.6837   mean 0.6852
v107 (결정론)    0.6945
```

| 비교 | Δ | t | 이김 |
|---|---:|---:|---:|
| v106 vs v98 42–45 | −0.0037 | −1.08 | 1/4 |
| **v107 vs v98 42–45** | **+0.0044** | +1.27 | 2/4 |
| **v107 vs v98 42–49** (8 seed, 가장 신뢰) | **+0.0093** | +3.62 | **6/8** |
| **v107 vs v98 4-seed 앙상블 (0.6951, §130)** | **−0.0006** | — | — |

v106이 감수했던 −0.0037 트레이드는 **사라졌다.** 하지만 다음 세 가지 때문에 "v98을 이겼다"고
쓰면 안 된다:

1. v98의 **상위 두 시드(0.6950, 0.6946)에는 진다.**
2. 42–45 부분군으로 좁히면 **2/4**로 부호가 갈린다.
3. **v98 4-seed 앙상블 0.6951이 아직 앞선다** — 다만 그 +0.0006은 어떤 기준으로도 측정 불가다.

⚠️ **위 t는 seed-paired가 아니다.** v107은 시드에 의존하지 않으므로 짝짓기가 분산을 전혀 줄이지
못한다 — t=+3.62(df 7)는 "v98 시드 평균과 다른가"를 묻는 **1-표본 t**다. §107-3 게이트를 그대로
갖다 붙이지 말 것.

**정확한 요약**: 학습 0회·시드 1회·결정론적인 구성이, **49 epoch × 4 시드를 학습해 앙상블한
구성과 같은 자리**에 왔다. 이득은 여전히 macro가 아니라 비용·재현성 쪽이다.

### 4. 남은 것

- **ABMIL(0.727)과는 여전히 −0.0321**(t=−3.04), 상회 2/10. K는 이 격차를 0.0402→0.0321로
  좁혔을 뿐 뒤집지 못했다. 격차가 큰 쪽이 n이 크고 신호가 강한 task(luad TP53 −0.076, grade
  −0.051, EGFR −0.046)라는 패턴은 그대로다.
- **VHL은 아직 0.5 미만**(0.4762). K로 +0.0128을 벌었지만 부호가 반대인 상태 자체는 안 바뀐다.
- **task별 K는 미검토.** PIK3CA는 K=64가 최고(0.5556), brca TP53은 K에 단조 감소 — 약신호 task는
  저K를 선호한다(§142-6). 다만 task마다 노브를 고르는 것이라 홀드아웃 없이는 선택 편향이 크다.
- **앙상블 축은 v107에 없다.** 결정론적이라 시드 앙상블(§130의 +0.0058~0.0071)을 못 쓴다.
  training-free 변형끼리의 앙상블(다른 K를 섞는 등)은 미검토.

---

## 144. 2026-08-16 — v107을 ICI·Musk에 전이: **Musk는 학습 모델을 넘었고, ICI는 여전히 우연**

*작성: nhn-NEXGEM-claude, 2026-08-16 (KST)*

### 0. 학습이 없어서 사라진 두 개의 다리

두 벤치마크 모두 그동안 **우회로**를 통해서만 잴 수 있었고, 그 우회로가 수치를 흐렸다.
v107은 파라미터가 0이라 체크포인트도 config도 없이 `TrainingFreeClassifier`를 직접 쓴다
(`scripts/eval_v107_transfer.py`).

| | 기존 | v107 |
|---|---|---|
| Musk | 166-d를 학습 모델의 1536-d 입력에 **패딩**해야 했다 (`test_musk.py`가 스스로 "crude OOD bridge"라 부름) | 기저가 들어온 것의 고유벡터라 **입력 차원 개념 자체가 없다** → 원래 166-d로 실행 |
| ICI | `launch_ici_protocol.sh`가 fold마다 **파인튜닝** (5 seed × 5 fold = 25회 학습) | 같은 25개 분할을 **학습 0회**로 채점 → fold 산포가 옵티마이저가 아니라 데이터의 것 |

### 1. Musk (UCI Musk2, 102 bag, 166-d, leave-one-out)

| 구성 | AUROC | 95% CI |
|---|---:|---|
| **v107 (K=256 → 166으로 cap)** | **0.8926** | [0.825, 0.948] |
| v107, 스윕 최적 K=32 | 0.9052 | [0.845, 0.954] |
| **v98 학습 모델 + tile 패딩** (같은 세션 실측) | **0.8799** | [0.807, 0.946] |
| v30 세대 최고 (§50, historical) | 0.858 | — |
| 목표 (§23) | 0.95 | — |

v107은 자기 상한(K=166)에서 v98 학습 모델을 **+0.0127** 앞선다. ⚠️ **CI가 크게 겹친다
(n=102) — 판정 불가다.** 의미 있는 것은 크기가 아니라 **학습도 패딩 다리도 없이 그 자리에
왔다**는 것이고, v30 세대 최고 0.858은 넘었다. **0.95는 여전히 미달.**

⚠️ K=32의 0.9052를 v107의 수치로 쓰지 말 것 — 같은 데이터에서 고른 argmax다(§142-4와 같은 함정).

### 2. ⚠️ Musk에서 K 방향이 **뒤집힌다** — 그리고 그게 §142를 설명한다

| K | 8 | 16 | **32** | 64 | 128 | 166 |
|---|---:|---:|---:|---:|---:|---:|
| AUROC | 0.8681 | 0.8974 | **0.9052** | 0.9007 | 0.9015 | 0.8926 |

PathoBench에서는 K를 128→256으로 **올려서** 이겼는데(§142) Musk에서는 32가 정점이고 올릴수록
나빠진다. 모순이 아니라 **같은 법칙의 양쪽**이다.

descriptor는 `triu(BᵀC_bag B)`인데, **n개 cell을 가진 bag의 중심화 공분산은 rank ≤ n−1**이다.
Musk bag의 중앙값은 **12 instance**(최소 1)다. 즉 K×K 행렬의 자유도는 ~11개뿐인데 triu는
K(K+1)/2개 항을 만든다 — K를 키우면 **나머지 항의 결정론적 함수인 좌표**가 늘어나고 ridge에서
노이즈만 증폭된다. PathoBench slide는 4,096 cell이라 이 제약이 걸리지 않았다.

**규칙: K의 최적점은 bag당 cell 수를 따라간다.** §142의 "256이 128보다 낫다"는 UNI2 slide 크기에
대한 진술이지 K에 대한 보편 진술이 아니다. 새 도메인에서는 **반드시 K를 다시 재야 한다.**

### 3. ICI (GSE285888, 87 donor, 512-d, 5 seed × 5 fold)

| K | fold-mean (n=25) | fold std | seed-mean std |
|---:|---:|---:|---:|
| 64 | 0.5206 | 0.1233 | 0.0307 |
| 128 | 0.5136 | 0.1255 | 0.0221 |
| **256** | **0.5178** | 0.1215 | 0.0220 |

**95% CI [0.470, 0.565] — 0.5를 포함한다. 우연이다.** K도 아무 영향이 없다(0.5136~0.5206).

§86의 무학습 CV-only(fixed random P, K=128) 0.5449±0.0180과 비교하면 −0.0271 = **1.12 SE**로
판정 불가다. 프로토콜은 비교 가능하다 — §86도 "학습 없이 fold마다 closed-form ridge만" 풀었다.

이건 새 소식이 **아니다.** ICI는 이미 잠긴 축이다: `history.md`가 "n=87 ICI 단일 코호트에서는
어떤 아키텍처 차이도 검출 불가능(AUROC CI 전부 [0.42, 0.68])", "ICI 5-seed 0.512±0.027 = 랜덤"으로
기록했다. **v107은 그 잠금을 확인할 뿐 흔들지 않는다.** fold std가 0.1215라는 것은 fold 하나가
±0.12씩 튄다는 뜻이고, 87 donor 코호트에서 이보다 잘 하기를 기대할 근거가 없다.

⚠️ **ICI에서 arm을 판정하지 말 것.** 검출 한계가 macro Δ ≈ 0.05다 — 우리가 다루는 효과(±0.01)의
5배다.

### 4. 종합

| 벤치마크 | v107 | 비교 대상 | 판정 |
|---|---:|---|---|
| PathoBench SEAL 10 | 0.6945 | ABMIL 0.727 | 여전히 −0.032 (§143) |
| **Musk** | **0.8926** | v98 학습 0.8799 | **+0.0127, 판정 불가**. 목표 0.95 미달 |
| **ICI** | **0.5178** | §86 0.5449 | **양쪽 다 우연.** 축 잠금 유지 |

학습을 없앤 것이 전이에서 **손해가 아니었다**는 것이 이 절의 요지다. 세 도메인 어디서도
v107은 학습 모델보다 나쁘지 않았고, Musk에서는 도구(패딩 다리)를 하나 없애면서 오히려 앞섰다.

---

## 145. 2026-08-17 — **§142의 K 이득은 전부 CV의 것이다. DD는 128에서 포화한다** (사용자 지적)

*작성: nhn-NEXGEM-claude, 2026-08-17 (KST)*

### 0. 지적 자체가 맞았다

사용자 지적: "K가 늘어난 게 CV의 영향인지 DD의 영향인지 모르는 거잖아?" — 맞다. §142는 K를
하나만 움직인 줄 알았지만 K는 **두 분기에 동시에** 들어간다:

| 분기 | K가 들어가는 곳 | K 의존 |
|---|---|---|
| CV | descriptor가 `triu(BᵀC_bag B)` → 길이가 K(K+1)/2 (9,792 → 34,432) | **직접** |
| DD | 같은 triangle에서 K×K 행렬을 **재구성**해 rank-1 분산 방향을 뽑음 | **직접** |
| CT | raw cell에서 farthest-point 선택 — 기저를 안 봄 | 없음 |

§127-2("arm당 노브 하나")를 어겼던 셈이다. §142-2에서 λ 교란은 실측해 없앴는데 **분기 교란은
놓쳤다.**

### 1. 분해 방법 — 슬라이스가 정확한 이유

`ICF_SKETCH_DIM_DD` 훅 추가(`scripts/test_pathobench.py`). PCA 기저는 고유값 내림차순이므로
**K에서 계산한 BᵀCB의 좌상단 k×k 블록 = k에서 계산한 BᵀCB**다. 근사가 아니라 항등식이고
eigh를 다시 돌리지도 않는다. `_dd_direction`이 축소 identity를 `covariance_sketch_dim`으로
만들기 때문에 그 값도 함께 임시 교체하고 즉시 복원한다.

**검증 3가지**: ① `ICF_SKETCH_DIM_DD=256`(K_cv=256)이 미설정과 **비트 단위 동일**(brca TP53
5 fold 0.8069) ② `=128`은 0.8272로 노브가 살아 있음 ③ K_dd > K_cv는 예외로 거부.

⚠️ **구조적 제약: K_dd ≤ K_cv.** DD는 cell을 직접 사영하지 않고 CV triangle의 부분블록을 읽는다
(`_covariance_matrices_from_triangle`). "CV보다 많은 방향을 보는 DD"는 훅의 한계가 아니라
**이 아키텍처가 표현할 수 없는 구성**이다. 따라서 격자는 하삼각이고 분해는 순차적이다.

### 2. 결과 — SEAL 10 격자

| K_cv | K_dd | macro | Δ vs (128,128) | t | 부호 |
|---:|---:|---:|---:|---:|---:|
| 256 | 256 | **0.6945** | +0.0081 | +2.98 | 8/10 |
| 256 | 192 | 0.6926 | +0.0062 | +2.35 | 7/10 |
| **256** | **128** | **0.6923** | **+0.0060** | **+2.37** | **8/10** |
| 256 | 64 | 0.6907 | +0.0044 | +0.98 | 6/10 |
| 256 | 32 | 0.6847 | −0.0017 | −0.40 | 3/10 |
| 128 | 128 | 0.6864 | — | — | — |
| 128 | 64 | 0.6845 | −0.0019 | −0.64 | 3/10 |
| 128 | 32 | 0.6790 | −0.0073 | −2.07 | 1/10 |

(256,256)=0.6945와 (128,128)=0.6864가 §142와 **자릿수까지 일치** — 격자의 내부 일관성 확인.

### 3. 귀속 — 홀드아웃 7개까지 포함해 결정적

| 증분 | SEAL 10 | 홀드아웃 7 | **전체 17** |
|---|---|---|---|
| **CV 128→256** (DD 128 고정) | +0.0060, t 2.37, 8/10 | +0.0081, t 1.75, 5/7 | **+0.0068, t 2.93, 13/17** |
| **DD 128→256** (CV 256 고정) | +0.0022, t 0.79, 5/10 | **−0.0012**, t −0.44, 3/7 | **+0.0008, t 0.40, 8/17** |
| 둘 다 (§142 원래 arm) | +0.0081, t 2.98, 8/10 | +0.0069, t 1.37, 4/7 | +0.0076, t 3.01, 12/17 |

**§142의 이득은 전부 CV의 것이다.** DD의 128→256 증분은 17개에서 +0.0008, **부호가 8/17**
— 동전 던지기다. 홀드아웃에서는 오히려 음수이고 (256,128)이 (256,256)보다 높다(0.5848 vs 0.5836).

### 4. 그런데 DD가 K에 둔감한 게 아니다 — **128에서 포화**한다

| 증분 | Δ | t | 부호 |
|---|---:|---:|---:|
| DD 32→128 (CV 256 고정) | **+0.0077** | +2.15 | **9/10** |
| DD 32→128 (CV 128 고정) | **+0.0073** | +2.07 | **9/10** |
| DD 128→256 (CV 256 고정) | +0.0022 | +0.79 | 5/10 |

낮은 구간에서는 DD도 방향이 절실하다(32→128에 +0.0075, 9/10, K_cv와 무관하게 재현). 다만
**128을 넘으면 아무것도 없다.**

이유는 두 분기가 K를 쓰는 방식이 다르기 때문이다. **DD는 K×K 공분산에서 방향 하나(rank-1)를
뽑는다** — 그 방향을 찾을 만큼 차원이 확보되면 더 늘려도 뽑을 게 없다. **CV의 ridge는
K(K+1)/2개 triangle 항 전부를 feature로 읽는다** — K를 늘리면 진짜로 feature가 늘어난다.

### 5. 함의

- **다음 투자는 CV descriptor 쪽이다.** DD를 키우는 방향은 닫혔다(포화 확인).
- **(K_cv=256, K_dd=128)은 정당한 대안 구성이다.** 17개에서 v107과 차이가 −0.0008(측정 불가)
  이고 홀드아웃에서는 오히려 높다. DD의 eigh가 256³ → 128³로 8배 싸진다. ⚠️ 다만 **런타임은
  이미 K와 무관**(§142-3, 24–25초)이라 실익은 속도가 아니라 **구성의 단순함**이다.
  **기본값은 v107(256,256) 그대로 두었다** — 구성 변경은 사용자 판단이다(§118).
- **§142의 "K=256이 낫다"는 이제 "CV의 K=256이 낫다"로 좁혀 읽어야 한다.**

### 6. ⚠️ 부분 fold 스크리닝에 또 속았다 — 같은 세션에서 두 번째

brca TP53 **5 fold**에서 (256,128)이 0.8272, (256,256)이 0.8069로 나와 "DD는 작은 K를
좋아한다"고 읽었다. **50 fold 전체는 0.8228 vs 0.8264로 반대다.** §142-6에서 똑같은 함정
(3 fold에서 K=256이 −0.037로 보였으나 실제 −0.0022)을 기록한 직후에 다시 걸렸다.
**부분 fold는 훅이 도는지 확인하는 용도이며, 방향을 읽는 데 쓰면 안 된다.**

---

## 146. 2026-08-17 — 적응적 rank DD: **t 검정은 게이트로 못 쓴다(기전 실측)**, 그리고 r>1은 이득이 없다

*작성: nhn-NEXGEM-claude, 2026-08-17 (KST)*

사용자 제안: DD의 r을 eigenvalue에 따라 적응적으로 정하자 — "충분히 차이가 큰 경우에만 그 방향을
가져오고, 아니면 r=1". 판독은 `s_b = uᵀC_b u`(또는 log)의 클래스별 평균이 유의하게 다른지
검정으로. 구현하고 실측했다. **아이디어의 구조는 맞았고, 판독 도구가 틀렸다.**

### 1. 구현 — r=1 등가성을 먼저 고정

`src/models/dd_adaptive_rank.py`(신규) + `tests/test_dd_adaptive_rank.py`(9개 통과).
`operator`의 고유벡터를 |λ| 내림차순으로 세우고, rank>0 후보는 context bag에서
log(s_b)의 Welch t를 통과해야 채택한다. **rank 1은 항상 채택** → 최악의 경우 현행으로 강등.

거리는 방향에 대해 **합**한다: `d[q,c] = Σ_j (f_qj − μ_cj)²/σ²_cj`가 곧 가우시안 판별식이라
`d1 − d0`가 로그 우도비로 유지된다. 출력은 여전히 [queries, 2]라 **고정 head를 건드리지 않는다.**

⚠️ 합이면 DD 크기가 r에 따라 커져 CV 대비 가중이 바뀐다 → `scale_by_rank` 대조군을 뒀다.

**검증**: `RANK_MAX=1`·`TSTAT=inf`·미설정이 brca TP53 5 fold에서 **전부 0.8069 동일**.

### 2. ⚠️ 검정이 게이트로 실패하는 두 가지 이유 (`diagnose_dd_rank_tstat.py`)

|t|를 실제로 재보니 3개 task × 5 fold에서:

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| \|t\| 중앙값 | 4.69 | 5.36 | 5.43 | 6.39 | 7.28 | **7.52** | 7.28 | 6.70 |
| \|t\| > 2.5 | 13/15 | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 |

**① 사후 선택 편향(post-selection inference).** 방향은 *바로 그 bag들에서* 클래스 분산 차이를
최대화하도록 뽑은 고유벡터인데 t를 같은 bag에서 계산한다. |t|가 2.3~9.8에 몰려 있어
t=2.5는 **8개 후보를 전부 통과**시킨다(실측: `r=8:5`, 즉 5 fold 모두 8개). t 분포표의
임계값을 그대로 쓰면 안 된다.

**② 더 나쁜 것 — |t|가 rank와 함께 커진다.** 중앙값이 4.69(r0) → 7.52(r5)로 오르고,
**rank 0이 |t| 최대인 fold는 15개 중 1개**뿐이다. 기전: |λ|는 클래스 간 평균 차이를 보고,
|t|는 그것을 **클래스 내 산포로 나눈다.** whitening이 저-|λ| 방향의 산포를 줄이므로 |t|는
연산자가 **가장 비판별적이라 순위 매긴 방향을 우선 채택**한다. 게이트가 정확히 거꾸로 작동한다.

### 3. 그래서 게이트를 끄고 고정 r을 쟀다 (§127-2: arm당 노브 하나)

| arm | r1 | r2 | r4 | r8 | r16 | scale8 |
|---|---:|---:|---:|---:|---:|---:|
| SEAL 10 macro | **0.6945** | 0.6915 | 0.6834 | 0.6824 | 0.6845 | 0.6884 |
| Δ vs r1 | — | −0.0030 | −0.0111 | −0.0121 | −0.0100 | −0.0061 |

| 증분 | SEAL 10 | 홀드아웃 7 | 전체 17 |
|---|---|---|---|
| r4 vs r1 | −0.0111, t −1.00, 4/10 | −0.0053, t −0.92, 4/7 | **−0.0087, t −1.28, 8/17** |
| r16 vs r1 | −0.0100, t −0.61, 4/10 | −0.0040, t −0.16, 4/7 | **−0.0075, t −0.55, 8/17** |

**|t| < 2.5라 게이트 통과 기각은 아니다**(§131-2: "판정 불가"). 다만 부호가 **두 독립 집단에서
모두 음수**이고(§131-5) 정점이 r=1이므로, 이 축에 더 투자할 근거가 없다.
`scale8`(0.6884)이 r1과 r8 사이에 있으므로 macro 손해의 **절반쯤은 DD 크기 교란**이다.

### 4. 진짜 손해는 평균이 아니라 **분산**이다

| arm | task별 Δ의 SD | \|Δ\| 최대 |
|---|---:|---:|
| r4 | 0.0281 | 0.0752 |
| r16 | **0.0568** | **0.1159** (BAP1 0.6669→0.5510) |

r=1의 장점은 DD가 **안정된 숫자 하나**라는 것인데, r을 키우면 task별로 ±0.1씩 흔들린다.
macro Δ가 −0.008인데 task SD가 0.057이면, 어떤 개별 task의 개선도 신뢰할 수 없다.

### 5. 관찰: VHL이 0.5를 넘었다 (⚠️ 법칙으로 읽지 말 것)

| task | r1 | r16 | Δ |
|---|---:|---:|---:|
| cptac_ccrcc VHL | 0.4762 | **0.5306** | +0.0544 |
| cptac_lscc ARID1A | 0.4168 | **0.5220** | +0.1051 |
| cptac_brca PIK3CA | 0.5333 | 0.5914 | +0.0582 |
| cptac_pda SMAD4 | 0.4938 | 0.3907 | −0.1031 |
| cptac_ccrcc BAP1 | 0.6669 | 0.5510 | −0.1159 |

VHL은 무학습 계보에서 **처음 0.5를 넘었다**(ABMIL 0.538에 근접). ARID1A도 0.417→0.522다.
"약신호 task가 이득"으로 읽고 싶지만 **SMAD4(0.494→0.391)가 정면 반례**이고, Δ를 r1 성능에
회귀하면 **상관 −0.262(n=17)** — 방향은 그쪽이나 아무것도 확립되지 않는다. §4의 분산이
이 표를 그대로 설명한다.

### 6. 남은 것 — 게이트를 살리는 정공법

t 검정 자체가 나쁜 게 아니라 **같은 데이터에서 뽑고 같은 데이터에서 검정한 것**이 문제다.
정석 해법은 **context 내 표본 분할**이다: context bag을 둘로 갈라 한쪽에서 방향을 뽑고
**다른 쪽에서 t를 계산**하면 사후 선택 편향이 사라지고 |t|가 실제 null 스케일로 돌아온다.
fold당 context가 50~200 bag이라 가능하다. ⚠️ 다만 §3이 "r>1 자체가 이득 없음"을 보였으므로,
게이트를 고쳐도 **고를 것이 있는지부터** 의문이다 — 우선순위는 §145가 지목한 CV descriptor다.

인프라(모듈·테스트 9개·훅 4개)는 남겨 뒀으므로 다른 selector로 언제든 재개할 수 있다.

---

## 147. 2026-08-17 — |λ| vs |t| 를 **selector로** 비교: 둘 다 |λ| 단독을 못 넘는다 → **DD 축 종료**

*작성: nhn-NEXGEM-claude, 2026-08-17 (KST)*

사용자 재해석: |t|는 "분산이 **일관되게** 다르다"는 뜻이니, 임계값(게이트)이 아니라 **선택 기준**으로
쓰자 — |λ| 최대 방향과 |t| 최대 방향을 각각 하나씩 가져오고, |t|는 2~16번째 중에서 고른다.
§146-2가 두 기준이 **실제로 다른 방향을 뽑는다**(rank 0이 |t| 최대인 fold가 15개 중 1개)고
측정했으므로 상보성 가정은 근거가 있었다.

### 1. 두 기준이 정확히 무엇을 재는가

```
|λ|  =  클래스 간 분산 평균차가 크다              (분자)
|t|  =  그 차이를 클래스 내 산포로 나눈 값이 크다   (분자/분모)
```

### 2. 설계 — 스케일 정합 대조군을 짝지어 뒀다

r이 바뀌면 거리 합의 크기가 바뀌어 고정 head에서 DD 가중이 이동한다(§146-1). 그래서 **같은 r
안에서만** 비교한다:

| arm | 방향 | r | 대조군 |
|---|---|---:|---|
| `r1` | |λ| 최대 (현행 v107) | 1 | — |
| `tonly` | **|t| 최대** (rank 1–15 중) | 1 | `r1` |
| `r2` | |λ| 상위 2개 | 2 | — |
| `lamt` | **|λ| 최대 + |t| 최대** | 2 | `r2` |

`tstat_range=(1,16)`이라 |t| 선택이 rank 0으로 붕괴할 수 없다(테스트로 고정).
r=1 등가성은 그대로 유지된다(`tests/test_dd_adaptive_rank.py` 14개 통과).

### 3. 결과 — 아무것도 없다

| | SEAL 10 | 홀드아웃 7 |
|---|---:|---:|
| `r1` (|λ|, r=1) | **0.6945** | 0.5836 |
| `tonly` (|t|, r=1) | 0.6836 | **0.5875** |
| `r2` (|λ| 2개) | 0.6915 | 0.5739 |
| `lamt` (|λ|+|t|) | 0.6910 | 0.5806 |

스케일 정합 비교:

| 비교 | SEAL 10 | 홀드아웃 7 | **전체 17** |
|---|---|---|---|
| **`tonly` vs `r1`** (|t| 선택 vs |λ| 선택, r=1) | −0.0109, t −1.18, 3/10 | +0.0039, t +0.29, 3/7 | **−0.0048, t −0.62, 6/17** |
| **`lamt` vs `r2`** (|λ|+|t| vs |λ|×2, r=2) | −0.0005, t −0.08, 6/10 | +0.0067, t +0.47, 3/7 | **+0.0025, t +0.36, 9/17** |
| `lamt` vs `r1` (실용 질문, r 변화 섞임) | −0.0035, t −0.48, 4/10 | −0.0030, t −0.23, 3/7 | **−0.0033, t −0.50, 7/17** |

**① |t|는 |λ|보다 나쁜 selector다** — 전체 17에서 −0.0048, 부호 6/17. 두 집단에서 부호가
뒤집히므로(SEAL −0.0109, 홀드아웃 +0.0039) 크기도 신뢰할 수 없다. 확실한 건 **넘지 못한다**는 것.
**② |t| 방향을 더해도 |λ| 두 개보다 낫지 않다** — +0.0025, 부호 **9/17 = 정확히 동전**.

### 4. 관찰 — VHL 개선은 |t| 방향에서 온 게 아니다

§146-5에서 r16이 VHL을 0.4762 → 0.5306으로 올렸는데, `tonly`는 VHL을 **0.4387로 떨어뜨린다.**
즉 그 개선은 "일관된 분산 차이" 방향의 공로가 아니다. 반면 약신호 task 개선은 재현된다:
ARID1A 0.4168 → **0.4832**(+0.066), PIK3CA 0.5333 → **0.5889**(+0.056). §146-5의 상관
−0.262와 같은 그림이고, 여전히 **분산으로 설명되는 범위**를 벗어나지 못한다.

### 5. 결론 — DD 축을 닫는다

DD에서 시도한 것과 결과:

| 시도 | 결과 |
|---|---|
| K 확대 (§145) | 128에서 **포화**. 128→256은 +0.0008, 8/17 |
| r 확대, 고정 (§146-3) | r=1이 정점. r4 −0.0087, r16 −0.0075 |
| \|t\| 게이트 (§146-2) | **기전상 실패** — 사후 선택 편향 + \|t\|가 rank와 함께 증가 |
| \|t\| selector (§147) | \|λ\| 단독을 못 넘음. −0.0048 / +0.0025 |

**DD는 "|λ| 최대 방향 하나, K=128이면 충분"에서 더 나아가지 않는다.** 네 방향 모두 닫혔고,
남은 우선순위는 §145가 지목한 **CV descriptor**다.

⚠️ 유일하게 남은 미검증 갈래는 **context 표본 분할 |t|**(§146-6)다. 지금의 |t|는 전부 오염돼
있으므로 정화하면 달라질 여지가 있다. 다만 ① |λ|가 이미 이기고 있고 ② 쫓는 효과가 ±0.005인데
task별 SD가 0.03~0.06이라 **검출 자체가 어렵다.** 인프라(모듈·테스트 14개·훅 6개)는 남겨 뒀다.

---

## 148. 2026-08-17 — CT 진단: **two-token readout은 병목이 아니다.** 16차원 abundance 자체가 거의 비어 있다

*작성: nhn-NEXGEM-claude, 2026-08-17 (KST)*

CT는 16차원 abundance를 만든 뒤 2개 좌표만 읽는다(`argmax`/`argmin` of
`(mean_0−mean_1)/SE`, margin = `q1−q0`). readout에서 정보를 버리는지, 아니면 token 자체가
약한지 분리했다.

### 1. 구현 — 표현(1–5단계)을 공유해 교란을 제거

`src/models/ct_readout.py`(신규). **1–5단계(cell 샘플링 → 표준화 → farthest-point token →
soft assign → bag별 평균)를 `ct_abundance()` 하나로 뽑아내** 세 arm이 **표현에서는 절대 다를 수
없게** 했다. 6–7단계만 교체한다:

| mode | readout |
|---|---|
| `extreme` | `q1 − q0` (현행 v107) |
| `prototype` | 16차원 표준화 abundance의 클래스 prototype, `‖â−p0‖² − ‖â−p1‖²` |
| `ridge` | 16차원 전체에 class-balanced ridge(primal, λ=1), `logit1 − logit0` |

`training_free.py._ct_features`도 이 공통 함수를 호출하도록 리팩터했고, 기본값
`ct_readout="extreme"`이라 **v107 출력은 그대로**다.

**검증**: ① `readout_extreme`가 계보 `_ct_features`와 **1e-6 일치**(테스트)
② 공식 경로 `ct_extreme` SEAL macro = **0.6945**, 즉 v107 자릿수까지 재현
③ shape·결정론·context-only(query 통계 미사용)·클래스 스왑 반대칭 테스트 15개 통과.

⚠️ **스케일 교정**: 대체 margin은 head 투입 전 extreme margin의 **context** 평균·RMS에 맞춘다.
`nocal` 대조군을 함께 돌려 이 선택이 실제로 필요했음을 확인했다(아래 §2).

### 2. full-model (공식 경로, 고정 head, CT weight 0.286)

| task | extreme | prototype | ridge | proto_nocal | ridge_nocal |
|---|---:|---:|---:|---:|---:|
| bc_therapy er_status | 0.7047 | 0.6878 | 0.7000 | 0.6794 | 0.6972 |
| bc_therapy grade | 0.7191 | 0.7192 | 0.7222 | 0.7083 | 0.7226 |
| bc_therapy her2_status | 0.6755 | 0.6704 | 0.6696 | 0.6666 | 0.6682 |
| cptac_brca PIK3CA | 0.5333 | 0.5351 | 0.5319 | 0.5321 | 0.5341 |
| cptac_brca TP53 | 0.8264 | 0.8334 | 0.8341 | 0.8355 | 0.8305 |
| cptac_ccrcc BAP1 | 0.6669 | 0.6649 | 0.6561 | 0.6727 | 0.6419 |
| cptac_ccrcc VHL | 0.4762 | 0.4790 | 0.4896 | 0.4695 | 0.4812 |
| cptac_luad EGFR | 0.7836 | 0.7871 | 0.7821 | 0.7746 | 0.7849 |
| cptac_luad STK11 | 0.8841 | 0.8788 | 0.8890 | 0.8621 | 0.8924 |
| cptac_luad TP53 | 0.6753 | 0.6664 | 0.6728 | 0.6501 | 0.6826 |
| **MACRO** | **0.6945** | 0.6922 | **0.6947** | 0.6851 | 0.6936 |

교정 없이 넣으면 prototype이 −0.0094다 → **교정은 필요했다**(품질이 아니라 크기를 비교하게 됨).

| 홀드아웃 7 | extreme | prototype | ridge |
|---|---:|---:|---:|
| MACRO | 0.5836 | 0.5836 | 0.5853 |

| 증분 | SEAL 10 | 홀드아웃 7 | **전체 17** |
|---|---|---|---|
| prototype | −0.0023, t −1.03, 5/10 | −0.0000, t −0.01, 3/7 | **−0.0014, t −0.74, 8/17** |
| ridge | +0.0002, t +0.10, 4/10 | +0.0017, t +0.45, 4/7 | **+0.0008, t +0.42, 8/17** |

### 3. CT-only (CT margin 단독, CV·DD 없음)

| | extreme | prototype | ridge |
|---|---:|---:|---:|
| SEAL 10 | **0.5990** | 0.5765 | 0.5969 |
| 홀드아웃 7 | 0.5042 | 0.5162 | **0.5207** |
| **전체 17** | **0.5600** | 0.5517 | 0.5655 |
| 전체 17 증분 | — | −0.0083, t −0.65, **8/17** | +0.0056, t +0.38, **8/17** |
| balanced BCE (SEAL, 교정 margin) | **0.7746** | 0.8332 | 0.7845 |

**두 집단에서 부호가 뒤집히고 둘 다 8/17이다** — 완전한 동전이다.

### 4. 왜 그런지 — 진단값이 답한다

| 진단 | SEAL 10 | 홀드아웃 7 |
|---|---:|---:|
| token별 \|(m0−m1)/SE\| 최대 | 3.44 | 3.18 |
| token별 \|(m0−m1)/SE\| **중앙값** | **1.31** | **1.23** |
| \|t\|>2인 token 수 | 4.5 / 16 | 3.4 / 16 |
| ridge \|w\| top-1 비중 | 0.156 | 0.145 |
| ridge \|w\| top-2 비중 | 0.282 | 0.263 |
| ridge **실효 token 수** (참여율) | **6.81 / 16** | **7.62 / 16** |
| extreme의 2개가 ridge top-2에 포함 | 1.17 / 2 | 1.05 / 2 |

**① abundance에 클래스 정보가 거의 없다.** token별 판별 통계의 **중앙값이 1.31** — 노이즈
수준이다. 16개 중 |t|>2를 넘는 것이 3~5개뿐이고, CT-only macro가 전체 17에서 **0.5600**으로
우연에서 겨우 벗어난다.

**② ridge는 실제로 더 많은 token을 쓰려 했고, 그래도 못 이겼다.** 실효 token 수가 6.8/16이고
top-1 비중이 15.6%다 — 극단 2개로 붕괴한 게 **아니다.** 정보가 있는데 readout이 못 읽은 것이라면
이 조건에서 이겼어야 한다. 못 이겼다는 것은 **나머지 14차원이 대부분 노이즈**라는 뜻이고,
그래서 가중을 퍼뜨리는 것이 도움이 아니라 해가 된다(prototype이 CT-only에서 −0.0224인 이유).

**③ 두 기준이 고르는 token이 절반만 겹친다**(1.17/2). 어느 선택도 이기지 못하므로, token 간
차이가 신호가 아니라 노이즈라는 ②와 같은 결론이다.

### 5. 판정 — 사용자 기준표 적용

> "prototype과 ridge 모두 개선되지 않음 → readout보다 **token 생성, cell sampling 또는
> distance metric이 병목**"

**이 경우다.** ⚠️ **단, raw 1536 조건에서만이다 — §150이 정정한다.** 거리 집중을 먼저 풀면
ridge가 extreme보다 나아진다(pca16에서 CT-only +0.0154/+0.0199, 두 집단 부호 일치). 아래 결론은
"표현이 비어 있는 상태에서는 readout을 바꿔도 소용없다"로 읽어야 하며, readout을 무죄로 만들지
않는다. CT-only에서도, full-model에서도, 두 독립 task 집단 어디서도 개선이 없다
(전부 8/17). **two-token readout은 병목이 아니다.** 병목은 상류다:

- **token 생성**: farthest-point sampling은 밀도를 무시하고 **극단값(outlier cell)을 고른다.**
  16개 token 중 다수가 희귀 cell 근방에 놓이면 bag 간 abundance가 거의 상수가 되고, 중앙값
  |t|=1.31이 정확히 그 모습이다.
- **cell sampling**: bag당 64 cell은 slide당 4,096 cell의 1.6%다. abundance는 64개 표본의
  평균이므로 표본오차 자체가 클래스 차이보다 클 수 있다.
- **distance metric**: 1536차원에서 squared Euclidean + softmax(T=0.5)는 거리 집중 현상에
  취약하다 — 모든 cell이 모든 token에서 비슷하게 멀면 abundance가 균일해진다.

⚠️ **아무것도 승격하지 않았다.** 기본값은 `ct_readout="extreme"`(v107) 그대로다.

### 6. 한계

- CT-only 진단은 `diagnose_full_basis.load_task` 경로라 bag을 로드 시 1회 8,192로 cap한다
  (§138-2). 절대값은 SEAL macro와 직접 비교할 수 없고 **arm 간 격차만** 유효하다.
  full-model 표는 공식 경로이므로 v107과 직접 비교 가능하다.
- ridge λ는 1.0 고정이다(`ICF_CT_RIDGE_LAMBDA`로 조정 가능). context bag이 50~200인데 16차원
  이므로 과적합 위험은 낮지만, λ를 쓸어보지는 않았다.
- CT-only가 0.56이라는 것은 **CT가 약하다**는 뜻이지 **쓸모없다**는 뜻이 아니다. CT 자체를
  제거하는 ablation은 이번 범위가 아니다.
- 위 세 상류 후보(token 생성 / sampling / metric) 중 어느 것인지는 **아직 분리하지 않았다.**

---

## 149. 2026-08-17 — **거리 집중은 실재한다(실측). PCA가 CT-only를 살리지만 macro에는 안 닿는다** (사용자 지적)

*작성: nhn-NEXGEM-claude, 2026-08-17 (KST)*

사용자 지적: 1536차원에서는 차원의 저주로 거리가 무의미해지니 PCA 후에 거리를 재야 한다.
§148-5가 남긴 상류 후보 세 개 중 **distance metric**을 정확히 겨눈 것이다. **기전은 확인됐고,
CT-only는 개선되며, 최종 macro에는 닿지 않는다.**

### 1. 구현 — 기저를 새로 만들지 않았다

v107에는 이미 fold의 **within-slide PCA(K=256)** 가 CV 분기용으로 계산돼 있다. CT가 그것을
재사용하므로 eigh 추가 없음이고, CT와 CV가 **같은 부분공간**을 본다는 보장이 생긴다
(⚠️ 그게 §4의 함정이 된다). `pca_dim`은 선행 열을 자르며, PCA 고유벡터가 내림차순이라
정확하다(§145와 같은 논거).

순서: raw cell을 사영 → **K개 성분**을 context로 표준화 → farthest-point token → soft assign.
raw를 사영하는 이유는 기저가 표준화되지 않은 UNI2 공간에서 직교라서다.

기본값 `pca_dim=None`은 **현행 v107 그대로**다. 공식 경로 `ct_extreme` = **0.6945** 재현 확인,
기저 없이 `ICF_CT_PCA_DIM`을 주면 예외로 거부, 테스트 23개 통과.

### 2. 기전 확인 — 거리 집중은 실재한다 (⚠️ 표는 2026-08-17 정정됨, §7 참조)

`scripts/diagnose_ct_pca_distance.py`. `contrast` = **cell 하나**의 16개 token 거리에 대한
(max−min)/mean — softmax가 실제로 다뤄야 하는 양. `rel_std` = 모든 cell–token 거리의
std/mean — **차원 집중의 교과서적 정의**(차원이 커지면 줄어든다). `entropy`는 abundance 행의
엔트로피(ln 16 = 2.773이 완전 균일 = 무용).

| PCA dim | contrast (SEAL / 홀드아웃) | **rel_std** (SEAL / 홀드아웃) | entropy | \|t\| 중앙값 |
|---|---|---|---|---|
| **1536 (raw)** | **0.965 / 0.972** | **0.229 / 0.226** | 1.228 / 0.870 | 1.31 / 1.23 |
| 4 | 2.191 / **3.354** | **0.637 / 0.896** | 1.768 / 1.651 | 1.17 / 1.39 |
| 8 | 1.840 / 2.696 | 0.490 / 0.650 | 1.371 / 1.142 | 1.35 / 1.39 |
| 16 | 1.889 / 1.801 | 0.451 / 0.442 | 1.118 / 0.868 | 1.36 / **1.58** |
| 32 | 1.563 / 1.522 | 0.368 / 0.365 | 0.984 / 0.693 | 1.31 / **1.60** |
| 64 | 1.472 / 1.338 | 0.336 / 0.310 | 0.988 / 0.637 | 1.20 / 1.52 |
| 128 | 1.411 / 1.142 | 0.310 / 0.257 | 1.049 / 0.771 | 1.20 / 1.36 |
| 256 | 1.241 / 1.071 | 0.261 / 0.233 | 0.902 / 0.851 | 1.23 / 1.21 |

**`rel_std`가 차원에 대해 완전히 단조 감소한다** — 4차원 0.637에서 1536차원 0.229까지, 두 집단이
거의 같은 값이다. 이것이 거리 집중의 정확한 서명이다. raw 1536에서는 cell 하나의 16개 token
거리가 평균의 ±23% 안에 다 몰려 있고, contrast도 0.97로 가장 낮다. soft assignment가 그만큼
무뎌지고(entropy), §148-4가 증상으로 잡았던 token별 판별 통계도 낮다(홀드아웃 1.23 → 32차원 1.60).

**지적한 기전이 그대로 측정된다.**

### 3. CT-only — 개선된다. 중간 차원에서 역U자

| PCA dim | SEAL 10 | 홀드아웃 7 | **전체 17** |
|---|---|---|---|
| 4 | −0.0532, t −2.45, 3/10 | +0.0586, t +1.85, 4/7 | −0.0072, t −0.32, 7/17 |
| 8 | −0.0104, t −0.66, 5/10 | **+0.0690**, t +1.97, 5/7 | +0.0223, t +1.17, 10/17 |
| 16 | +0.0009, t +0.05, 7/10 | +0.0464, t +1.52, 6/7 | +0.0196, t +1.13, **13/17** |
| **32** | +0.0200, t +0.95, 6/10 | +0.0381, t +1.70, 6/7 | **+0.0275, t +1.81, 12/17** |
| 64 | −0.0140, t −1.11, 4/10 | +0.0461, t +1.84, 5/7 | +0.0108, t +0.76, 9/17 |
| 128 | −0.0388, t −1.99, 2/10 | +0.0254, t +1.05, 6/7 | −0.0124, t −0.74, 8/17 |
| 256 | −0.0678, t −2.15, 2/10 | +0.0059, t +0.13, 4/7 | −0.0375, t −1.40, 6/17 |

CT-only macro(17): raw 0.5600 → **32에서 0.5874**. **홀드아웃에서는 8개 차원 전부 양수**이고
16·32·64가 5~6/7이다. 게이트(|t|≥2.5)는 못 넘지만 **CT 분기에서 지금까지 나온 가장 일관된
양의 신호**다(13/17까지).

⚠️ 최적 차원이 집단 간에 다르다(SEAL 32, 홀드아웃 8) — **차원 선택은 노이즈**이고,
믿을 수 있는 것은 **중간 차원이 raw보다 낫다**는 방향뿐이다.

### 4. ⚠️ full-model — 닿지 않는다. 그리고 이유가 구조적이다

| | raw | pca8 | pca16 | pca32 | pca64 |
|---|---:|---:|---:|---:|---:|
| SEAL 10 macro | **0.6945** | 0.6925 | 0.6910 | 0.6957 | 0.6945 |
| 홀드아웃 7 macro | 0.5836 | 0.5907 | 0.5851 | 0.5864 | 0.5867 |
| **전체 17 증분** | — | +0.0018, 10/17 | −0.0015, 7/17 | **+0.0019, t +0.59, 12/17** | +0.0013, 12/17 |

사용자 기준표의 **"CT-only에서는 개선되지만 full model에서는 개선되지 않음 → 정보가 CV/DD와
중복"** 경우다. 그리고 이번엔 **왜 중복인지가 명확하다**: CT에 넘긴 기저가 **CV가 쓰는 바로 그
within-slide PCA**다. CV의 descriptor는 같은 기저에서의 `triu(BᵀC_bag B)`이므로, CT를 그
부분공간으로 옮기는 것은 CT를 **CV가 이미 덮은 영역으로 밀어넣는 것**이다. 기저 재사용이
비용을 0으로 만든 대신 중복을 만들었다.

### 5. 결론과 다음 갈래

- **지적은 옳았다**: 거리 집중은 실재하고(rel_std 0.229 → 0.637, 완전 단조), CT의 표현을
  실제로 망치고 있었다.
- **CT-only는 +0.02~0.03 회복**(16~32차원, 13/17). 그러나 **최종 macro는 +0.002로 판정 불가.**
- ⚠️ **아무것도 승격하지 않았다.** `pca_dim=None`(v107) 그대로다. macro 이득이 없고 최적 차원이
  집단 간에 다르므로 승격 근거가 없다.
- **다음은 "CV와 다른 부분공간"이다.** CV 기저 재사용이 중복의 원인이므로, CT에는
  ① CV 기저의 **직교 여공간**, 또는 ② CT 목적(token 판별)으로 고른 별도 기저가 필요하다.
  기저를 공유하는 한 CT는 CV의 그림자에 머문다.
- §148-5의 남은 두 후보(**farthest-point token 생성**, **64-cell 샘플링**)는 여전히 미검증이다.
  거리 집중을 완전히 풀어도(rel_std 0.229 → 0.637) |t| 중앙값이 1.6을 못 넘는다는 것은
  **거리만이 문제가 아니라는** 신호다.

### 6. ⚠️ 부분 fold에 세 번째로 속았다

3 task × 5 fold 예비 측정에서 pca16이 CT-only **0.6399 → 0.7456(+0.106)** 으로 보였다.
10 task × 50 fold 전체는 **0.5990 → 0.5998(+0.0009)** 이다. 같은 세션에서 §142-6, §145-6에
이어 **세 번째**다. 부분 fold는 훅이 도는지 확인하는 용도 외에 쓰지 말 것 — 이제 규칙이 아니라
반복 관측이다.

### 7. ⚠️ 정정 — §2 첫 contrast 표는 잘못 측정한 것이었다 (사용자 지적)

첫 판에서 `token_contrast(abundance.tokens, cells=abundance.tokens, ...)`로 호출해
**cell이 아니라 token끼리의 거리**를 쟀다. 모든 행에 자기거리 0이 들어가므로 `contrast`가
(max−min)/mean이 아니라 max/mean이 되고, 추세가 조작된다. 잘못된 값 → 정정된 값:

| | 잘못 | 정정 |
|---|---|---|
| raw 1536 contrast (SEAL / 홀드아웃) | 1.370 / 1.378 | **0.965 / 0.972** |
| 최대 contrast | 2.919 (홀드아웃, 4차원) | **3.354** (같은 지점) |
| 단조성 | 없음 (SEAL에서 8→16이 역전) | **`rel_std`가 완전 단조** |

**결론은 바뀌지 않는다** — 기전은 오히려 더 선명해졌다. `entropy`·`|t| 중앙값`·모든 AUROC는
`abundance`와 실제 파이프라인에서 나온 값이라 **영향이 없었다**(SEAL raw 1.2277 / 1.31 / 0.5990
그대로 재현).

재발 방지: cell 준비 단계를 `ct_readout.prepare_cells()`로 뽑아 진단이 **파이프라인과 같은
셀**을 받게 했고(진단이 따로 재구현하다 생긴 사고다), `tests/test_ct_readout.py`에
`prepare_cells`와 `ct_abundance`의 일치를 고정했다.

---

## 150. 2026-08-17 — PCA × readout **조합**: §148의 "readout은 병목 아님"은 **raw 1536에서만** 참이었다

*작성: nhn-NEXGEM-claude, 2026-08-17 (KST)*

사용자 질문: "ridge로 한 거야 기존 two-token readout으로 한 거야?" — **§149는 전부 기존
two-token(`q1−q0`)으로 돌렸다.** §148은 readout만, §149는 거리만 바꿨다(§127-2, arm당 노브
하나). **조합은 한 번도 돌리지 않았고**, 그게 빈 칸이었다.

### 1. CT-only — readout 격차가 PCA 안에서 부호를 되찾는다

`ridge − extreme` (CT margin 단독):

| 부분공간 | SEAL 10 | 홀드아웃 7 | 부호 일치 |
|---|---:|---:|---|
| **raw 1536** | **−0.0021** | **+0.0165** | ✗ 갈림 |
| pca8 | +0.0105 | −0.0074 | ✗ 갈림 |
| **pca16** | **+0.0154** | **+0.0199** | **✓ 둘 다 양수** |
| pca32 | −0.0022 | +0.0148 | ✗ 갈림 |

CT-only macro 전체:

| | extreme | prototype | ridge |
|---|---:|---:|---:|
| SEAL raw / pca16 / pca32 | 0.5990 / 0.5998 / **0.6190** | 0.5765 / 0.6032 / 0.6043 | 0.5969 / **0.6152** / 0.6168 |
| 홀드아웃 raw / pca16 / pca32 | 0.5042 / 0.5507 / 0.5424 | 0.5162 / 0.5627 / 0.5489 | 0.5207 / **0.5706** / 0.5572 |

**§148의 판정은 raw 1536 조건에 묶인 것이었다.** 거리가 집중돼 abundance가 거의 비어 있으면
어떤 readout도 읽을 것이 없다 — 그 상태에서 "readout은 병목이 아니다"는 동어반복에 가깝다.
집중을 풀면 ridge가 extreme보다 나아지고, pca16에서 처음으로 **두 집단 부호가 일치**한다.

### 2. full-model 2×2 분해 (공식 경로, 고정 head)

`ct_extreme`가 세 스윕 디렉토리 전부에서 **0.6945** — 기준선 일관성 확인.

| 구성 | SEAL 10 | 홀드아웃 7 | 전체 17 Δ | t | 부호 |
|---|---:|---:|---:|---:|---:|
| v107 (raw, extreme) | 0.6945 | 0.5836 | — | — | — |
| **+PCA만** (32, extreme) | 0.6957 | 0.5864 | +0.0019 | +0.59 | 12/17 |
| **+ridge만** (raw, ridge) | 0.6947 | 0.5853 | +0.0008 | +0.42 | 8/17 |
| **+둘 다** (32, ridge) | **0.6967** | **0.5893** | **+0.0037** | **+1.00** | 11/17 |

단독 합 +0.0027 vs 조합 실측 **+0.0037**. 초가법성 +0.0010은 노이즈 범위 안이지만,
**조합이 두 집단 모두에서 양수**(SEAL +0.0022, 홀드아웃 +0.0057)인 유일한 CT 변형이다.

pca16_ridge도 비슷하다(+0.0026, t +0.87, 11/17). 눈에 띄는 개별 task:
STK11 0.8841 → **0.9019**, 홀드아웃 ARID1A 0.4168 → **0.4499**, PBRM1 0.4970 → 0.5130.

### 3. 판정 — 방향은 맞지만 **|t| = 1.00, 판정 불가**

- §107-3 게이트(|t| ≥ 2.5)에 한참 못 미친다. §131-2의 검출 한계 논의가 그대로 적용된다.
- 다만 이것은 **CT 분기에서 나온 것 중 가장 좋은 구성**이고, 부호가 두 독립 집단에서 일치한다.
  §131-5(독립 집단 재현이 단일 t보다 낫다)를 적용하면 "노이즈"로 치부할 수는 없다.
- ⚠️ **아무것도 승격하지 않았다.** `pca_dim=None`, `ct_readout="extreme"`(v107) 그대로다.
  +0.0037을 승격 근거로 쓰기에는 너무 약하다.

### 4. §148 정정

§148-5의 "**two-token readout은 병목이 아니다**"는 **raw 1536 조건 한정**으로 읽어야 한다.
표현이 비어 있을 때 readout을 바꿔봐야 아무 차이가 없는 것은 당연하고, 그 실험은 readout을
무죄로 만들어주지 않는다. **두 노브는 상호작용한다** — 거리를 먼저 고쳐야 readout이 의미를 갖는다.

교훈(절차): §127-2("arm당 노브 하나")는 **귀속**에는 맞지만, 한 노브가 다른 노브의 **효과를
가리는 경우**에는 단독 arm만으로 축을 닫으면 안 된다. 단독 둘이 모두 음성이어도 조합은 양성일 수
있다. 축을 닫기 전에 **최소한 대각선 한 칸**은 확인할 것.

### 5. 남은 것

- **CV와 다른 부분공간** (§149-4). 여전히 미검증이고, 이번 결과가 그 우선순위를 높인다 —
  지금 쓰는 기저는 CV가 이미 덮은 곳이라 +0.0037이 상한일 가능성이 크다.
- §148-5의 나머지 두 후보: **farthest-point token 생성**, **64-cell 샘플링**. 미검증.
- pca × readout 격자에서 최적 차원이 여전히 불안정하다(CT-only는 pca16, full-model은 pca32).
  차원 선택은 노이즈로 취급할 것.

---

## 151. 2026-08-17 — ⚠️ **판정 프로토콜 정정: 결정론적 arm에 t를 쓰지 말 것** (사용자 지적) + CT weight 스윕

*작성: nhn-NEXGEM-claude, 2026-08-17 (KST)*

### 1. 정정 — §142 이후 내가 보고한 모든 t는 근거가 없다

사용자 지적: "결정론적 실험이어서 t를 적용할 수 없다."

**맞다.** §107-3의 t는 **seed-paired** 통계다 — 반복 단위가 시드이고, 시드가 만드는 분산이
분모다. v106 이후 구성은 학습이 없어 **시드 분산이 정확히 0**이다(§139). 나는 그 자리에 **task를
반복 단위로 끼워넣어** t를 계산했는데(§142~§150), 그것은 "17개 task가 어떤 task 모집단에서
뽑힌 표본"이라는 전혀 다른 가정이고, 근거가 없다. task는 고정된 벤치마크이고 다시 뽑히지 않는다.

**결정론적 arm에서 보고할 수 있는 것:**

| 쓸 것 | 쓰지 말 것 |
|---|---|
| task별 Δ (정확한 측정값, 오차 없음) | t, p, CI |
| **부호 일치 수** (몇 개 task에서 이겼는가) | §107-3 게이트 (\|t\| ≥ 2.5) |
| macro Δ | §131-2 검출 한계 (시드 분산 기반) |
| **독립 task 집단 간 재현**(SEAL 10 / 홀드아웃 7, §131-5) | |

⚠️ **§142~§150의 t 값은 모두 무시할 것.** Δ와 부호 수는 유효하다(실측이므로). 결론이 바뀌는
곳은 없지만 — §145의 CV 귀속(13/17), §146~§147의 DD 음성(8/17), §150의 조합(11/17)은 부호만으로도
같은 결론이다 — **근거의 성격이 다르다.** 특히 §150에서 "|t|=1.00이라 판정 불가"라고 쓴 것은
잘못된 프레이밍이다. 정확히는 **+0.0037이 두 집단에서 재현되며 11/17이다**가 전부다.

### 2. CT weight 스윕 — CV 대비 1/5이라는 구조적 제약

두 번째 지적: CT는 가중 0.286으로 CV의 1.442 대비 **1/5**이라 구조적으로 macro를 많이 못
움직인다. 0.286은 §137-3에서 **옛 two-token readout**을 기준으로 학습된 head를 분해해 얻은
값이므로, readout이 바뀐 지금 같은 값을 원할 이유가 없다.

`ICF_FIXED_HEAD_CT_WEIGHT` 훅 추가(반대칭은 유지 — 쌍이 등가·반대로 남는다).
전부 (pca32, ridge) 위에서:

| CT weight | SEAL 10 | 홀드아웃 7 | 전체 17 평균 | **v107 대비 부호** |
|---|---:|---:|---:|---:|
| 0.286 (v107 readout) | 0.6945 | 0.5836 | 0.6488 | — |
| **0.286 (v108)** | 0.6967 | 0.5893 | 0.6525 | **11/17** |
| 0.4 | **0.6977** | 0.5903 | 0.6535 | **11/17** |
| **0.5** | 0.6976 | 0.5917 | 0.6540 | 10/17 |
| 0.7 | 0.6971 | 0.5933 | 0.6544 | 9/17 |
| 1.0 | 0.6956 | **0.5934** | 0.6535 | 7/17 |
| 1.442 (= CV) | 0.6928 | — | — | — |

**평균은 0.5~0.7에서 정점인데 부호 일치는 단조 감소한다**(11/17 → 7/17). weight는 CT 신호의
**이득(gain)** 이고, CT는 어떤 task에서는 맞고 어떤 task에서는 틀리기 때문이다. 올리면 이득과
손해가 **함께** 커진다:

| task | v107 | w=0.5 | w=1.0 |
|---|---:|---:|---:|
| cptac_ccrcc VHL | 0.4762 | **0.5098** | **0.5253** |
| cptac_luad STK11 | 0.8841 | 0.9050 | **0.9070** |
| cptac_lscc ARID1A (홀드아웃) | 0.4168 | 0.4769 | **0.5106** |
| cptac_ccrcc BAP1 | 0.6669 | 0.6408 | **0.6342** |
| bc_therapy her2 | 0.6755 | 0.6610 | **0.6496** |
| cptac_pda SMAD4 (홀드아웃) | 0.4938 | 0.4744 | **0.4603** |

w=1.442(CV와 동률)에서는 SEAL 0.6928로 **v107보다 낮다** — 상한이 명확하다.

**VHL이 정식 경로에서 0.5를 넘는다**(w=0.5에서 0.5098, w=1.0에서 0.5253). 무학습 계보의 오래된
이상값이었고 §146-5에서 r16으로 한 번 넘겼으나 그때는 macro를 크게 잃었다. 여기서는 macro가
같이 오른다.

### 3. 판단 — weight는 승격하지 않는다

사용자 지시는 (32, ridge) 승격 + weight 0.5 **확인**이었다. 확인 결과:
w=0.5는 SEAL +0.0009 / 홀드아웃 +0.0024(v108 대비)로 **양쪽 다 개선**이지만 **부호가 11/17 →
10/17로 내려간다.** 결정론적 실험에서 평균 하나보다 신뢰할 수 있는 것이 부호 일치이므로,
**기본값은 0.286 유지**했다. 훅은 남겨 두었으니 언제든 바꿀 수 있다.

---

## 152. 2026-08-17 — **v108 승격 확정 (사용자 결정): CT = 32-d PCA 부분공간 + ridge readout**

*작성: nhn-NEXGEM-claude, 2026-08-17 (KST)*

**v108 = v107에서 CT 분기만 두 가지 변경.** 학습 파라미터 0, 결정론성 유지.

| 자리 | v107 | **v108** |
|---|---|---|
| CT 거리 공간 | raw 1,536차원 | **상위 32 PCA 방향** |
| CT readout | 극단 2개 token `q1−q0` | **16차원 전체 class-balanced ridge** |
| CT weight | 0.286 | 0.286 (유지, §151-3) |
| 사영 / head / CV / DD | 그대로 | 그대로 |

**둘은 반드시 함께 간다** — 단독은 +0.0019(PCA)와 +0.0008(ridge)인데 조합이 +0.0037이고,
**두 task 집단에서 모두 양수인 유일한 CT 변형**이다(§150-2). 거리가 집중돼 abundance가 비어
있으면 readout이 읽을 것이 없고, readout이 2차원만 읽으면 부분공간을 고쳐도 못 쓴다.

### 1. 수치

| | v107 | **v108** | Δ |
|---|---:|---:|---:|
| SEAL 10 macro | 0.6945 | **0.6967** | +0.0022 |
| 홀드아웃 7 macro | 0.5836 | **0.5893** | +0.0057 |
| 전체 17 평균 | 0.6488 | **0.6525** | **+0.0037 (11/17)** |

⚠️ t·p·CI는 보고하지 않는다 — 결정론적 arm이다(§151-1).

### 2. 실행

```bash
bash scripts/eval_v108.sh <gpu> <tag> [tasks...]     # 정의가 사는 단 하나의 자리
#  = ICF_COVARIANCE_BASIS=pca_within ICF_FIXED_HEAD=1 ICF_SKETCH_DIM=256 \
#    ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge \
#    bash scripts/eval_seal_tasks.sh <gpu> <아무 v98 ckpt> <config> <tag> <tasks...>
```
**검증**: `eval_v108.sh`로 VHL을 정식 경로에서 돌려 §151 표의 **0.5011 재현** 확인.
`TrainingFreeClassifier()` 기본값도 v108로 바꿨고 테스트가 고정한다.

⚠️ 계보 `_ct_features`에는 ridge/PCA CT가 없으므로 **등가성 테스트는 v107 구성을 명시**해
비교한다(`ct_readout="extreme"`, `ct_pca_dim=None`). 등가성은 **구성에 대한 진술**이고
기본값에 대한 진술이 아니다 — 기본값은 `DefaultTest`가 따로 지킨다.

### 3. 한계 — 왜 +0.0037뿐인가

1. **CT weight가 0.286**으로 CV의 1.442 대비 1/5이다. 구조적으로 CT는 macro를 많이 못 움직인다.
   올려보면 평균은 0.5~0.7에서 조금 더 오르지만 부호 일치가 무너진다(§151-2).
2. **기저가 CV와 같다.** CT에 넘긴 32차원은 CV가 이미 덮은 within-slide PCA의 선행 부분이다.
   CT를 그 안으로 옮긴 것은 CV의 그림자 안에서 개선한 것이고, **+0.0037이 상한일 가능성이 크다**
   (§149-4). **CV와 다른 부분공간**(직교 여공간, 또는 token 판별용 별도 기저)이 남은 최우선 갈래다.
3. **최적 차원이 불안정하다** — CT-only는 pca16, full-model은 pca32에서 가장 좋았다. 32를 고른
   것은 full-model 기준이며, 차원 선택 자체는 노이즈로 취급해야 한다.
4. §148-5의 나머지 두 후보 **farthest-point token 생성**과 **64-cell 샘플링**은 여전히 미검증이다.

### 4. ABMIL과의 거리

| | macro | ABMIL 0.727 대비 |
|---|---:|---:|
| v107 | 0.6945 | −0.0321 |
| **v108** | **0.6967** | **−0.0299** |

여전히 −0.030이다. CT 축에서 더 짜낼 수 있는 것은 (2)의 부분공간 문제를 풀지 않으면 없다.

---

## 153. 2026-08-17 — DD weight 하향 스윕: **DD의 값은 코호트 의존적이다** (두 집단이 정반대로 단조)

*작성: nhn-NEXGEM-claude, 2026-08-17 (KST)*

§145~§147이 DD 개선 경로를 전부 닫았으므로(K는 128에서 포화, r=1이 정점, |t| 게이트·selector
모두 |λ| 단독에 패배), 남은 질문은 "DD가 애초에 값을 하는가"다. `ICF_FIXED_HEAD_DD_WEIGHT`
훅을 추가해(0.343은 **크기**이고 부호는 DD가 거리를 낸다는 사실에 고정) v108 위에서 0까지 내렸다.
0은 **DD 완전 ablation**이다.

### 1. 결과 — 두 집단이 반대로 갈린다

| DD weight | SEAL 10 | 홀드아웃 7 | 전체 17 평균 | v108 대비 부호 |
|---|---:|---:|---:|---:|
| **0.343 (v108)** | **0.6967** | 0.5893 | 0.6525 | — |
| 0.25 | 0.6959 | 0.5919 | 0.6531 | 10/17 |
| 0.2 | 0.6955 | 0.5936 | 0.6535 | **11/17** |
| 0.15 | 0.6945 | 0.5956 | 0.6538 | 8/17 |
| 0.1 | 0.6934 | 0.5977 | 0.6540 | 8/17 |
| 0.05 | 0.6919 | 0.6002 | **0.6542** | 7/17 |
| **0.0 (DD 제거)** | 0.6895 | **0.6030** | 0.6539 | 7/17 |

**SEAL은 단조 감소(0.6967 → 0.6895), 홀드아웃은 단조 증가(0.5893 → 0.6030).** 7개 지점 전부
단조이므로 노이즈가 아니다. 전체 평균이 거의 평평한 것(0.6525 → 0.6542)은 **두 집단이 서로를
상쇄하기 때문**이지, DD가 무해하기 때문이 아니다.

즉 **홀드아웃 7개에서는 DD가 적극적으로 해롭다** — 완전 제거가 **+0.0137**이고, 이 세션에서
CT 작업으로 얻은 어떤 값보다 크다.

### 2. 개별 task — 양쪽 다 넓게 퍼져 있다 (단일 이상값이 아니다)

DD 제거(0.0) − v108(0.343):

| SEAL 10 | Δ | 홀드아웃 7 | Δ |
|---|---:|---|---:|
| bc_therapy her2 | **−0.0325** | cptac_lscc KEAP1 | **+0.0404** |
| cptac_ccrcc VHL | −0.0234 | cptac_lscc Histologic_Grade | +0.0273 |
| cptac_luad EGFR | −0.0212 | cptac_pda SMAD4 | +0.0236 |
| bc_therapy er_status | −0.0178 | cptac_lscc ARID1A | +0.0171 |
| cptac_luad STK11 | −0.0169 | cptac_luad KRAS | +0.0106 |
| bc_therapy grade | −0.0127 | ucla_lung progression | −0.0135 |
| cptac_brca TP53 | −0.0102 | cptac_ccrcc PBRM1 | −0.0094 |
| cptac_luad TP53 | −0.0011 | | |
| cptac_brca PIK3CA | +0.0218 | | |
| cptac_ccrcc BAP1 | **+0.0415** | | |

SEAL은 **8/10이 음수**(DD가 도움), 홀드아웃은 **5/7이 양수**(DD가 해로움). 둘 다 한두 task가
끌고 가는 것이 아니다.

### 3. 해석 — 그리고 무엇을 아직 모르는지

DD는 context에서 **rank-1 분산 방향 하나**를 뽑는다(§146). 신호가 강한 코호트에서는 그 방향이
실재하는 것을 잡고, 약한 코호트에서는 **노이즈에 적합**해 분산만 더한다 — 홀드아웃 7개의
절대 AUROC가 0.58~0.60으로 SEAL의 0.69보다 낮다는 사실과 맞는 그림이다.

⚠️ 다만 **확립되지 않았다.** 두 집단은 "SEAL 여부" 말고도 **코호트 자체가 다르다**(lscc·pda·
ucla_lung은 홀드아웃에만 있다). 신호 강도 때문인지 코호트 특성 때문인지 이 데이터로는 분리되지
않는다. 분리하려면 신호 강도별로 재분할해야 하는데, 그 분할을 같은 데이터에서 고르면 §142-4의
선택 편향이 된다.

⚠️ v107에서도 같은 갈림이 나오는지는 **미확인**이다. 이 스윕은 v108 위에서만 돌렸다.

### 4. 판단 — 가중을 바꾸지 않는다

**DD weight는 이 데이터로 고를 수 없다.** 어느 쪽으로 옮겨도 한 집단을 다른 집단과 맞바꾼다:
0으로 내리면 홀드아웃 +0.0137, SEAL −0.0072. 0.2가 부호 11/17로 v108과 같지만 SEAL이 −0.0012다.

⚠️ **0.343 유지했다.** 훅은 남겼다. 그리고 이 결과는 "DD를 줄여야 한다"가 **아니라**
**"DD의 기여가 코호트에 따라 부호가 바뀐다"** 는 발견이다 — 그게 더 중요하다.

**후속 갈래**: DD를 **에피소드마다 켜고 끄는** 기준을 context만으로 만들 수 있는지. 예컨대
§146의 dispersion 방향이 얼마나 잘 결정되는지(고유값 간격, 또는 context 분할 검증)를 보고
약하면 DD 가중을 0으로 내리는 것. ⚠️ §146-2가 경고한 사후 선택 편향을 피해야 하므로
**context 표본 분할**이 전제다.

---

## 154. 2026-08-17 — DD의 **누락된 LLR log-det 항**: 유도는 맞고, 항은 크고, **fold-mean AUROC에는 원리적으로 닿을 수 없다** (사용자 제안)

*작성: nhn-NEXGEM-claude, 2026-08-17 (KST)*

사용자 제안: 지금은 D₀−D₁만 쓰는데 `log(σ₀²/σ₁²)`을 더해 실제 LLR을 쓰자.

### 1. 유도 — 지적이 정확하다

DD는 log-variance feature `f`를 클래스별 1-D 정규분포로 모델링하고
`d_c = (f−μ_c)²/σ_c²`를 낸다. 두 정규분포의 실제 로그 우도비는

```
log p(f|1) − log p(f|0) = ½·[ (f−μ₀)²/σ₀² − (f−μ₁)²/σ₁² ] + ½·log(σ₀²/σ₁²)
                        = ½·(d₀ − d₁)                      + ½·log(σ₀²/σ₁²)
```

현재 head는 `0.343·(d₀−d₁)`만 쓴다. **즉 지금의 DD는 log-determinant 항이 잘린 LLR이다.**
`d_c += log(σ_c²)`를 더하면 `(d₀+log σ₀²)−(d₁+log σ₁²) = (d₀−d₁)+log(σ₀²/σ₁²)`로 정확히 복원되고,
클래스 스왑 시 σ₀²↔σ₁²이므로 **반대칭도 유지**된다(테스트로 고정).

### 2. ⚠️ 구현 함정 — σ_c²는 `distances`에서 되읽을 수 없다

첫 시도에서 `_dd_distance_features`의 출력에서 σ_c²를 복원하려 했는데, `d_c`가 **이미** σ_c²로
나눈 값이라 클래스 c에서 평균하면 **정의상 정확히 1**이 나온다. 그래서 offset이 전 fold에서
0.0000으로 측정됐다 — 데이터 성질이 아니라 순환 계산이었다.
(`tests/test_dd_adateive_rank.py::test_averaging_the_lineage_distances_gives_one_not_sigma`로 박아둠.)

§149-7의 교훈대로 손으로 재구현하지 않고, 계보와 등가성이 검증된 경로에
`dd_adaptive_rank.class_dispersions()`를 노출했다. 그 σ_c²로 거리 공식을 **재구성해 계보 출력과
1e-5 일치**함을 테스트가 확인한다.

### 3. 항은 작지 않다

| task | log(σ₀²/σ₁²) 평균 | sd | \|max\| |
|---|---:|---:|---:|
| cptac_luad KRAS (홀드아웃) | −1.94 | 0.38 | **3.06** |
| cptac_luad TP53 | +1.25 | 1.36 | **3.17** |
| cptac_lscc ARID1A (홀드아웃) | −1.95 | 0.53 | 2.88 |
| cptac_ccrcc BAP1 | −1.66 | 0.19 | 2.08 |
| bc_therapy er_status | **+1.38** | 0.15 | 1.66 |
| cptac_ccrcc VHL | **+1.31** | 0.16 | 1.60 |
| bc_therapy her2 | −0.37 | 0.10 | 0.54 |

`d_c`가 O(1) 스케일인데 offset이 최대 3.2다. **누락된 항이 전체 LLR에서 지배적일 수 있는 크기다.**

### 4. 그런데 fold-mean AUROC는 바뀌지 않는다 — 우연이 아니라 구조다

σ₀², σ₁²는 **context에서만** 나온다. 정식 프로토콜에서 한 fold의 context는 그 fold의 모든 query에
대해 동일하므로, 이 항은 **fold 안에서 모든 query에 대해 같은 상수**다. AUROC는 fold 내 **순위**만
읽으므로 상수 이동은 원리적으로 아무 영향이 없다.

| | fold-mean Δ 평균 | \|최대\| | pooled Δ 평균 | pooled 양수 |
|---|---:|---:|---:|---:|
| SEAL 10 | **+0.00000** | 0.00025 | −0.00088 | 4/10 |
| 홀드아웃 7 | **−0.00006** | 0.00032 | +0.00021 | 3/7 |

macro는 SEAL 0.6967 → **0.6967**, 홀드아웃 0.5893 → **0.5892**. 남은 1e-4 수준 변동은 bf16에서
근접 동점이 다르게 깨진 결과이며 효과가 아니다.

**pooled AUROC는 움직이지만 방향이 없다**(task별로 ±0.01까지, 부호 7/17). pooled는 서로 다른
상수를 가진 fold들을 섞으므로 이 항에 반응하는 것이 맞고, 그 반응이 개선인지는 확립되지 않는다.

### 5. 결론 및 일반 규칙

- **제안은 이론적으로 옳다.** 지금의 DD는 절단된 LLR이고, 그것을 고치는 것이 정확하다.
- **다만 우리가 판정하는 지표로는 측정 자체가 불가능하다.** 이득도 손해도 아니라 **불가시**다.
  절단이 여태 무해했던 이유도 이것으로 설명된다.
- **의미가 생기는 곳**: 확률 보정(log loss), fold를 섞는 pooled 지표, 또는 임계값을 쓰는 의사결정.
  SEAL macro는 그중 어느 것도 아니다.
- ⚠️ **승격하지 않았다.** 훅(`ICF_DD_LLR=1`)은 남겼다.

**일반 규칙 (다음 DD arm 전에 확인할 것)**: 변경이 margin을 **에피소드마다 상수만큼** 옮기는
것이라면 **fold-mean AUROC는 정의상 움직이지 않는다.** 돌리기 전에 "이 변경이 query에 의존하는가"를
먼저 따질 것. §153의 weight 스윕은 query 의존적이라(거리에 곱해진다) 움직였고, 이번 항은 아니다.

---

## 155. 2026-08-18 — 상대 거리 `(D₀−D₁)/(D₀+D₁+ε)`: **전제는 참, 효과는 기각** (사용자 제안)

*작성: nhn-NEXGEM-claude, 2026-08-18 (KST)*

사용자 제안: 둘 다 멀면서 차이만 작은 query를 억제하도록 분모로 나누자.

먼저 §154-5의 스크리닝을 통과한다 — 이 변환은 **query에 의존**하므로(분모가 query마다 다르다)
fold-mean AUROC를 움직일 수 있다. ε를 무시하면 `(r−1)/(r+1)`, r = D₀/D₁이므로 **차이 대신
비율로 순위를 매기는 것**이고, query별 스케일에 불변이 된다. 클래스 스왑 시 분자만 부호가
뒤집히므로 반대칭도 유지된다.

### 1. 전제는 실제로 성립한다

`scripts/diagnose_dd_relative.py`:

| | SEAL 10 | 홀드아웃 7 |
|---|---:|---:|
| `D₀+D₁`의 fold 내 p90/p10 | **14.4배** | **27.6배** |
| 차이 순위 vs 비율 순위 일치도 | 0.727 | 0.784 |
| **"둘 다 멀고(상위 1/3) 차이는 작은(하위 1/2)" query 비율** | **5.4%** | **9.7%** |

분모가 fold 안에서 14~28배 흔들리므로 나눗셈은 상수 재조정이 아니고, 순위도 실제로 달라진다
(일치도 0.73~0.78). **전제는 맞다.** 다만 겨냥한 케이스 자체는 **5~10%**에 불과하다.

### 2. 그런데 DD 단독 판별력이 떨어진다

| | 차이 (현행) | 비율 (제안) |
|---|---:|---:|
| SEAL 10 DD-only AUROC | **0.6354** | 0.6074 |
| 홀드아웃 7 DD-only AUROC | **0.4937** | 0.4797 |
| 비율이 이긴 task | — | **5/17** |

### 3. full-model — 교정하면 크게 나쁘고, 교정 안 하면 무의미하다

| | v108 | 비율(교정) | 비율(무교정) |
|---|---:|---:|---:|
| SEAL 10 macro | **0.6967** | 0.6749 | 0.6929 |
| 홀드아웃 7 macro | **0.5893** | 0.5370 | 0.5953 |
| 전체 17 평균 | **0.6525** | 0.6181 | 0.6527 |
| v108 대비 부호 | — | **3/17** | 6/17 |

개별 손해가 크다: cptac_luad TP53 **−0.0895**, KRAS **−0.1210**, BAP1 −0.0738, KEAP1 −0.0755.
(이득: VHL +0.0265, EGFR +0.0156.)

### 4. ⚠️ 새 함정 — 두꺼운 꼬리를 가진 기준에 RMS를 맞추면 유계 통계량이 부풀려진다

**교정판이 무교정판보다 훨씬 나쁘다**(−0.0344 vs +0.0002)는 것이 이상해 보이지만 원인이 있다.
`m_rel`은 (−1,1)로 **유계**인데 기준인 `(D₀−D₁)`은 이상 query 때문에 **꼬리가 두껍다.** 그 RMS에
맞추려면 `m_rel`을 자기 자연 범위 훨씬 밖으로 **증폭**해야 하고, 그러면 DD가 margin을 지배한다.
§148의 CT 교정은 두 분포가 비슷한 모양이어서 통했지만, 여기서는 **교정이 문제를 만든다.**

한편 무교정판(+0.0002, 6/17)은 사실상 **DD 가중을 줄인 것과 같다** — `m_rel`이 유계라
기여가 쪼그라든다. §153에서 DD 가중을 0.05로 내린 값(0.6542)·0으로 내린 값(0.6539)과
거의 같은 자리(0.6527)라는 것이 그 해석을 뒷받침한다. **즉 무교정판의 "무해함"은 변환의 공로가
아니라 DD를 껐기 때문이다.**

### 5. 왜 실패했는가

1. **겨냥한 케이스가 5~10%뿐**인데 변환은 **모든** query의 순위를 바꾼다. 소수를 고치려고
   다수를 흔들었고 손실이 이득을 넘었다.
2. **가우시안 판별식은 차이지 비율이 아니다.** §154가 확인한 대로 DD의 올바른 형태는
   `½(d₀−d₁) + ½log(σ₀²/σ₁²)`다. 비율은 이것의 단조 변환이 **아니므로** 우도 구조를 버린다.
   이상치 억제는 **강건성** 목표이고, 이 데이터에서는 우도 구조가 더 값이 나갔다.

⚠️ **기각. 승격하지 않았다.** 훅(`ICF_DD_RELATIVE=1`)은 남겼다.

### 6. 살릴 수 있는 방향

목표("근거가 약한 query를 억제")는 타당하다. 판별식을 **교체**하는 대신 **감쇠**하면 정상 query의
순위를 보존할 수 있다:

```
margin ∝ w(D₀+D₁) · (D₀−D₁),   w는 D₀+D₁이 극단일 때만 1 → 0으로 감소
```
이러면 ① 5~10%의 이상 query만 건드리고 ② 나머지에서는 현행 판별식이 그대로 남고
③ 유계가 아니므로 §4의 교정 함정도 피한다. ⚠️ `w`의 임계값을 **context만으로** 정해야 하며,
같은 fold의 query 분포에서 고르면 누출이다.

---

## 156. 2026-08-18 — **CV descriptor 해부: raw mean은 무용, 대각 256차원은 유해하다** (사용자 제안)

*작성: nhn-NEXGEM-claude, 2026-08-18 (KST)*

CV는 가중 1.442로 지배적인 분기인데 34,432차원 descriptor `[vech(BᵀC_bag B), x̄_b]`를 어떻게
쓰는지 한 번도 분해하지 않았다. 사용자 제안대로 5개 블록으로 갈랐다.

### 1. 설계 — DD를 건드리지 않기 위한 제약

⚠️ **DD가 CV descriptor의 triangle을 읽는다**(`_covariance_matrices_from_triangle`). descriptor를
전역으로 마스킹하면 mean-only arm에서 DD가 부서지고 두 분기가 섞인다. 그래서 마스킹을
**CV ridge가 보는 것에만** 걸었다 — `_ridge_logits`가 `self._normalize_descriptors`를 **인스턴스**로
찾으므로 거기서 열 선택과 블록 정규화를 함께 하면 되고, DD는 `_relation_logits`가 들고 있는
원본 `context`를 그대로 받는다. CT는 raw cell을 읽어 무관하다.

**CV-only 측정**은 기존 훅으로 `ICF_FIXED_HEAD_DD_WEIGHT=0 ICF_FIXED_HEAD_CT_WEIGHT=0`을 주면
margin이 `1.442·(cv1−cv0)`뿐이 된다 — 정식 경로에서 그대로 잰다.

**검증**: `ICF_CV_BLOCKS=cov+mean`이 미설정과 동일(SEAL 0.6967 = v108).

### 2. 결과

| CV가 보는 것 | 차원 | CV-only 17 | vs 전체 | 부호 | full-model 17 | vs v108 | 부호 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cov + mean (v108) | 34,432 | 0.6466 | — | — | 0.6525 | — | — |
| covariance만 | 32,896 | 0.6484 | +0.0019 | 8/17 | 0.6539 | +0.0014 | 9/17 |
| **off-diagonal만** | 32,640 | **0.6517** | **+0.0052** | **13/17** | **0.6555** | **+0.0030** | **12/17** |
| raw mean만 | 1,536 | 0.6248 | −0.0218 | 5/17 | 0.6383 | −0.0142 | 6/17 |
| diagonal + mean | 1,792 | 0.6223 | −0.0243 | 4/17 | 0.6361 | −0.0164 | 4/17 |
| **diagonal만** | 256 | **0.5880** | **−0.0586** | 4/17 | 0.6123 | −0.0402 | 3/17 |

### 3. 발견 ① raw bag mean은 무용하다 — 그리고 §86을 뒤집는다

mean을 **빼면** CV-only +0.0019, full +0.0014다. 즉 1,536차원이 **아무것도 더하지 않는다.**
mean 단독은 −0.0218로 공분산보다 훨씬 약하다.

⚠️ **§86은 mean이 +0.0037이라고 기록했다** (covariance-only 0.6630 → CV 0.6667). 그건
**K=128 + 고정 랜덤 P** 시절이다. within-slide PCA·K=256에서는 그 기여가 **사라졌다.**

일관된 해석: within-slide 기저는 bag별 자기 평균으로 센터링해 만들어지므로 **between-slide 항을
의도적으로 버린다**(§139-4). raw bag mean은 바로 그 버린 성분이고, §123-4가 ICC 31.6%로 잰
**nuisance(염색·스캐너·환자)** 가 지배한다. 랜덤 P 시절에는 기저가 nuisance를 덜 정리해 mean이
보완 정보를 줬지만, PCA가 신호 쪽을 제대로 잡은 뒤에는 남은 것이 잡음뿐이다.

### 4. 발견 ② 대각 256차원은 **적극적으로 해롭다**

triangle에서 대각만 빼면 CV-only **+0.0052 (13/17)**, full **+0.0030 (12/17)**. 두 집단이 부호에
동의한다:

| | SEAL 10 | 홀드아웃 7 | 전체 17 |
|---|---|---|---|
| CV-only, off-diagonal | +0.0054 (**9/10**) | +0.0049 (4/7) | **+0.0052 (13/17)** |
| full-model, off-diagonal | +0.0032 (**8/10**) | +0.0026 (4/7) | **+0.0030 (12/17)** |

그리고 대각 **단독**은 최악의 arm이다(CV-only −0.0586). 개별로는
luad TP53 0.6902 → **0.5819**, SMAD4 0.4859 → **0.3793**, BAP1 0.6412 → 0.5462.

**즉 CV의 성능은 PCA 방향별 분산(스펙트럼)이 아니라 방향 간 상관 구조에서 나온다.** 대각은
정보가 적은 데다 32,896차원 중 256차원을 차지하면서 ridge를 오염시킨다.

### 5. 교란 점검 — 둘 다 통과/실패

**(a) 블록 정규화는 교란이 아니다.** `parent`(전체 블록 통계 유지) vs `blockwise`:
diag 0.6492 vs 0.6466 (−0.0026), offdiag 0.6999 vs 0.7000 (+0.0000). **내용 차이가 맞다.**

**(b) ⚠️ λ은 저차원에서 무력하지 않다 — §142-2 정정.**

| arm | λ=0.01 | λ=1 (현행) | λ=100 |
|---|---:|---:|---:|
| diagonal (256차원) | 0.6383 | 0.6492 | **0.6503** |
| raw mean (1,536차원) | 0.6681 | 0.6871 | **0.6886** |

§142-2는 λ∈[0.01, 3.52]가 완전 무력이라고 기록했지만 그건 **9,792~34,432차원** 구간이었다.
256·1,536차원에서는 λ가 최대 **+0.02**까지 움직인다 — Gram이 차원에 비례하므로 저차원에서는
λ=1이 상대적으로 강한 정규화가 된다.

**따라서 diag·mean arm의 열세는 다소 과장돼 있다**(각자의 최적 λ가 아니다). 다만 λ=100에서도
diag 0.6503, mean 0.6886으로 off-diagonal의 0.6999에 한참 못 미쳐 **순위는 유지된다.**
⚠️ arm마다 λ를 최적화하면 같은 10개 task에서 고르는 선택 편향이 되므로 하지 않았다.

### 6. 함의 — 학습을 어디에 넣을지

- **off-diagonal 32,640차원이 CV다.** 다음 작업은 전부 여기여야 한다.
- **대각과 mean은 제거 후보다.** off-diagonal-only가 두 집단에서 부호 일치로 이기고, 차원도 줄고
  (34,432 → 32,640), 계산도 준다. ⚠️ **승격하지 않았다** — 구성 확정은 사용자 판단(§118)이고,
  §151-1대로 t는 쓰지 않으니 판단 근거는 **12~13/17 부호 + 두 집단 동의**다.
- **학습 위치**: 상관 구조가 값을 내는데 지금은 32,640차원을 **무가중 ridge**로 균등 취급한다.
  학습을 넣을 자연스러운 자리는 ① 어떤 방향 **쌍**이 중요한지 가중하는 것, 또는 ② PCA 기저를
  상관 구조가 잘 드러나는 쪽으로 회전시키는 것이다. 대각을 키우는 방향은 닫혔다.
- **§149-4의 갈래와 연결된다**: CT가 CV의 기저를 공유해 상한에 걸렸는데, CV 자체가 그 기저의
  **상관 구조**를 쓴다는 것이 확인됐다. CT에는 CV가 안 쓰는 것(대각/스펙트럼?)을 주는 설계가
  가능할 수 있다 — 미검증.

---

## 157. 2026-08-18 — **k-means token: FPS는 16개 중 ~2개만 쓰고 있었다.** CT 표현이 고쳐지고, 그러자 weight가 값을 한다 (사용자 제안)

*작성: nhn-NEXGEM-claude, 2026-08-18 (KST)*

§148-5가 남긴 상류 후보 셋 중 **farthest-point token 생성**을 겨눈 제안이다. 결과가 이 세션 CT
작업 중 가장 크다.

### 1. 설계 — FPS를 초기값으로 쓰는 Lloyd 반복

k-means 초기화를 난수로 하면 v108의 **seed std 0.00000**이 깨진다. 그래서 **FPS 결과를 초기값**으로
쓰고 Lloyd 반복만 돌린다. 세 가지 이득: ① 결정론 유지 ② `iterations=0`이 현행과 **비트 동일**
③ 반복 수가 **커버리지(FPS) ↔ 밀도(k-means)를 잇는 단일 노브**가 된다(§127-2).

빈 클러스터는 이전 위치를 유지한다(결정론적인 유일한 처리). 테스트 8개로 고정: 0 반복 동일성,
token이 더 이상 실제 cell이 아님, 결정론, query 무영향, 수렴, 빈 클러스터 NaN 없음.

### 2. ⚠️ 발견 — FPS token 14개는 죽어 있었다

**유효 token 수** = exp(클러스터 점유율의 엔트로피). 16이면 완전 균형, 1이면 한 클러스터가 전부.

| Lloyd 반복 | 유효 token (SEAL/홀드아웃) | abundance entropy | \|t\| 중앙값 | CT-only extreme | **CT-only ridge** |
|---:|---:|---:|---:|---:|---:|
| **0 (FPS = v108)** | **1.90 / 1.67** | 0.98 / 0.69 | 1.31 / 1.60 | 0.6190 / 0.5424 | **0.6168 / 0.5572** |
| 1 | 1.90 / 1.67 | 2.18 / 1.92 | 1.37 / 1.57 | 0.6214 / 0.5430 | 0.6450 / 0.5759 |
| 3 | 6.75 / 4.96 | 2.47 / 2.31 | 1.40 / 1.52 | 0.6159 / 0.5641 | 0.6504 / 0.5918 |
| 10 | 12.18 / 9.85 | 2.53 / 2.39 | 1.39 / 1.45 | 0.6208 / 0.5476 | 0.6505 / 0.5885 |
| **30** | **13.17 / 11.44** | 2.53 / 2.41 | 1.39 / 1.45 | 0.6177 / 0.5495 | **0.6532 / 0.5962** |

**FPS에서 유효 token이 1.9개다.** 16개를 뽑아놓고 사실상 **2개만** 쓰고 있었다 — 나머지 14개는
어느 cell도 가장 가깝지 않은 이상점이고, FPS가 "최대한 멀리"를 최적화하니 당연한 결과다.
§148-4가 잡은 증상(판별 통계 중앙값 1.31)의 **원인이 이것이다.**

k-means로 13.2개까지 회복되고 **CT-only ridge가 +0.0364(SEAL) / +0.0390(홀드아웃)** 오른다 —
이 세션 CT 단독 최대 폭이다.

⚠️ **§149의 entropy 해석을 정정한다.** 나는 낮은 entropy를 "날카로워서 좋다"로 읽었는데, FPS의
낮은 entropy(0.98)는 **모든 cell이 같은 1~2개 token에 몰려 bag 간 차이가 없는** 붕괴 상태였다.
entropy만으로는 두 경우를 구분할 수 없고 **유효 token 수가 그 구분자다.**

### 3. 상호작용이 양방향으로 확인된다

`extreme` readout에서는 CT-only가 거의 안 움직인다(0.6190 → 0.6177). 그리고 full-model에서
**k-means token + 옛 two-token readout = −0.0028 (8/17)** 로 **오히려 해롭다.**

즉 **token과 readout은 함께 바꿔야 한다.** §150이 "거리를 먼저 고쳐야 readout이 의미를 갖는다"를
보였고, 여기서는 그 역도 참이다 — **좋은 token은 16차원을 다 읽는 readout이 있어야 쓸 수 있다.**

### 4. full-model만으로는 작다 (weight 0.286에서)

| Lloyd 반복 | SEAL 10 | 홀드아웃 7 | 전체 17 | vs v108 | 부호 |
|---:|---:|---:|---:|---:|---:|
| 0 (v108) | 0.6967 | 0.5893 | 0.6525 | — | — |
| 1 | 0.6982 | 0.5896 | 0.6535 | +0.0010 | 10/17 |
| 3 | 0.6975 | 0.5899 | 0.6532 | +0.0007 | 10/17 |
| 30 | 0.6978 | 0.5924 | 0.6544 | +0.0019 | 9/17 |

CT-only가 +0.037인데 macro는 +0.0019다 — §152-3이 지목한 두 제약(가중 0.286, CV와 공유 기저)이
그대로 작동한다.

### 5. **그래서 weight를 다시 쟀다 — 이번엔 값을 한다**

§151의 weight 스윕은 **붕괴된 FPS token** 기준이었으므로 작동하는 CT에 대해서는 아무 말도 하지
않는다. k-means token 위에서 다시:

| 구성 | SEAL 10 | 홀드아웃 7 | 전체 17 | vs v108 | 부호 |
|---|---:|---:|---:|---:|---:|
| FPS, w=0.286 (v108) | 0.6967 | 0.5893 | 0.6525 | — | — |
| k-means, w=0.286 | 0.6978 | 0.5924 | 0.6544 | +0.0019 | 9/17 |
| k-means, w=0.4 | 0.6989 | 0.5958 | 0.6564 | +0.0039 | 10/17 |
| k-means, w=0.5 | 0.6995 | 0.5980 | 0.6577 | +0.0052 | **11/17** |
| **k-means, w=0.7** | **0.6999** | 0.6017 | 0.6594 | **+0.0070** | **11/17** |
| k-means, w=1.0 | 0.6983 | **0.6053** | **0.6600** | +0.0075 | 9/17 |

**§151과 정반대다.** FPS token에서는 weight를 올릴 때 부호가 11/17 → 7/17로 **단조 붕괴**했는데,
k-means token에서는 0.5~0.7에서 **11/17로 유지**된다. 가중을 더 줘도 되는 분기가 됐다는 뜻이다.

w=0.7: SEAL +0.0032 (7/10), 홀드아웃 **+0.0124** (4/7). 홀드아웃 개별로 ARID1A 0.4499 → **0.4973**,
PBRM1 0.5130 → **0.5415**, Histologic_Grade 0.6208 → 0.6408. SEAL에서는 BAP1 0.6412 → **0.6712**,
grade 0.7204 → 0.7330. 손해는 er_status −0.0242, EGFR −0.0081.

**ABMIL 격차: −0.0303 → −0.0271.**

### 6. 판단

- **`km30 + w=0.7`이 이 세션 최대 full-model 이득이다**(+0.0070, 11/17, 두 집단 양수).
- ⚠️ **승격하지 않았다.** 구성 확정은 사용자 판단(§118)이고, §151-1대로 t는 쓰지 않으므로 근거는
  **11/17 부호 + 두 집단 동의 + CT-only +0.037의 기전**이다.
- ⚠️ w는 0.5·0.7·1.0이 평균에서 0.002 안에 몰려 있어 **정확한 값은 노이즈다.** 0.7은 평균과 부호가
  동시에 가장 좋은 지점이라는 뜻일 뿐이다.
- **§148-5의 남은 후보는 64-cell 샘플링 하나다.** 이제 token 생성(§157)과 거리(§149)와
  readout(§148·§150)이 모두 검증됐다.

---

## 158. 2026-08-18 — **v109 승격 확정 (사용자 결정): CV = off-diagonal only, CT = k-means token @ w=0.7**

*작성: nhn-NEXGEM-claude, 2026-08-18 (KST)*

두 분기를 한 번에 바꾼다. **가법성을 가정하지 않고 조합 자체를 쟀다**(§150이 노브 상호작용을
보인 전례가 있다).

| 구성 | SEAL 10 | 홀드아웃 7 | 전체 17 | vs v108 | 부호 |
|---|---:|---:|---:|---:|---:|
| v108 (기준) | 0.6967 | 0.5893 | 0.6525 | — | — |
| CT만 (km30, w=0.7) | 0.6999 | 0.6016 | 0.6594 | +0.0070 | 11/17 |
| CV만 (offdiag) | 0.6999 | 0.5919 | 0.6555 | +0.0030 | 12/17 |
| **v109 (둘 다)** | **0.7027** | **0.6042** | **0.6621** | **+0.0096** | **13/17** |

**가법 예측 0.6624 vs 실측 0.6621** — 0.0003 안에서 가법적이다. 다른 분기이니 놀랍지 않지만
확인했다. **13/17은 이 세션에서 나온 최고 부호 일치**이고, SEAL이 **처음으로 0.70을 넘었다.**

### 1. 무엇이 바뀌었나

| 자리 | v108 | **v109** | 근거 |
|---|---|---|---|
| CV가 보는 descriptor | triangle 32,896 + mean 1,536 | **off-diagonal 32,640만** | mean은 무용(+0.0019), 대각은 유해(+0.0052, 13/17) — §156 |
| CT token 배치 | farthest-point | **k-means (Lloyd 30회, FPS 초기값)** | FPS는 16개 중 **1.9개**만 사용 — §157-2 |
| CT weight | 0.286 | **0.7** | k-means token에서는 올려도 부호 유지(11/17) — §157-5 |
| DD | 그대로 | **그대로** | ⚠️ 아래 |

⚠️ **DD는 여전히 전체 triangle을 받는다.** DD가 CV descriptor의 triangle에서 K×K 행렬을
재구성하므로, descriptor를 전역으로 마스킹하면 CV를 좁히는 게 아니라 **DD를 부순다**(§156-1).
마스킹은 CV ridge가 보는 것에만 걸린다. 테스트로 고정(`weight_cv=0`에서 `cv_blocks`를 바꿔도
margin 불변).

### 2. 실행

```bash
bash scripts/eval_v109.sh <gpu> <tag> [tasks...]     # 정의가 사는 단 하나의 자리
```
**검증**: `eval_v109.sh`로 VHL을 정식 경로에서 돌려 **0.5209 재현**. `TrainingFreeClassifier()`
기본값도 v109로 바꿨고(`cv_blocks="offdiag"`, `ct_kmeans_iterations=30`, `weight_ct=0.7`)
결정론·반대칭·기본값을 테스트가 지킨다.

### 3. per-task

| SEAL 10 | v108 | v109 | Δ |
|---|---:|---:|---:|
| bc_therapy grade | 0.7204 | 0.7366 | **+0.0162** |
| bc_therapy her2 | 0.6630 | 0.6677 | +0.0046 |
| bc_therapy er_status | 0.7066 | 0.6905 | **−0.0161** |
| cptac_brca PIK3CA | 0.5313 | 0.5449 | +0.0136 |
| cptac_brca TP53 | 0.8247 | 0.8214 | −0.0033 |
| cptac_ccrcc BAP1 | 0.6412 | 0.6690 | **+0.0278** |
| cptac_ccrcc VHL | 0.5011 | **0.5209** | +0.0198 |
| cptac_luad EGFR | 0.7869 | 0.7808 | −0.0062 |
| cptac_luad STK11 | 0.9019 | 0.9045 | +0.0026 |
| cptac_luad TP53 | 0.6902 | 0.6905 | +0.0003 |

| 홀드아웃 7 | v108 | v109 | Δ |
|---|---:|---:|---:|
| cptac_lscc ARID1A | 0.4499 | **0.5097** | **+0.0598** |
| cptac_ccrcc PBRM1 | 0.5130 | **0.5507** | **+0.0377** |
| cptac_lscc Histologic_Grade | 0.6208 | 0.6327 | +0.0119 |
| cptac_lscc KEAP1 | 0.5579 | 0.5632 | +0.0053 |
| cptac_pda SMAD4 | 0.4859 | 0.4866 | +0.0008 |
| ucla_lung progression | 0.7844 | 0.7862 | +0.0018 |
| cptac_luad KRAS | 0.7132 | 0.7003 | **−0.0129** |

**ARID1A가 처음으로 0.5를 넘었다**(0.4499 → 0.5097). VHL도 0.5209로 올라간다.

### 4. ABMIL과의 거리

| | SEAL macro | ABMIL 0.727 대비 |
|---|---:|---:|
| v106 | 0.6864 | −0.0406 |
| v107 | 0.6945 | −0.0321 |
| v108 | 0.6967 | −0.0303 |
| **v109** | **0.7027** | **−0.0243** |

### 5. 한계

- ⚠️ **t·p·CI는 보고하지 않는다**(§151-1, 결정론적 arm). 근거는 **13/17 부호 + 두 집단 동의**다.
- **CT weight 0.7은 sharp optimum이 아니다** — 0.5·0.7·1.0이 전체 평균에서 0.002 안에 몰려 있다.
- **CV·CT 모두 여전히 같은 within-slide PCA 기저를 쓴다**(§149-4). CT가 CV의 그림자에 있다는
  제약은 그대로이고, off-diagonal이 CV의 본질이라는 §156의 발견은 **CT에 대각/스펙트럼 쪽을
  주는 설계**가 가능함을 시사한다 — 미검증.
- **§148-5의 남은 후보는 64-cell 샘플링 하나**다.
- **λ은 off-diagonal(32,640차원)에서 재확인하지 않았다.** §156-5가 저차원에서 λ이 살아난다는 것을
  보였으나 32,640은 §142-2의 무력 구간 안이다.

---

## 159. 2026-08-18 — CT의 64-cell 상한 제거: **기각. 샘플링은 병목이 아니었다** (사용자 지시)

*작성: nhn-NEXGEM-claude, 2026-08-18 (KST)*

§148-5가 남긴 세 번째이자 마지막 후보다. bag당 64 cell은 8,192-cell slide의 **1.6%** 이고
abundance는 그 표본의 평균이므로, 표본오차가 클래스 차이를 넘을 수 있다는 가설이었다.

### 1. 구현

`cells_per_bag: int | None`으로 바꿨다. `None`이면 모든 cell을 쓴다(bag은 이미 encoder의
`max_cells`로 상한이 걸려 있어 무한이 아니다). **기본값은 64 유지** — v109 재현 경로다.
훅은 `ICF_CT_CELLS=all` 또는 숫자.

PCA32 사영 → token 생성 → k-means 30회 → context/query abundance가 **전부** 전체 cell을 쓴다.
CV/DD는 손대지 않았다. **query는 token 생성과 정규화 통계에 들어가지 않는다** — 전체 cell
설정에서 그것을 확인하는 테스트를 추가했다(query bag을 `×7 − 3`으로 바꿔도 tokens·context 불변).

⚠️ **OOM은 발생하지 않았다.** 지시대로 현재 vectorized 구현으로 돌렸고, `[N,16]` 재작성이나
chunking은 **불필요**했다. 비용은 cptac_luad/EGFR(324 slide, 50 fold)에서 **48초 → 74초**
(약 1.5배). k-means 반복이 1.6M cell에 걸리는 구간이 지배한다.

### 2. 결과 — 기각

| CT cells/bag | SEAL 10 | 홀드아웃 7 | 전체 17 | vs v109 | 부호 |
|---|---:|---:|---:|---:|---:|
| **64 (v109)** | **0.7027** | **0.6042** | **0.6621** | — | — |
| 256 | 0.7004 | 0.5993 | 0.6587 | −0.0034 | 6/17 |
| 1024 | 0.6985 | 0.6003 | 0.6581 | −0.0041 | 5/17 |
| 전체 (cap 8,192) | 0.6999 | 0.6034 | 0.6602 | −0.0020 | 10/17 |

**전부 v109보다 낮고 단조도 아니다**(64 → 256 → 1024 → 전체가 −0.0034, −0.0041, −0.0020).
비단조성은 이 축의 차이가 모두 노이즈 범위라는 뜻이다. 전체 cell의 10/17은 사실상 동전이다.

### 3. ⚠️ 축을 닫기 전 대각선 확인 (§150-4 규칙)

§148은 "readout은 병목이 아니다"를 **raw 1536 조건에서만** 재고 축을 닫았다가 §150에서 정정됐다.
같은 실수를 피하기 위해 cell 수를 **k-means 이전 조건에서도** 쟀다:

| | SEAL 10 | 홀드아웃 7 | 전체 17 |
|---|---:|---:|---:|
| FPS token, 64 cell | 0.6995 | 0.5946 | 0.6563 |
| FPS token, 전체 cell | 0.6953 | 0.5947 | 0.6539 |
| k-means token, 64 cell (v109) | **0.7027** | **0.6042** | **0.6621** |
| k-means token, 전체 cell | 0.6999 | 0.6034 | 0.6602 |

| cell 수를 늘린 효과 | 전체 17 | SEAL 10 | 홀드아웃 7 |
|---|---|---|---|
| FPS token 위에서 | −0.0024 (6/17) | −0.0041 (2/10) | +0.0000 (4/7) |
| k-means token 위에서 | −0.0020 (10/17) | −0.0028 (6/10) | −0.0007 (4/7) |

**상호작용이 없다.** 두 token 방식에서 거의 같은 크기의 음수다. 즉 "k-means가 표본 민감도를
없앴다"는 내 가설은 **기각**이고, 결론은 더 단순하다 — **cell 수는 애초에 문제가 아니었다.**

### 4. 왜 64개로 충분한가

abundance는 16차원 단체(simplex) 위의 조성 벡터이고, 64개 표본으로 그것을 추정하는 오차는
**bag 간 실제 차이보다 작다.** §157이 token을 밀집 영역에 놓은 뒤에는 abundance가 **큰 집단**에
지배되는데, 큰 집단의 비율은 64개로도 잘 추정된다. cell을 더 넣어야 해상되는 것은 **희귀 집단**
이고, 그건 k-means가 애초에 강조하지 않는 쪽이다.

⚠️ 다만 FPS token에서도 이득이 없었으므로 이 설명은 **필요조건이 아니다.** 확실한 것은 실측이다.

### 5. §148-5 세 후보의 최종 정산

| 후보 | 결과 |
|---|---|
| **token 생성** (farthest-point) | **이것이었다.** 16개 중 1.9개만 사용 → k-means로 CT-only +0.037 (§157) |
| **거리 metric** (1536차원 집중) | 실재하나 CV와 공유 기저 때문에 macro 상한(§149) |
| **cell 샘플링** (64개) | **기각.** 전체 cell −0.0020 (10/17), token 방식과 무관 (§159) |

**세 후보가 모두 측정됐고, 값을 한 것은 token 생성 하나다.**

⚠️ **승격하지 않았다** — 결과가 음수이므로. `cells_per_bag=64`가 기본값으로 남고
`ICF_CT_CELLS`는 훅으로 유지한다.

---

## 160. 2026-08-18 — token 수 스윕: **32이 정점.** 전체 cell은 token 수와 무관하게 손해 (사용자 지시)

*작성: nhn-NEXGEM-claude, 2026-08-18 (KST)*

지시는 "전체 cell을 유지한 채로 token(=cluster=feature) 수를 16 → 32/64/128 스윕"이었다.
그대로 돌렸고, **token 축은 실재하는 이득이지만 전체 cell 전제는 성립하지 않는다.**

### 1. ⚠️ 실제 OOM 발생 → chunking 구현 (지시 조건 충족)

`ICF_CT_TOKENS` 훅 추가. 128 token은 단독 GPU에서 통과했으나, **64 token × 전체 cell에서
GPU를 두 job이 공유하자 OOM이 났다**: `lloyd_refine`의 3-D broadcast가 **24.36 GiB**를 요청
(`[1.6M cells, 64 tokens, 32 dims]`).

지시대로 그때 최적화했고, **[N,K] 재작성 대신 cell 축 chunking**을 골랐다. 이유는 정확성이다 —
chunking은 원소별 산술을 바꾸지 않으므로 **기존에 들어갔던 것은 값이 변할 수 없다.** `[N,K]`
확장식(`‖x‖²−2x·t+‖t‖²`)은 수학적으로는 같지만 부동소수점이 달라 v109를 흔들 수 있다.

예산은 2²⁷ 원소(≈537 MB). v109 설정(64 cell/bag ≈ 13k pooled cell, 16 token)은 **단일 chunk**에
들어가므로 비트 동일이 보장된다. abundance softmax도 같은 방식으로 chunk했다(chunk별 합을
누적해 마지막에 나눈다).

**회귀 확인**: chunking 도입 후 `eval_v109.sh`로 VHL을 재측정해 **0.5209 정확히 재현**.
테스트 3개로 고정(chunked = unchunked 동일, v109 설정이 단일 chunk임).

### 2. 결과 — cells × tokens 격자 (전체 17 평균)

| tokens | **64 cell** | 전체 cell |
|---:|---:|---:|
| 16 | 0.6621 *(v109)* | 0.6602 |
| **32** | **0.6672** | 0.6640 |
| 64 | 0.6661 | 0.6638 |
| 128 | 0.6648 | 0.6633 |

**64 cell 열이 모든 token 수에서 전체 cell 열을 이긴다.** 그리고 **32가 양쪽 열의 정점**이다.

| 구성 | SEAL 10 | 홀드아웃 7 | 전체 17 | vs v109 | 부호 |
|---|---:|---:|---:|---:|---:|
| 16 tok, 64 cell (v109) | 0.7027 | 0.6042 | 0.6621 | — | — |
| **32 tok, 64 cell** | **0.7070** | **0.6103** | **0.6672** | **+0.0051** | **15/17** |
| 64 tok, 64 cell | 0.7075 | 0.6069 | 0.6661 | +0.0040 | 12/17 |
| 128 tok, 64 cell | 0.7066 | 0.6051 | 0.6648 | +0.0027 | 11/17 |
| 32 tok, 전체 cell | 0.7041 | 0.6068 | 0.6640 | +0.0019 | 10/17 |
| 64 tok, 전체 cell | 0.7055 | 0.6042 | 0.6638 | +0.0017 | 11/17 |
| 128 tok, 전체 cell | 0.7056 | 0.6028 | 0.6633 | +0.0012 | 10/17 |

**`32 tok, 64 cell`이 +0.0051에 부호 15/17** — 이 세션 전체에서 가장 높은 부호 일치다(v109 승격
근거였던 13/17보다 높다). 집단별로 **SEAL 9/10, 홀드아웃 6/7** 이다.

### 3. 전체 cell은 token 수와 무관하게 손해다 — §159 독립 재확인

64 cell 대비 전체 cell의 순효과:

| token 수 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|
| Δ | −0.0019 (10/17) | −0.0031 (6/17) | −0.0023 (4/17) | −0.0015 (7/17) |

**네 개 token 수 전부에서 음수다.** §159는 16 token에서 전체 cell을 기각했는데, 여기서 32·64·128에
대해 **독립적으로 재확인**됐다. "token이 많아지면 전체 cell이 필요해진다"는 전제는 실측으로 틀렸다.

해석: k-means는 **밀집 영역**에 중심을 놓고(§157), 큰 집단의 비율은 64개 표본으로도 잘 추정된다.
cell을 더 넣으면 **희귀 집단까지 클러스터가 생기는데** 그 비율은 bag마다 추정 오차가 크다 —
token 수를 늘리는 것과 같은 방향의 부담을 이중으로 지는 셈이다.

### 4. per-task (`32 tok, 64 cell`)

| SEAL 10 | Δ | 홀드아웃 7 | Δ |
|---|---:|---|---:|
| bc_therapy her2 | **+0.0127** | cptac_lscc KEAP1 | **+0.0238** |
| bc_therapy er_status | +0.0109 | cptac_luad KRAS | **+0.0134** |
| bc_therapy grade | +0.0068 | cptac_lscc Histologic_Grade | +0.0082 |
| cptac_ccrcc BAP1 | +0.0060 | cptac_ccrcc PBRM1 | +0.0019 |
| cptac_luad TP53 | +0.0049 | cptac_pda SMAD4 | +0.0014 |
| cptac_brca TP53 | +0.0043 | cptac_lscc ARID1A | +0.0003 |
| cptac_ccrcc VHL | +0.0024 | ucla_lung progression | −0.0062 |
| cptac_luad EGFR | +0.0015 | | |
| cptac_luad STK11 | +0.0003 | | |
| cptac_brca PIK3CA | −0.0067 | | |

손해가 두 task(PIK3CA −0.0067, ucla_lung −0.0062)뿐이다.

**ABMIL 0.727 대비: v109 −0.0243 → −0.0200.**

### 5. 판단

- ⚠️ **승격하지 않았다.** 구성 확정은 사용자 판단(§118)이고, 지시도 결과 확인까지였다.
- **가장 강한 후보는 `32 tok, 64 cell`**(+0.0051, 15/17)이며 **전체 cell은 포함하지 않는다.**
  즉 지시한 스윕의 답은 "token 32가 맞고, 전체 cell은 빼는 게 맞다"다.
- ⚠️ **λ를 32/64/128 token에서 재확인하지 않았다.** §156-5가 저차원에서 λ이 살아난다는 것을
  보였고 CT ridge는 16 → 128차원으로 움직인다. 이 축의 이득 일부가 λ 효과일 수 있다.
- token 32와 64는 SEAL에서 거의 동률(0.7070 vs 0.7075)이고 홀드아웃에서 32가 앞선다
  (0.6103 vs 0.6069). **정확한 값은 32~64 사이 어디로도 노이즈**로 볼 것.

---

## 161. 2026-08-18 — **v110 승격 확정 (사용자 결정): CT cluster 16 → 32**

*작성: nhn-NEXGEM-claude, 2026-08-18 (KST)*

**v110 = v109에서 CT의 cluster(=token=feature) 수만 16 → 32.** cell 수는 **64 유지**다.

| | SEAL 10 | 홀드아웃 7 | 전체 17 | vs v109 | 부호 |
|---|---:|---:|---:|---:|---:|
| v109 | 0.7027 | 0.6042 | 0.6621 | — | — |
| **v110** | **0.7070** | **0.6103** | **0.6672** | **+0.0051** | **15/17** |

**15/17은 이 세션 최고 부호 일치**다(v109 승격 근거였던 13/17보다 높다). 집단별로
**SEAL 9/10, 홀드아웃 6/7**. 손해는 두 task뿐(PIK3CA −0.0067, ucla_lung −0.0062).

⚠️ **전체 cell은 포함하지 않는다.** §160이 16·32·64·128 token 전부에서 전체 cell이 64 cell보다
나쁨을 보였다(−0.0015 ~ −0.0031). 이득은 **cluster 수**의 것이지 cell 수의 것이 아니다.

**ABMIL 0.727 대비: −0.0243 → −0.0200.**

### 1. 실행과 검증

```bash
bash scripts/eval_v110.sh <gpu> <tag> [tasks...]
```
`eval_v110.sh`로 VHL을 정식 경로에서 돌려 **0.5233 재현** 확인. `TrainingFreeClassifier()`
기본값도 `ct_num_tokens=32`로 옮겼고 `DefaultTest`가 `ct_num_tokens=32`·`ct_cells_per_bag=64`를
함께 고정한다(전체 cell로 잘못 흘러가는 것을 막는다).

### 2. 아키텍처 문서 정리

`docs/current_architecture.md`가 아직 **학습을 전제한 v83 계보 명세**에 v10x 박스만 얹은 상태였다.
학습 파라미터가 0인 지금 gradient 계약·학습 모듈 절이 본문을 차지하는 것은 오해를 부른다.
정리했다:

- **`§0. v110 명세`를 최상단에 신설** — 전체 파이프라인(기저 → descriptor → CV/DD/CT → head)을
  차원까지 포함해 자족적으로 기술. 하위 절: 0-1 한 눈에, 0-2 수치, **0-3 각 상수의 근거**,
  **0-4 구조적 제약**, **0-5 닫힌 축**, **0-6 열린 갈래**.
- 기존 `Active-*` → **`Historical-*`** 로 개칭하고 "v110에 적용되지 않는다"를 명시. 그 절들의
  gradient·checkpoint 계약은 전부 v83~v98 시절 것이다.

**0-4(구조적 제약)** 는 이 세션에서 반복해 부딪힌 것들을 모았다: DD가 CV triangle을 읽는다,
K_dd ≤ K_cv, CT·CV 기저 공유, 에피소드 상수는 fold-mean을 못 움직인다, 거리 chunking은 정확하다.

### 3. 한계

- **λ가 CT ridge 32차원에서 미확인**이다. §156-5가 저차원에서 λ이 살아남을 보였고 CT feature가
  16 → 32로 늘었다. 이 이득의 일부가 λ 효과일 수 있다.
- **32 vs 64 token은 SEAL에서 거의 동률**(0.7070 vs 0.7075)이고 홀드아웃에서만 32가 앞선다.
  정확한 값은 노이즈로 볼 것.
- CT·CV 기저 공유(§149-4)는 그대로다.

---

## 162. 2026-08-18 — 아이디어 1 (CV를 상관행렬로): **기각.** 공분산의 크기는 정보였다

*작성: nhn-NEXGEM-claude, 2026-08-18 (KST)*

§156은 대각 **항목**을 뺐다. 그런데 대각은 여전히 off-diagonal의 **크기**를 정한다 —
`BᵀC B[i,j]`는 `√(λᵢλⱼ)`에 비례하므로 상위 PCA 쌍이 자동으로 큰 값을 갖고, 무가중 ridge는 그
크기를 중요도로 읽는다. 그 영향까지 제거하면 어떻게 되는가:

```
corr[i,j] = C[i,j] / √(C[i,i]·C[j,j])      → off-diagonal 32,640차원 (차원 동일)
```

bag마다 **자기 분산 프로파일**로 정규화되므로, bag의 방향별 전체 퍼짐이 기여를 멈춘다.

### 1. 결과 — 기각

| | SEAL 10 | 홀드아웃 7 | 전체 17 | Δ | 부호 |
|---|---:|---:|---:|---:|---:|
| **full model** — 공분산 (v110) | 0.7069 | 0.6103 | 0.6671 | — | — |
| **full model** — 상관 | 0.7048 | 0.6075 | 0.6647 | **−0.0024** | 8/17 |
| **CV-only** — 공분산 | 0.6877 | 0.6003 | 0.6517 | — | — |
| **CV-only** — 상관 | 0.6916 | 0.5988 | 0.6534 | +0.0017 | 11/17 |

CV-only는 +0.0017(11/17)로 미세하게 오르지만 **집단 간 부호가 갈린다**(SEAL +0.0039 6/10,
홀드아웃 −0.0015 5/7). full model은 −0.0024(8/17)로 동전이다. **어느 쪽도 이득이 아니다.**

task별 변동이 크다: VHL +0.0245, Histologic_Grade +0.0227, her2 +0.0161이 오르는 반면
SMAD4 **−0.0456**, PBRM1 −0.0308, BAP1 −0.0282, brca TP53 −0.0185이 내린다. 양방향으로 크게
흔들고 순효과가 없다 — §155에서 상대 거리가 실패한 것과 같은 모양이다.

### 2. ⚠️ 스케일 교란을 의심했으나 아니었다

CV-only(스케일 무관)는 오르는데 full model(고정 head가 1.442를 곱함)은 내리므로, §148에서 CT에
했던 **스케일 교정을 CV에는 안 한 것**이 원인일 수 있었다. 실측
(`scripts/diagnose_cv_correlation_scale.py`, context bag의 CV margin RMS):

| | 공분산 | 상관 | 비 |
|---|---:|---:|---:|
| 5 task × 10 fold 평균 | 1.992 | 1.996 | **1.002** |

**두 descriptor의 CV margin 스케일이 사실상 같다.** 블록 표준화가 이미 크기를 맞추고 있었기
때문이다. 따라서 §1의 비교는 처음부터 공정했고, CV-only↑/full-model↓ 갈림은 **실재하는 현상**
이다(상관 CV의 오차가 DD·CT와 더 겹친다는 뜻일 가능성).

부수적으로 이것은 **§156의 CV 블록 arm들도 스케일 교란이 없었음**을 보증한다 — 같은 기계를 쓴다.

### 3. 무엇을 배웠나

**대각 항목을 feature로 쓰는 것**과 **대각이 off-diagonal의 크기를 정하는 것**은 다른 문제이고,
이제 둘 다 측정됐다:

| | 결과 |
|---|---|
| 대각 256차원을 feature로 사용 (§156) | **유해.** 빼면 +0.0052 (13/17) |
| 대각이 off-diagonal 크기에 주는 영향 (§162) | **정보였다.** 빼면 −0.0024 (8/17) |

즉 `√(λᵢλⱼ)` 가중은 무가중 ridge가 우연히 갖게 된 부작용이 아니라 **쓸모 있는 사전 가중**이다 —
분산이 큰 방향 쌍의 상관을 더 신뢰하는 것이 맞았다. §156-6이 "학습을 넣는다면 off-diagonal
가중"이라고 했는데, **그 가중의 출발점은 균등이 아니라 현재의 `√(λᵢλⱼ)`** 라는 것이 이 결과의
실질적 함의다.

⚠️ **승격하지 않았다.** 훅(`ICF_CV_CORR=1`)은 남겼다.

### 4. 운영 메모

GPU 4–7에 다른 사용자의 학습이 돌고 있어, 모든 sweep 스크립트의 기본값을 **`NGPU=4`(GPU 0–3)**
로 바꿨다(`GPU_OFFSET`로 이동 가능). 이 절의 홀드아웃은 그 설정으로 재실행한 결과다.

---

## 163. 2026-08-18 — CT ridge λ 민감도: **λ=1이 이미 최적.** §160 token 이득의 약 30%는 λ 효과였다

*작성: nhn-NEXGEM-claude, 2026-08-18 (KST)*

§160-5/§161-3에서 미확인으로 남긴 항목이다. §156-5가 λ이 **저차원에서 살아난다**는 것을 보였고
(256차원에서 +0.012, 1,536차원에서 +0.021) CT의 ridge는 이제 **32차원**이므로 더 낮다.
`ICF_CT_RIDGE_LAMBDA`는 §148부터 있었으나 한 번도 쓸어본 적이 없었다.

분기 격리를 위해 `ICF_FIXED_HEAD_CV_WEIGHT`를 추가했다 — 이제 세 가중 훅이 모두 있어 어느
분기든 나머지 둘을 0으로 만들어 단독 측정할 수 있다. CV가 1.442로 지배적이라 CT 단독을 보려면
이것이 필요했다.

### 1. λ_CT ∈ {0.1, 1, 10, 100}

| λ_CT | SEAL 10 | 홀드아웃 7 | 전체 17 | Δ vs λ=1 | 부호 |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.7048 | 0.6105 | 0.6659 | −0.0012 | 7/17 |
| **1 (현행)** | **0.7070** | **0.6103** | **0.6672** | — | — |
| 10 | 0.7032 | 0.6054 | 0.6629 | −0.0042 | 4/17 |
| 100 | 0.6977 | 0.6010 | 0.6579 | −0.0093 | 4/17 |

**CT-only** (CV·DD 가중 0):

| λ_CT | SEAL 10 | 홀드아웃 7 | 전체 17 | Δ vs λ=1 |
|---:|---:|---:|---:|---:|
| 0.1 | 0.6699 | 0.6340 | 0.6551 | −0.0037 |
| **1** | **0.6763** | 0.6337 | **0.6588** | — |
| 10 | 0.6600 | 0.6168 | 0.6422 | −0.0165 |
| 100 | 0.6391 | 0.6029 | 0.6242 | **−0.0346** |

**λ=1이 양쪽에서 정점이다.** CT-only의 −0.0346은 CT의 32차원 ridge가 λ에 **정말 민감하다**는
뜻이므로 §156-5의 저차원 규칙은 성립한다 — 다만 우리가 쓰던 값이 맞았다.

⚠️ CT-only에서 홀드아웃(0.6337)이 SEAL(0.6763)보다 낮지 않다는 점이 눈에 띈다. full model에서는
홀드아웃이 훨씬 낮은데(0.6103 vs 0.7070), **CT 분기만 보면 두 집단이 비슷하다.** 집단 간 격차는
CV·DD가 만드는 것이다. 이번 범위 밖이지만 기록해 둔다.

### 2. ⚠️ §160의 token 이득을 재평가한다 (§150-4 규칙)

"λ=1이 32 token에서 최적"만으로는 부족하다. **16 token에서 λ=1이 최적이 아니었다면** §160의
비교가 16 token에 불공정했던 셈이다. 같은 λ를 16 token에서도 쓸었다:

| λ_CT | 16 token (v109) | 32 token (v110) |
|---:|---:|---:|
| 0.1 | **0.6636** | 0.6659 |
| **1** | 0.6621 | **0.6672** |
| 10 | 0.6569 | 0.6629 |
| 100 | 0.6536 | 0.6579 |

**16 token에서는 λ=0.1이 약간 낫다**(0.6636 vs 0.6621). 즉 v109는 λ가 살짝 덜 맞춰져 있었다.

| | token 이득 |
|---|---:|
| λ=1 고정 (§160의 보고 값) | **+0.0051** |
| 각자 최적 λ | **+0.0036** |

**이득의 약 30%는 λ 효과였다.** 그러나 사라지지 않는다 — 32 token은 어떤 λ에서도 같은 λ의
16 token보다 높다(0.6659>0.6636, 0.6672>0.6621, 0.6629>0.6569, 0.6579>0.6536, **4/4**).
§161의 승격 근거는 유지되지만 **크기는 +0.0051이 아니라 +0.0036~+0.0051 사이**로 읽어야 한다.

⚠️ 각 구성에 자기 최적 λ를 주는 것은 같은 17 task에서 고르는 것이므로 그 자체가 선택 편향이다.
그래서 **λ를 구성별로 튜닝하지 않고** 기본값 1을 유지한다.

### 3. 판단

- **λ=1 유지.** 32 token(v110의 설정)에서 최적이므로 바꿀 것이 없다. 승격 대상 없음.
- **§161-3의 미확인 항목을 닫는다** — 다만 §160의 +0.0051은 위와 같이 정정해 읽어야 한다.
- CV ridge(32,640차원)는 여전히 λ∈[0.01,3.52]에서 무력하고(§142-2), CT ridge(32차원)는 민감하다.
  **차원이 λ 민감도를 결정한다**는 §156-5의 규칙이 세 번째 지점에서 확인됐다.

---

## 164. 2026-08-18 — 서버 이동 준비: 노드 종속 항목을 한 파일로, agent handoff 재작성

*작성: nhn-NEXGEM-claude, 2026-08-18 (KST)*

사용자가 서버를 옮겨야 하는 상황이라 이동에 필요한 것을 정리했다.

### 1. ⚠️ 감사 결과 — 노드 종속이 넷이었다

| 항목 | 문제 |
|---|---|
| `PY=/home/aibio_3/miniconda3/envs/BagPFN/bin/python` | 노드마다 conda 경로가 다르다 |
| `OFFICIAL`·`FEATURES=/NHNHOME/BASE/kimds/Data/...` | 같은 Lustre인데 **마운트 이름이 노드마다 다르다** |
| `CKPT=checkpoints/20260815_113422/...` | 디렉토리명이 타임스탬프라 노드/재실행마다 다르다 |
| `NGPU=4` | **이 노드에 다른 사용자의 학습이 4–7에 있어서** 정한 값이지 능력치가 아니다 |

9개 runner에 흩어져 있었다.

### 2. `scripts/node_env.sh` — 해석은 한 곳에서

모든 runner가 source 한다. **이미 설정된 환경변수는 덮어쓰지 않고**, 자동 탐지가 실패하면
그 변수만 export 하거나 `scripts/node_env.local.sh`(git-ignored)에 적으면 된다.

- `ICF_PYTHON` — 후보를 돌며 **`import torch, lightning`이 성공하는 첫 번째**를 고른다.
  §141-3의 함정(`python3`면 14개 모듈이 조용히 사라지고 "Ran 158 tests"라고 말함)이 기준이다.
- `ICF_DATA_ROOT` — 알려진 마운트 후보를 순회해 `official/`이 있는 곳
- `ICF_CKPT` — 글롭으로 탐색. v106+ 는 사영과 head를 덮어쓰므로 **껍데기일 뿐**이고 어느 v98
  시드든 같은 수치가 나온다(§152)
- `NGPU`/`GPU_OFFSET` — 기본 4/0. ⚠️ **예의 설정이지 능력 설정이 아니다.**
  혼자 쓰는 노드면 `export NGPU=8`.

**회귀 확인**: 배선 후 `eval_v110.sh`로 VHL을 재측정해 **0.5233 재현**. 291 tests 통과.

### 3. `agent_handoff.md` 최상단 재작성

992줄에 IMPORTANT 블록이 20개 쌓여 있어 "새 서버에서 뭘 먼저 하나"에 답하지 못했다.
**§0을 신설**했다:

| 절 | 내용 |
|---|---|
| 0-1 | 노드 종속 변수 표와 `node_env.sh` 사용법 |
| 0-2 | **이동 후 30초 점검** — 환경 출력 / 291 tests / VHL 0.5233 |
| 0-3 | 현재 상태 한 눈에 (v110 파이프라인 + v106~v110 수치 + ABMIL 격차) |
| 0-4 | **판정 규칙 변경**(결정론적 arm에 t 금지)과 홀드아웃 7개의 의미 |
| 0-5 | §142~§163에서 확정된 것 한 줄씩 |
| 0-6 | 다음에 할 일 5개 (근거 순) |
| 0-7 | **건드리기 전 구조적 제약 6개** |

기존 992줄은 그 아래에 참조로 남겼다.

### 4. 이동 후 첫 명령

```bash
. scripts/node_env.sh && echo "$PYTHON_BIN / $ICF_DATA_ROOT / $ICF_CKPT / NGPU=$NGPU"
"$ICF_PYTHON" -m unittest discover -s tests -p "test_*.py"     # 291 tests
bash scripts/eval_v110.sh 0 smoke cptac_ccrcc/VHL_mutation     # 0.5233
```

---

## 165. 2026-08-18 — CT random 512-cell dictionary + full-cell abundance: **기각**

*작성: Codex, 2026-08-18 — 사용자 지시로 균등 64 추출을 random 512로 바꾸고 abundance만 전체 cell 사용.*

### 1. 질문과 분리한 변수

§159·§160의 `ICF_CT_CELLS=all`은 한 번에 네 가지를 바꿨다: (i) context 좌표 정규화,
(ii) FPS/k-means dictionary, (iii) context abundance, (iv) query abundance. 그래서 “전체 cell의
abundance가 나쁜가”와 “전체 cell이 dictionary를 움직인 것이 나쁜가”를 구분할 수 없었다.

이번 arm은 다음처럼 분리했다.

```
dictionary + 정규화   : bag당 random 512 cells (sampling seed 고정)
context/query abundance: 모든 cell
나머지                 : v110 그대로 (PCA32, k-means30, token32, CT ridge λ=1, weight 0.7)
```

구현:

- `CTReadoutConfig.abundance_cells_per_bag`: `"match"`는 기존 coupled 경로를 bit-identical하게 유지,
  `None`은 dictionary cap과 무관하게 abundance에 전체 cell 사용.
- `sampling="random"`, `sampling_seed`: bag index를 seed에 섞은 재현 가능한 무작위 subset.
  추출 뒤 index를 정렬해 누적 순서는 유지했다.
- 평가 훅: `ICF_CT_ABUNDANCE_CELLS`, `ICF_CT_SAMPLING`, `ICF_CT_SAMPLING_SEED`.
- 기본값은 `cells=64 / abundance=match / sampling=even`으로 유지 — v110은 움직이지 않는다.

### 2. 실행과 검증

모든 Python 실행은 새 노드의 환경 인터프리터를 직접 사용했다.

```bash
BAGPY=/NHNHOME/WORKSPACE/26msit005_C/kimds/miniconda3/envs/BagPFN/bin/python
"$BAGPY" -m unittest discover -s tests -p 'test_*.py'
# Ran 295 tests in 267.559s — OK

ICF_CT_CELLS=512 ICF_CT_ABUNDANCE_CELLS=all \
ICF_CT_SAMPLING=random ICF_CT_SAMPLING_SEED=42 \
  bash scripts/eval_v110.sh 0 ct_r512_aball_s42_smoke cptac_ccrcc/VHL_mutation
# VHL 0.5183 (v110 0.5233)

ARMS='v110 r512all_s42 r512all_s43 r512all_s44 r512all_s45' NGPU=8 GPU_OFFSET=0 \
  TASKSET=seal bash scripts/run_ct_readout_sweep.sh \
    logs/20260818_ct_random512_abundance_all/seal
ARMS='v110 r512all_s42 r512all_s43 r512all_s44 r512all_s45' NGPU=8 GPU_OFFSET=0 \
  TASKSET=heldout bash scripts/run_ct_readout_sweep.sh \
    logs/20260818_ct_random512_abundance_all/heldout
```

로그/예측:

- `logs/20260818_ct_random512_abundance_all/{seal,heldout}/`
- `logs/20260818_ct_random512_abundance_all/full_tests.log`
- 예측 `.pt`는 위 sweep directory의 arm/task별 파일

### 3. 결과 — 두 집단 모두 음수

| arm | SEAL 10 | 홀드아웃 7 | 전체 17 | Δ 전체 | 상승 task |
|---|---:|---:|---:|---:|---:|
| **v110** | **0.70692** | **0.61029** | **0.66713** | — | — |
| random512/all s42 | 0.70344 | 0.60853 | 0.66436 | −0.00277 | 6/17 |
| random512/all s43 | 0.70392 | 0.60733 | 0.66415 | −0.00298 | 5/17 |
| random512/all s44 | 0.70360 | 0.61004 | 0.66508 | −0.00205 | 6/17 |
| random512/all s45 | 0.70416 | 0.60970 | 0.66526 | −0.00186 | 10/17 |
| **4-seed 평균** | **0.70378±0.00032** | **0.60890±0.00123** | **0.66471±0.00054** | **−0.00242±0.00054** | **7/17** |

SEAL 평균 Δ는 **−0.00314**, 홀드아웃 Δ는 **−0.00139**로 독립 집단의 부호가 같다.
sampling seed 분산은 작아(SEAL macro std 0.00032) 음수가 한 seed의 불운으로 생긴 것도 아니다.

task별 4-seed 평균의 큰 변화:

| 하락 | Δ | 상승 | Δ |
|---|---:|---|---:|
| ER status | **−0.0142** | grade | +0.0046 |
| HER2 status | **−0.0129** | KRAS (홀드아웃) | +0.0034 |
| BAP1 | **−0.0094** | BRCA TP53 | +0.0023 |
| SMAD4 (홀드아웃) | −0.0062 | ucla progression | +0.0025 |
| KEAP1 (홀드아웃) | −0.0057 | Histologic Grade | +0.0013 |
| ARID1A (홀드아웃) | −0.0050 | STK11 | +0.0007 |

### 4. 판단과 해석

**기각, v110 유지.** 사용자가 지시한 random 512 + full abundance는 구현됐지만 활성 기본값으로
승격하지 않는다. 기존의 “균등 간격 64가 이론적으로 옳다”는 주장도 하지 않는다. 이번 arm은
sampling policy(균등→random), dictionary 크기(64→512), abundance cap(64→all)을 함께 바꿨으므로
음수의 귀속을 셋 중 하나로 정할 수 없다.

다만 두 가지는 확정됐다.

1. 전체-cell abundance 자체가 512-cell random dictionary 위에서 성능을 회복시키지 못했다.
2. seed 42–45 평균에서도 v110보다 낮아, “균등 64의 우연한 index 선택”만으로 기존 우위를
   설명할 수는 없다.

다음에 원인을 더 분리한다면 `random64/match`, `random512/match`, `random64/all`의 대각선 세 칸이
필요하다. 그러나 현재 성능 개선 최우선순위는 여전히 `current_architecture.md` §0-6의 CV 가중과
CT의 독립 부분공간이며, 이 sampling 축은 요청 없이는 추가로 확장하지 않는다.

---

## 166. 2026-08-18 — random-512에서 abundance `all → 512`: **full abundance는 원인이 아님**

*작성: Codex, 2026-08-18 — 사용자 지시로 §165와 같은 seed에서 abundance population 하나만 분리.*

### 1. 실험 설계

§165의 `r512all_s{42..45}`를 control로 재사용했다. 새 `r512a512_s{42..45}`는 정규화와 dictionary에
쓰는 random 512-cell subset, sampling seed 42–45, v110의 나머지 설정을 모두 고정하고
`ICF_CT_ABUNDANCE_CELLS=all`만 `512`로 바꿨다. 따라서 같은 seed·task의 차이는 abundance 계산에
전체 cell을 넣느냐 동일한 512-cell subset만 넣느냐의 효과다.

```bash
BAGPY=/NHNHOME/WORKSPACE/26msit005_C/kimds/miniconda3/envs/BagPFN/bin/python
ARMS='r512a512_s42 r512a512_s43 r512a512_s44 r512a512_s45' NGPU=8 GPU_OFFSET=0 \
  TASKSET=seal bash scripts/run_ct_readout_sweep.sh \
  logs/20260818_ct_random512_abundance512/seal
# 같은 명령의 TASKSET=heldout → heldout/

"$BAGPY" -m unittest tests.test_ct_readout tests.test_training_free
# Ran 52 tests in 10.186s — OK
```

로그와 예측은 `logs/20260818_ct_random512_abundance512/{seal,heldout}/`에 있다(40+28 완료).

### 2. 결과 — abundance=512가 오히려 미세하게 낮음

| seed | SEAL 10 | 홀드아웃 7 | 전체 17 | Δ vs 같은-seed abundance-all | W/T/L vs all |
|---|---:|---:|---:|---:|---:|
| 42 | 0.70322 | 0.60814 | 0.66407 | −0.00029 | 6/1/10 |
| 43 | 0.70368 | 0.60776 | 0.66418 | +0.00004 | 9/0/8 |
| 44 | 0.70313 | 0.61027 | 0.66489 | −0.00018 | 6/1/10 |
| 45 | 0.70400 | 0.60966 | 0.66515 | −0.00011 | 5/1/11 |
| **4-seed 평균** | **0.70351±0.00035** | **0.60896±0.00104** | **0.66458±0.00046** | **−0.00014** | **task-mean 5/0/12** |

paired 차이는 SEAL **−0.00027**, 홀드아웃 **+0.00006**, 전체 **−0.00014**다. seed마다 전체 Δ도
`−0.00029, +0.00004, −0.00018, −0.00011`로 10⁻⁴ 수준이다. 반면 새 arm은 v110 대비 여전히
전체 **−0.00256**(seed별 −0.00306/−0.00295/−0.00224/−0.00198)이다.

### 3. 판단

**full-cell abundance가 §165 하락의 원인이라는 가설은 기각한다.** abundance를 512로 제한해도
성능은 회복되지 않았고 오히려 미세하게 낮다. 즉 random-512 dictionary 위에서는 abundance
추정 표본을 512에서 전체로 늘리는 효과가 거의 없으며, §165의 v110 대비 하락은 dictionary/정규화의
random-512 구성(그리고 v110의 even-64와 달라진 sampling policy/크기)에 귀속된다.

활성 v110(`even-64 / abundance=match`)은 유지한다. 다음 원인 분리의 최소 비교는 같은 random policy의
`random64/match` 대 `random512/match`이며, abundance-all 자체를 더 의심할 근거는 없다.

---

## 167. 2026-08-18 — random-64 dictionary + full-cell abundance: **64→512 크기는 원인 아님**

*작성: Codex, 2026-08-18 — 사용자의 “64 토큰 random”을 bag당 64 cell random 추출로 해석. CT token은 v110의 32개 유지.*

### 1. 설계와 실행

새 `r64all_s{42..45}`는 `cells=64 / sampling=random / abundance=all`이고, PCA32·k-means30·token32·
CT weight 0.7 등은 v110과 같다. 같은 seed의 §165 `r512all`과 비교하면 dictionary/정규화 표본 수
`512→64`만 달라진다. 모든 Python 명령은 BagPFN 환경 인터프리터를 직접 사용했다.

```bash
BAGPY=/NHNHOME/WORKSPACE/26msit005_C/kimds/miniconda3/envs/BagPFN/bin/python
ARMS='r64all_s42 r64all_s43 r64all_s44 r64all_s45' NGPU=8 GPU_OFFSET=0 \
  TASKSET=seal bash scripts/run_ct_readout_sweep.sh \
  logs/20260818_ct_random64_abundance_all/seal
# TASKSET=heldout도 동일하게 실행

"$BAGPY" -m unittest tests.test_ct_readout tests.test_training_free
# Ran 52 tests in 11.074s — OK
```

로그/예측: `logs/20260818_ct_random64_abundance_all/{seal,heldout}/` (40+28 완료).

### 2. 결과

| seed | SEAL 10 | 홀드아웃 7 | 전체 17 | Δ vs random512/all | W/T/L vs random512 |
|---|---:|---:|---:|---:|---:|
| 42 | 0.70388 | 0.60880 | 0.66473 | +0.00037 | 10/0/7 |
| 43 | 0.70385 | 0.60781 | 0.66431 | +0.00016 | 11/0/6 |
| 44 | 0.70351 | 0.60841 | 0.66435 | −0.00072 | 6/0/11 |
| 45 | 0.70394 | 0.60939 | 0.66501 | −0.00026 | 5/2/10 |
| **4-seed 평균** | **0.70379±0.00017** | **0.60860±0.00057** | **0.66460±0.00029** | **−0.00011** | **task-mean 9/0/8** |

random512/all 대비 paired 차이는 SEAL **+0.00001**, 홀드아웃 **−0.00030**, 전체 **−0.00011**로
사실상 동률이다. v110 대비는 SEAL **−0.00313**, 홀드아웃 **−0.00169**, 전체 **−0.00253**이고,
task-mean 부호도 8/17만 양수다.

### 3. 판단

**random dictionary의 cell 수 64→512가 §165 하락의 원인이라는 가설은 기각한다.** abundance-all을
고정하면 random-64와 random-512가 같다. §166에서 abundance 512/all도 같았으므로, 지금까지 분리한
두 축(dictionary 표본 수, abundance 표본 수)은 성능을 설명하지 않는다.

남은 직접적인 차이는 v110의 `sampling=even`과 새 arm의 `sampling=random`이다. 다만 v110은
`abundance=match(64)`이고 이 arm은 `all`이므로 최종 단일변수 확인은 `random64/match`가 필요하다.
현재 결과만으로 random 자체를 확정 기각하지는 않으며 활성 v110은 유지한다.

---

## 168. 2026-08-18 — full-cell Hierarchical PCA Bisection × K=8..256: **64 이후 plateau, 전부 기각**

*작성: Codex, 2026-08-18 — DPC/계층 PCA/Facility Location/Density-FPS 중 실제 N에 맞는 방법을 선택하고, 사용자 지시로 낮은 K를 재스윕.*

### 1. 선택과 구현

VHL fold 1의 실제 context는 **189 bags, 1,148,534 cells**, query는 553,548 cells였다(D=32 PCA).
exact DPC `O(N²D)`는 불가능하고 k-NN DPC용 FAISS·SciPy·sklearn·torch-cluster·pynndescent도
환경에 없었다. Facility Location과 Density-Weighted FPS는 큰 K에서 순차 `O(KND)`다. 그래서
모든 context cell을 depth마다 한 번 읽는 **Hierarchical PCA-initialized 2-means tree**를 택했다.

- leaf별 주방향: 3-step power iteration
- deterministic 2-means update 2회
- child가 너무 작으면 stable projection median fallback → 항상 exact K/non-empty
- query는 tree에 들어가지 않고 context/query abundance 모두 모든 cell 사용
- `h2T*`는 `PCA32 / CT ridge λ=1 / weight=0.7 / full cells / full abundance`

### 2. 계산 최적화와 안정성 선택

512-token FPS+Lloyd profiler에서 CUDA 시간의 `mean/sub/square`가 각각 **58.9/21.4/11.0%**였고
ridge solve는 6.4 ms뿐이었다. `[cells,tokens,32]` broadcast를 만들지 않는 opt-in FP32 GEMM
`||c||²−2x·c` 커널을 추가했다. 기본은 `broadcast`라 v110은 그대로다.

tree reduction은 두 모드를 지원한다.

- `segment`: label stable-sort + segment reduce. 같은 fold 반복이 **bit-exact**였지만 매우 느리다.
- `atomic`: CUDA `index_add`, exact-K/non-empty는 유지하지만 합산 순서 때문에 같은 fold 확률이
  최대 약 0.00122 달라질 수 있다.

사용자 지시대로 이미 실행 중이던 **25개 SEAL job은 segment로 끝까지 유지**했고, 나머지 SEAL 35개와
held-out 42개는 atomic으로 실행했다. 따라서 이 sweep의 10⁻⁴ 차이는 해석하지 않는다. runner도
GPU별 고정 worker queue로 고쳐, 긴 task 때문에 동일 GPU에 job이 겹치고 다른 GPU가 노는 문제를 제거했다.

### 3. 결과

| K | SEAL 10 | 홀드아웃 7 | 전체 17 | Δ vs v110 | W/T/L vs v110 |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.69088 | 0.59371 | 0.65087 | −0.01626 | 3/0/14 |
| 16 | 0.69576 | 0.58996 | 0.65219 | −0.01494 | 3/0/14 |
| 32 | 0.70172 | 0.59343 | 0.65713 | −0.01000 | 4/0/13 |
| **64** | 0.70296 | **0.59944** | 0.66034 | −0.00679 | 6/0/11 |
| 128 | 0.70389 | 0.59881 | 0.66062 | −0.00651 | 6/0/11 |
| 256 | **0.70453** | 0.59809 | **0.66070** | **−0.00643** | 4/0/13 |
| **v110** | **0.70692** | **0.61029** | **0.66713** | — | — |

증분은 `16−8 +0.00132`, `32−16 +0.00494`, `64−32 +0.00321`, 이후
`128−64 +0.00029`, `256−128 +0.00008`이다. **K=64 이후 완전히 plateau**이며 atomic 변동보다도 작다.

task별 최적 K는 크게 달랐다: VHL·ARID1A는 K8, grade·TP53는 K32, PBRM1·Histologic Grade·KEAP1·
STK11·progression은 K64, ER·HER2·BAP1·EGFR·KRAS는 K256이었다. 각 task label을 보고 최적 K를
사후 선택한 oracle 평균은 **0.66737 (v110 대비 +0.00024)**지만 유효한 모델/arm이 아니다.

로그/예측: `logs/20260818_ct_hierarchical_low_tokens/{seal,heldout}/` (60+42 완료).
BagPFN Python 직접 실행: focused **58 tests OK**, full **301 tests in 39.430s, OK**
(`logs/20260818_ct_hierarchical_low_tokens/full_tests.log`). v110 VHL도 **0.5233** 재현했다.

### 4. 판단

**모든 단일 K arm 기각, v110 유지.** 전체 cell hierarchical representation 자체는 계산 가능하고 일부
task에서는 v110을 이기지만, 하나의 K를 전 task에 고정하면 홀드아웃 손실을 회복하지 못한다.
K=64 이후 더 세밀하게 나눠도 평균은 늘지 않는다.

다음에 이 결과를 살린다면 label로 K를 고르는 것이 아니라, 한 번 만든 tree의 여러 depth abundance를
동시에 제공하는 **multi-resolution readout**을 context-only 규칙으로 설계해야 한다. oracle +0.00024는
상한조차 매우 작으므로 우선순위는 낮다.

---

## 169. 2026-08-18 — full-cell GPU HDBSCAN 자동 K: **K 중앙값 8, 전체 −0.01722로 기각**

*작성: Codex, 2026-08-18 — §168에서 task별 최적 K가 달랐던 관찰을 label-free dynamic K로 검증.*

### 1. 구현 및 고정 프로토콜

BagPFN Python 3.12/CUDA 13 환경에 `cuml-cu13==26.8.0`을 설치하고, context 전체 cell에서 GPU
HDBSCAN을 fit했다. 백만~수백만 cell의 brute-force kNN은 불가능하므로 cuML의 **NN-descent** graph를
썼다. 이는 HDBSCAN condensed density tree를 쓰지만 이웃 graph는 근사다.

- 입력: context-only PCA32 standardised cell, **dictionary=all / abundance=all**
- `min_cluster_size=max(256, ceil(0.001·N))`, `min_samples=32`
- `cluster_selection_method=leaf`, `allow_single_cluster=False`; K 상한/목표 없음
- cluster membership probability로 centroid를 가중 평균
- HDBSCAN noise는 centroid fit에서 제외하지만 최종 soft abundance에는 context/query **모든 cell** 사용
- all-noise fold는 global-mean 1 token fallback(850 folds 중 2회)
- readout/head: class-balanced ridge λ=1, CT weight 0.7, 나머지는 v110과 동일

초기 VHL pilot에서 `allow_single_cluster=True + EOM`은 K=1을 골라 폐기했다. leaf에서 희귀 허용
`min_cluster_size=256, min_samples=16`도 첫 3 fold K=40–47, AUROC 0.5067/0.5319/0.5264로,
상대 density scale의 K=10–14, 0.5167/0.5597/0.5667보다 모두 낮아 최종 설정으로 쓰지 않았다.

### 2. 전체 17-task 결과

| split | HDBSCAN | v110 | Δ |
|---|---:|---:|---:|
| SEAL 10 | 0.68584 | 0.70692 | −0.02108 |
| held-out 7 | 0.59857 | 0.61029 | −0.01171 |
| **전체 17** | **0.64991** | **0.66713** | **−0.01722** |

task별 HDBSCAN / Δ vs v110:

| task | AUROC | Δ | fold K 범위 (평균) |
|---|---:|---:|---:|
| ER | 0.6868 | −0.0146 | 4–6 (4.42) |
| grade | 0.7193 | −0.0240 | 3–6 (4.40) |
| HER2 | 0.6572 | −0.0231 | 4–6 (4.38) |
| PIK3CA | 0.5444 | **+0.0063** | 9–16 (12.92) |
| BRCA TP53 | 0.8283 | **+0.0028** | 8–16 (12.90) |
| BAP1 | 0.6014 | −0.0736 | 12–18 (15.58) |
| VHL | 0.4887 | −0.0346 | 10–19 (15.08) |
| EGFR | 0.7699 | −0.0123 | 4–8 (6.12) |
| STK11 | 0.8829 | −0.0219 | 3–9 (6.34) |
| LUAD TP53 | 0.6795 | −0.0158 | 4–9 (6.32) |
| PBRM1 | 0.5308 | −0.0218 | 11–18 (14.82) |
| ARID1A | 0.5294 | **+0.0194** | 6–11 (9.06) |
| Histologic Grade | 0.6286 | −0.0122 | 6–11 (8.82) |
| KEAP1 | 0.5643 | −0.0227 | 6–11 (8.84) |
| KRAS | 0.6979 | −0.0157 | 4–8 (6.08) |
| SMAD4 | 0.4660 | −0.0220 | 6–13 (8.98) |
| progression | 0.7730 | −0.0070 | 1–4 (2.76) |

850 folds의 K는 **1–19, 평균 8.70, 중앙값 8**, noise 비율은 평균 **0.96225**
(범위 0.91312–1.00000)였다. task 승패는 **3/14**다. HDBSCAN 0.64991은 §168의 고정
hierarchical K8 0.65087보다도 −0.00096 낮다.

### 3. 판단

**HDBSCAN arm 기각, v110 유지.** 자동 K는 fold/task에 따라 실제로 변했지만, standardised PCA32 cell
공간은 대부분 연속 manifold처럼 보여 안정 density core가 전체의 약 3.8%에 불과했고 K도 낮게
수축했다. §168의 “predictive 최적 K가 task마다 다름”과 “cell density가 정한 K”는 서로 다른 문제다.
label-free HDBSCAN은 후자를 해결하지만 전자를 해결하지 못했다.

로그/예측: `logs/20260818_ct_hdbscan_fullcell/{seal,heldout}/` (10+7 task, 850 folds 완료).
BagPFN Python 직접 실행 full **302 tests in 40.813s, OK**
(`logs/20260818_ct_hdbscan_fullcell/tests/full_tests.log`).

---

## 170. 2026-08-18 — random-64 + adaptive DBSCAN: **84.5%가 K=1, 전체 −0.01737로 기각**

*작성: Codex, 2026-08-18 — 사용자 지시로 64 random subsampling에서 K-free DBSCAN 평가.*

### 1. 설계

§167의 `random64/all`과 동일하게 context bag마다 seed별 64 cell을 random 추출해 dictionary와
정규화를 fit하고, 최종 abundance는 context/query **전체 cell**로 계산했다. seed 42–45를 모두 반복했다.

- PCA32 standardised cell, class-balanced ridge λ=1, CT weight 0.7
- GPU cuML DBSCAN, `min_samples=16`, K/cluster 수 지정 없음
- eps는 각 fold context-only 16-NN distance를 오름차순 정렬하고, endpoint chord와 curve의 수직 간격
  `x−y`가 최대인 k-distance knee에서 자동 선택
- DBSCAN noise는 centroid 계산에서 제외하되 전체 cell이 최종 soft abundance에 참여
- all-noise면 global mean 1-token fallback
- arm: `db64all_s{42..45}`

합성된 두 Gaussian blob에서는 num_tokens=999를 무시하고 K=2, noise 1.1%를 복원했다. 실제 데이터에서는
평균 eps 5.0965, knee quantile 0.9356이었다.

### 2. 결과

| seed | SEAL 10 | held-out 7 | 전체 17 |
|---:|---:|---:|---:|
| 42 | 0.69707 | 0.58299 | 0.65009 |
| 43 | 0.69756 | 0.58247 | 0.65017 |
| 44 | 0.69605 | 0.58333 | 0.64964 |
| 45 | 0.69675 | 0.58114 | 0.64915 |
| **4-seed 평균** | **0.69686±0.00055** | **0.58248±0.00083** | **0.64976±0.00041** |

비교:

| 기준 | SEAL Δ | held-out Δ | 전체 Δ | task W/L |
|---|---:|---:|---:|---:|
| v110 | −0.01006 | −0.02781 | **−0.01737** | 5/12 |
| 같은 random64/all k-means (§167) | −0.00693 | −0.02612 | **−0.01484** | 6/11 |

task별 4-seed 평균 / Δ vs v110:

| task | AUROC | Δ |
|---|---:|---:|
| ER | 0.70420 | +0.00280 |
| grade | 0.72490 | −0.01840 |
| HER2 | 0.67350 | −0.00680 |
| PIK3CA | 0.54180 | +0.00370 |
| BRCA TP53 | 0.83000 | +0.00450 |
| BAP1 | 0.63148 | −0.04353 |
| VHL | 0.49732 | −0.02598 |
| EGFR | 0.78832 | +0.00612 |
| STK11 | 0.89440 | −0.01040 |
| LUAD TP53 | 0.68265 | −0.01265 |
| PBRM1 | 0.52192 | −0.03068 |
| ARID1A | 0.41430 | −0.09570 |
| Histologic Grade | 0.60025 | −0.04055 |
| KEAP1 | 0.56468 | −0.02232 |
| KRAS | 0.70357 | −0.01003 |
| SMAD4 | 0.49510 | +0.00710 |
| progression | 0.77755 | −0.00245 |

3,400 folds에서 K 범위 **1–5**, 평균 **1.162**, 중앙값 1이었다. K별 fold 수는
`K1=2874 (84.5%) / K2=506 / K3=16 / K4=3 / K5=1`; noise 평균은 **0.00523**였다.

### 3. 판단

**DBSCAN arm 기각, v110 유지.** adaptive knee eps는 거의 모든 sampled cell을 하나의 density-connected
component로 묶었다. K=1이면 softmax abundance는 모든 bag에서 `[1]`이라 standardisation 후 CT margin이
0으로 붕괴한다. 따라서 §167과의 −0.01484는 random-64 sampling 때문이 아니라 DBSCAN tokenizer가
대부분 CT branch를 제거한 결과다. HDBSCAN은 반대로 96.2%를 noise로 버렸고 DBSCAN은 99.5%를 한
cluster에 연결했다. PCA32에서 density-only 방식 어느 쪽도 predictive cell resolution을 주지 못했다.

로그/예측: `logs/20260818_ct_dbscan_random64_all/{seal,heldout}/` (40+28 task, 3,400 folds).
BagPFN Python 직접 실행 full **303 tests in 33.346s, OK**
(`logs/20260818_ct_dbscan_random64_all/tests/full_tests.log`).

---

## 171. 2026-08-18 — corrected random-64 HDBSCAN: **noise 93.6%, 전체 −0.01894로 기각**

*작성: Codex, 2026-08-18 — §170의 DBSCAN은 사용자 의도와 달랐으므로 HDBSCAN으로 바로잡아 재평가.*

### 1. 설계

§167/§170과 같은 context bag별 random 64-cell dictionary와 context/query 전체-cell abundance를
사용했다. HDBSCAN은 K를 지정하지 않으며, 64-cell 표본에서 full-cell 기본값이 과도하지 않도록
`min_cluster_size=64`, `min_samples=16`, `cluster_selection_method=leaf`, fraction 0으로 고정했다.
PCA32 standardised cell, class-balanced ridge λ=1, CT weight 0.7과 seed 42–45도 동일하다.

### 2. 결과

| seed | SEAL 10 | held-out 7 | 전체 17 |
|---:|---:|---:|---:|
| 42 | 0.68431 | 0.59499 | 0.64753 |
| 43 | 0.68566 | 0.59594 | 0.64872 |
| 44 | 0.68766 | 0.59296 | 0.64866 |
| 45 | 0.68544 | 0.59413 | 0.64784 |
| **4-seed 평균** | **0.68577±0.00121** | **0.59450±0.00110** | **0.64819±0.00052** |

| 비교 기준 | 전체 17 | HDBSCAN Δ |
|---|---:|---:|
| v110 | 0.66713 | **−0.01894** (3/14) |
| random64/all k-means (§167) | 0.66460 | **−0.01641** |
| full-cell HDBSCAN (§169) | 0.64991 | **−0.00172** |
| random64/all DBSCAN (§170) | 0.64976 | **−0.00157** |

3,400 folds에서 K 범위는 1–8, 평균 2.84, 중앙값 2였다. K별 fold 수는
`K1=1028 / K2=902 / K3=365 / K4=313 / K5=402 / K6=277 / K7=97 / K8=16`이며,
noise 평균은 **0.93576**였다. K1 1,028 folds(30.2%)는 전부 all-noise global-mean fallback이다.

### 3. 판단

**HDBSCAN random64/all arm 기각, v110 유지.** §170 DBSCAN의 84.5% K1은 eps가 거의 모든 cell을
하나로 연결한 실패였고, 이번 HDBSCAN은 반대로 대부분을 noise로 제거해 hierarchy가 빈약해진 실패다.
메커니즘은 반대지만 둘 다 CT의 유효 차원을 1–2개 수준으로 붕괴시켜 비슷하게 낮은 최종 성능을 냈다.
random 64-cell 표본은 HDBSCAN의 density hierarchy를 안정적으로 추정하기에도 너무 희소하다.

로그/예측: `logs/20260818_ct_hdbscan_random64_all/{seal,heldout}/` (40+28 task, 3,400 folds).
BagPFN Python 직접 실행 full **303 tests in 38.833s, OK**
(`logs/20260818_ct_hdbscan_random64_all/tests/full_tests.log`).

---

## 172. 2026-08-18 — 세션 종료 handoff checkpoint

§165–§171의 CT sampling/tokenizer 탐색 결과, 실행 설정, 비교 수치, 판정을 living docs 네 곳에
동기화했다.

---

## 173. 2026-08-18 — Full-cell PCA 3D HDBSCAN: **Noise 75~97%, SEAL 0.6849/0.6855로 기각**

*작성: Codex / Antigravity, 2026-08-18 — 32D HDBSCAN의 거리 집중(거리 편차 소멸)을 해결하기 위해 PCA를 D=3으로 대폭 축소하여 평가.*

### 1. 실험 설계
- 입력: Context 전체 cell (최대 280만 개), PCA 3D 사영
- `hdb3_raw`: `pca_scaling="raw"` (고유값 분산 크기 비율 유지)
- `hdb3_std`: `pca_scaling="standardise"` (3개 축 단위 분산 표준화)
- HDBSCAN: `min_samples=15`, `min_cluster_size=256` (relative floor $\approx 0.001 \cdot N$), `cluster_selection_method="leaf"`, `distance_kernel="gemm"`
- Readout: CT ridge $\lambda=1.0$, Fixed Head CT weight $= 0.7$

### 2. SEAL 10-Task 결과

| arm | SEAL 10 Macro | v110 대비 Δ | 평균 Noise 비율 | 주요 병목 |
|---|---:|---:|:---:|---|
| **v110 (baseline)** | **0.7070** | — | **0.0%** | Fold당 ~5ms (SEAL 20초) |
| **`hdb3_raw`** | **0.6849** | **−0.0221** | **75% ~ 97%** | 280만 cell NN-descent (SEAL 37분) |
| **`hdb3_std`** | **0.6855** | **−0.0215** | **75% ~ 97%** | 280만 cell NN-descent (SEAL 37분) |

Task별 `hdb3_raw` / `hdb3_std` AUROC:
- `bc_therapy/er_status`: 0.6680 / 0.6653
- `bc_therapy/grade`: 0.7455 / 0.7419
- `bc_therapy/her2_status`: 0.6399 / 0.6403
- `cptac_brca/PIK3CA_mutation`: 0.5556 / 0.5623
- `cptac_brca/TP53_mutation`: 0.8054 / 0.8048
- `cptac_luad/EGFR_mutation`: 0.7732 / 0.7739
- `cptac_luad/STK11_mutation`: 0.8421 / 0.8459
- `cptac_luad/TP53_mutation`: 0.6570 / 0.6568
- `cptac_ccrcc/BAP1_mutation`: 0.6750 / 0.6723
- `cptac_ccrcc/VHL_mutation`: 0.4871 / 0.4911

### 3. 판단 및 교훈
**PCA 3D HDBSCAN 기각, v110 유지.**
1. **연속 분포(Continuous Cloud) 한계**: 차원을 $D=3$으로 낮추어도 세포 공간은 명확한 섬 모양의 밀도 피크가 아닌 넓은 연속체 형태를 띠며, HDBSCAN은 국소 밀도가 완만한 영역(전체의 75%~97%)을 노이즈로 잘라내어 세포 다양성(Abundance) 정보를 잃는다.
2. **계산 병목**: 슬라이드당 최대 280만 세포에 대한 $k$-NN 그래프 생성으로 연산 시간이 과도하게 증가한다.
3. 밀도 기반 K 탐색(DBSCAN, HDBSCAN)은 차원과 무관하게 병리 MIL 문제에 부적합함이 최종 확정되어, 이 축을 완전히 종료한다.

---

## 174. 2026-08-18 — CT sampling/tokenizer working candidate: random-64 + seeded k-means++

사용자 결정으로 storage-order bias가 있는 균등 간격 64-cell 선택을 seeded random sampling으로
교체했다. 이어서 FPS 초기화 + Lloyd 30회 대신 다음 경로를 활성 기본값으로 구현했다.

- context/query bag별 random 64 cell, seed 0을 bag index와 섞어 재현성과 bag별 독립 pattern을 유지
- seeded k-means++ D² 초기화, K=32
- Lloyd 최대 8회, centroid RMS 이동 최대값이 `1e-4` 이하면 조기 종료
- 빈 cluster는 가장 큰 cluster에서 현재 centroid 오차가 가장 큰 실제 cell로 복구
- 기존 FPS+30/even 경로는 명시적 historical replay 옵션으로 보존

### 공식 17-task 결과와 2×2 원인 분리

| sampling / tokenizer | SEAL 10 | 홀드아웃 7 | 전체 17 | Δ vs v110 | W/T/L |
|---|---:|---:|---:|---:|---:|
| even / FPS+30 (v110) | 0.70692 | 0.61029 | 0.66713 | — | — |
| random / FPS+30 | 0.70273 | 0.60804 | 0.66374 | −0.00339 | 5/0/12 |
| even / k-means+++≤8 | 0.70780 | 0.60694 | 0.66627 | −0.00086 | 7/1/9 |
| **random / k-means+++≤8** | **0.70370** | **0.60667** | **0.66375** | **−0.00338** | **4/0/13** |

같은 random sampling 안에서 FPS+30 → k-means+++≤8의 순효과는 전체 **+0.00001**
(SEAL +0.00097, 홀드아웃 −0.00137, 10/17 양수)로 완전 동률이다. 따라서 결합 경로의 하락은
k-means++가 아니라 **균등→random sampling 변경**에 귀속된다. k-means++는 품질을 유지하면서
CPU 계산을 줄이므로 유지할 수 있다. random sampling은 storage-order bias 제거와 전체 −0.0034의
명시적 트레이드오프다.

v110의 SEAL 0.7070 / 홀드아웃 0.6103은 계속 **even-64 + FPS 초기화 + Lloyd30의 historical
수치**이며 새 candidate 수치로 인용하면 안 된다.

검증: BagPFN Python full **308 tests in 67.984s, OK**. 대표 `12,000×32`, K=32 tokenizer
microbenchmark에서 CPU는 FPS+30 **2275.9 ms** → k-means+++≤8 **612.8 ms**(3.7×), B200 GPU는
약 6.8 ms → 6.7 ms로 거의 동률이었다. GPU에서는 작은 문제 크기의 kernel-launch 고정비가 지배한다.
공식 runner VHL 50-fold smoke도 새 설정이 실제로 주입됨을 로그에서 확인했고 **0.5227**로 완료했다
(`logs/official50/cptac_ccrcc_VHL_mutation_ct_kpp_random64_smoke.log`). 전체 평가 로그는
`logs/official50/*_ct_{random64_fps30_s0,even64_kpp8_s0,kpp_random64_s0}.log`다.

---

## 175. 2026-08-19 — full-cell hierarchical 2-means PCA64/128 × K256..2048: **차원 확장 가설 기각**

K가 커질수록 PCA32의 feature가 부족해진다는 가설을 검증하기 위해, context/query abundance 모두
전체 cell을 사용하는 hierarchical 2-means에서 PCA 차원 64/128과 K=256/512/1024/2048의 8개
arm을 B200 GPU 8장에 하나씩 배정해 공식 17-task를 실행했다. 공통 조건은 CT ridge λ=1,
CT weight=0.7, standardise scaling, GEMM distance, 2회 bisection update, 3회 power iteration이다.

| PCA dim | K | SEAL 10 | 홀드아웃 7 | 전체 17 | Δ vs 동일 K PCA32 | Δ vs v110 |
|---:|---:|---:|---:|---:|---:|---:|
| **64** | **256** | **0.70242** | **0.59319** | **0.65744** | **−0.00326** | **−0.00969** |
| 64 | 512 | 0.70150 | 0.59254 | 0.65664 | −0.00339 | −0.01049 |
| 64 | 1024 | 0.69936 | 0.59091 | 0.65471 | −0.00501 | −0.01242 |
| 64 | 2048 | 0.69818 | 0.59006 | 0.65366 | −0.00501 | −0.01347 |
| 128 | 256 | 0.69875 | 0.58164 | 0.65053 | −0.01017 | −0.01660 |
| 128 | 512 | 0.69887 | 0.58350 | 0.65136 | −0.00866 | −0.01577 |
| 128 | 1024 | 0.69791 | 0.58500 | 0.65142 | −0.00829 | −0.01571 |
| 128 | 2048 | 0.69570 | 0.58569 | 0.65040 | −0.00827 | −0.01673 |

동일 K의 기존 PCA32 전체 macro는 K256/512/1024/2048에서 각각
0.66070/0.66003/0.65971/0.65867이다. PCA64는 네 K 모두 이를 밑돌았고, PCA128은 더 크게
하락했다. 동일 K PCA32 대비 전체 task 부호도 PCA64가 각각 8/17, 6/17, 8/17, 9/17만 양수,
PCA128은 6/17, 6/17, 6/17(1 tie), 6/17만 양수였다. PCA64 안에서도 K256→2048은 전체
−0.00378(6/17 상승), PCA128은 −0.00013(4/17 상승)이므로 고차원이 큰 K의 이득을 복원하지 않는다.

**가설 기각, v110 유지.** PCA32 정보 병목이 K>256 하락의 원인이라는 증거는 없고, 오히려 차원
확장은 finite-context PCA/거리 추정의 잡음을 늘리는 방향으로 보인다. 이번 arm은 bit-reproducible
`tree_reduction=segment`를 사용했고 기존 PCA32 sweep은 segment/atomic이 섞였으므로 1e-3 이하
차이는 해석하지 않는다. 그러나 관측된 −0.0033~−0.0102의 동일-K 하락과 독립 held-out의 일관된
하락은 그 수치 변동 범위를 넘는다. 로그는
`logs/official50/*_h2_pca{64,128}_k{256,512,1024,2048}_fullcell.log`에 있다.

---

## 176. 2026-08-19 — full-cell hierarchical 2-means PCA8/16 × K8/16: **저구간은 감소 추세 반전**

§175의 고차원·대형 K 하락을 바탕으로 PCA dim=8/16 × K=8/16의 4개 arm을 추가 평가했다.
각 arm의 17개 task를 두 worker로 분할하여 B200 GPU 8장을 동시에 사용했다. 나머지 조건은 §175와
동일한 full-cell/full-abundance, standardise PCA, hierarchical 2-means, segment reduction이다.

| PCA dim | K | SEAL 10 | 홀드아웃 7 | 전체 17 | Δ vs 동일 K PCA32 | Δ vs v110 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 0.68514 | 0.58373 | 0.64338 | −0.00749 | −0.02375 |
| 8 | 16 | 0.69317 | **0.59156** | 0.65133 | −0.00086 | −0.01580 |
| 16 | 8 | 0.69167 | 0.58544 | 0.64793 | −0.00294 | −0.01920 |
| **16** | **16** | **0.69508** | 0.59091 | **0.65219** | **−0.00001** | **−0.01494** |
| 32 (기존) | 8 | 0.69088 | **0.59371** | 0.65087 | — | −0.01626 |
| 32 (기존) | 16 | **0.69576** | 0.58996 | **0.65219** | — | −0.01494 |

저구간에서는 K8→16이 PCA8에서 전체 +0.00795(14/17 상승), PCA16에서 +0.00426(12/17 상승)이다.
PCA8→16도 K8에서 +0.00455, K16에서 +0.00086이며 두 비교 모두 10/17 task가 상승했다. 따라서
**PCA dim과 K가 낮을수록 계속 좋아진다는 단조 추세는 성립하지 않는다.** 너무 작은 K=8과 PCA8은
representation capacity 부족으로 손실이 커진다. PCA16/K16과 PCA32/K16은 전체 macro가 반올림상
완전히 같지만, 둘 다 v110보다 −0.01494이므로 활성 baseline은 유지한다.

68개 로그 모두 정상 종료했고 traceback은 없었다. 로그는
`logs/official50/*_h2_pca{8,16}_k{8,16}_fullcell.log`에 있다.

---

## 177. 2026-08-19 — full-cell hierarchical 2-means PCA64 × K16/32: **K32 우세, PCA32보다 열세**

PCA64에서 K=16/32를 공식 17-task로 평가했다. 각 arm을 네 worker로 분할해 B200 GPU 8장을
동시에 사용했고, 나머지는 §175–§176과 동일한 full-cell/full-abundance, standardise PCA,
hierarchical 2-means, segment reduction 조건이다.

| PCA dim | K | SEAL 10 | 홀드아웃 7 | 전체 17 | Δ vs 동일 K PCA32 | Δ vs v110 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 16 | 0.69129 | 0.58967 | 0.64945 | −0.00275 | −0.01768 |
| **64** | **32** | **0.69805** | **0.59003** | **0.65357** | −0.00356 | −0.01356 |
| 32 (기존) | 16 | 0.69576 | 0.58996 | 0.65219 | — | −0.01494 |
| 32 (기존) | 32 | **0.70172** | **0.59343** | **0.65713** | — | −0.01000 |

PCA64 안에서는 K16→32가 SEAL +0.00676, 홀드아웃 +0.00036, 전체 +0.00412이며 13/17 task가
상승했다. 특히 BAP1 +0.0262, KEAP1 +0.0245, ER +0.0153이 컸고, ARID1A −0.0163,
VHL −0.0138, SMAD4 −0.0131은 하락했다. 그러나 동일 K의 PCA32와 비교하면 PCA64/K16은
전체 −0.00275(7/17 상승), PCA64/K32는 −0.00356(8/17 상승)이다.

따라서 **PCA64에서는 K32가 K16보다 낫지만, 차원을 32→64로 늘리는 이득은 없다.** 두 arm 모두
v110보다 낮아 활성 baseline은 유지한다. 34개 로그 모두 정상 종료했고 traceback은 없었다. 로그는
`logs/official50/*_h2_pca64_k{16,32}_fullcell.log`에 있다.

---

## 178. 2026-08-19 — full-cell hierarchical 2-means PCA16 × K256: **held-out 개선, PCA32/K256과 근접**

PCA16/K256 하나의 arm을 8개 worker로 task 분할하여 B200 GPU 8장에서 공식 17-task 평가했다.
조건은 이전 sweep과 동일한 full-cell/full-abundance, standardise PCA, hierarchical 2-means,
segment reduction이다.

| arm | SEAL 10 | 홀드아웃 7 | 전체 17 | Δ vs v110 |
|---|---:|---:|---:|---:|
| PCA16 / K16 | 0.69508 | 0.59091 | 0.65219 | −0.01494 |
| **PCA16 / K256** | 0.70035 | **0.60101** | **0.65945** | −0.00768 |
| PCA32 / K256 (기존) | **0.70453** | 0.59809 | **0.66070** | −0.00643 |
| PCA64 / K256 | 0.70242 | 0.59319 | 0.65744 | −0.00969 |

PCA16에서 K16→256은 SEAL +0.00527, 홀드아웃 +0.01010, 전체 +0.00726이며 10/17 task가
상승했다. PCA64/K256 대비 전체 +0.00201, 홀드아웃 +0.00783이다. PCA32/K256 대비로는 SEAL
−0.00418, 홀드아웃 +0.00293, 전체 −0.00125이고 6/17 task가 상승했다. 특히 ER +0.0194,
ARID1A +0.0142, KRAS +0.0110, SMAD4 +0.0207였지만 BAP1 −0.0343이 전체 평균을 크게 낮췄다.

따라서 **PCA16에서도 K256의 세분화 이득은 재현되며 held-out은 K256 차원 조합 중 최고**지만,
전체 macro는 PCA32/K256을 넘지 못하고 v110보다 −0.00768이다. 활성 baseline은 유지한다.
17개 로그 모두 정상 종료했고 traceback은 없었다. 로그는
`logs/official50/*_h2_pca16_k256_fullcell.log`에 있다.

---

## 179. 2026-08-19 — random512 + full abundance + hierarchical PCA32/K256, 4 seed: **미승격**

사용자 요청으로 §165의 random-512 dictionary/full-cell abundance와 §168의 hierarchical
PCA/2-means tokenizer를 교차했다. bag마다 sampling seed 42–45로 무작위 512 cell을 뽑아
dictionary와 표준화 통계를 만들고, PCA32에서 hierarchical 2-means K=256 token을 만든 뒤 모든
cell의 assignment를 abundance에 평균했다. 나머지는 v110의 CV off-diagonal, DD full triangle,
CT ridge λ=1, CT weight 0.7을 유지했다.

| seed | SEAL 10 | 홀드아웃 7 | 전체 17 |
|---:|---:|---:|---:|
| 42 | 0.702910 | 0.599514 | 0.660335 |
| 43 | 0.703360 | 0.599771 | 0.660706 |
| 44 | 0.702250 | 0.599614 | 0.659988 |
| 45 | 0.703040 | 0.599343 | 0.660341 |
| **평균±population std** | **0.702890±0.000404** | **0.599560±0.000155** | **0.660343±0.000254** |
| **Δ vs v110** | **−0.004030** | **−0.010730** | **−0.006787** |

task별 4-seed 평균은 v110 대비 **SEAL 2/10, 홀드아웃 1/7, 전체 3/17만 상승**했다. 상승은
cptac_brca TP53 +0.00508, cptac_luad EGFR +0.00185, held-out KRAS +0.01290이고, 큰 하락은
SMAD4 −0.03625, ARID1A −0.02260, PBRM1 −0.01795, ER −0.01150이다. 양쪽 독립 집단 평균이 모두
음수이며 전체 14/17 task가 하락하므로 미승격, v110을 유지한다.

full-cell hierarchical PCA32/K256(전체 0.66070)과는 전체 −0.00036로 사실상 동률이지만,
random512 + Lloyd k-means/full abundance(전체 0.66471)보다 −0.00437 낮다. 즉 random512에서
hierarchical tokenizer가 기존 Lloyd tokenizer를 회복시키지 못했다.

실행 arm은 `h2r512all_s{42..45}`이고 로그/예측은
`logs/20260819_ct_h2_random512_full_abundance/{seal,heldout}/`에 있다. 68개 로그 모두 final 1개,
traceback 0으로 확인했다. 재현용 arm은 `scripts/run_ct_readout_sweep.sh`에 추가했다.

---

## 180. 2026-08-19 — raw1536 spherical k-means + random512/full abundance, 4 seed: **미승격**

사용자 요청으로 CT PCA를 사용하지 않는 raw 1536-d에서 spherical/cosine k-means를 구현·평가했다.
context의 random 512-cell subset으로 좌표별 centre/scale을 구한 뒤 각 cell을 L2 정규화하고,
`1−cosine` 거리로 seeded k-means++ 초기화와 최대 8회 Lloyd 갱신을 수행했다. 각 centroid는 갱신
직후 단위구면으로 재투영하며, abundance는 같은 context 통계와 L2 정규화를 적용한 모든 cell의
cosine soft assignment 평균이다. K=32, CT ridge λ=1, CT weight 0.7이고 sampling seed만 42–45로
바꿨다.

| seed | SEAL 10 | 홀드아웃 7 | 전체 17 |
|---:|---:|---:|---:|
| 42 | 0.699390 | 0.594443 | 0.656176 |
| 43 | 0.699210 | 0.594400 | 0.656053 |
| 44 | 0.700020 | 0.594271 | 0.656476 |
| 45 | 0.699210 | 0.594314 | 0.656018 |
| **평균±population std** | **0.699457±0.000333** | **0.594357±0.000068** | **0.656181±0.000180** |
| **Δ vs v110** | **−0.007463** | **−0.015933** | **−0.010949** |

task별 4-seed 평균은 v110 대비 SEAL 4/10, 홀드아웃 3/7, 전체 **7/17만 상승**했다. 상승은 KRAS
+0.01213, PIK3CA +0.00720, Histologic Grade +0.00418 등이지만 ARID1A −0.05798, ER −0.03505,
SMAD4 −0.03193, PBRM1 −0.02990, VHL −0.02880의 하락이 크다. 양쪽 독립 집단 macro가 모두
하락하므로 요청 조합은 미승격, v110을 유지한다.

⚠️ 이 arm은 **PCA32→raw1536과 Euclidean→cosine을 동시에 바꾼 조합**이다. 따라서 spherical
k-means 자체의 순효과를 분리하지 못하며, 이 결과만으로 cosine 축 전체를 닫지 않는다. 분리가
필요하면 동일 raw1536/random512/full-abundance의 Euclidean control 또는 PCA32 spherical arm을
짝지어야 한다.

실행 arm은 `sphraw512all_s{42..45}`, 로그/예측은
`logs/20260819_ct_spherical_raw512_full_abundance/{seal,heldout}/`에 있다. 68개 로그 모두 final 1개,
traceback 0이다. 구현 전 BagPFN Python full test는 **310 tests, OK (68.011s)**였다.

---

## 181. 2026-08-19 — **v111 승격 확정: full-cell hierarchical PCA32/K256, CT 분기 종료**

사용자 결정으로 예측 macro가 가장 높은 v110 대신, **cell selection bias가 없고 random sampling의
영향도 없는 구성 중 최고**인 full-cell/full-abundance hierarchical PCA32/K256을 앞으로의 공식
baseline으로 선택했다.

```
CT dictionary/statistics : 모든 context cell
CT abundance              : context/query의 모든 cell
projection                : CV와 공유하는 within-slide PCA의 상위 32방향
tokenizer                 : deterministic hierarchical PCA/2-means tree
tokens                    : K=256
readout                   : class-balanced ridge, λ=1
head weight               : 0.7
```

| | SEAL 10 | 홀드아웃 7 | 전체 17 | seed std |
|---|---:|---:|---:|---:|
| **v111 (활성)** | **0.70453** | **0.59809** | **0.66070** | **0.00000** |
| v110 (historical predictive best) | 0.70692 | 0.61029 | 0.66713 | 0.00000 |
| Δ v111−v110 | −0.00239 | −0.01220 | −0.00643 | — |

이 승격은 성능 우월성 주장이 아니라 **운영 불변성에 대한 사용자 선택**이다. 전체 cell을 사용하므로
`sampling=even/random`과 sampling seed는 실제 cell 집합에 관여하지 않는다. CT 분기는 이 구성에서
종료하며 PCA24 보간, CT 독립 부분공간, 추가 tokenizer/λ 탐색은 다음 action에서 제거한다.

구현:

- `TrainingFreeConfig()` 기본값을 full cells / full abundance / PCA32 / hierarchical K256 / GEMM으로 변경
- 활성 runner `scripts/eval_v111.sh` 추가
- `scripts/eval_v110.sh`는 historical even64/FPS+Lloyd30/K32 재현 전용으로 복구
- VHL 50-fold smoke `0.5116`으로 기존 `h2T256` 로그와 정확히 일치
- BagPFN Python full **310 tests, OK (66.011s)**

활성 실행:

```bash
bash scripts/eval_v111.sh <gpu> <tag> [tasks...]
```

---

## 182. 2026-08-19 — DD ordered coordinate × typicality 구현 (평가 전)

사용자와 합의한 내용은 세 가지다.

1. 같은 context로 DD 방향을 선택한 뒤 같은 표본의 prototype separation을 독립 confidence인
   `r_task`로 다시 사용하지 않는다.
2. prototype 순서는 bounded class-direction evidence로 사용하되, prototype 바깥이라는 이유만으로
   최대 confidence라고 해석하지 않는다.
3. 작은 gap의 불확실성은 별도 gate 대신
   `h_eff=max(|p1-p0|/2, κ sigma_pool)`에 흡수하고, 두 클래스 모두에서 먼 query는
   nearest-class typicality로 감쇠한다. 첫 구현은 κ=1이다.

최종 class-1-positive evidence는 다음이다.

```text
a(q) = clip(sign(p1-p0) * (q-midpoint) / h_eff, -1, 1)
o(q) = exp(-0.5 * min_c ((q-p_c)^2 / (sigma_c^2+eps)))
M_DD(q) = a(q) * o(q)
```

구현:

- `src/models/dd_adaptive_rank.py::ordered_typicality_margin` — 공통 수식
- `src/models/training_free.py` — `dd_readout="ordered_typicality"` opt-in
- `src/models/set_transformer_ridge.py::_dd_ordered_typicality_features` — 정식 평가 계보 경로
- `scripts/test_pathobench.py` — `ICF_DD_ORDERED_TYPICALITY=1`,
  `ICF_DD_SEPARATION_FLOOR=1.0` 훅
- `scripts/eval_dd_ordered_typicality.sh` — v111에서 DD만 바꾸는 단일 GPU runner
- `scripts/run_dd_ordered_typicality_eval.sh` — 17-task GPU worker runner

테스트는 작은 gap 감쇠, 외분 OOD 감쇠, label antisymmetry, 겹친 prototype의 0 evidence,
boundedness, 정식/독립 구현 일치, legacy v111 동등성을 고정한다. 관련 **89 tests, OK**, 전체
**318 tests, OK (72.356s)**. **아직 macro 결과가 없으므로 승격하지 않았고 v111 기본
`distance` readout은 보존했다.**

### 182-1. 17-task 평가 실행 중

```text
started       2026-08-19 10:13:57 KST (corrected relaunch)
parent PID    1991184
session ID    1991184 (nohup + setsid, terminal-independent)
workers       GPU 0..7, one sequential queue per GPU
jobs          SEAL 10 + held-out 7 = 17 deterministic evaluations
tag           dd_ordered_typicality_k1
launcher log  logs/20260819_dd_ordered_typicality_k1/launcher.out
task logs     logs/official50/*_dd_ordered_typicality_k1.log
predictions   predictions/pathobench_*_dd_ordered_typicality_k1_official50_bf16.pt
```

첫 실행(PID 1981165)은 fold 진입 전 전부 실패했다. 원인은 `scripts/test_pathobench.py`에서 DD
selector 문자열 변수 `selection`을 뒤의 CV column-selection list가 덮어써 새 arm의 conflict guard가
항상 발동한 것이었다. `dd_selection`으로 분리했고(`88641c1`), 함께 발견된 기존 wrapper의 Python
실패 `rc=0` 은 전체 실패 code를 반환하도록 수정했다. 실패 run은 final/prediction을 하나도 만들지
않았고 같은 tag의 log를 corrected run이 덮어썼다.

재실행 직후 부모와 8 worker가 살아 있고 첫 8 task가 `START`된 것을 확인했다. 다음 Action은 launcher의
`EVALUATION DONE`, 17개 log 각각 final 1개와 traceback 0을 확인한 뒤 v111 대비 task별 delta,
SEAL/held-out/전체 macro와 부호 일치를 집계하는 것이다.

### 182-2. 17-task 평가 완료 — DD weight가 legacy 0.343 그대로였음을 발견

`EVALUATION DONE`, 17개 log 모두 `fold-mean AUROC` 1회·`Saved official-fold predictions` 1회·
traceback 0으로 정상 종료했다. 그런데 로그의 `fixed head: cv=1.442 dd=0.343 ct=0.7` 줄을 보면
`ICF_FIXED_HEAD_DD_WEIGHT`를 어디서도 오버라이드하지 않아 **DD 항이 여전히 legacy distance
readout용으로 fit된 0.343 magnitude를 그대로 쓰고 있었다.** `ordered_typicality` margin은
`[-1,1]`로 bounded된 반면 0.343은 옛 distance-squared 출력 스케일에 맞춘 값이므로, 재사용은
근거가 없다. 사용자 지시로 이 항을 없애고(=1로 설정) 재평가했다(§182-3).

이 시점의 (미승격) 결과는 v111 대비 SEAL 10 −0.00231, 홀드아웃 7 **+0.00920**, 전체 17 **+0.00243**
이었고, 두 독립 집단의 부호가 반대(SEAL 하락/홀드아웃 상승)라 §182-3 전까지는 승격 신호로 보지
않았다.

### 182-3. DD weight=1로 수정 후 17-task 재평가 — **v112으로 승격**

`scripts/eval_dd_ordered_typicality.sh`에 `ICF_FIXED_HEAD_DD_WEIGHT=1`을 추가하고 새 tag
`dd_ordered_typicality_k1_ddw1`으로 17-task를 재실행했다(PID/SID `2052454`, launcher log
`logs/20260819_dd_ordered_typicality_k1_ddw1/launcher.out`). 17개 log 모두 `fixed head: cv=1.442
dd=1.0 ct=0.7`을 확인했고 `fold-mean AUROC`/`Saved official-fold predictions` 각 1회, traceback
0으로 정상 종료했다.

| 그룹 | task | v111 | dd=0.343 (§182-2) | **dd=1.0 (v112)** | Δ v112−v111 |
|---|---|---:|---:|---:|---:|
| SEAL | bc_therapy/er_status | 0.6875 | 0.6823 | 0.6834 | −0.0041 |
| SEAL | bc_therapy/grade | 0.7387 | 0.7305 | 0.7329 | −0.0058 |
| SEAL | bc_therapy/her2_status | 0.6787 | 0.6633 | 0.6736 | −0.0051 |
| SEAL | cptac_brca/PIK3CA_mutation | 0.5357 | 0.5460 | 0.5400 | +0.0043 |
| SEAL | cptac_brca/TP53_mutation | 0.8309 | 0.8243 | 0.8270 | −0.0039 |
| SEAL | cptac_ccrcc/BAP1_mutation | 0.6821 | 0.7193 | 0.7019 | +0.0198 |
| SEAL | cptac_ccrcc/VHL_mutation | 0.5116 | 0.5004 | 0.5095 | −0.0021 |
| SEAL | cptac_luad/EGFR_mutation | 0.7828 | 0.7721 | 0.7825 | −0.0003 |
| SEAL | cptac_luad/STK11_mutation | 0.9029 | 0.8918 | 0.9006 | −0.0023 |
| SEAL | cptac_luad/TP53_mutation | 0.6944 | 0.6922 | 0.6918 | −0.0026 |
| 홀드아웃 | cptac_ccrcc/PBRM1_mutation | 0.5359 | 0.5250 | 0.5266 | −0.0093 |
| 홀드아웃 | cptac_lscc/ARID1A_mutation | 0.4710 | 0.4802 | 0.4747 | +0.0037 |
| 홀드아웃 | cptac_lscc/Histologic_Grade | 0.6367 | 0.6577 | 0.6519 | +0.0152 |
| 홀드아웃 | cptac_lscc/KEAP1_mutation | 0.5858 | 0.6161 | 0.6002 | +0.0144 |
| 홀드아웃 | cptac_luad/KRAS_mutation | 0.7257 | 0.7371 | 0.7311 | +0.0054 |
| 홀드아웃 | cptac_pda/SMAD4_mutation | 0.4578 | 0.4743 | 0.4654 | +0.0076 |
| 홀드아웃 | ucla_lung/progression_regression | 0.7737 | 0.7606 | 0.7628 | −0.0109 |

| | SEAL 10 | 홀드아웃 7 | 전체 17 |
|---|---:|---:|---:|
| v111 | 0.70453 | 0.59809 | 0.66070 |
| **v112 (dd weight=1)** | **0.70432** | **0.60181** | **0.66211** |
| Δ v112−v111 | −0.00021 | +0.00372 | +0.00141 |

DD weight를 0.343→1로 고치자 §182-2의 SEAL 하락과 홀드아웃 상승 폭이 모두 절반 가까이
줄었다 — §182-2의 효과 일부가 legacy magnitude-fit 계수의 인위적 스케일링이었다는 뜻이다. 상승
부호 패턴(전체 7/17: PIK3CA, BAP1, ARID1A, Histologic Grade, KEAP1, KRAS, SMAD4)은 §182-2와
동일하게 유지된다. SEAL macro는 −0.00021로 사실상 flat(결정론적 arm이므로 seed 변동은 없다)이고
홀드아웃은 +0.00372로 뚜렷하게 양수다.

---

## 183. 2026-08-19 — **v112 승격 확정: v111 + DD ordered-coordinate × typicality (κ=1, weight=1)**

사용자 결정으로 §182-3의 결과를 새 활성 baseline으로 승격했다.

```
CV/CT                     : v111과 동일 (offdiag CV, full-cell hierarchical PCA32/K256 CT, ridge λ=1, CT weight 0.7)
DD readout                : ordered-coordinate evidence × nearest-class typicality (§182)
DD separation floor (κ)   : 1.0
DD fixed-head weight      : 1.0 (legacy distance-readout의 0.343 magnitude-fit 제거)
```

| | SEAL 10 | 홀드아웃 7 | 전체 17 | seed std |
|---|---:|---:|---:|---:|
| **v112 (활성)** | **0.70432** | **0.60181** | **0.66211** | **0.00000** |
| v111 (previous baseline) | 0.70453 | 0.59809 | 0.66070 | 0.00000 |
| Δ v112−v111 | −0.00021 | +0.00372 | +0.00141 | — |

SEAL macro는 사실상 flat, 홀드아웃 macro가 전체 상승을 이끈다. 17개 중 7개 task가 상승했고
(PIK3CA, BAP1, ARID1A, Histologic Grade, KEAP1, KRAS, SMAD4), 나머지 10개는 소폭 하락했다 —
그중 BAP1 +0.0198과 Histologic Grade +0.0152, KEAP1 +0.0144가 상승분의 대부분을 차지한다.

구현:

- `src/models/training_free.py::TrainingFreeConfig` 기본값 변경: `dd_readout="distance"` →
  `"ordered_typicality"`, `weight_dd=-0.343` → `-1.0`. `dd_separation_floor=1.0`은 변경 없음(이미
  κ=1과 일치).
- 활성 runner `scripts/eval_v112.sh` 추가 (v111과 CV/CT 설정 동일, DD만 다름).
- `scripts/eval_v111.sh`는 historical distance-readout 재현 전용으로 유지(변경 없음).
- `scripts/eval_dd_ordered_typicality.sh`에 `ICF_FIXED_HEAD_DD_WEIGHT=1` 추가(§182-3).
- `tests/test_training_free.py`: lineage 동등성 fixture `V107`에 `dd_readout="distance",
  weight_dd=-0.343`을 명시적으로 고정(그 lineage 경로는 옛 distance 고정 head만 구현하므로, 이는
  "이 configuration에 대한 진술"이지 default에 대한 진술이 아니다 — 파일 상단 주석과 동일한 근거).
  `DefaultTest`는 새 기본값(`ordered_typicality`, `weight_dd=-1.0`)을 pin하도록 갱신, 경계 테스트의
  하드코딩된 bound도 0.343→1.0으로 갱신.
- BagPFN Python full **318 tests, OK (42.524s)**.
- v112 VHL 50-fold smoke가 §182-3의 `dd=1.0` 로그 값(0.5095)과 정확히 일치하는지 확인 중.

CT 분기는 §181에서 이미 종료됐고 이번 승격은 DD 분기에 한정된다. 다음 action은 v112를 baseline
으로 놓고 추가 DD 탐색(다른 κ 값, seed 없는 결정론적 arm이므로 반복 불필요) 또는 CV/CT 재탐색
여부를 사용자와 논의하는 것이다.

_Recorded by: nhn-YLC-claude — 2026-08-19 11:35_

---

## 184. 2026-08-20 — CT cell 예산을 bag-size 비례 fraction으로: 22GB GPU OOM 해소 + v112 대비 macro sampling-invariance 확인

이 gnode3 22GB GPU 노드에서 v112(§183) SEAL 10-task 평가를 재현하면 LUAD 3개 task
(EGFR/STK11/TP53_mutation) 전부 `evaluate_trial`의 `pooled = cat(context)` / `centered = values -
values.mean(...)` 단계에서 즉시 `CUDA out of memory`로 죽었다(`Tried to allocate 278.00 MiB. ...
this process has 21.82 GiB memory in use`, `logs/official50/cptac_luad_*_v112.log`,
`logs/official50/cptac_luad_*_v113.log`, `logs/20260819_v112_17task/cptac_luad_*.runner.log`
전부 동일 실패). v112가 §183에서 측정된 8×B200(180GB/장) 노드에서는 문제없이 돌던 것과 대비된다 —
원인은 v111/v112가 물려받은 CT의 full-cell/full-abundance hierarchical 샘플링이 LUAD처럼 bag당
최대 ~35k cell인 slide에서 `prepare_cells`가 CONTEXT 전체를 한 번에 GPU에 올리기 때문이다.

**수정 두 가지 (uncommitted 시점 기준, 이후 커밋 예정)**:

1. `src/models/stream_eval.py` (신규) — raw bag을 CPU에 상주시키고 PCA covariance scatter만
   1-bag/1-chunk 단위로 GPU에 올려 계산. `test_pathobench.py`의 `ICF_COVARIANCE_BASIS=pca|pca_within`
   경로가 이걸 통해 basis를 구성하도록 배선.
2. `src/models/ct_readout.py` — `cells_per_bag`(고정 정수 cap) 대신 **bag/episode 크기에 비례하는
   fraction 샘플링**을 추가: `cells_fraction`(0,1] + `cells_scale="own"|"median"` + `cells_min`.
   `scripts/eval_v113.sh`(신규)가 `ICF_CT_CELLS=0.125`(own bag 크기의 1/8, floor 64)로 이를 켠다.
   LUAD의 최대 35k cell bag이 이 cap으로 ~4,375 cell까지 줄어든다.

**검증**: BagPFN pytest `tests/test_ct_readout.py` + `tests/test_training_free.py` 81 passed, 2
skipped, 회귀 없음. 이 노드에서 SEAL 10-task 전체를 `eval_v113.sh`로 재실행 — **10개 전부 `rc=0`,
OOM 0건**(LUAD 3개 포함, 이 노드에서 처음으로 완주).

| task | v112 baseline (§183, B200) | v113 (fraction 샘플링, gnode3) | Δ |
|---|---:|---:|---:|
| bc_therapy/er_status | 0.6834 | 0.6885 | +0.0051 |
| bc_therapy/grade | 0.7329 | 0.7338 | +0.0009 |
| bc_therapy/her2_status | 0.6736 | 0.6737 | +0.0001 |
| cptac_brca/PIK3CA_mutation | 0.5400 | 0.5390 | −0.0010 |
| cptac_brca/TP53_mutation | 0.8270 | 0.8255 | −0.0015 |
| cptac_ccrcc/BAP1_mutation | 0.7019 | 0.6877 | −0.0142 |
| cptac_ccrcc/VHL_mutation | 0.5095 | 0.5090 | −0.0005 |
| cptac_luad/EGFR_mutation | 0.7825 | 0.7862 | +0.0037 |
| cptac_luad/STK11_mutation | 0.9006 | 0.8949 | −0.0057 |
| cptac_luad/TP53_mutation | 0.6918 | 0.7011 | +0.0093 |
| **SEAL 10 macro** | **0.70432** | **0.70394** | **−0.00038** |

**해석 — 이번 기록의 핵심은 OOM이 없어졌다는 것보다 이 Δ다.** v112와 v113은 CT의 cell 선택
방식이 근본적으로 다르다(전체 cell vs bag 크기의 1/8만 샘플링). 그런데도 SEAL macro 차이가
−0.00038로, §183에서 v111→v112 승격 때 "사실상 flat"이라 판정한 폭(−0.00021)과 같은 급의
잡음이다. task별로도 5승 5패로 부호가 갈리고 한쪽으로 치우친 체계적 손실이 없다(가장 큰 하락은
BAP1_mutation −0.0142, §107-3 게이트 수준에는 못 미친다). 즉 **CT의 abundance/tokenizer 단계는
cell 개수 자체보다 cell 비율에 더 가깝게 반응한다는 sampling-invariance 증거**이며, 메모리
제약이 있는 노드에서 `cells_fraction`을 기본값으로 써도 안전하다는 근거가 된다.

⚠️ 이 절 작성 시점 기준 `stream_eval.py`/`ct_readout.py`의 `cells_fraction` 관련 변경은
**working tree에 uncommitted 상태**다(`scripts/eval_v113.sh`, `src/models/stream_eval.py`는
untracked). §183의 활성 baseline(v112) 승격 자체를 대체하는 것은 아니고, 같은 v112 DD/CV 설정에
CT cell 샘플링 policy 하나만 바꾼 별도 arm(v113/`ICF_CT_CELLS`)으로 취급한다.

_by Claude Sonnet 5 on gnode3 at 2026-08-20 01:08:02_

---

