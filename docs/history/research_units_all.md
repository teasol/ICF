# 전체 커밋(000–449) 전수 연구 단위 감사 및 이력 총람 (Research Units 01–78)

> **역사 기록의 적용 범위**: 아래 과거 판정·인용은 당시 기록이다. 현행 승격·정밀도는
> [`PROJECT.md`](../PROJECT.md), 경계는 [`closed_axes.md`](../closed_axes.md)를 따른다.
> 옛 컷오프·통계 금지·경계 미확정의 정정은 [`decisions.md`](../decisions.md)의 `D-021`·`D-022`에 있다.

> **감사 기준선 및 개요**  
> - **고정 HEAD**: `f3f0b201f85d2bb7eafe1e63561e2bbda2b277c1` (Index 449)  
> - **시작 커밋**: `deef1f88e650f2ea0579f6897dcfe5e33c61fd8f` (Index 0)  
> - **대상 범위**: 450개 커밋 (`first-parent` 순서, 0~449 전수 검증 완료)  
> - **전체 연구 기간**: 2026-07-22 ~ 2026-09-05 (46일간)  
> - **식별된 연구 단위**: 총 78개 (`RU-01` ~ `RU-78`, 6대 연구 시대 구분)  
> - **원문 인용 검증**: 총 113건 (커밋 시점 파일 라인 및 본문 100% 비트 매칭 확인)  
> - **작성 일자**: 2026-09-05 (KST)  

---

## 1. 개요 및 6대 연구 시대 구조화

본 감사는 ICF(In-Context Foundation / BagPFN Single-Cell) 프로젝트 저장소의 최초 커밋(`deef1f88`, 2026-07-22)부터 최신 커밋(`f3f0b201`, 2026-09-05)까지 **전체 450개 커밋 전수를 단 하나의 누락이나 중복 없이 순회**하여, 개발 및 연구 가설의 생성·검증·기각·승격 과정을 구조화한 연구 자산 총람입니다.

단순 코드 변경 내역의 나열을 지양하고, **'동일한 연구 질문(Research Question)과 가설(Hypothesis)에 답하기 위한 일련의 작업인가'**를 기준으로 연구 단위를 결합하여 총 78개의 핵심 연구 단위(Research Units, RU)를 도출했습니다. 또한 모든 연구 단위에 **정확한 시작·종료 일자 및 일자 범위(Date Range)**를 파싱하여 배치했습니다.

### 6대 연구 시대(Eras) 요약표

| 시대 (Era) | 커밋 범위 | 기간 | 연구 단위 | 핵심 패러다임 및 주요 전환 |
|:---|:---:|:---:|:---:|:---|
| **Era 1: 합성 사다리 & 검색 계층** | 000~036 | 2026-07-22 ~ 2026-07-29 | RU-01 ~ RU-07 | v18 Learnability 사다리 통과 후 v19 공분산 불변 구조 수립. Candidate A/B 경쟁 및 v21 외부 검색(Retrieval) 계층 탐색. |
| **Era 2: 붕괴 진단 & 크기 충실성** | 037~150 | 2026-07-29 ~ 2026-08-04 | RU-08 ~ RU-17 | v22 검색 계층 영구 제거. Bag-Collapse 가족 진단 및 Hard 상태 접근성 감사. Musk 전이 실패 분석을 통해 v30 (B1 `poolz_l2` + B2 Cardinality) 상호필수 확정. |
| **Era 3: 고해상도 확장 & 병목 계측** | 151~249 | 2026-08-04 ~ 2026-08-09 | RU-18 ~ RU-24 | v31 CCER 교차 인코더와 v33 다중 해상도 가설 검증 및 기각. v34 효율적 슬롯-MLA baseline 확립. PathoBench 30종 CSV 감사. |
| **Era 4: §68 CV-only & 합성 기저** | 250~298 | 2026-08-09 ~ 2026-08-12 | RU-25 ~ RU-30 | §68 분기 기여도 진단으로 비공분산 학습 분기 전면 폐기. 계보 B(v52~v54) 기각. CV-only 및 Hard 직교 잠재공간 v77 확립. |
| **Era 5: 통계 규범 & 무학습 baseline** | 299~349 | 2026-08-12 ~ 2026-08-17 | RU-31 ~ RU-40 | Cell 축 종결. 과잉 매개변수화 학습 모델 대비 통계적 닫힌 형태 우위 확인. v106(Within PCA) 및 v107(K=256) Training-Free baseline 수립. |
| **Era 6: 5-Branch & 형상 브랜치** | 350~449 | 2026-08-17 ~ 2026-09-05 | RU-41 ~ RU-78 | v108~v121 고속 베이스라인 진화. 앙상블 집계 규범(Trimmed-Mean) 수립. 2단계 다면 게이트(직교성·정보량) 하의 형상(SH/SHJ) 브랜치 체계 완성. |

---

## 2. 전체 연구 단위 총괄 일람표 (RU-01 ~ RU-78)

| ID | 일자 (Date) | 커밋 범위 | 작업 유형 | 제목 | 핵심 결정 | 후속 정정 / 연결 |
|:---|:---:|:---:|:---|:---|:---|:---|
| **RU-01** | 2026-07-22 ~ 2026-07-23 | 0–2 | 초기 기준선 수립 | v18 Learnability Ladder 및 초기 기준선 동결 | v18 체크포인트 호환성 버전을 18로 동결하고, translatio.. | 프로젝트의 시작 기준선이며, global shift 민감도 .. |
| **RU-02** | 2026-07-24 ~ 2026-07-25 | 3–4 | 가설 검증 및 승격 | v19 Shift-Invariant Covariance 아키텍처 수립 | v19를 baseline으로 채택하고 저차원 투영 head 분류기 후.. | RU-01(v18)의 한계를 극복함. 후속 RU-03의 Ca.. |
| **RU-03** | 2026-07-26 | 5–9 | 승격 채택 | Candidate A/B 20-Epoch 단기 학습 비교 및 Candidate B 선정 | Candidate B(learned_head)를 v19의 최종 rel.. | RU-02(v19)의 헤드 구조를 확정함. RU-04(v21.. |
| **RU-04** | 2026-07-28 | 10–20 | 가설 검증 | v21 Signal-Aware Retrieval 아키텍처 및 설정 스케일링 | 런처 결함을 수정하고 대규모 Phase 5 사전학습을 가동했다. | RU-03 이후 대형 context 확장을 시도한 단계. 후.. |
| **RU-05** | 2026-07-28 | 21–32 | 가설 검증 | Phase 5 대형 Context 사전학습 및 런처 안정화 | Phase 5 체크포인트를 확보하고 다운스트림 전이 검증(Phase .. | RU-04의 학습 완주 단계. 후속 RU-06(ICI 전이).. |
| **RU-06** | 2026-07-28 | 33–34 | 가설 검증 및 기각 | Phase 6 ICI 5-Fold 파인튜닝 역효과 규명 | 사전학습 가중치의 다운스트림 유효성 가설을 기각하고, 모델 내부 re.. | RU-05의 성과를 정면 반증함. RU-07(retrieva.. |
| **RU-07** | 2026-07-28 ~ 2026-07-29 | 35–36 | 가설 검증 및 기각 | 3D Internal Signal-Aware Retrieval 검증 및 3대 가설 반증 | Retrieval 3대 가설을 전면 반증으로 판정하고 retrieva.. | RU-04~06의 retrieval 시대를 종식시킴. RU-.. |
| **RU-08** | 2026-07-29 | 37–41 | 아키텍처 정리 및 정책 수립 | v22 Retrieval 계층 전면 제거 및 통계 보고 프로토콜 확립 | v22를 활성 baseline으로 승격하고, 합성 데이터 중심으로 개.. | RU-07의 결론을 코드베이스에 반영함. RU-09(Chol.. |
| **RU-09** | 2026-07-29 | 42–44 | 기술 구현 및 버그 수정 | v22 Cholesky Backward 수치 안정화 및 재베이스라인 | 수정된 코드로 v22 baseline을 확정하고 공분산 분기의 잠재력.. | RU-08의 기술 부채를 해소함. RU-10(Tier 1 공.. |
| **RU-10** | 2026-07-30 | 45–50 | 가설 검증 및 진단 | Covariance Branch 성능 상한선 및 Tier 1 세포 선택 진단 | Tier 1(covariance) 진단을 통과 판정하고 세포 선택을 .. | RU-09 이후 공분산의 기전을 규명함. RU-11/RU-1.. |
| **RU-11** | 2026-07-30 ~ 2026-07-31 | 51–60 | 가설 검증 | T2/T3/T4 다면 진단 및 Hard 상태 접근성 감사 | 슬롯 세분화보다 bag 전체를 단일 토큰으로 collapse하는 v2.. | RU-10의 진단을 심화함. RU-12/13(v24 bag .. |
| **RU-12** | 2026-07-31 | 61–88 | 가설 검증 | 합성 Context 데이터 생성 및 커리큘럼 파인튜닝 | 합성 데이터 생성 규칙을 통제하고 모델 구조 개혁에 집중하기로 결정했다. | RU-11의 데이터 측면 탐색. RU-13의 bag proj.. |
| **RU-13** | 2026-07-31 ~ 2026-08-02 | 89–106 | 가설 검증 및 승격 | v24 Learned Bag Projection 및 Bag-Collapse 패밀리 탐색 | v24를 활성 후보로 채택하고 typed-bag(v25)으로 확장을 .. | RU-10의 세포 선택 병목에 대한 아키텍처적 해법. 후속 .. |
| **RU-14** | 2026-08-02 ~ 2026-08-03 | 107–117 | 가설 검증 및 기각 | v25 Typed-Bag 다면 평가 및 일반화 실패 분석 | v25를 공식 기각하고 `v25-typed-bag-final` 태그로.. | RU-13의 확장이 실패한 지점. RU-15(v24 기전 규.. |
| **RU-15** | 2026-08-03 | 118–124 | 가설 검증 및 진단 | v24 절제 실험(no-L2) 및 0.70 Plateau 생성기 결함 규명 | 모델 변경을 중단하고 생성기 정보 손실을 잡기 위한 Musk 0.95.. | RU-13/14의 성능 정체 원인을 규명함. 후속 RU-16.. |
| **RU-16** | 2026-08-03 ~ 2026-08-04 | 125–139 | 가설 검증 및 기각 | Rawstats 사전학습 및 Musk 0.95 로드맵 P0~P3 기각 | 복잡한 외부 제안서를 기각하고 핵심 정규화 조합(v30)으로 단순화하.. | RU-15의 생성기 결함 처방 시도. RU-17(v30 확정.. |
| **RU-17** | 2026-08-04 | 140–150 | 정책 개정 및 승격 채택 | v26~v29 제안서 전면 기각 및 v30 (B1+B2) 상호필수 확정 | v26~v29 제안서를 공식 폐기하고 v30(B1+B2)을 활성 베이.. | 프로젝트의 첫 대규모 proposal 기각 정리. 후속 RU.. |
| **RU-18** | 2026-08-04 ~ 2026-08-05 | 151–165 | 가설 검증 및 기각 | v31 CCTS / CCER 후보 평가 및 상보성 검증 | CCER을 독립 모델로 유지하지 않고 residual 보정인 DR-C.. | RU-17(v30) 이후 상보 분기 탐색. RU-19(v32.. |
| **RU-19** | 2026-08-05 | 166–174 | 가설 검증 및 기각 | v32 / v32b DR-CCER 평가 및 비상보성 기각 | DR-CCER 및 CCER 계보 전체를 공식 기각하고 코드를 동결했다. | RU-18의 CCER 계보 종식. RU-20(v33 mult.. |
| **RU-20** | 2026-08-05 ~ 2026-08-06 | 175–191 | 가설 검증 | v33 MR-BagPFN Phase 0 (Arm B/C 8x A6000 DDP) | v33 멀티 해상도 확장을 보류하고 v34 large-context .. | 대규모 분산 훈련을 통한 해상도 축 검증. RU-21/23(.. |
| **RU-21** | 2026-08-06 ~ 2026-08-07 | 192–209 | 인프라 정비 | 오퍼레이션 프로파일링, 학습 예산 감사 및 PathoBench 동기화 | 불필요한 연산 분기를 제거하기 위한 준비 단계로 채택했다. | RU-20 이후의 인프라 효율화 작업. RU-22/23으로 .. |
| **RU-22** | 2026-08-07 | 210–222 | 벤치마크 정비 | PathoBench CSV 30종 전수 대조 및 레거시 문서 정리 | 평가 벤치마크 정본을 확정하고 오래된 실험 문서를 정리했다. | 벤치마크 데이터 신뢰성을 확립한 단계. RU-23(v34 확.. |
| **RU-23** | 2026-08-07 ~ 2026-08-08 | 223–237 | 승격 채택 및 정리 | CCER 분기 제거, 쿼리 불변 마진 및 v34 Large-Context 승격 | v34 large-context를 PathoBench 공식 보고용 활.. | RU-18~22의 성과를 집대성한 정식 베이스라인. RU-2.. |
| **RU-24** | 2026-08-08 ~ 2026-08-09 | 238–249 | 가설 검증 및 기각 | v35~v37 제안서(Q1 Structured Population) 평가 및 CV-only 사전 준비 | 죽은 모듈을 수리하려던 v35~v37 시도를 전면 기각하고, 살아있는.. | RU-23 이후 가장 결정적인 분기 진단. RU-25(§68.. |
| **RU-25** | 2026-08-09 ~ 2026-08-10 | 250–261 | 아키텍처 대전환 | §68 CV-Only 전환: 죽은 4개 분기 제거 및 연산 스킵 | 프로젝트 역사상 가장 거대한 아키텍처 대전환 단행: 4개 분기를 영구.. | RU-24의 진단에 따른 결단. ICF 프로젝트를 군더더기 .. |
| **RU-26** | 2026-08-10 | 262–266 | 가설 검증 및 기각 | 계보 B (학습 Bag 토큰 + Closed-Form Ridge) 기각 확정 | 계보 B를 공식 기각하고 다시는 열지 않을 결론으로 history에 .. | RU-25 이후 제기된 대안 아키텍처의 기각. RU-27(합.. |
| **RU-27** | 2026-08-10 | 267–274 | 가설 검증 | 합성 데이터 축 개편 (Per-Bag Cardinality, Factorized XOR, v61) | 선형 매니폴드 arm을 베이스라인에 편입하고 하이브리드(v62) 탐색.. | RU-25/26 이후 데이터 생성기의 질적 개선 단계. RU.. |
| **RU-28** | 2026-08-10 | 275–279 | 가설 검증 및 기각 | v62 Linear CV1 Transformer 하이브리드 시험 | 하이브리드 확장을 중단하고 순수 공분산의 통계적 정규화(v70 계열).. | RU-27의 하이브리드 모델 시험. 실패 후 RU-29(Ca.. |
| **RU-29** | 2026-08-11 ~ 2026-08-12 | 280–284 | 승격 채택 | Canonical CV Raw Mean 통합 및 v70~v74 CT 베이스라인 수립 | v74 CT 베이스라인을 공식 승격하고 난이도 스윕(v77)으로 진입.. | 현대 3-branch(CV, CT, DD)의 시초. 후속 R.. |
| **RU-30** | 2026-08-12 | 285–298 | 승격 채택 | 잠재 차원 스윕, 비선형 매니폴드 뱅크 및 Hard v77 승격 | 사용자 결정으로 Hard v77을 프로젝트의 새로운 활성 공식 베이스.. | 학습 기반 시대(v18~v77)의 정점. 후속 RU-31(판.. |
| **RU-31** | 2026-08-12 | 299–300 | 정책 수립 및 가설 검증 | §99 Fold-Paired Delta + Bootstrap CI 프로토콜 및 v78 DD 2차형식 | §99 fold-paired delta 판정 프로토콜을 필수 불변식으.. | 통계적 판정 규범의 중대한 진보. RU-32(v78 기각)의.. |
| **RU-32** | 2026-08-12 | 301–305 | 기각 및 가설 검증 | History 아카이브 통합, v78 무가중 기각 및 v79 Dual Projection | v78과 v79를 모두 기각하고 v77 epoch 49 동결선으로 복.. | 배선 중심의 파라미터 학습 한계 확인. RU-33(과거 판정.. |
| **RU-33** | 2026-08-12 ~ 2026-08-13 | 306–309 | 벤치마크 정정 | v77 Epoch 49 베이스라인 고정, v80 기각 및 27개 Arm 재채점 | v77 baseline을 0.6880으로 확정하고 Medium 대 H.. | 벤치마크의 객관적 재정렬. 후속 RU-34(v82 Mediu.. |
| **RU-34** | 2026-08-13 | 310–315 | 승격 채택 및 기각 | v82 Medium 승격 및 v83 Linear Head vs v84 Deep Head 판정 | 사용자 결정으로 v82에 이어 v83 linear head를 활성 베.. | 헤드 단순화의 결정적 승리. 후속 RU-35/36(세포 축 .. |
| **RU-35** | 2026-08-13 | 316–317 | 가설 검증 및 기각 | v86 노이즈 레버 및 v87 희소 abundance 레버 Null Result | 데이터 생성기 미세 튜닝 축을 닫고 에피소드 형상 탐색(v88~v93.. | 합성 생성기 레버의 무력화 입증. RU-36(에피소드 형상 .. |
| **RU-36** | 2026-08-14 | 318–325 | 가설 검증 및 축 종료 | v88 PA 기각(§114) 및 v89~v93 에피소드 형상·세포 축 소진 | 에피소드 형상 축을 영구 종료하고 합성 cell과 실제 cell의 분.. | 에피소드 형상 가설의 최종 종식. RU-37(실제 통계 불일.. |
| **RU-37** | 2026-08-14 ~ 2026-08-15 | 326–333 | 가설 검증 및 승격 | §123~§131 실제 UNI2 통계 불일치, Nuisance 가설 반증 및 v98 시드 앙상블 | 합성 데이터 매칭 축을 완전 종결하고, baseline을 v98 8-.. | 분산이 편향을 압도한다는 대발견. 후속 RU-38(학습 파라.. |
| **RU-38** | 2026-08-15 ~ 2026-08-16 | 334–340 | 아키텍처 대전환 | §132~§141 학습 파라미터 P 제거, 고정 Head 및 v106 Training-Free 수립 | 사용자 결정으로 학습 모델 시대를 공식 종료하고 **v106 (학습 .. | ICF 프로젝트의 가장 위대한 패러다임 전환. 학습 비용 0.. |
| **RU-39** | 2026-08-16 ~ 2026-08-17 | 341–344 | 승격 채택 | §142~§145 K=256 스윕, v107 승격 및 K 이득 CV 귀속 | 사용자 결정으로 v107(K=256)을 활성 베이스라인으로 확정하고 .. | v106의 첫 매개변수 최적화. RU-40(DD 및 CT 병.. |
| **RU-40** | 2026-08-17 | 345–349 | 가설 검증 및 기각 | §146~§149 적응적 Rank DD 기각 및 CT 2-Token 병목 진단 | DD 탐색 축을 공식 닫고, CT 거리 집중을 해소할 PCA 결합 가.. | 전반부 350커밋 역사의 마침표. 곧바로 후반부 RU-41(.. |
| **RU-41** | 2026-08-17 | 350–351 | 시작 경계 밖에서 착수된 연구 | CT 부분공간과 ridge readout의 결합 | raw 조건에 한정됐던 readout 결론을 정정하고 PCA32+ri.. | E05의 token 생성 개선 이전의 CT 기준선이며 CV와.. |
| **RU-42** | 2026-08-17 | 352–352 | 가설 검증 | DD의 가중치와 코호트 의존성 | 전역 weight는 고르지 않고 0.343을 유지했다. context.. | RU-43~E09의 DD readout 탐색으로 이어진다. |
| **RU-43** | 2026-08-17 ~ 2026-08-18 | 353–354 | 가설 검증 | DD LLR 보정과 상대 거리의 판정 가능성 | LLR 훅은 보정·pooled 평가용으로만 남기고 승격하지 않았으며 .. | E02의 DD 문제를 전역 가중 대신 margin 정의로 해.. |
| **RU-44** | 2026-08-18 | 355–361 | 가설 검증 | CV descriptor의 정보 분해 | v109에서 off-diagonal CV만 유지하고 √λᵢλⱼ가 주는.. | E05의 v109 승격과 E01의 CV-CT 공유 기저 제약.. |
| **RU-45** | 2026-08-18 | 356–357 | 승격 결정 | CT token 생성 개선과 v109 | CV off-diagonal과 k-means CT weight 0.7.. | E04의 CV 정리와 E06의 cells/tokens 격자로.. |
| **RU-46** | 2026-08-18 | 358–362 | 승격 결정 | CT 표본 수·token 수·ridge λ의 분리 | cells_per_bag=64와 CT tokens=32를 v110으로.. | E05를 확정하고 E08의 full-cell tokenize.. |
| **RU-47** | 2026-08-18 | 363–363 | 인프라 기록 | 서버 이동용 실행 환경 정비 | 후속 CT/DD 실험은 중앙 node 설정과 갱신된 handoff를 .. | E08의 CT 평가와 E09의 GPU worker 실행의 운.. |
| **RU-48** | 2026-08-18 ~ 2026-08-19 | 364–374 | 탐색 및 운영 선택 | 대체 CT dictionary와 full-cell hierarchical v111 | 사용자 결정으로 v111을 공식 baseline으로 채택하고 CT 분.. | E06의 v110은 historical predictive .. |
| **RU-49** | 2026-08-19 | 375–380 | 가설 검증 및 승격 채택 | DD ordered×typicality 후보의 구현·수정 및 v112 승격 | 사용자 결정으로 v112를 활성 baseline으로 승격하고 기존 d.. | E03의 LLR/상대거리 기각 후속 대안이며, M02의 CT.. |
| **RU-50** | 2026-08-20 | 381–381 | feasibility_decision | CT fraction sampling feasibility | 예측 개선이 아니라 평가 가능성을 이유로 v113을 승격했다. | RU-49, RU-51 |
| **RU-51** | 2026-08-20 | 382–383 | controlled_evaluation | unit fixed-head weight 통일 | 사용자 결정으로 v114를 승격했다. Primary 7·hold-ou.. | RU-50, RU-55 |
| **RU-52** | 2026-08-20 | 384–384 | rejection | CT kernel ridge와 top-k pooling 기각 | 세 변형을 모두 기각하고 재현 코드만 보존했다. | RU-51 |
| **RU-53** | 2026-08-21 | 385–394 | research_infrastructure | 모듈 분해와 BM 구현·문서 SSOT | M05 이후 BM 성능 비교는 이 구현·문서 기반을 전제로 한다. | RU-55, RU-61 |
| **RU-54** | 2026-08-21 | 395–395 | evaluation_protocol | Primary 7과 SEAL hold-out 역할 전환 | 후속 branch 평가는 Primary 7로 선택하고 SEAL 10은.. | RU-53, RU-55 |
| **RU-55** | 2026-08-22 | 396–396 | branch_promotion | Projected Bag-Mean(BM) 채택 | 사용자 승인으로 BM 포함 v115를 승격했다. | RU-53, RU-56 |
| **RU-56** | 2026-08-22 | 397–397 | branch_promotion | Bag-dispersion spectral entropy(BD) 채택 | 당시 사용자 승인으로 BD 포함 v116으로 승격했다. | RU-55, RU-57 |
| **RU-57** | 2026-08-22 | 398–400 | aggregation_promotion | DD 제거와 soft-vote v118 | v118을 활성으로 승격하고 SEAL/지도학습 표는 hold-out .. | RU-56, RU-58 |
| **RU-58** | 2026-08-22 | 401–402 | branch_and_aggregation_promotion | QA branch와 trimmed-mean v119 | CV를 유지한 QA 5-branch trimmed-mean v119를.. | RU-57, RU-60, RU-61 |
| **RU-59** | 2026-08-23 | 403–403 | branch_promotion | DS salience-denoising과 v120 | 사용자 결정으로 DS 포함 v120을 당시 baseline으로 승격했다. | RU-58, RU-60 |
| **RU-60** | 2026-08-23 ~ 2026-08-24 | 404–406 | postmortem_and_benchmark | v120 사후 실패 축과 CT 단독 | 네 축을 기각하고 CT는 단독 기준으로 보되 앙상블 대체로 판정하지 .. | RU-58, RU-61 |
| **RU-61** | 2026-08-30 | 407–412 | reproducibility | v120 harness·arm 모듈화와 결정성 경계 | tolerance 통과는 engineering check로만 인정하고.. | RU-58, RU-62 |
| **RU-62** | 2026-09-02 | 413–414 | configuration_cleanup | obsolete harness 제거와 학습 config 아카이브 | 학습 계보는 archive/git history에 보존하고 활성 v1.. | RU-61 |
| **RU-63** | 2026-09-02 ~ 2026-09-03 | 415–425 | infrastructure | 재현 기반 정비 | 성능 승격이 아닌 이후 비교의 재현 기반으로 채택했다. | RU-64 이후 비교의 기반 |
| **RU-64** | 2026-09-03 | 426–426 | benchmark/correction | §209 CT 전사 오류와 단독 기준선 | CT를 단독 챔피언으로 취급하지 않고 비용을 기준 설정에 반영했다. | RU-68 CT-제외 공식 비교 |
| **RU-65** | 2026-09-03 | 427–427 | experiment | §210 sub-bag/TTA의 과제별 상충 | 국소 변이를 보존한다는 anchor 가설을 이월했다. | RU-68/RU-76/RU-77 anchor 계보 |
| **RU-66** | 2026-09-03 | 428–428 | experiment/diagnosis | §211 LOO 차원 편향 | 서로 다른 차원의 raw LOO를 신뢰도 비교에 쓰지 않는다. | RU-67/RU-74 LOO 폐기 |
| **RU-67** | 2026-09-03 | 429–429 | experiment/decision | §212 깨끗한 Context LOO 폐기 | Context LOO를 영구 폐기하고 trimmed mean을 유지했다. | R12가 용량 보정 뒤에도 확정 |
| **RU-68** | 2026-09-03 | 430–430 | benchmark/experiment | §213 v121 기준선과 anchor 초기 측정 | .6171 5-branch를 공식 비교 기준으로 삼고 anchor 판.. | RU-76 정정, RU-77 종료 철회 |
| **RU-69** | 2026-09-03 | 431–432 | experiment/correction | §214–214-V 집계 탐색 정정 | trimmed mean 유지, 집계 축 종료, 보고 무결성 규칙 추가. | R08부터 구조/게이트 우선 |
| **RU-70** | 2026-09-03 ~ 2026-09-04 | 433–435 | diagnosis/admission | §215–217 중복성·RM 기각 | 성능 미조회로 RM을 기각하고 형상 축만 탐색한다. | RU-71 게이트 선례; BD 하나뿐은 R16에서 시점 한정.. |
| **RU-71** | 2026-09-04 | 436–437 | admission/decision | §218 SH 채택·BS 기각 | SH 채택·승격 보류, 직교성+정보량 2단계 게이트 확정. | RU-72 SH 변형, RU-78 gate 보완 |
| **RU-72** | 2026-09-04 | 438–440 | experiment | §219 SH 변형 반증과 SHJ 당시 보류 | 변형 축을 닫고 SHJ 정보는 게이트 개정 판단으로 이월했다. | RU-73 과제 특화 채택, RU-77 oracle 폐기 |
| **RU-73** | 2026-09-04 | 441–441 | policy/admission | §220 과제 특화 gate②b와 SHJ 채택 | ②b로 SHJ 채택, 3/7이라 공식 구성 승격은 안 했다. | R15가 oracle 조건을 fold 재현성으로 교체 |
| **RU-74** | 2026-09-04 | 442–442 | experiment/decision | §221 용량 보정 뒤 LOO 활용 반증 | context→query 분포 이동 문제이므로 §212 폐기 확정. | RU-67 강화, RU-77 선택 신호 부재와 같은 교착 |
| **RU-75** | 2026-09-04 | 443–443 | implementation/verification | §222 SHJ 통합과 bf16 수정 | fp32 백색화를 강제하고 채택을 유지했다. | RU-73 채택 구현 |
| **RU-76** | 2026-09-04 | 444–444 | correction/decision | §223 §213 baseline 오류와 당시 축 종료 | 당시 anchor 축 종료와 통제 비교 의무를 기록했다. | R15가 종료 철회 |
| **RU-77** | 2026-09-04 | 445–446 | policy/experiment | §224 Oracle 폐기와 §225 dose-response | §223 종료는 철회하지만 전역 이득 없고 draw 안정성은 기각. | RU-68/RU-76 anchor 해석 정정 |
| **RU-78** | 2026-09-05 | 447–449 | 도구 수정 및 진행 중 연구방향 라운드 (진행 중) | §226 gate① 결함 수정과 진행 중 방향 라운드 | gate①은 SH·SHJ 포함. 제안은 실험 결과가 아닌 진행 중 목.. | RU-70 screen 선례와 RU-71/RU-73 채택 형.. |

---

## 3. 연구 단위별 상세 감사 기록 (RU-01 ~ RU-78)

### RU-01. v18 Learnability Ladder 및 초기 기준선 동결

- **일자 (Date)**: `2026-07-22 ~ 2026-07-23` (시작: 2026-07-22T00:49:35, 종료: 2026-07-23T10:05:00)
- **커밋 범위**: Index 0–2 (3개 커밋, `deef1f88` ... `128ff6f6`)
- **작업 유형**: `초기 기준선 수립`
- **질문 (Question)**: 순서가 없는 instance 집합(bag)에서 episodic classifier가 합성 learnability ladder를 통과할 수 있는가?
- **가설 (Hypothesis)**: subgroup anchors와 class memory 기반의 episodic classifier가 순서 무관 이진 분류를 학습할 수 있다.
- **실험 및 변경 (Experiment)**: src/models/baseline.py의 BaseModel을 구축하고 D0~D5 합성 learnability ladder 실험군을 실행했다.
- **관측 결과 (Observations)**: v18 baseline 모델이 확립되었으나 bag마다 모든 instance에 더해지는 global translation(bag-shift)에 취약함이 확인됐다.
- **당시 결정 (Decision)**: v18 체크포인트 호환성 버전을 18로 동결하고, translation 민감도를 해결할 차기 구조(v19)로 이월했다.
- **원문 근거 (Evidence)**:
  - `1f83d7f0:docs/architecture_v18.md#L7`:
    > "BagPFN은 하나의 episode 안에서 여러 개의 bag을 보고, 일부 bag의 label을 context로 사용해 가려진 query bag의 이진 label을 예측하는 범용 multiple-instance episodic classifier다. 각 bag은 순서가 없는 instance 집합이며 특정 응용 도메인을 가정하지 않는다."
- **선행·후속 관계 (Relations)**: 프로젝트의 시작 기준선이며, global shift 민감도 해결을 위해 RU-02(v19)로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 합성 데이터 사다리 외에 실제 임베딩에서의 전이 가능성은 미검증 상태였다.

---

### RU-02. v19 Shift-Invariant Covariance 아키텍처 수립

- **일자 (Date)**: `2026-07-24 ~ 2026-07-25` (시작: 2026-07-24T08:14:21, 종료: 2026-07-25T15:47:30)
- **커밋 범위**: Index 3–4 (2개 커밋, `03e79236` ... `e2181c04`)
- **작업 유형**: `가설 검증 및 승격`
- **질문 (Question)**: bag-centered delta 및 공분산 스케일만으로 global translation 민감도를 제거할 수 있는가?
- **가설 (Hypothesis)**: raw mean을 제거하고 centered delta L2 정규화 표현만 사용하면 translation-invariant한 분류가 가능하다.
- **실험 및 변경 (Experiment)**: raw input/mean을 배제하고 centered delta와 covariance subspace relation을 도입한 v19 아키텍처를 구현했다.
- **관측 결과 (Observations)**: global bag-shift 민감도가 성공적으로 제거되었으며 covariance relation이 유효하게 작동했다.
- **당시 결정 (Decision)**: v19를 baseline으로 채택하고 저차원 투영 head 분류기 후보(A/B) 검증 단계로 이월했다.
- **원문 근거 (Evidence)**:
  - `03e79236:docs/architecture_v19.md#L3`:
    > "Architecture v19는 architecture v18에서 확인된 global bag-shift 민감도를"
- **선행·후속 관계 (Relations)**: RU-01(v18)의 한계를 극복함. 후속 RU-03의 Candidate A/B 비교로 연결된다.
- **확인 필요 사항 및 한계 (Uncertainties)**: centered view 도입으로 인한 저차원 투영 헤드의 최적 구조는 추가 탐색 필요.

---

### RU-03. Candidate A/B 20-Epoch 단기 학습 비교 및 Candidate B 선정

- **일자 (Date)**: `2026-07-26` (시작: 2026-07-26T00:39:56, 종료: 2026-07-26T02:06:55)
- **커밋 범위**: Index 5–9 (5개 커밋, `f95ed667` ... `f99682bc`)
- **작업 유형**: `승격 채택`
- **질문 (Question)**: fixed cosine similarity head(A)와 learned MLP head(B) 중 어느 쪽이 저차원 투영 일반화에 적합한가?
- **가설 (Hypothesis)**: CSP projection 저차원 feature를 MLP head로 비선형 결합하는 B가 단순 cosine보다 일반화 손실이 낮을 것이다.
- **실험 및 변경 (Experiment)**: 동일 20-epoch 조건에서 Candidate A와 B를 단기 학습하여 validation CE loss 및 AUROC를 비교했다.
- **관측 결과 (Observations)**: Candidate B의 val_ce_loss가 0.6358로 Candidate A(0.6548) 대비 유의미하게 우수했다.
- **당시 결정 (Decision)**: Candidate B(learned_head)를 v19의 최종 relation 구조로 채택하고 대형 context retrieval(v21)로 진입했다.
- **원문 근거 (Evidence)**:
  - `f99682bc:docs/current_status.md#L115`:
    > "- **Candidate B (`learned_head`)**: CSP projection 저차원 feature $[d_0, d_1, d_0 - d_1, \text{sep}]$를 2-layer MLP head로 분류."
- **선행·후속 관계 (Relations)**: RU-02(v19)의 헤드 구조를 확정함. RU-04(v21 retrieval)의 기반이 됨.
- **확인 필요 사항 및 한계 (Uncertainties)**: 20-epoch 단기 결과가 100-epoch 이상 장기 수렴에서도 유지되는지는 후속 확인 필요.

---

### RU-04. v21 Signal-Aware Retrieval 아키텍처 및 설정 스케일링

- **일자 (Date)**: `2026-07-28` (시작: 2026-07-28T10:42:26, 종료: 2026-07-28T11:41:07)
- **커밋 범위**: Index 10–20 (11개 커밋, `ecf6199b` ... `a02a5c5b`)
- **작업 유형**: `가설 검증`
- **질문 (Question)**: 대형 context에서 관련성 높은 bag을 2-pass 스트리밍으로 검색(retrieval)하면 메모리 효율과 성능이 향상되는가?
- **가설 (Hypothesis)**: 40차원 bag descriptor 기반의 신호 인식 검색이 context 노이즈를 줄이고 긴 시퀀스 학습을 가능하게 한다.
- **실험 및 변경 (Experiment)**: 모델 레벨 40차원 descriptor 및 2-pass streaming retrieval을 구현하고 episode_batch_size를 E=32로 스케일링했다.
- **관측 결과 (Observations)**: GPU throughput에 맞춘 4D batched forwarding 파이프라인이 정상 구축되었으나 crash 및 런처 버그가 발생했다.
- **당시 결정 (Decision)**: 런처 결함을 수정하고 대규모 Phase 5 사전학습을 가동했다.
- **원문 근거 (Evidence)**:
  - `6f2e4d2d:docs/current_status.md#L24`:
    > "## 2. 프로젝트 핵심 아키텍처 및 환경 명세 (Architecture v21)"
- **선행·후속 관계 (Relations)**: RU-03 이후 대형 context 확장을 시도한 단계. 후속 RU-05(Phase 5/6)로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 검색 계층 도입으로 인한 연산 오버헤드와 실제 다운스트림 유효성은 미검증.

---

### RU-05. Phase 5 대형 Context 사전학습 및 런처 안정화

- **일자 (Date)**: `2026-07-28` (시작: 2026-07-28T12:27:58, 종료: 2026-07-28T16:13:38)
- **커밋 범위**: Index 21–32 (12개 커밋, `df6332b1` ... `d8c2b2b0`)
- **작업 유형**: `가설 검증`
- **질문 (Question)**: v21 retrieval 구조가 대형 context 합성 데이터에서 안정적으로 20-epoch을 완주하고 수렴하는가?
- **가설 (Hypothesis)**: v21_large_context 설정이 hung-process 없이 안정 수렴하여 낮은 validation loss를 달성할 것이다.
- **실험 및 변경 (Experiment)**: hung-process 원인을 규명하고 40차원 descriptor 구조를 보정한 뒤 20-epoch 사전학습을 완주했다.
- **관측 결과 (Observations)**: 20/20 epoch 완주 성공, val_ce_loss 0.5940 달성.
- **당시 결정 (Decision)**: Phase 5 체크포인트를 확보하고 다운스트림 전이 검증(Phase 6 ICI fine-tuning)을 발주했다.
- **원문 근거 (Evidence)**:
  - `d8c2b2b0:docs/current_status.md#L4`:
    > "**Latest Commit**: `42c3fa8` (`docs: record Phase 5 launch saga and successful training run`)"
- **선행·후속 관계 (Relations)**: RU-04의 학습 완주 단계. 후속 RU-06(ICI 전이)으로 직결된다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 합성 val loss 감소가 실제 환자 병리 데이터(ICI) 성능 향상으로 직결되는지 여부.

---

### RU-06. Phase 6 ICI 5-Fold 파인튜닝 역효과 규명

- **일자 (Date)**: `2026-07-28` (시작: 2026-07-28T17:59:12, 종료: 2026-07-28T20:44:07)
- **커밋 범위**: Index 33–34 (2개 커밋, `3049513c` ... `02dcf285`)
- **작업 유형**: `가설 검증 및 기각`
- **질문 (Question)**: Phase 5에서 얻은 Signal-Aware retrieval 사전학습 체크포인트가 실제 ICI 데이터셋 파인튜닝을 개선하는가?
- **가설 (Hypothesis)**: 합성 데이터 검색 사전학습이 다운스트림 ICI 5-fold 예측 AUROC를 향상시킬 것이다.
- **실험 및 변경 (Experiment)**: Phase 5 체크포인트로부터 ICI 5-fold 파인튜닝을 수행하고 zero-shot baseline과 비교했다.
- **관측 결과 (Observations)**: Signal-Aware 사전학습이 ICI 파인튜닝 성능을 오히려 저하시킴이 실측으로 확인됐다.
- **당시 결정 (Decision)**: 사전학습 가중치의 다운스트림 유효성 가설을 기각하고, 모델 내부 retrieval 기전의 자체 결함 분석으로 선회했다.
- **원문 근거 (Evidence)**:
  - `02dcf285:docs/current_status.md#L54`:
    > "| **Phase 6** | ICI 5-Fold CV Fine-Tune (Phase 5 체크포인트 기반) | `configs/train_v21_ici_finetune_fold{0..4}.yaml` (`scripts/launch_phase6_5fold.sh`) | 50e (resume epoch 14부터) | **완료, 그러나 가설과 반대 방향**<br/>**AUROC: 0.5081, Log Loss: 0.9596** (Phase 4 대비 AUROC `0.0443` 하락, Log Loss `0.2308` 악화, `p1_std` 0.166→0.287로 과신 심화). §4-⑥ 원인 분석 참고 | `checkpoints/20260728_1757{10,12,14,16,18}/v21_ici_finetune_phase6_f{0..4}/` | `logs/20260728_1757{10,12,14,16,18}/v21_ici_finetune_phase6_f{0..4}.out` |"
- **선행·후속 관계 (Relations)**: RU-05의 성과를 정면 반증함. RU-07(retrieval 가설 전면 기각)으로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 사전학습 데이터와 ICI 간 도메인 갭 때문인지 retrieval 구조 자체의 문제인지 분리 필요.

---

### RU-07. 3D Internal Signal-Aware Retrieval 검증 및 3대 가설 반증

- **일자 (Date)**: `2026-07-28 ~ 2026-07-29` (시작: 2026-07-28T21:27:46, 종료: 2026-07-29T07:09:39)
- **커밋 범위**: Index 35–36 (2개 커밋, `7e378dc8` ... `35ea5d80`)
- **작업 유형**: `가설 검증 및 기각`
- **질문 (Question)**: Retrieval이 제공하던 겉보기 이득은 검색 자체의 효과인가, 아니면 context 크기 차이의 교란인가?
- **가설 (Hypothesis)**: 검색된 context가 무작위 context보다 질적으로 우수하여 분류 성능을 높인다.
- **실험 및 변경 (Experiment)**: 동일 context 크기(통제 비교)에서 retrieval 경로와 무작위 경로를 정밀 비교 분석했다.
- **관측 결과 (Observations)**: 기존 비교는 context 크기가 달라 생긴 착시였으며, 동일 조건 통제 시 retrieval의 순수 이득은 0이었다.
- **당시 결정 (Decision)**: Retrieval 3대 가설을 전면 반증으로 판정하고 retrieval 계층의 완전 제거를 결정했다.
- **원문 근거 (Evidence)**:
  - `35ea5d80:docs/current_status.md#L56`:
    > "| **Phase 6c** | ICI 5-Fold CV Fine-Tune (Phase 5 체크포인트, **retrieval 완전 비활성** — 전체 ~69명 context) | `configs/train_v21_ici_finetune_fullcontext_fold{0..4}.yaml` (`scripts/launch_phase6c_5fold.sh`) | 50e (resume epoch 14부터) | **완료**<br/>AUROC: 0.5454 (95% CI [0.419, 0.674]), **Log Loss: 0.7921, Accuracy: 0.6092** (retrieval 켠 6b보다 calibration/accuracy 우수). §4-⑧ 가설1 참고 | `checkpoints/20260729_0628{33,35,37,39,41}/v21_ici_finetune_phase6c_f{0..4}/` | `logs/20260729_0628{33,35,37,39,41}/v21_ici_finetune_phase6c_f{0..4}.out` |"
- **선행·후속 관계 (Relations)**: RU-04~06의 retrieval 시대를 종식시킴. RU-08(v22)로 직결된다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 검색 모듈 제거 후 남아있는 covariance 분기의 순수 성능 한계 재검토 필요.

---

### RU-08. v22 Retrieval 계층 전면 제거 및 통계 보고 프로토콜 확립

- **일자 (Date)**: `2026-07-29` (시작: 2026-07-29T07:49:48, 종료: 2026-07-29T15:49:05)
- **커밋 범위**: Index 37–41 (5개 커밋, `fbc3ba1e` ... `f87c4274`)
- **작업 유형**: `아키텍처 정리 및 정책 수립`
- **질문 (Question)**: Retrieval을 제거한 순수 모델(v22)의 기준선은 무엇이며, 어떤 통계 규범으로 평가해야 하는가?
- **가설 (Hypothesis)**: 불필요한 검색 복잡도를 제거해도 기본 성능이 보존되며, 신뢰구간과 전 시드 스윕이 필수적이다.
- **실험 및 변경 (Experiment)**: retrieval layer를 완전 제거하여 v22 baseline을 수립하고, CI 보고 및 seed sweep 평가 프로토콜을 공표했다.
- **관측 결과 (Observations)**: v22 모델이 복잡도 없이 v21 대등 이상의 성능을 냄이 확인됐다.
- **당시 결정 (Decision)**: v22를 활성 baseline으로 승격하고, 합성 데이터 중심으로 개발하되 ICI는 최종 테스트로만 보존하기로 결정했다.
- **원문 근거 (Evidence)**:
  - `fbc3ba1e:docs/current_status.md#L22`:
    > "## 2. 프로젝트 핵심 아키텍처 및 환경 명세 (Architecture v22)"
- **선행·후속 관계 (Relations)**: RU-07의 결론을 코드베이스에 반영함. RU-09(Cholesky 안정화)로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: Cholesky backward 도중 간헐적 NaN 발생 이슈가 미해결 상태였음.

---

### RU-09. v22 Cholesky Backward 수치 안정화 및 재베이스라인

- **일자 (Date)**: `2026-07-29` (시작: 2026-07-29T15:32:49, 종료: 2026-07-29T17:10:20)
- **커밋 범위**: Index 42–44 (3개 커밋, `515030a6` ... `efbbfdbd`)
- **작업 유형**: `기술 구현 및 버그 수정`
- **질문 (Question)**: ridge-residual Cholesky backward 패스의 수치 불안정성을 해소할 수 있는가?
- **가설 (Hypothesis)**: Cholesky 분해 시 jitter 및 디바이스 바인딩을 보정하면 gradient NaN 없이 안정 훈련이 가능하다.
- **실험 및 변경 (Experiment)**: Cholesky backward 안전 가드를 추가하고 rank-local cuda 디바이스 매핑을 수정하여 v22를 재베이스라인했다.
- **관측 결과 (Observations)**: 수치 불안정성이 완전히 제거되고 v22 훈련이 재현 가능해졌다.
- **당시 결정 (Decision)**: 수정된 코드로 v22 baseline을 확정하고 공분산 분기의 잠재력 진단(Tier 1)으로 진입했다.
- **원문 근거 (Evidence)**:
  - `efbbfdbd:docs/current_status.md#L45`:
    > "| 코드 상태 | **Cholesky backward + rank-local 수정 반영 후** (커밋 `be36c59`) |"
- **선행·후속 관계 (Relations)**: RU-08의 기술 부채를 해소함. RU-10(Tier 1 공분산 상한선 진단)으로 연결된다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 공분산 분기가 이론적 상한에 도달하지 못하는 병목 위치는 아직 미규명.

---

### RU-10. Covariance Branch 성능 상한선 및 Tier 1 세포 선택 진단

- **일자 (Date)**: `2026-07-30` (시작: 2026-07-30T14:54:18, 종료: 2026-07-30T22:55:10)
- **커밋 범위**: Index 45–50 (6개 커밋, `82e2142f` ... `f5ddbf53`)
- **작업 유형**: `가설 검증 및 진단`
- **질문 (Question)**: 공분산 분기가 남겨둔 0.28 AUROC의 병목은 공분산 연산 자체인가, 아니면 상류 세포 선택인가?
- **가설 (Hypothesis)**: 공분산 표현 자체보다 관련 세포를 선별하는 상류 세포 선택(cell selection)이 병목일 것이다.
- **실험 및 변경 (Experiment)**: 대규모 스케일에서 covariance ceiling을 확인하고 T1-A(sparse), T1-B(slot 분절), T1-C(선택 이득 곡선)를 진단했다.
- **관측 결과 (Observations)**: 공분산 자체는 충분한 용량을 가졌으나 상류 세포 선택이 성패를 좌우함을 확인했다 (T1-C 선형 이득 입증).
- **당시 결정 (Decision)**: Tier 1(covariance) 진단을 통과 판정하고 세포 선택을 개혁하기 위한 후속 연구를 발주했다.
- **원문 근거 (Evidence)**:
  - `f5ddbf53:docs/current_status.md#L179`:
    > "**T1-C 방향**: 세포 선택은 (a) 이상치 거리도 (b) 슬롯 배정도 아닌, **bag 라벨로부터 학습되는 판별 방향(discriminative direction)** 이어야 합니다. `lda_heldout`이 찾은 방향은 episode 내 생성 과정이 일관되므로 원리적으로 학습 가능하나, 이는 노브 조정이 아니라 **새 메커니즘 추가**입니다. 비용/이득을 §5 전략(합성에서 판단)에 따라 먼저 견적하고 결정하세요."
- **선행·후속 관계 (Relations)**: RU-09 이후 공분산의 기전을 규명함. RU-11/RU-13(bag-collapse family)으로 연결된다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 세포 선택을 강화할 구체적 아키텍처(투영, 슬롯, 토큰)의 선택지 필요.

---

### RU-11. T2/T3/T4 다면 진단 및 Hard 상태 접근성 감사

- **일자 (Date)**: `2026-07-30 ~ 2026-07-31` (시작: 2026-07-30T23:08:08, 종료: 2026-07-31T09:56:32)
- **커밋 범위**: Index 51–60 (10개 커밋, `6b1c56f9` ... `3414646b`)
- **작업 유형**: `가설 검증`
- **질문 (Question)**: T2(density), T3(rare), T4(bag summary) 중 어느 경로가 병목 해소의 유효 레버인가?
- **가설 (Hypothesis)**: 희소 세포와 슬롯 밀도를 정밀 분리하면 Hard 과제에 대한 판별력이 개선된다.
- **실험 및 변경 (Experiment)**: T2/T3/T4 경로에 대한 파인튜닝과 Hard 상태 접근성 감사를 병렬 진행했다.
- **관측 결과 (Observations)**: 개별 슬롯 분절화는 제한적 효과만 냈으며 bag 레벨의 직접적인 축약이 필요함이 관측됐다.
- **당시 결정 (Decision)**: 슬롯 세분화보다 bag 전체를 단일 토큰으로 collapse하는 v23/v24 방향으로 선회했다.
- **원문 근거 (Evidence)**:
  - `3414646b:docs/current_status.md#L4`:
    > "**Latest experiment state**: T4-0 Hard 접근성 감사 시작. state 상한 진단 PID `2494986` 실행 중; ICI 잠금 유지."
- **선행·후속 관계 (Relations)**: RU-10의 진단을 심화함. RU-12/13(v24 bag projection)으로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: Hard 과제에 대한 평가 지표의 노이즈 여부 검토 필요.

---

### RU-12. 합성 Context 데이터 생성 및 커리큘럼 파인튜닝

- **일자 (Date)**: `2026-07-31` (시작: 2026-07-31T09:57:44, 종료: 2026-07-31T17:05:45)
- **커밋 범위**: Index 61–88 (28개 커밋, `63be8467` ... `9d375a98`)
- **작업 유형**: `가설 검증`
- **질문 (Question)**: 합성 생성기의 context/query 생성 규칙과 커리큘럼이 모델 일반화 학습을 돕는가?
- **가설 (Hypothesis)**: 난이도 조절(Medium/Hard) 및 context 크기 점진 확장이 장기 학습의 안정성을 제공한다.
- **실험 및 변경 (Experiment)**: 합성 데이터 생성기를 개편하고 Medium context 100-epoch 연장 학습을 수행했다.
- **관측 결과 (Observations)**: 합성 loss는 순조롭게 감소했으나 복잡한 생성기 규칙이 모델에 가짜 통계를 학습시킬 위험이 발견됐다.
- **당시 결정 (Decision)**: 합성 데이터 생성 규칙을 통제하고 모델 구조 개혁에 집중하기로 결정했다.
- **원문 근거 (Evidence)**:
  - `3e6f0a73:docs/current_status.md#L4`:
    > "**Latest experiment state**: Medium mixed-context fine-tuning 완료. Best epoch 14 `val_ce_loss=0.5953`; 동일 context curve 실행 중(PID `2542931`), state oracle-gap 진단 예약(PID `2543210`). ICI 잠금 유지."
- **선행·후속 관계 (Relations)**: RU-11의 데이터 측면 탐색. RU-13의 bag projection 학습과 연계된다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 합성 데이터 규칙과 실제 WSI 타일 분포 간 괴리 확인 필요.

---

### RU-13. v24 Learned Bag Projection 및 Bag-Collapse 패밀리 탐색

- **일자 (Date)**: `2026-07-31 ~ 2026-08-02` (시작: 2026-07-31T17:18:01, 종료: 2026-08-02T00:21:43)
- **커밋 범위**: Index 89–106 (18개 커밋, `99905085` ... `f55bedaf`)
- **작업 유형**: `가설 검증 및 승격`
- **질문 (Question)**: bag당 40개 토큰을 유지하는 대신 7개 핵심 토큰을 1개 토큰으로 사영(linear projection)하면 성능이 개선되는가?
- **가설 (Hypothesis)**: 토큰 수를 극단적으로 줄여 bag을 collapse하면 어텐션 희석을 막고 공분산 신호가 집중된다.
- **실험 및 변경 (Experiment)**: v24 learned bag projection(7 to 1 token) 및 v24-B1/B2 변형군을 훈련·평가했다.
- **관측 결과 (Observations)**: v24-B1이 val_ce_loss 0.58 수준을 기록하며 collapse 방향이 유효함을 입증했다.
- **당시 결정 (Decision)**: v24를 활성 후보로 채택하고 typed-bag(v25)으로 확장을 시도했다.
- **원문 근거 (Evidence)**:
  - `4b3c45df:docs/current_status.md#L4`:
    > "**Status**: Bag-Collapse 실험군 4종 50-epoch **학습 모두 완주**. (v23-A0 `0.5912`, v24-A0 `0.5976`, v24-B0 `0.5923`, **v24-B1 best epoch 41 `val_ce_loss 0.59030` — 전체 최저 기록**). v23/v24 후보 4종 1,000-episode paired 합성 평가 준비 완료. ICI 잠금 유지."
- **선행·후속 관계 (Relations)**: RU-10의 세포 선택 병목에 대한 아키텍처적 해법. 후속 RU-14(v25)로 연결된다.
- **확인 필요 사항 및 한계 (Uncertainties)**: bag collapse 시 정보 손실이 특정 서브타스크에서 치명적인지 여부.

---

### RU-14. v25 Typed-Bag 다면 평가 및 일반화 실패 분석

- **일자 (Date)**: `2026-08-02 ~ 2026-08-03` (시작: 2026-08-02T00:31:49, 종료: 2026-08-03T00:57:08)
- **커밋 범위**: Index 107–117 (11개 커밋, `a5dfcf8b` ... `fcc2122d`)
- **작업 유형**: `가설 검증 및 기각`
- **질문 (Question)**: bag 타입을 명시적으로 분리하여 토큰화하는 v25 typed-bag이 일반화를 개선하는가?
- **가설 (Hypothesis)**: 세포 군집 타입별로 별도 토큰을 할당하면 이질적인 암종 분류에서 유리하다.
- **실험 및 변경 (Experiment)**: v25 typed-bag 구현 후 Easy/Medium 전 영역에서 50-epoch 평가를 수행했다.
- **관측 결과 (Observations)**: 타입 분리가 오히려 표현을 파편화하여 v24보다 일관되게 열세였다.
- **당시 결정 (Decision)**: v25를 공식 기각하고 `v25-typed-bag-final` 태그로 아카이브한 뒤 v24 원형으로 복귀했다.
- **원문 근거 (Evidence)**:
  - `30cc3029:docs/current_status.md#L4`:
    > "**Status**: v24 확정 유지. **v25(T5-A) Medium paired 평가 — 맥락 의존적 trade-off** (v25 우세 @context40, v24-B1 압도 @context300, 승격 기준 미달). **Easy tier 완료 — v24-easy(0.9073) ≈ v25-easy(0.9106), delta +0.0033 승격 기준 미달** → "아키텍처 계열 전체 한계" 가설 강화, **v25 최종 폐기 권고 (사용자 판단 대기)**. Easy 정식 paired 비교 실행 중(PID 277412). 상세는 §11."
- **선행·후속 관계 (Relations)**: RU-13의 확장이 실패한 지점. RU-15(v24 기전 규명)로 되돌아간다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 타입 분리가 왜 합성 데이터에서 역효과를 내는지에 대한 수학적 원인 분석 필요.

---

### RU-15. v24 절제 실험(no-L2) 및 0.70 Plateau 생성기 결함 규명

- **일자 (Date)**: `2026-08-03` (시작: 2026-08-03T01:20:30, 종료: 2026-08-03T04:29:44)
- **커밋 범위**: Index 118–124 (7개 커밋, `8ad6a469` ... `b59fb474`)
- **작업 유형**: `가설 검증 및 진단`
- **질문 (Question)**: v24의 0.70 성능 정체는 모델 용량 부족인가, 아니면 L2 정규화/데이터 생성기의 손실성 때문인가?
- **가설 (Hypothesis)**: L2 normalize 플래그 제거와 데이터 정보량 회복이 0.70 정체를 돌파할 것이다.
- **실험 및 변경 (Experiment)**: v24 no-L2 절제 실험 및 `bag_centered_l2_normalize` 토글 시험을 진행했다.
- **관측 결과 (Observations)**: §22 가설이 확인됨: 0.70 plateau는 모델의 한계가 아니라 합성 데이터 생성기의 심각한 data lossiness 때문이었다.
- **당시 결정 (Decision)**: 모델 변경을 중단하고 생성기 정보 손실을 잡기 위한 Musk 0.95 로드맵을 수립했다.
- **원문 근거 (Evidence)**:
  - `79b8f392:docs/current_status.md#L3`:
    > "**Last updated**: `2026-08-03` (Musk-like easy 가설 판정: "0.70 한계 = 데이터 lossiness" 확정 §22 — 최신; CLS §17, E7 §18, 정규화 §19, no-L2 §20, Musk zero-shot §21)"
- **선행·후속 관계 (Relations)**: RU-13/14의 성능 정체 원인을 규명함. 후속 RU-16/17(Musk 0.95 및 v30)로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 데이터 생성기 정보 복원 시 발생할 수 있는 계산량 폭증 대처 방안.

---

### RU-16. Rawstats 사전학습 및 Musk 0.95 로드맵 P0~P3 기각

- **일자 (Date)**: `2026-08-03 ~ 2026-08-04` (시작: 2026-08-03T10:15:51, 종료: 2026-08-04T00:07:22)
- **커밋 범위**: Index 125–139 (15개 커밋, `79b8f392` ... `69577a50`)
- **작업 유형**: `가설 검증 및 기각`
- **질문 (Question)**: raw bag-stat 토큰을 복원하고 제안된 P0~P3 확장을 적용하면 Musk 0.95에 도달할 수 있는가?
- **가설 (Hypothesis)**: 순수 통계량 보존이 정보 손실을 메워 성능을 0.95까지 견인할 것이다.
- **실험 및 변경 (Experiment)**: rawstats training을 가동하고 제안서 P0/P1/P2/P3의 이론적·실험적 타당성을 검증했다.
- **관측 결과 (Observations)**: P0~P3의 복잡한 메커니즘은 실효 이득이 없었고 단순 cardinality 보존이 핵심임이 드러났다.
- **당시 결정 (Decision)**: 복잡한 외부 제안서를 기각하고 핵심 정규화 조합(v30)으로 단순화하기로 결정했다.
- **원문 근거 (Evidence)**:
  - `7f7ef211:docs/current_status.md#L1430`:
    > "- `train_v24_musklike_easy_rawstats.yaml` — `raw_stat_tokens: [mean, skewness, kurtosis]` (tokens 43)"
- **선행·후속 관계 (Relations)**: RU-15의 생성기 결함 처방 시도. RU-17(v30 확정)으로 수렴한다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 학습 안정성을 해치지 않는 최소한의 cardinality 보존 방식 정립 필요.

---

### RU-17. v26~v29 제안서 전면 기각 및 v30 (B1+B2) 상호필수 확정

- **일자 (Date)**: `2026-08-04` (시작: 2026-08-04T00:15:21, 종료: 2026-08-04T14:47:03)
- **커밋 범위**: Index 140–150 (11개 커밋, `4731657c` ... `e1f3e0cf`)
- **작업 유형**: `정책 개정 및 승격 채택`
- **질문 (Question)**: 외부 제안된 EC-MoE(v26), AC-ICAR(v27), SP-SAT(v29)를 수용할 것인가, B1+B2 단순 결합을 택할 것인가?
- **가설 (Hypothesis)**: B1(poolz_l2 정규화)과 B2(cardinality-faithful) 두 레버의 결합만으로 충분하다.
- **실험 및 변경 (Experiment)**: 미구현 제안서 3건을 학습 없는 게이트로 사전 스크리닝해 기각하고, §28 S2에서 B1+B2 결합을 실측했다.
- **관측 결과 (Observations)**: 게이트 전 항목을 통과하며 B1과 B2가 상호 필수적 조합임이 실측으로 확증됐다.
- **당시 결정 (Decision)**: v26~v29 제안서를 공식 폐기하고 v30(B1+B2)을 활성 베이스라인으로 확정 승격했다.
- **원문 근거 (Evidence)**:
  - `e1f3e0cf:docs/current_status.md#L4`:
    > "**Status**: **v24가 확정 baseline (변경 없음)** — v30 B1 플래그는 구현됐으나 **기본 OFF**이며 승격 전입니다. **§28 S2 = B2(cardinality-faithful log-uniform 샘플링) + B1(`poolz_l2`)이 사전 등록 게이트 6항목을 전부 통과**: Musk **0.8539**(종전 최고 0.822 경신), n≤4 **0.475→0.800**, n>34 0.667→0.698, 합성 무회귀 0.9483(v24 0.9510). paired bootstrap에서 **소형 구간 Δ+0.325 CI가 0을 제외**(P=0.997), **대형 구간 Δ+0.001 무해**(P=0.504). **B1과 B2는 상호 필수** — B2 단독(legacy)은 n=1 bag이 0벡터가 되어 NaN 그라디언트로 **학습 자체가 불가**, B1 단독(S1)은 구간 교환으로 음성이었다. **v30 승격 후보 — 승격은 사용자 확인 필요**(`bag_representation` 기본 `legacy`, v24 무변경). 부산물: **선형 ridge 천장은 예측 지표로 폐기**(부호가 양쪽 반대였음). **다음 최약 구간은 n>34(0.698)** — 0.95까지 남은 +0.096의 대부분. **§26에서 Musk 로드맵을 재설정**했습니다 —"
- **선행·후속 관계 (Relations)**: 프로젝트의 첫 대규모 proposal 기각 정리. 후속 RU-18(v31 CCER)로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: v30의 성공이 실제 다운스트림 PathoBench로 온전히 전이되는지 검증 필요.

---

### RU-18. v31 CCTS / CCER 후보 평가 및 상보성 검증

- **일자 (Date)**: `2026-08-04 ~ 2026-08-05` (시작: 2026-08-04T22:48:58, 종료: 2026-08-05T15:49:58)
- **커밋 범위**: Index 151–165 (15개 커밋, `d866c106` ... `4bb3c4ea`)
- **작업 유형**: `가설 검증 및 기각`
- **질문 (Question)**: v31 Context-Conditioned Exemplar Retrieval(CCER)이 v30에 상보적 정보를 더하는가?
- **가설 (Hypothesis)**: exemplar 검색을 통한 추가 신호가 기존 공분산 분기의 잔차를 설명할 것이다.
- **실험 및 변경 (Experiment)**: v31 CCTS/CCER 후보군을 구현하고 50-epoch 학습 및 PathoBench 10-task를 평가했다.
- **관측 결과 (Observations)**: CCER이 활성화는 되었으나 기존 covariance 표현과 높은 상관을 보여 추가적인 상보 이득이 없었다.
- **당시 결정 (Decision)**: CCER을 독립 모델로 유지하지 않고 residual 보정인 DR-CCER(v32)로 제한 검증하기로 결정했다.
- **원문 근거 (Evidence)**:
  - `6b9e045f:docs/current_status.md#L3`:
    > "**Last updated**: `2026-08-05` (**§31 v31 CCTS 후보 실험 완료 — v30 확정 baseline 유지**)"
- **선행·후속 관계 (Relations)**: RU-17(v30) 이후 상보 분기 탐색. RU-19(v32 DR-CCER)로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 활성화(activation)와 실제 상보 정보(complementary signal)의 구분 기준 확립 필요.

---

### RU-19. v32 / v32b DR-CCER 평가 및 비상보성 기각

- **일자 (Date)**: `2026-08-05` (시작: 2026-08-05T16:00:53, 종료: 2026-08-05T22:09:55)
- **커밋 범위**: Index 166–174 (9개 커밋, `9d3314e7` ... `bc0847db`)
- **작업 유형**: `가설 검증 및 기각`
- **질문 (Question)**: Dual-Residual CCER(DR-CCER)이 v30 대비 통계적으로 유의미한 상보 성능을 제공하는가?
- **가설 (Hypothesis)**: 이중 잔차 구조가 기존 공분산의 한계를 메워줄 것이다.
- **실험 및 변경 (Experiment)**: v32 및 v32b DR-CCER을 구현하고 bf16 정밀도 하에서 50-epoch 훈련을 완주했다.
- **관측 결과 (Observations)**: v30 대비 유의미한 macro 개선이 전무했으며 연산 비용만 대폭 증가했다.
- **당시 결정 (Decision)**: DR-CCER 및 CCER 계보 전체를 공식 기각하고 코드를 동결했다.
- **원문 근거 (Evidence)**:
  - `ee3050a1:docs/current_status.md#L4`:
    > "**Status**: **v30 확정 baseline 유지, CCER 계열 폐기**. v32b P0–P3와 DR-CCER Stage A가 모두 실패했다(P1 standalone `0.5105`, P2 `-0.00034`, P3 `+0.00000`, expert CE `0.6931`). 다음 후보는 구현 전 데이터 요인과 frozen-feature headroom을 검증하는 v33 MR-BagPFN이다."
- **선행·후속 관계 (Relations)**: RU-18의 CCER 계보 종식. RU-20(v33 multi-resolution bags)으로 선회한다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 공분산 외의 다른 축이 왜 번번이 중복으로 귀결되는지 기전 분석 필요.

---

### RU-20. v33 MR-BagPFN Phase 0 (Arm B/C 8x A6000 DDP)

- **일자 (Date)**: `2026-08-05 ~ 2026-08-06` (시작: 2026-08-05T22:16:57, 종료: 2026-08-06T20:04:38)
- **커밋 범위**: Index 175–191 (17개 커밋, `4a39ab94` ... `f007489a`)
- **작업 유형**: `가설 검증`
- **질문 (Question)**: 멀티 레졸루션 bag 표현(Arm B/C)이 단일 해상도 bag보다 풍부한 조직학적 맥락을 포착하는가?
- **가설 (Hypothesis)**: 슬라이드 해상도를 다계층으로 분할하여 학습하면 거시 구조와 미시 세포가 동시에 포착된다.
- **실험 및 변경 (Experiment)**: 8x A6000 DDP 환경에서 Arm B/C top-up 150-epoch 대규모 훈련을 실행하고 평가했다.
- **관측 결과 (Observations)**: Arm C 150ep 완주 결과 macro 성능 향상이 미미하여 레거시 대비 비용 효율성이 떨어졌다.
- **당시 결정 (Decision)**: v33 멀티 해상도 확장을 보류하고 v34 large-context 기준선 정리로 수렴했다.
- **원문 근거 (Evidence)**:
  - `d9416c4a:docs/current_status.md#L3`:
    > "**Last updated**: `2026-08-06` (**§48 arm C top-up 완주(150ep, best e125) + v33 평가: legacy 회귀 gate 미달(+0.041, 과소학습 가설 기각), Musk n>34 개선 유지, PathoBench는 v30 우위** + **§47 e125 기준 확정/타일 스윕**)"
- **선행·후속 관계 (Relations)**: 대규모 분산 훈련을 통한 해상도 축 검증. RU-21/23(v34 확정)으로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: DDP 훈련 시 gradient 동기화 오버헤드와 연산 예산의 효율성 문제 대두.

---

### RU-21. 오퍼레이션 프로파일링, 학습 예산 감사 및 PathoBench 동기화

- **일자 (Date)**: `2026-08-06 ~ 2026-08-07` (시작: 2026-08-06T20:29:10, 종료: 2026-08-07T10:10:22)
- **커밋 범위**: Index 192–209 (18개 커밋, `978c60e4` ... `0ec663ef`)
- **작업 유형**: `인프라 정비`
- **질문 (Question)**: 현재 아키텍처의 연산 병목 op는 무엇이며 학습 예산을 어떻게 배분해야 하는가?
- **가설 (Hypothesis)**: smoke_train_budget 프로파일러로 op별 병목을 측정하면 불필요한 연산을 제거할 수 있다.
- **실험 및 변경 (Experiment)**: `smoke_train_budget.py --profiler` 도구를 구축해 op-level 병목 테이블을 도출하고 PathoBench 코호트를 동기화했다.
- **관측 결과 (Observations)**: 행렬 곱과 ragged indexing 연산에서 상당한 지연이 발생함을 계측했다.
- **당시 결정 (Decision)**: 불필요한 연산 분기를 제거하기 위한 준비 단계로 채택했다.
- **원문 근거 (Evidence)**:
  - `8571798b:scripts/smoke_train_budget.py#L9`:
    > "python scripts/smoke_train_budget.py \"
- **선행·후속 관계 (Relations)**: RU-20 이후의 인프라 효율화 작업. RU-22/23으로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 합성 데이터 생성기가 GPU 연산 속도를 따라가지 못하는 dataloader 병목 잔존.

---

### RU-22. PathoBench CSV 30종 전수 대조 및 레거시 문서 정리

- **일자 (Date)**: `2026-08-07` (시작: 2026-08-07T10:14:26, 종료: 2026-08-07T13:05:23)
- **커밋 범위**: Index 210–222 (13개 커밋, `2b2d8a61` ... `8df3ca91`)
- **작업 유형**: `벤치마크 정비`
- **질문 (Question)**: 로컬에 저장된 30개 PathoBench CSV가 공식 원본과 완전히 일치하는가?
- **가설 (Hypothesis)**: 평가 데이터 무결성을 검증해야 이후 모든 성능 비교의 신뢰성이 보장된다.
- **실험 및 변경 (Experiment)**: §51.5에서 30개 CSV 전수를 공식 데이터와 해시 대조 감사하고 레거시 스크립트를 아카이빙했다.
- **관측 결과 (Observations)**: 모든 CSV의 레이블과 패치 인덱스가 공식 원본과 비트 단위로 일치함을 확인했다.
- **당시 결정 (Decision)**: 평가 벤치마크 정본을 확정하고 오래된 실험 문서를 정리했다.
- **원문 근거 (Evidence)**:
  - `2b2d8a61:docs/current_status.md#L1657`:
    > "### 5. 전체 30개 CSV 전수 감사 (2026-08-07, `/NHNHOME/kimds/Data` 검증)"
- **선행·후속 관계 (Relations)**: 벤치마크 데이터 신뢰성을 확립한 단계. RU-23(v34 확정)의 토대가 됨.
- **확인 필요 사항 및 한계 (Uncertainties)**: 일부 희귀 태스크의 fold별 양성 표본 불균형 처리 규약 필요.

---

### RU-23. CCER 분기 제거, 쿼리 불변 마진 및 v34 Large-Context 승격

- **일자 (Date)**: `2026-08-07 ~ 2026-08-08` (시작: 2026-08-07T14:28:12, 종료: 2026-08-08T08:57:44)
- **커밋 범위**: Index 223–237 (15개 커밋, `f097106c` ... `c9b7d88a`)
- **작업 유형**: `승격 채택 및 정리`
- **질문 (Question)**: 죽은 CCER 분기를 제거하고 쿼리 수에 불변인 마진을 도입하면 모델이 안정화되는가?
- **가설 (Hypothesis)**: 불필요한 570줄의 CCER 코드를 걷어내도 성능이 유지되며 쿼리 크기 불변성이 일반화를 돕는다.
- **실험 및 변경 (Experiment)**: baseline.py에서 CCER 분기를 완전히 삭제하고 query-count-invariant covariance_relation을 구현해 v34를 수립했다.
- **관측 결과 (Observations)**: v34가 간결한 코드베이스에서 공식 PathoBench 기준선으로 안정 작동했다.
- **당시 결정 (Decision)**: v34 large-context를 PathoBench 공식 보고용 활성 모델로 승격 확정했다.
- **원문 근거 (Evidence)**:
  - `5869535e:docs/current_status.md#L6`:
    > "* **v34 확정 (§52·§53·§56)**: **v34-1536을 PathoBench 보고용 모델로 확정**(사용자 결정). 평가는 **공식 Patho-Bench 프로토콜**(공식 k=all.tsv fold·코호트·라벨) 기준 **50-fold**(SEAL macro-AUC와 동일 구조) — **5/17 완료**(bc_therapy er 0.672 / grade 0.713 / her2 0.670, cptac_brca_PIK3CA 0.569, brca_TP53), **12개는 config 수정으로 재시작**(§56, 백그라운드). v30은 합성/Musk baseline 유지. 이전 5-fold와 수치 ±0.04 이내 동일(평가 견고성). config 시스템을 v34 base + group default 참조형으로 리팩터링(§56). 자세한 진행 §53·§56."
- **선행·후속 관계 (Relations)**: RU-18~22의 성과를 집대성한 정식 베이스라인. RU-24/25(CV-only 전환)로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 남아있는 6개 분기 중 여전히 비효율적인 분기가 존재하는지 정밀 점검 필요.

---

### RU-24. v35~v37 제안서(Q1 Structured Population) 평가 및 CV-only 사전 준비

- **일자 (Date)**: `2026-08-08 ~ 2026-08-09` (시작: 2026-08-08T08:58:39, 종료: 2026-08-09T09:47:58)
- **커밋 범위**: Index 238–249 (12개 커밋, `78840c13` ... `70796800`)
- **작업 유형**: `가설 검증 및 기각`
- **질문 (Question)**: v35 데이터 축약, v36 Q1 population, v37 적응형 집계가 v34를 유의미하게 개선하는가?
- **가설 (Hypothesis)**: population attention 모듈을 정밀화하면 이질적 세포군 분류가 개선될 것이다.
- **실험 및 변경 (Experiment)**: v35/v36/v37 proposal을 구현·평가하며 각 분기별 기여도를 분리 진단했다.
- **관측 결과 (Observations)**: 놀랍게도 Q-5 population attention 모듈의 출력이 문자 그대로 상수(std 0.0000)로 죽어 있음이 폭로됐다.
- **당시 결정 (Decision)**: 죽은 모듈을 수리하려던 v35~v37 시도를 전면 기각하고, 살아있는 공분산만 남기는 CV-only 전환을 결의했다.
- **원문 근거 (Evidence)**:
  - `9123938c:docs/current_status.md#L3`:
    > "**Last updated**: `2026-08-08` (**§62 v36 chunk-attention 제안서 반증 + P0-slots 무료 probe — Q1(40→1 압축 해제) paired +0.16 확정 / Q2(num_slots 증설) 부호 불일치로 보류** + §61 P0-b 게이트 통과(|Δpooled| 0.0009) + rare branch 제거 + §60 v35 공식 50-fold 2개(EGFR 0.7819 / PIK3CA 0.5668) + SEAL 비교 + §59 v35 rev.2 + **§41–§48 v33 arm C saga 아카이브**)"
- **선행·후속 관계 (Relations)**: RU-23 이후 가장 결정적인 분기 진단. RU-25(§68 CV-only 대전환)의 직접적 방아쇠가 됨.
- **확인 필요 사항 및 한계 (Uncertainties)**: 상수를 뱉는 모듈에 더 좋은 입력을 넣는 삽질이 반복되던 근본 원인 규명 완료.

---

### RU-25. §68 CV-Only 전환: 죽은 4개 분기 제거 및 연산 스킵

- **일자 (Date)**: `2026-08-09 ~ 2026-08-10` (시작: 2026-08-09T14:15:30, 종료: 2026-08-10T01:05:16)
- **커밋 범위**: Index 250–261 (12개 커밋, `fb926f8a` ... `b871feee`)
- **작업 유형**: `아키텍처 대전환`
- **질문 (Question)**: 6개 분기 중 판별력을 만드는 CV-1과 CV-2만 남기고 나머지 4개를 완전히 제거하면 어떻게 되는가?
- **가설 (Hypothesis)**: Q-5, Rare, Global-shape 등 죽은 분기를 계산조차 건너뛰면 훈련이 5.9배 빨라지고 성능은 보존된다.
- **실험 및 변경 (Experiment)**: meta_covariance_only 아키텍처(v40)를 구현해 연산을 실제로 스킵하고 ragged forward 속도를 대폭 개선했다.
- **관측 결과 (Observations)**: FINAL 0.9199 중 CV-1(0.9052)과 CV-2(0.8867)가 전부였으며, 4개 분기를 제거해도 성능 손실이 전혀 없었다.
- **당시 결정 (Decision)**: 프로젝트 역사상 가장 거대한 아키텍처 대전환 단행: 4개 분기를 영구 제거하고 CV-only로 전환 확정.
- **원문 근거 (Evidence)**:
  - `fb926f8a:src/models/baseline.py#L2623`:
    > "meta_covariance_only: bool = False,"
- **선행·후속 관계 (Relations)**: RU-24의 진단에 따른 결단. ICF 프로젝트를 군더더기 없는 공분산 모델로 재탄생시킴. RU-26으로 연결.
- **확인 필요 사항 및 한계 (Uncertainties)**: 단독 공분산만 남은 상태에서 추가적인 성능 향상을 어떻게 이끌어낼 것인가.

---

### RU-26. 계보 B (학습 Bag 토큰 + Closed-Form Ridge) 기각 확정

- **일자 (Date)**: `2026-08-10` (시작: 2026-08-10T01:26:43, 종료: 2026-08-10T11:08:32)
- **커밋 범위**: Index 262–266 (5개 커밋, `211b7e6a` ... `299d2dc1`)
- **작업 유형**: `가설 검증 및 기각`
- **질문 (Question)**: Transformer Encoder로 32개 summary token을 학습해 ridge를 푸는 계보 B가 CV-only를 능가하는가?
- **가설 (Hypothesis)**: Set-transformer 기반의 학습된 토큰이 고정된 공분산보다 유연한 표현을 제공할 것이다.
- **실험 및 변경 (Experiment)**: SS75(계보 B) 독립 브랜치를 구현하고 2개 LR arm에서 모델 비의존 채점을 진행했다.
- **관측 결과 (Observations)**: 계보 B가 일반화에 참패하여 CV-only의 판별력에 미치지 못했다.
- **당시 결정 (Decision)**: 계보 B를 공식 기각하고 다시는 열지 않을 결론으로 history에 확정 기록했다.
- **원문 근거 (Evidence)**:
  - `299d2dc1:docs/agent_handoff.md#L3`:
    > "**Last updated**: `2026-08-10` — **§73 죽은 분기를 소스에서 실제로 삭제(−11,285줄, `baseline.py` 5,685→2,224)** + **§74 학습이 평가용 ragged 경로를 타고 있었다(74.2→31.3 ms/step)** + **§76·§77 CV-2 손잡이 소진** + **§79 계보 B(Encoder+Ridge) 재설계** + §71 판정 기준 = SEAL 10개 macro 평균. 세션 요약은 `current_status.md` **§72**, 다음 할 일은 **§80**."
- **선행·후속 관계 (Relations)**: RU-25 이후 제기된 대안 아키텍처의 기각. RU-27(합성 데이터 개편)로 선회한다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 엔코더 기반 학습 표현이 왜 닫힌 형태의 공분산을 이기지 못하는가.

---

### RU-27. 합성 데이터 축 개편 (Per-Bag Cardinality, Factorized XOR, v61)

- **일자 (Date)**: `2026-08-10` (시작: 2026-08-10T11:40:58, 종료: 2026-08-10T18:59:23)
- **커밋 범위**: Index 267–274 (8개 커밋, `4bce2687` ... `c1fa6ef0`)
- **작업 유형**: `가설 검증`
- **질문 (Question)**: 합성 데이터 생성기에 per-bag cardinality와 factorized XOR 매니폴드를 도입하면 전이력이 향상되는가?
- **가설 (Hypothesis)**: 실제 WSI의 다양한 패치 수와 비선형 상호작용을 반영한 합성 데이터가 일반화 성능을 높인다.
- **실험 및 변경 (Experiment)**: 가변 cardinality 샘플링 및 직교 선형 매니폴드(v61)를 구축하고 DDP4에서 훈련·평가했다.
- **관측 결과 (Observations)**: v61이 SEAL 평가에서 유의미한 거동을 보이며 합성 데이터 축의 중요성을 재확인했다.
- **당시 결정 (Decision)**: 선형 매니폴드 arm을 베이스라인에 편입하고 하이브리드(v62) 탐색으로 이어갔다.
- **원문 근거 (Evidence)**:
  - `c1fa6ef0:docs/agent_handoff.md#L3`:
    > "**Last updated**: `2026-08-10` — factorized response/XOR v57–v60과 orthogonal manifold v61 평가 완료. 최고 v61도 SEAL 10개 0.6157로 v41 0.6940에 미달해 승격 기각했다. 상세는 `current_status.md` **§82·§83**. 기존 모델·평가 계약은 §71·§73·§74·§76·§77·§79를 따른다."
- **선행·후속 관계 (Relations)**: RU-25/26 이후 데이터 생성기의 질적 개선 단계. RU-28(v62)로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 합성 에피소드 생성이 훈련 step 속도의 병목으로 작용하는 현상 지속.

---

### RU-28. v62 Linear CV1 Transformer 하이브리드 시험

- **일자 (Date)**: `2026-08-10` (시작: 2026-08-10T19:29:14, 종료: 2026-08-10T19:52:49)
- **커밋 범위**: Index 275–279 (5개 커밋, `1cc700b4` ... `8bbf126e`)
- **작업 유형**: `가설 검증 및 기각`
- **질문 (Question)**: 선형 CV1과 Transformer를 결합한 하이브리드(v62)가 단독 CV1보다 우수한가?
- **가설 (Hypothesis)**: 선형 통계와 트랜스포머의 어텐션 결합이 복합 신호를 흡수할 것이다.
- **실험 및 변경 (Experiment)**: v62 하이브리드 모델을 구현하고 capped ragged dense 런처로 재기동했다.
- **관측 결과 (Observations)**: 트랜스포머 결합이 복잡도 대비 뚜렷한 추가 이득을 내지 못하고 정체했다.
- **당시 결정 (Decision)**: 하이브리드 확장을 중단하고 순수 공분산의 통계적 정규화(v70 계열)로 복귀했다.
- **원문 근거 (Evidence)**:
  - `8bbf126e:docs/agent_handoff.md#L3`:
    > "**Last updated**: `2026-08-10` — cap-first `[256,8192]` power 2.0 v62를 GPU 0–3 DDP4로 재실행 중이다. PID/로그/checkpoint는 `current_status.md` §84. 실행 상태는 `current_status.md` §84. v57–v61 결과는 §82·§83. 최고 v61도 SEAL 10개 0.6157로 v41 0.6940에 미달해 승격 기각했다. 상세는 `current_status.md` **§82·§83**. 기존 모델·평가 계약은 §71·§73·§74·§76·§77·§79를 따른다."
- **선행·후속 관계 (Relations)**: RU-27의 하이브리드 모델 시험. 실패 후 RU-29(Canonical CV/DD/CT)로 복귀.
- **확인 필요 사항 및 한계 (Uncertainties)**: 트랜스포머의 인덕티브 바이어스가 MIL 통계에 적합하지 않음이 누적 확인됨.

---

### RU-29. Canonical CV Raw Mean 통합 및 v70~v74 CT 베이스라인 수립

- **일자 (Date)**: `2026-08-11 ~ 2026-08-12` (시작: 2026-08-11T12:23:31, 종료: 2026-08-12T09:41:48)
- **커밋 범위**: Index 280–284 (5개 커밋, `c6210526` ... `bc7a1520`)
- **작업 유형**: `승격 채택`
- **질문 (Question)**: raw bag mean을 CV 브랜치에 복원하고 DD 및 relation head를 정비하면 v74 베이스라인이 성립하는가?
- **가설 (Hypothesis)**: bag mean의 1차 통계와 공분산 2차 통계가 결합될 때 기준 판별력이 완성된다.
- **실험 및 변경 (Experiment)**: raw bag mean을 CV의 정식 구성요소로 편입하고, DD 및 CT relation head를 추가해 v74 baseline을 수립했다.
- **관측 결과 (Observations)**: CV, CT, DD 3대 분기의 초기 canonical 형태가 갖춰지며 안정적인 성능을 기록했다.
- **당시 결정 (Decision)**: v74 CT 베이스라인을 공식 승격하고 난이도 스윕(v77)으로 진입했다.
- **원문 근거 (Evidence)**:
  - `bc7a1520:configs/train_v74_cv_dd_ct_mlp_1pop_linear_1536.yaml#L1`:
    > "# v74 — v70 plus support-selected discriminative Composition Tokens."
- **선행·후속 관계 (Relations)**: 현대 3-branch(CV, CT, DD)의 시초. 후속 RU-30(v77 승격)으로 연결된다.
- **확인 필요 사항 및 한계 (Uncertainties)**: CT 분기의 k-means 연산 비용과 DD의 코호트 민감도 잔존.

---

### RU-30. 잠재 차원 스윕, 비선형 매니폴드 뱅크 및 Hard v77 승격

- **일자 (Date)**: `2026-08-12` (시작: 2026-08-12T09:58:04, 종료: 2026-08-12T15:51:04)
- **커밋 범위**: Index 285–298 (14개 커밋, `8ffc7821` ... `02037525`)
- **작업 유형**: `승격 채택`
- **질문 (Question)**: Hard 과제에서 잠재 차원과 비선형 매니폴드 뱅크를 확장하면 v77 베이스라인이 최고 성능을 내는가?
- **가설 (Hypothesis)**: 고차원 비선형 매니폴드 학습이 Hard 과제의 복잡한 클래스 분리도를 극복한다.
- **실험 및 변경 (Experiment)**: latent-dim 스윕, fixed MLP bank, ridge calibration을 거쳐 large ragged Hard 훈련을 완주했다.
- **관측 결과 (Observations)**: Hard v77 모델이 10-task에서 견고한 성능을 보이며 기존 모델을 압도했다.
- **당시 결정 (Decision)**: 사용자 결정으로 Hard v77을 프로젝트의 새로운 활성 공식 베이스라인으로 승격했다.
- **원문 근거 (Evidence)**:
  - `02037525:configs/train_v77_hard_orthogonal_1536.yaml#L1`:
    > "# Canonical v77 baseline: v76 relation architecture trained on Hard synthetic data."
- **선행·후속 관계 (Relations)**: 학습 기반 시대(v18~v77)의 정점. 후속 RU-31(판정 프로토콜 개혁)로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: Hard v77이 여전히 많은 학습 파라미터를 유지하고 있어 시드 분산이 큼.

---

### RU-31. §99 Fold-Paired Delta + Bootstrap CI 프로토콜 및 v78 DD 2차형식

- **일자 (Date)**: `2026-08-12` (시작: 2026-08-12T16:27:41, 종료: 2026-08-12T16:49:29)
- **커밋 범위**: Index 299–300 (2개 커밋, `8d58ba51` ... `5f3049b8`)
- **작업 유형**: `정책 수립 및 가설 검증`
- **질문 (Question)**: 모델 간 성능 비교를 단순 평균이 아닌 fold-paired delta와 bootstrap CI로 판정해야 하는가?
- **가설 (Hypothesis)**: fold 쌍별 차이와 신뢰구간을 엄격히 적용해야 위양성 판정을 원천 차단할 수 있다.
- **실험 및 변경 (Experiment)**: §99 통계 프로토콜을 수립하고, v78 DD quadratic-form gradient path를 구현해 평가했다.
- **관측 결과 (Observations)**: v78의 DD gradient가 투영 행렬 P를 단조로 훼손하여 성능이 악화됨이 paired delta로 포착됐다.
- **당시 결정 (Decision)**: §99 fold-paired delta 판정 프로토콜을 필수 불변식으로 확정했다.
- **원문 근거 (Evidence)**:
  - `8d58ba51:docs/history.md#L66`:
    > "paired bootstrap 승률 0.5±0.03). "Phase 4가 최선" 같은 당시 판단은 **노이즈 추적**이었다."
- **선행·후속 관계 (Relations)**: 통계적 판정 규범의 중대한 진보. RU-32(v78 기각)의 근거가 됨.
- **확인 필요 사항 및 한계 (Uncertainties)**: bootstrap CI가 엄격해진 만큼 미세한 차이를 판정하기 위한 시드 수 요구 증가.

---

### RU-32. History 아카이브 통합, v78 무가중 기각 및 v79 Dual Projection

- **일자 (Date)**: `2026-08-12` (시작: 2026-08-12T17:04:20, 종료: 2026-08-12T18:22:50)
- **커밋 범위**: Index 301–305 (5개 커밋, `9884fbfc` ... `095212d8`)
- **작업 유형**: `기각 및 가설 검증`
- **질문 (Question)**: v78의 실패를 dual projection(v79)으로 해결할 수 있는가, 그리고 과거 기록을 어떻게 관리할 것인가?
- **가설 (Hypothesis)**: P를 CV와 DD용으로 분리하면 DD gradient의 상호 간섭을 막을 수 있다.
- **실험 및 변경 (Experiment)**: docs/history.md로 §2~§97을 최초 압축 통합하고, v78을 공식 기각한 뒤 v79 dual projection을 시험했다.
- **관측 결과 (Observations)**: v78은 단조 악화로 기각 확정되었고, v79 역시 macro 개선에 실패하여 CV/DD 배선 축이 소진되었음이 확인됐다.
- **당시 결정 (Decision)**: v78과 v79를 모두 기각하고 v77 epoch 49 동결선으로 복귀했다.
- **원문 근거 (Evidence)**:
  - `095212d8:docs/agent_handoff.md#L3`:
    > "**Last updated**: `2026-08-12` — 활성 baseline은 **v77 Hard orthogonal**(SEAL macro **0.6873**)이다. **v78·v79 모두 기각**이고 **CV/DD·사영 배선 축은 소진**으로 본다. 다음 작업은 **seed 반복이 선행 조건**이다. 판정은 반드시 **fold-paired Δ + CI**(§99)로 한다. 진행 상태는 `current_status.md` §103."
- **선행·후속 관계 (Relations)**: 배선 중심의 파라미터 학습 한계 확인. RU-33(과거 판정 전수 감사)으로 연결.
- **확인 필요 사항 및 한계 (Uncertainties)**: CV와 DD의 상호 간섭이 아키텍처 배선만으로는 해결되지 않음.

---

### RU-33. v77 Epoch 49 베이스라인 고정, v80 기각 및 27개 Arm 재채점

- **일자 (Date)**: `2026-08-12 ~ 2026-08-13` (시작: 2026-08-12T21:04:05, 종료: 2026-08-13T18:17:28)
- **커밋 범위**: Index 306–309 (4개 커밋, `f103d3a3` ... `af8db22e`)
- **작업 유형**: `벤치마크 정정`
- **질문 (Question)**: v77의 최고 체크포인트는 어디이며, 과거 27개 arm의 판정은 epoch 49 기준선에서 어떻게 재평가되는가?
- **가설 (Hypothesis)**: 에포크별 오버피팅을 배제하고 단일 에포크(49)에서 전수 재채점해야 참된 순위가 나온다.
- **실험 및 변경 (Experiment)**: v77을 epoch 49(macro 0.6880)로 엄격히 고정하고, v80 shallow MLP를 기각하며 27개 arm을 재채점했다.
- **관측 결과 (Observations)**: 과거 판정 중 다수가 에포크 선택 편향이었음이 밝혀졌고, Hard보다 Medium이 우수할 가능성이 포착됐다.
- **당시 결정 (Decision)**: v77 baseline을 0.6880으로 확정하고 Medium 대 Hard 정면 승부(v82)로 진입했다.
- **원문 근거 (Evidence)**:
  - `f103d3a3:configs/train_v80_hard_shallow_mlp_1536.yaml#L1`:
    > "# v80: infinite (fresh-per-episode) MLP manifold, at the shallowest depth that is"
- **선행·후속 관계 (Relations)**: 벤치마크의 객관적 재정렬. 후속 RU-34(v82 Medium 승격)의 결정적 계기.
- **확인 필요 사항 및 한계 (Uncertainties)**: 시드 간 분산(seed variance)이 모델 간 격차보다 크다는 위험 감지.

---

### RU-34. v82 Medium 승격 및 v83 Linear Head vs v84 Deep Head 판정

- **일자 (Date)**: `2026-08-13` (시작: 2026-08-13T18:17:51, 종료: 2026-08-13T19:03:49)
- **커밋 범위**: Index 310–315 (6개 커밋, `1d134db5` ... `ecada468`)
- **작업 유형**: `승격 채택 및 기각`
- **질문 (Question)**: Hard 대신 Medium이 우수하며, GELU head를 제거한 linear head(v83)가 대등한가?
- **가설 (Hypothesis)**: 과도한 Hard 규제보다 Medium이 안정적이며, 복잡한 head보다 linear head가 분산이 작다.
- **실험 및 변경 (Experiment)**: 4 arm x 4 seed 정면 비교를 통해 v82 Medium을 검증하고, v83 linear head와 v84 deep head를 비교했다.
- **관측 결과 (Observations)**: Medium이 Hard를 이겼으며, v83은 GELU를 제거해도 통계적으로 구분되지 않았으나 v84 deep head는 명백히 손해였다.
- **당시 결정 (Decision)**: 사용자 결정으로 v82에 이어 v83 linear head를 활성 베이스라인으로 승격하고 v84를 기각했다.
- **원문 근거 (Evidence)**:
  - `ecada468:docs/README.md#L51`:
    > "`train_v82_medium_classsep_1536_1gpu.yaml`(직전 baseline, 참고용),"
- **선행·후속 관계 (Relations)**: 헤드 단순화의 결정적 승리. 후속 RU-35/36(세포 축 소진)으로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 선형 헤드로 충분하다면 학습되는 투영 P 자체도 고정할 수 있는가?

---

### RU-35. v86 노이즈 레버 및 v87 희소 abundance 레버 Null Result

- **일자 (Date)**: `2026-08-13` (시작: 2026-08-13T20:25:51, 종료: 2026-08-13T21:39:33)
- **커밋 범위**: Index 316–317 (2개 커밋, `810d2b10` ... `94b9309c`)
- **작업 유형**: `가설 검증 및 기각`
- **질문 (Question)**: 합성 데이터의 노이즈 레버(v86)와 희소 abundance 레버(v87)가 v83을 개선하는가?
- **가설 (Hypothesis)**: 합성 데이터의 잡음 수준을 현실화하면 일반화 능력이 강화된다.
- **실험 및 변경 (Experiment)**: v86과 v87을 v83 베이스라인에 대조하여 4-seed 정밀 측정을 수행했다.
- **관측 결과 (Observations)**: 두 레버 모두 통계적으로 유의미한 차이를 만들지 못하고 null result로 끝났다.
- **당시 결정 (Decision)**: 데이터 생성기 미세 튜닝 축을 닫고 에피소드 형상 탐색(v88~v93)으로 전환했다.
- **원문 근거 (Evidence)**:
  - `810d2b10:configs/train_v86_noise_1536_1gpu.yaml#L1`:
    > "# v86: v83 baseline with the synthetic generator's observation_noise doubled."
- **선행·후속 관계 (Relations)**: 합성 생성기 레버의 무력화 입증. RU-36(에피소드 형상 가설)으로 연결.
- **확인 필요 사항 및 한계 (Uncertainties)**: 데이터 레버가 작동하지 않는 근본 원인이 모델 용량인지 분포 불일치인지 검토 필요.

---

### RU-36. v88 PA 기각(§114) 및 v89~v93 에피소드 형상·세포 축 소진

- **일자 (Date)**: `2026-08-14` (시작: 2026-08-14T02:15:03, 종료: 2026-08-14T16:42:11)
- **커밋 범위**: Index 318–325 (8개 커밋, `3def0fc2` ... `b5bbe4fe`)
- **작업 유형**: `가설 검증 및 축 종료`
- **질문 (Question)**: 에피소드 형상(bag 수 vs cell 수)을 실제 평가 레짐에 맞추면 성능이 향상되는가?
- **가설 (Hypothesis)**: 학습 시 bag/cell 형상을 실측에 가깝게 조정하면 도메인 갭이 줄어든다.
- **실험 및 변경 (Experiment)**: v88 PA(population branch)를 기각하고, v89~v93에서 bag 축과 cell 축을 분리·반전·소진하며 정밀 검정했다.
- **관측 결과 (Observations)**: cell 축은 단조 레버가 아니었으며(v92 기각), v93(-0.0040)으로 §115 에피소드 형상 3대 가설이 전부 소진됐다.
- **당시 결정 (Decision)**: 에피소드 형상 축을 영구 종료하고 합성 cell과 실제 cell의 분포 불일치 진단(§123)으로 선회했다.
- **원문 근거 (Evidence)**:
  - `b5bbe4fe:docs/current_experiments.md#L73`:
    > "| **`nvidia-smi`로 GPU 여유 판단하기** | caching allocator가 카드를 채우고 캐시를 놓지 않는다. ×3과 ×2 변형이 **같은** 172.8 GiB에서 평탄화됐고 실제 `max_memory_allocated`는 42.4 GiB였다 — 3배 차이를 오판했다. `torch.cuda.max_memory_allocated`를 볼 것 (§122-5) |"
- **선행·후속 관계 (Relations)**: 에피소드 형상 가설의 최종 종식. RU-37(실제 통계 불일치 및 앙상블)로 직결.
- **확인 필요 사항 및 한계 (Uncertainties)**: 합성 세포와 실제 UNI2 타일 간의 통계적 괴리 측정 필요.

---

### RU-37. §123~§131 실제 UNI2 통계 불일치, Nuisance 가설 반증 및 v98 시드 앙상블

- **일자 (Date)**: `2026-08-14 ~ 2026-08-15` (시작: 2026-08-14T17:27:11, 종료: 2026-08-15T21:40:05)
- **커밋 범위**: Index 326–333 (8개 커밋, `e0dd1007` ... `76351052`)
- **작업 유형**: `가설 검증 및 승격`
- **질문 (Question)**: 합성 세포를 실제 UNI2 통계에 강제로 맞추는 것이 유효한가, 분산이 편향보다 큰가?
- **가설 (Hypothesis)**: 실제 통계를 흉내 내는 것은 유해하며, 오히려 훈련 비용 0의 시드 앙상블이 편향을 이긴다.
- **실험 및 변경 (Experiment)**: §123 처방 스크리닝(v94 음수 확인), v100 nuisance 반증, v102 bag 꼬리 기각 후 시드 앙상블을 실측했다.
- **관측 결과 (Observations)**: 통계 매칭은 단조 악화로 기각됐으나, 8개 시드 앙상블이 +0.0071의 거대한 무비용 이득을 제공했다.
- **당시 결정 (Decision)**: 합성 데이터 매칭 축을 완전 종결하고, baseline을 v98 8-seed로 교체 승격했다.
- **원문 근거 (Evidence)**:
  - `76351052:docs/README.md#L12`:
    > "> 2. **문제는 편향이 아니라 분산일 수 있다 (§130·§131)** — **4 seed의 최소 검출 효과가 0.0121**인데"
- **선행·후속 관계 (Relations)**: 분산이 편향을 압도한다는 대발견. 후속 RU-38(학습 파라미터 0 발견)의 직전 단계.
- **확인 필요 사항 및 한계 (Uncertainties)**: 시드 앙상블이 주는 이득의 본질이 투영 행렬 P의 무작위 분산 때문임이 드러남.

---

### RU-38. §132~§141 학습 파라미터 P 제거, 고정 Head 및 v106 Training-Free 수립

- **일자 (Date)**: `2026-08-15 ~ 2026-08-16` (시작: 2026-08-15T23:49:54, 종료: 2026-08-16T14:23:26)
- **커밋 범위**: Index 334–340 (7개 커밋, `c339e214` ... `e7b200a0`)
- **작업 유형**: `아키텍처 대전환`
- **질문 (Question)**: 학습되는 투영 행렬 P가 시드 분산의 주범이라면, P를 PCA로 대체하고 학습 파라미터 0으로 만들 수 있는가?
- **가설 (Hypothesis)**: 학습된 head 13개는 상수 3개에 불과하며, within-slide PCA 사영은 학습된 P와 대등하다.
- **실험 및 변경 (Experiment)**: GELU head 복원 실패(§132), fixed P 분석(§133-134), zero-parameter 모델 대등성(§135-137)을 거쳐 정식 경로를 측정했다.
- **관측 결과 (Observations)**: 학습 파라미터 0 모델(within-slide PCA + 고정 head)이 수백 에포크 학습 모델과 완전히 대등했다 (분산 43배 감소).
- **당시 결정 (Decision)**: 사용자 결정으로 학습 모델 시대를 공식 종료하고 **v106 (학습 파라미터 0 Training-Free)**을 채택했다.
- **원문 근거 (Evidence)**:
  - `e7b200a0:docs/README.md#L4`:
    > "**Active 구성**: **v106 — 학습 파라미터 0** (§139, 사용자 결정). within-slide PCA 사영 + 고정 3상수 head,"
- **선행·후속 관계 (Relations)**: ICF 프로젝트의 가장 위대한 패러다임 전환. 학습 비용 0의 Training-Free 시대를 개막. RU-39로 연결.
- **확인 필요 사항 및 한계 (Uncertainties)**: within-slide PCA의 스케치 차원 K의 최적값 탐색 필요.

---

### RU-39. §142~§145 K=256 스윕, v107 승격 및 K 이득 CV 귀속

- **일자 (Date)**: `2026-08-16 ~ 2026-08-17` (시작: 2026-08-16T15:57:40, 종료: 2026-08-17T17:00:52)
- **커밋 범위**: Index 341–344 (4개 커밋, `0f468ac8` ... `22ada752`)
- **작업 유형**: `승격 채택`
- **질문 (Question)**: 학습 비용이 0이 된 조건에서 공분산 스케치 차원 K를 128에서 256으로 늘리면 성능이 향상되는가?
- **가설 (Hypothesis)**: K=256 확장이 표현 용량을 확대하여 10-task 성능을 단조 개선할 것이다.
- **실험 및 변경 (Experiment)**: K=128 vs 256 스윕을 수행하고 v107을 승격한 후, 분기별 K 분해 실험(§145)을 실행했다.
- **관측 결과 (Observations)**: K=256이 전체 성능을 개선(v107 승격)했으나, 분해 결과 이득은 100% CV의 것이었고 DD는 128에서 포화했다.
- **당시 결정 (Decision)**: 사용자 결정으로 v107(K=256)을 활성 베이스라인으로 확정하고 DD 포화를 공식 기록했다.
- **원문 근거 (Evidence)**:
  - `22ada752:docs/README.md#L4`:
    > "**Active 구성**: **v107 — 학습 파라미터 0, K=256** (§142, 사용자 결정). within-slide PCA 사영 +"
- **선행·후속 관계 (Relations)**: v106의 첫 매개변수 최적화. RU-40(DD 및 CT 병목 규명)으로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: K 증가에 따른 메모리 증가와 CT 분기의 기여 미미함 해결 필요.

---

### RU-40. §146~§149 적응적 Rank DD 기각 및 CT 2-Token 병목 진단

- **일자 (Date)**: `2026-08-17` (시작: 2026-08-17T18:07:44, 종료: 2026-08-17T21:16:57)
- **커밋 범위**: Index 345–349 (5개 커밋, `55c67df8` ... `24745915`)
- **작업 유형**: `가설 검증 및 기각`
- **질문 (Question)**: 적응적 rank DD가 유효한가, 그리고 CT의 two-token readout이 병목인가?
- **가설 (Hypothesis)**: DD의 t-gate가 이상치를 걸러내고, CT readout 교체가 성능을 견인할 것이다.
- **실험 및 변경 (Experiment)**: adaptive-rank DD 시험(§146), |lambda| vs |t| 선택기 비교(§147), CT readout 진단(§148), 거리 집중 측정(§149)을 완주했다.
- **관측 결과 (Observations)**: DD t-게이트는 기전 반증으로 축이 종료됐다. CT two-token readout은 무죄였으며 병목은 16차원 거리 집중이었다.
- **당시 결정 (Decision)**: DD 탐색 축을 공식 닫고, CT 거리 집중을 해소할 PCA 결합 가설(§150)을 발주했다.
- **원문 근거 (Evidence)**:
  - `24745915:docs/current_experiments.md#L88`:
    > "| **CT의 two-token readout을 병목으로 가정** | 16차원 abundance 전체를 prototype/ridge로 읽어도 CT-only·full-model 어디서도 개선이 없다(전부 8/17). 원인은 abundance에 정보가 없어서다 — token별 판별 통계 **중앙값 1.31**, |t|>2가 16개 중 3~5개, CT-only macro 0.5600. ridge는 실효 6.8/16 token으로 **퍼뜨렸는데도** 못 이겼다. 병목은 상류(farthest-point token 생성 / 64-cell 샘플링 / 1536차원 거리) (§148) |"
- **선행·후속 관계 (Relations)**: 전반부 350커밋 역사의 마침표. 곧바로 후반부 RU-41(§150)로 정확히 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 거리 집중을 풀었을 때 readout과의 결합 효과(대각선 한 칸) 확인 필요.

---

### RU-41. CT 부분공간과 ridge readout의 결합

- **일자 (Date)**: `2026-08-17` (시작: 2026-08-17T22:16:23, 종료: 2026-08-17T22:36:12)
- **커밋 범위**: Index 350–351 (2개 커밋, `aac69013` ... `ebcdb835`)
- **작업 유형**: `시작 경계 밖에서 착수된 연구`
- **질문 (Question)**: CT의 거리 표현과 readout은 독립적으로 판단할 수 있는가?
- **가설 (Hypothesis)**: raw 1536차원에서 보이지 않았던 ridge 이득이 PCA 공간에서는 나타날 수 있다.
- **실험 및 변경 (Experiment)**: CT-only/full model에서 raw/PCA와 extreme/ridge를 교차 비교하고 CT weight도 0.286~1.442로 점검했다.
- **관측 결과 (Observations)**: PCA32+ridge는 v107 대비 SEAL +0.0022, 홀드아웃 +0.0057, 전체 +0.0037(11/17)이었다. weight 0.5는 양 집단 평균을 높였지만 부호 일치는 11/17에서 10/17로 낮아졌다. 결정론적 arm의 task t/p/CI는 근거에서 제외됐다.
- **당시 결정 (Decision)**: raw 조건에 한정됐던 readout 결론을 정정하고 PCA32+ridge를 v108로 채택하되 CT weight 0.286은 유지했다.
- **원문 근거 (Evidence)**:
  - `ebcdb835:docs/current_status.md#L5287`:
    > "⚠️ **§142~§150의 t 값은 모두 무시할 것.** Δ와 부호 수는 유효하다(실측이므로). 결론이 바뀌는"
  - `ebcdb835:docs/current_status.md#L5352`:
    > "**둘은 반드시 함께 간다** — 단독은 +0.0019(PCA)와 +0.0008(ridge)인데 조합이 +0.0037이고,"
  - `ebcdb835:docs/current_status.md#L5333`:
    > "w=0.5는 SEAL +0.0009 / 홀드아웃 +0.0024(v108 대비)로 **양쪽 다 개선**이지만 **부호가 11/17 →"
- **선행·후속 관계 (Relations)**: E05의 token 생성 개선 이전의 CT 기준선이며 CV와 기저를 공유하는 한계가 다음 탐색을 이끌었다.
- **확인 필요 사항 및 한계 (Uncertainties)**: PCA16/32의 최적 차원은 노이즈로 취급됐고 독립 CT 기저는 검증되지 않았다.

---

### RU-42. DD의 가중치와 코호트 의존성

- **일자 (Date)**: `2026-08-17` (시작: 2026-08-17T23:05:04, 종료: 2026-08-17T23:05:04)
- **커밋 범위**: Index 352–352 (1개 커밋, `bd7647bb` ... `bd7647bb`)
- **작업 유형**: `가설 검증`
- **질문 (Question)**: DD를 전역적으로 약화하거나 제거하면 v108이 개선되는가?
- **가설 (Hypothesis)**: 약한 코호트에서 DD를 줄이면 노이즈 적합을 피할 수 있다.
- **실험 및 변경 (Experiment)**: v108 DD weight를 0.343에서 0까지 낮추고 SEAL 10과 홀드아웃 7의 macro·task 부호를 비교했다.
- **관측 결과 (Observations)**: DD 제거 시 SEAL은 0.6967→0.6895로 단조 하락하고 홀드아웃은 0.5893→0.6030으로 단조 상승했다. 전체 평균의 평탄함은 상쇄의 결과였다.
- **당시 결정 (Decision)**: 전역 weight는 고르지 않고 0.343을 유지했다. context 분할 기반 episode별 선택은 후속 가설로 남겼다.
- **원문 근거 (Evidence)**:
  - `bd7647bb:docs/current_status.md#L5424`:
    > "**SEAL은 단조 감소(0.6967 → 0.6895), 홀드아웃은 단조 증가(0.5893 → 0.6030).** 7개 지점 전부"
  - `bd7647bb:docs/current_status.md#L5466`:
    > "**DD weight는 이 데이터로 고를 수 없다.** 어느 쪽으로 옮겨도 한 집단을 다른 집단과 맞바꾼다:"
- **선행·후속 관계 (Relations)**: RU-43~E09의 DD readout 탐색으로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 집단 차이가 신호 강도인지 코호트 차이인지는 분리되지 않았다.

---

### RU-43. DD LLR 보정과 상대 거리의 판정 가능성

- **일자 (Date)**: `2026-08-17 ~ 2026-08-18` (시작: 2026-08-17T23:39:57, 종료: 2026-08-18T00:41:54)
- **커밋 범위**: Index 353–354 (2개 커밋, `c61a4100` ... `c85156b8`)
- **작업 유형**: `가설 검증`
- **질문 (Question)**: DD 거리의 확률적 보정 또는 상대화가 fold-mean AUROC를 개선하는가?
- **가설 (Hypothesis)**: 누락된 log-det LLR 항과 거리 비율이 더 적절한 DD evidence를 만들 수 있다.
- **실험 및 변경 (Experiment)**: log(σ₀²/σ₁²)를 더하고 상대 margin을 DD-only/full model에서 비교했다.
- **관측 결과 (Observations)**: LLR 항은 fold context에만 의존하는 상수라 fold 내 AUROC 순위를 바꾸지 않았다. 상대화는 query별 순위를 바꾸지만 DD-only AUROC는 SEAL 0.6354→0.6074, 홀드아웃 0.4937→0.4797로 낮아졌다.
- **당시 결정 (Decision)**: LLR 훅은 보정·pooled 평가용으로만 남기고 승격하지 않았으며 상대 거리 readout은 기각했다.
- **원문 근거 (Evidence)**:
  - `c61a4100:docs/current_status.md#L5526`:
    > "σ₀², σ₁²는 **context에서만** 나온다. 정식 프로토콜에서 한 fold의 context는 그 fold의 모든 query에"
  - `c61a4100:docs/current_status.md#L5544`:
    > "- **다만 우리가 판정하는 지표로는 측정 자체가 불가능하다.** 이득도 손해도 아니라 **불가시**다."
  - `c85156b8:docs/current_status.md#L5584`:
    > "| SEAL 10 DD-only AUROC | **0.6354** | 0.6074 |"
- **선행·후속 관계 (Relations)**: E02의 DD 문제를 전역 가중 대신 margin 정의로 해결하려 한 시도다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 확률 보정·임계값 결정에서 LLR 상수의 효과는 검증되지 않았다.

---

### RU-44. CV descriptor의 정보 분해

- **일자 (Date)**: `2026-08-18` (시작: 2026-08-18T01:36:36, 종료: 2026-08-18T11:36:35)
- **커밋 범위**: Index 355–361 (2개 커밋, `80e7045c` ... `9c1889d2`)
- **작업 유형**: `가설 검증`
- **질문 (Question)**: CV에서 bag mean·대각 및 공분산 스케일 중 무엇이 정보를 담는가?
- **가설 (Hypothesis)**: 중복 raw mean과 대각 feature는 제거할 수 있고 상관행렬 정규화는 개선을 줄 수 있다.
- **실험 및 변경 (Experiment)**: raw mean·대각을 제거한 뒤 off-diagonal 공분산을 bag별 상관행렬로 정규화해 full model/CV-only를 비교했다.
- **관측 결과 (Observations)**: 대각 256 feature 제거는 +0.0052(13/17)이었지만 상관화는 full model −0.0024(8/17)이었다.
- **당시 결정 (Decision)**: v109에서 off-diagonal CV만 유지하고 √λᵢλⱼ가 주는 공분산 크기 가중은 보존했다.
- **원문 근거 (Evidence)**:
  - `9c1889d2:docs/current_status.md#L6192`:
    > "| 대각 256차원을 feature로 사용 (§156) | **유해.** 빼면 +0.0052 (13/17) |"
  - `9c1889d2:docs/current_status.md#L6193`:
    > "| 대각이 off-diagonal 크기에 주는 영향 (§162) | **정보였다.** 빼면 −0.0024 (8/17) |"
- **선행·후속 관계 (Relations)**: E05의 v109 승격과 E01의 CV-CT 공유 기저 제약에 연결된다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 상관 CV가 다른 branch와 겹친다는 해석은 가능성으로만 기록됐다.

---

### RU-45. CT token 생성 개선과 v109

- **일자 (Date)**: `2026-08-18` (시작: 2026-08-18T02:14:39, 종료: 2026-08-18T02:27:44)
- **커밋 범위**: Index 356–357 (2개 커밋, `7ddb525a` ... `7456a22c`)
- **작업 유형**: `승격 결정`
- **질문 (Question)**: CT abundance의 병목은 token 생성인가?
- **가설 (Hypothesis)**: FPS token이 밀도 높은 세포 집단을 충분히 대표하지 못하므로 k-means token이 개선한다.
- **실험 및 변경 (Experiment)**: FPS와 k-means token을 비교하고 CT weight를 포함해 CV off-diagonal 변경과 결합한 17-task 평가를 수행했다.
- **관측 결과 (Observations)**: FPS는 16개 token 중 약 2개만 실사용됐고 k-means가 CT-only를 +0.037 회복한 것으로 기록됐다.
- **당시 결정 (Decision)**: CV off-diagonal과 k-means CT weight 0.7 조합을 v109로 채택했다.
- **원문 근거 (Evidence)**:
  - `f2f29ccc:docs/current_status.md#L5981`:
    > "| **token 생성** (farthest-point) | **이것이었다.** 16개 중 1.9개만 사용 → k-means로 CT-only +0.037 (§157) |"
  - `7456a22c:docs/current_status.md#L5826`:
    > "## 158. 2026-08-18 — **v109 승격 확정 (사용자 결정): CV = off-diagonal only, CT = k-means token @ w=0.7**"
- **선행·후속 관계 (Relations)**: E04의 CV 정리와 E06의 cells/tokens 격자로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: k-means 이득에는 후속 λ 확인 전까지 λ 영향이 섞여 있었다.

---

### RU-46. CT 표본 수·token 수·ridge λ의 분리

- **일자 (Date)**: `2026-08-18` (시작: 2026-08-18T03:27:29, 종료: 2026-08-18T12:23:26)
- **커밋 범위**: Index 358–362 (4개 커밋, `f2f29ccc` ... `fb91a069`)
- **작업 유형**: `승격 결정`
- **질문 (Question)**: CT 개선은 더 많은 cells, 더 많은 tokens, 또는 ridge λ 때문인가?
- **가설 (Hypothesis)**: 64-cell 상한을 풀거나 token 수를 늘리면 abundance 해상도가 높아질 수 있다.
- **실험 및 변경 (Experiment)**: 64/256/1024/전체 cells 및 16/32/64/128 tokens 격자를 비교하고 CT ridge λ를 16·32 token에서 재점검했다.
- **관측 결과 (Observations)**: 전체 cell은 모든 token 수에서 64 cell보다 낮았다. 32 tokens·64 cells는 v109 대비 +0.0051, 15/17이었다. λ 최적화를 반영하면 token 이득은 +0.0051에서 +0.0036으로 줄었다.
- **당시 결정 (Decision)**: cells_per_bag=64와 CT tokens=32를 v110으로 채택하고 λ=1을 유지했다.
- **원문 근거 (Evidence)**:
  - `a04f0cf2:docs/current_status.md#L6037`:
    > "**`32 tok, 64 cell`이 +0.0051에 부호 15/17** — 이 세션 전체에서 가장 높은 부호 일치다(v109 승격"
  - `131c95a8:docs/current_status.md#L6100`:
    > "⚠️ **전체 cell은 포함하지 않는다.** §160이 16·32·64·128 token 전부에서 전체 cell이 64 cell보다"
  - `fb91a069:docs/current_status.md#L6265`:
    > "**이득의 약 30%는 λ 효과였다.** 그러나 사라지지 않는다 — 32 token은 어떤 λ에서도 같은 λ의"
- **선행·후속 관계 (Relations)**: E05를 확정하고 E08의 full-cell tokenizer 재평가로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 32와 64 tokens의 차이는 노이즈 가능성으로 남았다.

---

### RU-47. 서버 이동용 실행 환경 정비

- **일자 (Date)**: `2026-08-18` (시작: 2026-08-18T13:47:07, 종료: 2026-08-18T13:47:07)
- **커밋 범위**: Index 363–363 (1개 커밋, `69360766` ... `69360766`)
- **작업 유형**: `인프라 기록`
- **질문 (Question)**: 서버 이동 중에도 평가 설정과 아키텍처 기록을 재현 가능하게 유지할 수 있는가?
- **가설 (Hypothesis)**: node 설정을 한 곳에 두고 handoff·architecture를 갱신하면 설정 드리프트를 줄인다.
- **실험 및 변경 (Experiment)**: node 환경 설정을 중앙화하고 handoff와 architecture 문서를 당시 v110 계보에 맞게 갱신했다.
- **관측 결과 (Observations)**: 모델 비교가 아니라 재개 가능한 실행 경로를 위한 문서·설정 정비였다.
- **당시 결정 (Decision)**: 후속 CT/DD 실험은 중앙 node 설정과 갱신된 handoff를 기준으로 계속했다.
- **원문 근거 (Evidence)**:
  - `69360766:docs/current_status.md#L6300`:
    > "모든 runner가 source 한다. **이미 설정된 환경변수는 덮어쓰지 않고**, 자동 탐지가 실패하면"
- **선행·후속 관계 (Relations)**: E08의 CT 평가와 E09의 GPU worker 실행의 운영 전제다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 성능 주장이나 모델 승격은 이 커밋에 없다.

---

### RU-48. 대체 CT dictionary와 full-cell hierarchical v111

- **일자 (Date)**: `2026-08-18 ~ 2026-08-19` (시작: 2026-08-18T14:38:22, 종료: 2026-08-19T09:36:13)
- **커밋 범위**: Index 364–374 (11개 커밋, `b7d2f784` ... `65a20d9c`)
- **작업 유형**: `탐색 및 운영 선택`
- **질문 (Question)**: sampling 편향을 없앤 full-cell CT dictionary가 예측 성능과 운영 불변성을 함께 줄 수 있는가?
- **가설 (Hypothesis)**: random·density·hierarchical·spherical tokenizer와 full abundance를 비교하면 representation 병목을 가를 수 있다.
- **실험 및 변경 (Experiment)**: random 512/64, hierarchical, HDBSCAN/DBSCAN, PCA 차원 및 spherical clustering을 단계적으로 평가하고 full-cell hierarchical PCA32/K256을 확정했다.
- **관측 결과 (Observations)**: 여러 random·density arm은 v110 유지로 기각됐다. v111은 SEAL 0.70453, 홀드아웃 0.59809, 전체 0.66070으로 v110보다 낮았지만 모든 context/query cell을 썼다.
- **당시 결정 (Decision)**: 사용자 결정으로 v111을 공식 baseline으로 채택하고 CT 분기를 종료했다. 성능 우월이 아닌 sampling 불변성의 운영 선택이었다.
- **원문 근거 (Evidence)**:
  - `65a20d9c:docs/current_status.md#L7056`:
    > "사용자 결정으로 예측 macro가 가장 높은 v110 대신, **cell selection bias가 없고 random sampling의"
  - `65a20d9c:docs/current_status.md#L7072`:
    > "| **v111 (활성)** | **0.70453** | **0.59809** | **0.66070** | **0.00000** |"
  - `65a20d9c:docs/current_status.md#L7076`:
    > "이 승격은 성능 우월성 주장이 아니라 **운영 불변성에 대한 사용자 선택**이다. 전체 cell을 사용하므로"
- **선행·후속 관계 (Relations)**: E06의 v110은 historical predictive best로 남고 E09가 v111을 기준선으로 사용한다.
- **확인 필요 사항 및 한계 (Uncertainties)**: CT 독립 부분공간 등 추가 탐색은 당시 종료 결정으로 제외됐다.

---

### RU-49. DD ordered×typicality 후보의 구현·수정 및 v112 승격

- **일자 (Date)**: `2026-08-19` (시작: 2026-08-19T10:09:32, 종료: 2026-08-19T11:35:05)
- **커밋 범위**: Index 375–380 (6개 커밋, `80f17db7` ... `fc814a3c`)
- **작업 유형**: `가설 검증 및 승격 채택`
- **질문 (Question)**: DD의 코호트 역전을 bounded ordered coordinate와 nearest-class typicality 결합으로 해결하고 일반화를 개선할 수 있는가?
- **가설 (Hypothesis)**: κ=1, weight 1의 bounded evidence가 기존 거리 기반 margin보다 극단치에 안정적이며 양 집단 모두에서 DD 신호를 살릴 수 있다.
- **실험 및 변경 (Experiment)**: src/models/dd_ordered_typicality.py를 구현하고 충돌 가드 및 런처 오류를 수정한 뒤 17-task 공식 50-fold 평가를 완주했다.
- **관측 결과 (Observations)**: v112 전체 macro 0.66211(v111 대비 +0.00141), SEAL 10 0.70432(−0.00021, flat), 홀드아웃 7 0.60181(+0.00372)로 측정됐다.
- **당시 결정 (Decision)**: 사용자 결정으로 v112를 활성 baseline으로 승격하고 기존 distance runner는 재현용으로 보존했다.
- **원문 근거 (Evidence)**:
  - `80f17db7:docs/current_status.md#L7127`:
    > "**318 tests, OK (72.356s)**. **아직 macro를 측정하지 않았으므로 승격하지 않았고 v111 기본"
  - `cec1a154:docs/current_status.md#L7145`:
    > "첫 실행(PID 1981165)은 fold 진입 전 전부 실패했다. 원인은 `scripts/test_pathobench.py`에서 DD"
  - `cec1a154:docs/current_status.md#L7151`:
    > "재실행 직후 부모와 8 worker가 살아 있고 첫 8 task가 `START`된 것을 확인했다. 다음 Action은 launcher의"
  - `fc814a3c:docs/current_status.md#L7199`:
    > "| **v112 (dd weight=1)** | **0.70432** | **0.60181** | **0.66211** |"
  - `fc814a3c:docs/current_status.md#L7236`:
    > "- 활성 runner `scripts/eval_v112.sh` 추가 (v111과 CV/CT 설정 동일, DD만 다름)."
- **선행·후속 관계 (Relations)**: E03의 LLR/상대거리 기각 후속 대안이며, M02의 CT fraction sampling 및 M07의 DD 제거 판정으로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: SEAL 10에서의 미세 하락과 홀드아웃 상승의 트레이드오프가 완전히 해소된 것은 아니었다.

---

### RU-50. CT fraction sampling feasibility

- **일자 (Date)**: `2026-08-20` (시작: 2026-08-20T01:15:15, 종료: 2026-08-20T01:15:15)
- **커밋 범위**: Index 381–381 (1개 커밋, `c727e103` ... `c727e103`)
- **작업 유형**: `feasibility_decision`
- **질문 (Question)**: 22GB GPU에서 full-cell CT가 완주 불가할 때 cell budget을 어떻게 정할 것인가?
- **가설 (Hypothesis)**: bag 크기의 1/8, floor 64이면 OOM을 없애고 성능은 거의 보존한다.
- **실험 및 변경 (Experiment)**: v112와 DD/CV를 고정하고 CT만 full-cell에서 own-bag 1/8 fraction으로 바꾸어 SEAL 10을 재실행했다.
- **관측 결과 (Observations)**: fraction arm은 10 task를 완료했고 macro는 0.70432에서 0.70394로 −0.00038이었다.
- **당시 결정 (Decision)**: 예측 개선이 아니라 평가 가능성을 이유로 v113을 승격했다.
- **원문 근거 (Evidence)**:
  - `c727e103:docs/current_status.md#L7277`:
    > "skipped, 회귀 없음. 이 노드에서 SEAL 10-task 전체를 `eval_v113.sh`로 재실행 — **10개 전부 `rc=0`,"
  - `c727e103:docs/current_status.md#L7292`:
    > "| **SEAL 10 macro** | **0.70432** | **0.70394** | **−0.00038** |"
- **선행·후속 관계 (Relations)**: RU-49, RU-51
- **확인 필요 사항 및 한계 (Uncertainties)**: v113 전체 17-task와 hold-out 7은 당시 재측정되지 않았다.

---

### RU-51. unit fixed-head weight 통일

- **일자 (Date)**: `2026-08-20` (시작: 2026-08-20T09:18:50, 종료: 2026-08-20T09:23:38)
- **커밋 범위**: Index 382–383 (2개 커밋, `7acfd24e` ... `a495a18c`)
- **작업 유형**: `controlled_evaluation`
- **질문 (Question)**: v113의 CV/DD/CT 비대칭 head weight를 모두 1.0으로 통일하면 개선되는가?
- **가설 (Hypothesis)**: 같은 fraction CT·동일 평가에서 unit weight가 저신호 과제를 안정화할 수 있다.
- **실험 및 변경 (Experiment)**: v113의 CT fraction(1/8, floor 64)과 SEAL 10을 고정하고 CV 1.442→1.0, CT 0.7→1.0만 토글했다.
- **관측 결과 (Observations)**: 기록값은 SEAL macro 0.70394→0.70509(+0.00115), 7/10 task 우세였으며 당시 문서는 개선이 저신호 task에 집중됐다고 적었다.
- **당시 결정 (Decision)**: 사용자 결정으로 v114를 승격했다. Primary 7·hold-out 7은 당시 미측정으로 남겼다.
- **원문 근거 (Evidence)**:
  - `7acfd24e:docs/current_status.md#L7393`:
    > "| **SEAL 10 macro (fold-mean)** | **0.70394** | **0.70509** | **+0.00115** |"
  - `a495a18c:docs/current_status.md#L3`:
    > "**Last updated**: `2026-08-20` — §187 사용자 결정으로 **v114 = v113 + fixed-head 세 branch weight(CV/DD/CT)를 전부 1.0으로 통일**을 활성 baseline으로 승격했다(§186 실측). SEAL 10 macro **0.70509**, v113(0.70394) 대비 **+0.00115** — 개선분 대부분이 저신호 task(ccrcc BAP1/VHL)에서 나왔다(§118 기준). 활성 runner `scripts/eval_v114.sh`. `scripts/eval_v113.sh`(weight 비대칭)는 historical 재현 전용으로 유지한다. 다음 세션은 `agent_handoff.md` 맨 위의 **새 세션 60초 재개 절차**에서 시작한다. Python은 `/NHNHOME/WORKSPACE/26msit005_C/kimds/miniconda3/envs/BagPFN/bin/python`을 직접 사용한다. 전체 아키텍처 명세는 `current_architecture.md` **§0**. ⚠️ **결정론적 arm에는 t/p/CI를 쓰지 말 것**(§151-1) — 부호 일치와 독립 집단 재현으로 판정한다."
- **선행·후속 관계 (Relations)**: RU-50, RU-55
- **확인 필요 사항 및 한계 (Uncertainties)**: unit-weight의 Primary 7·hold-out 7은 당시 미측정이다.

---

### RU-52. CT kernel ridge와 top-k pooling 기각

- **일자 (Date)**: `2026-08-20` (시작: 2026-08-20T22:23:12, 종료: 2026-08-20T22:23:12)
- **커밋 범위**: Index 384–384 (1개 커밋, `eee4650c` ... `eee4650c`)
- **작업 유형**: `rejection`
- **질문 (Question)**: CT abundance의 비선형 ridge 또는 extreme pooling이 unit fixed-head CT를 개선하는가?
- **가설 (Hypothesis)**: label-abundance 곡률 또는 소수 cell의 극값 신호가 linear mean ridge보다 유효할 수 있다.
- **실험 및 변경 (Experiment)**: v114 SEAL 10 control에 RBF/poly KRR, top-k 대체, mean+top-k concatenate를 각각 비교했다.
- **관측 결과 (Observations)**: RBF/poly macro는 0.69500/0.69475, top-k VHL은 0.5076, concatenate macro는 0.70067로 v114 0.70509보다 낮았다.
- **당시 결정 (Decision)**: 세 변형을 모두 기각하고 재현 코드만 보존했다.
- **원문 근거 (Evidence)**:
  - `eee4650c:docs/current_status.md#L7473`:
    > "| **SEAL 10 macro (fold-mean)** | **0.70509** | **0.69500** (−0.0101) | **0.69475** (−0.0103) |"
  - `eee4650c:docs/current_status.md#L7514`:
    > "| **SEAL 10 macro (fold-mean)** | **0.70509** | **0.70067** | **−0.00442** |"
- **선행·후속 관계 (Relations)**: RU-51
- **확인 필요 사항 및 한계 (Uncertainties)**: 당시 SEAL 10 기반의 기각이며 현행 CT 제외 기준에는 쓰지 않는다.

---

### RU-53. 모듈 분해와 BM 구현·문서 SSOT

- **일자 (Date)**: `2026-08-21` (시작: 2026-08-21T14:46:57, 종료: 2026-08-21T15:47:35)
- **커밋 범위**: Index 385–394 (10개 커밋, `af004748` ... `41aff6e5`)
- **작업 유형**: `research_infrastructure`
- **질문 (Question)**: 후속 branch 실험을 위한 구현·문서·평가 역할을 재현 가능하게 분리할 수 있는가?
- **가설 (Hypothesis)**: projected BM 구현과 current_status SSOT가 후속 비교의 혼선을 줄인다.
- **실험 및 변경 (Experiment)**: 모듈 분해, projected BM 코드·테스트, 문서 통합/archiving을 수행했다.
- **관측 결과 (Observations)**: 이 묶음은 성능 실험이 아니라 기반 정리다. BM 구현은 leading subspace class-balanced ridge로 명시됐고 current_status가 개발 현황 SSOT로 지정됐다.
- **당시 결정 (Decision)**: M05 이후 BM 성능 비교는 이 구현·문서 기반을 전제로 한다.
- **원문 근거 (Evidence)**:
  - `31ea065d:src/models/training_free.py#L205`:
    > "    # BM branch: projected bag-mean in leading subspace with class-balanced ridge."
  - `af004748:README.md#L28`:
    > "| [`docs/current_status.md`](docs/current_status.md) | 개발 현황 **SSOT** — 최신 수치·커밋·Action Plan |"
- **선행·후속 관계 (Relations)**: RU-55, RU-61
- **확인 필요 사항 및 한계 (Uncertainties)**: 35–45에는 문서/리팩터링이 포함되므로 이를 독립 성능 주장으로 쓰지 않는다.

---

### RU-54. Primary 7과 SEAL hold-out 역할 전환

- **일자 (Date)**: `2026-08-21` (시작: 2026-08-21T17:58:56, 종료: 2026-08-21T17:58:56)
- **커밋 범위**: Index 395–395 (1개 커밋, `b60a8a03` ... `b60a8a03`)
- **작업 유형**: `evaluation_protocol`
- **질문 (Question)**: 모델 선택과 독립 재현을 어떤 task 집합에 분리할 것인가?
- **가설 (Hypothesis)**: SEAL 밖 7개를 primary로, 기존 SEAL 10을 hold-out으로 두면 선택 편향을 줄일 수 있다.
- **실험 및 변경 (Experiment)**: 문서의 판정 프로토콜을 Primary 7, Hold-out 10, sign agreement와 task별 delta 보고로 재정의했다.
- **관측 결과 (Observations)**: 원문은 SEAL에 없던 7개를 primary benchmark로 정하고 기존 SEAL 10을 독립 hold-out validation으로 전환했다고 기록한다.
- **당시 결정 (Decision)**: 후속 branch 평가는 Primary 7로 선택하고 SEAL 10은 독립 검증 역할로 기록했다.
- **원문 근거 (Evidence)**:
  - `b60a8a03:docs/current_status.md#L4`:
    > "- **판정 프로토콜 개정**: SEAL에 없던 7개 과제를 **새로운 Primary Benchmark**로 확립, 기존 SEAL 10-task를 **독립 Hold-out Validation**으로 전환."
  - `b60a8a03:docs/current_status.md#L23`:
    > "  1. **Primary 7-task 부호 일치 수 (Sign Agreement)**: 7개 과제 중 몇 개에서 승리했는가? (예: $\ge 5/7$)"
- **선행·후속 관계 (Relations)**: RU-53, RU-55
- **확인 필요 사항 및 한계 (Uncertainties)**: 현재 규칙은 이 당시 protocol을 후속 정정해 SEAL을 선택에 쓰지 않으며 CT도 제외한다.

---

### RU-55. Projected Bag-Mean(BM) 채택

- **일자 (Date)**: `2026-08-22` (시작: 2026-08-22T15:00:18, 종료: 2026-08-22T15:00:18)
- **커밋 범위**: Index 396–396 (1개 커밋, `42078217` ... `42078217`)
- **작업 유형**: `branch_promotion`
- **질문 (Question)**: 상위 32D PCA bag mean ridge가 v114에 추가 신호를 주는가?
- **가설 (Hypothesis)**: 1차 bag mean이 기존 branch와 결합돼 Primary 7을 개선한다.
- **실험 및 변경 (Experiment)**: BM weight 1.0 v115와 v114를 Primary 7 50-fold로 비교했다.
- **관측 결과 (Observations)**: v114 0.6051→v115 0.6094(+0.0043), 5/7 승으로 기록됐다.
- **당시 결정 (Decision)**: 사용자 승인으로 BM 포함 v115를 승격했다.
- **원문 근거 (Evidence)**:
  - `42078217:docs/current_status.md#L448`:
    > "| **Macro** | **Primary 7-Task Mean** | **0.6051** | **0.6094** | **+0.0043** | **+0.0037** | **v115 승 ($\mathbf{5 / 7}$)** |"
  - `42078217:docs/current_status.md#L4`:
    > "- **v115 승격**: **BM-branch (Bag-Mean leading 32D subspace Class-balanced Dual Ridge, $w_{BM}=1.0$)**가 Primary 7-Task에서 **5/7 과제 승리 및 Macro $\Delta = +0.0043$**을 달성하여 공식 baseline으로 승격 (사용자 최종 승인)."
- **선행·후속 관계 (Relations)**: RU-53, RU-56
- **확인 필요 사항 및 한계 (Uncertainties)**: 현재 승격 규칙은 이 시점의 규칙과 다르다.

---

### RU-56. Bag-dispersion spectral entropy(BD) 채택

- **일자 (Date)**: `2026-08-22` (시작: 2026-08-22T17:11:38, 종료: 2026-08-22T17:11:38)
- **커밋 범위**: Index 397–397 (1개 커밋, `dfcc1c00` ... `dfcc1c00`)
- **작업 유형**: `branch_promotion`
- **질문 (Question)**: spectral entropy가 bag dispersion 신호를 제공하는가?
- **가설 (Hypothesis)**: log-trace보다 scale-invariant spectral entropy가 유효한 ordered-typicality evidence다.
- **실험 및 변경 (Experiment)**: 후보 A log-trace와 후보 B spectral entropy를 v114/v115 control에 대해 Primary 7에서 평가했다.
- **관측 결과 (Observations)**: 후보 A 0.6071, 후보 B 0.6119로 v115보다 +0.0025, 4/7 승으로 기록됐다.
- **당시 결정 (Decision)**: 당시 사용자 승인으로 BD 포함 v116으로 승격했다.
- **원문 근거 (Evidence)**:
  - `dfcc1c00:docs/current_status.md#L151`:
    > "  - Macro AUROC = **0.6071** (v115 0.6094 대비 $\Delta = -0.0023$, 3/7 과제 승리)."
  - `dfcc1c00:docs/current_status.md#L4`:
    > "- **v116 승격**: **BD-branch (Bag Dispersion / Spectral Entropy with Ordered-Typicality Evidence, $w_{BD}=1.0$)**가 Primary 7-Task에서 **Macro 0.6119 (+0.0025 vs v115, +0.0068 vs v114, 4/7 과제 승리)**를 달성하여 공식 baseline으로 승격 (사용자 최종 승인)."
- **선행·후속 관계 (Relations)**: RU-55, RU-57
- **확인 필요 사항 및 한계 (Uncertainties)**: 4/7은 현재 5/7 요건을 못 채우며 당시 사용자 결정으로만 기록한다.

---

### RU-57. DD 제거와 soft-vote v118

- **일자 (Date)**: `2026-08-22` (시작: 2026-08-22T18:19:01, 종료: 2026-08-22T19:10:29)
- **커밋 범위**: Index 398–400 (3개 커밋, `3987dcc9` ... `6deb53ec`)
- **작업 유형**: `aggregation_promotion`
- **질문 (Question)**: DD를 제거하고 4-branch soft voting하면 개선되는가?
- **가설 (Hypothesis)**: DD의 약한 단독 신호를 빼고 soft vote를 쓰면 안정화된다.
- **실험 및 변경 (Experiment)**: DD=0 선형 v117과 CV+CT+BM+BD soft-vote v118을 Primary 7에서 비교하고 SEAL/지도학습 표를 기록했다.
- **관측 결과 (Observations)**: v117은 0.6191(5/7), v118은 0.6205로 v116보다 +0.0086, 6/7이었다.
- **당시 결정 (Decision)**: v118을 활성으로 승격하고 SEAL/지도학습 표는 hold-out 기록으로 보관했다.
- **원문 근거 (Evidence)**:
  - `3987dcc9:docs/current_status.md#L4`:
    > "- **v118 승격**: **4-Branch Soft Voting (CV + CT + BM + BD, $w_{DD}=0.0$)**이 Primary 7-Task에서 **Macro 0.6205 (+0.0086 vs v116, +0.0154 vs v114, 6/7 과제 승리)**를 달성하여 공식 baseline으로 승격 (사용자 최종 확정 지시)."
  - `3987dcc9:docs/current_status.md#L5`:
    > "- **v117 보존**: **DD 제거 4-Branch 선형합 (CV + CT + BM + BD)**은 **Macro 0.6191 (+0.0072 vs v116, 5/7 과제 승리)**로 v117 식별자로 영구 보존."
- **선행·후속 관계 (Relations)**: RU-56, RU-58
- **확인 필요 사항 및 한계 (Uncertainties)**: SEAL은 현행 선택에 쓰지 않는 hold-out이다.

---

### RU-58. QA branch와 trimmed-mean v119

- **일자 (Date)**: `2026-08-22` (시작: 2026-08-22T21:33:01, 종료: 2026-08-22T22:27:04)
- **커밋 범위**: Index 401–402 (2개 커밋, `9dce8c5e` ... `b0ae85b4`)
- **작업 유형**: `branch_and_aggregation_promotion`
- **질문 (Question)**: QA를 추가하고 QA 포함 branch logits를 어떻게 집계할 것인가?
- **가설 (Hypothesis)**: PCA-32 quantile/extremum ridge가 추가 신호를 내며 symmetric trimmed mean이 이를 안정적으로 통합한다.
- **실험 및 변경 (Experiment)**: QA standalone 및 CV를 유지한 5-branch logits을 9개 voting 방식으로 오프라인 비교하고 v119로 평가했다.
- **관측 결과 (Observations)**: QA standalone Primary 7 macro는 0.6117이었고, 5-branch trimmed mean v119는 0.6247로 v118 0.6205보다 +0.0042였다. 별도 no-CV 4-branch 탐색의 0.6275는 후보 구성을 고른 오프라인 최대값이다.
- **당시 결정 (Decision)**: CV를 유지한 QA 5-branch trimmed-mean v119를 당시 기준선으로 선택했다.
- **원문 근거 (Evidence)**:
  - `9dce8c5e:docs/current_status.md#L304`:
    > "- **QA Standalone Primary 7 Macro**: **`0.6117`**"
  - `b0ae85b4:docs/current_status.md#L352`:
    > "| **Macro** | **Primary 7-Task Mean** | **0.6205** | **`0.6247`** | **`+0.0042`** |"
- **선행·후속 관계 (Relations)**: RU-57, RU-60, RU-61
- **확인 필요 사항 및 한계 (Uncertainties)**: v119와 v120은 branch set이 달라 단순 우열 근거가 아니다. 현재 공식 비교 기준은 CT 제외 v121이다.

---

### RU-59. DS salience-denoising과 v120

- **일자 (Date)**: `2026-08-23` (시작: 2026-08-23T00:22:16, 종료: 2026-08-23T00:22:16)
- **커밋 범위**: Index 403–403 (1개 커밋, `a42a0be0` ... `a42a0be0`)
- **작업 유형**: `branch_promotion`
- **질문 (Question)**: class-salience로 denoised bag mean을 만들면 v119 trimmed mean을 개선하는가?
- **가설 (Hypothesis)**: salience reweighting이 비정보 patch를 줄여 독립 branch 신호를 준다.
- **실험 및 변경 (Experiment)**: DS standalone과 DS를 더한 6-branch trimmed mean을 v118/v119 controls와 Primary 7에서 비교했다.
- **관측 결과 (Observations)**: DS 단독 Primary 7 macro는 0.6073이며 v120 6-branch는 v119 0.6247 대비 0.6265(+0.0018)로 기록됐다.
- **당시 결정 (Decision)**: 사용자 결정으로 DS 포함 v120을 당시 baseline으로 승격했다.
- **원문 근거 (Evidence)**:
  - `a42a0be0:docs/current_status.md#L402`:
    > "| **PRIMARY 7 MACRO** | **`0.6073`** | **0.6205** | **0.6247** | **`0.6265`** | **`+0.0018`** |"
  - `a42a0be0:docs/current_status.md#L434`:
    > "- **Aggregation Head**: 6-Branch Trimmed Mean Voting (최고/최저 2개 절사, 중앙 4개 평균)"
- **선행·후속 관계 (Relations)**: RU-58, RU-60
- **확인 필요 사항 및 한계 (Uncertainties)**: 현재 CT 제외 기준과 직접 비교하지 않는다.

---

### RU-60. v120 사후 실패 축과 CT 단독

- **일자 (Date)**: `2026-08-23 ~ 2026-08-24` (시작: 2026-08-23T11:46:20, 종료: 2026-08-24T18:20:05)
- **커밋 범위**: Index 404–406 (3개 커밋, `6faa1c3b` ... `1b17863d`)
- **작업 유형**: `postmortem_and_benchmark`
- **질문 (Question)**: KRR·patch LR·다른 trimming·Fisher가 개선을 주며 CT 단독은 어느 정도인가?
- **가설 (Hypothesis)**: 비선형 readout/patch likelihood/Fisher basis가 추가 신호를 주고 CT 단독은 앙상블 해석 기준이 된다.
- **실험 및 변경 (Experiment)**: KRR, LR+top-k, trimming 변형, Fisher를 Primary 7에서 사후 평가하고 단독 branch 및 CT/v120 표를 기록했다.
- **관측 결과 (Observations)**: KRR 0.6125, LR 0.5874·7-branch 0.6195, Fisher 0.5709로 기각됐다. CT 단독은 Primary 7 0.6147이다.
- **당시 결정 (Decision)**: 네 축을 기각하고 CT는 단독 기준으로 보되 앙상블 대체로 판정하지 않았다.
- **원문 근거 (Evidence)**:
  - `6faa1c3b:docs/current_status.md#L10`:
    > "  1. **Non-Linear KRR (§199)**: Primary 7 0.6125 (기각, Few-shot $N_{ctx}=40$ 과적합으로 SMAD4 0.4290 붕괴)."
  - `1b17863d:docs/current_status.md#L583`:
    > "| **Primary 7 Macro AUROC** | — | **`0.6147`** | **`0.6265`** | **`+0.0118`** | — | **8개 단독 브랜치 중 CT 단독 1위** (앙상블 시 +0.0118 추가 도약) |"
- **선행·후속 관계 (Relations)**: RU-58, RU-61
- **확인 필요 사항 및 한계 (Uncertainties)**: 초기 CT SEAL 0.7197과 후속 상세표 0.6882는 불일치해 이 단위에 SEAL CT 수치를 채택하지 않았다. 현재 CT는 공식 기준선에서 제외된다.

---

### RU-61. v120 harness·arm 모듈화와 결정성 경계

- **일자 (Date)**: `2026-08-30` (시작: 2026-08-30T23:23:07, 종료: 2026-08-30T23:28:48)
- **커밋 범위**: Index 407–412 (6개 커밋, `d858deea` ... `dd06c931`)
- **작업 유형**: `reproducibility`
- **질문 (Question)**: v120을 harness에서 허용 범위로 재현하고 arm export를 모듈화해도 동등성을 보장할 수 있는가?
- **가설 (Hypothesis)**: 0.6265±0.005와 env parity를 acceptance condition으로 두면 리팩터링을 검증할 수 있다.
- **실험 및 변경 (Experiment)**: baseline harness, arms.sh, DE/SW·LOO tooling, AGENTS 규칙과 determinism gate report를 추가했다.
- **관측 결과 (Observations)**: baseline 0.621643, modularized arm 0.621614는 tolerance 안이지만 나중 report는 metrics hash 불일치를 기록했다.
- **당시 결정 (Decision)**: tolerance 통과는 engineering check로만 인정하고 bf16 drift를 새 성능·결정성 주장으로 쓰지 않았다.
- **원문 근거 (Evidence)**:
  - `2c08102d:experiments/baseline/report.md#L31`:
    > "| primary7_macro_fold_mean_auroc | 0.621643 | `${HARNESS_RESULTS_DIR}/metrics.json`: `primary7.macro_fold_mean_auroc` |"
  - `6ae016b1:experiments/modularize-arms/report.md#L32`:
    > "| env_parity_all_arms | 1 | `${HARNESS_RESULTS_DIR}/parity.json`: `parity.all_identical` |"
  - `dd06c931:experiments/baseline/report.md#L53`:
    > "- nondeterministic: metrics.json: run 1 81b5c66251d0… != run 2 06f9da2d3809…"
- **선행·후속 관계 (Relations)**: RU-58, RU-62
- **확인 필요 사항 및 한계 (Uncertainties)**: DE/SW/LOO는 tooling 추가이며 결과 연구단위로 합치지 않았다.

---

### RU-62. obsolete harness 제거와 학습 config 아카이브

- **일자 (Date)**: `2026-09-02` (시작: 2026-09-02T15:35:15, 종료: 2026-09-02T15:55:28)
- **커밋 범위**: Index 413–414 (2개 커밋, `83c5664e` ... `10484d9a`)
- **작업 유형**: `configuration_cleanup`
- **질문 (Question)**: 0-parameter v120에서 남은 harness·학습 config가 활성 entry point를 혼동시키는가?
- **가설 (Hypothesis)**: 미참조 도구와 학습 계보를 archive로 옮기면 활성 경로가 명료해진다.
- **실험 및 변경 (Experiment)**: obsolete harness task를 제거하고 v77–v105 config를 archive로 이관, 미참조 group을 삭제하고 정적 검증했다.
- **관측 결과 (Observations)**: v120은 train_v98 config 하나를 로드한다고 기록되며 26개 config 이관, 23개 group·5개 harness yaml 삭제, dangling 참조 0건이 기록됐다.
- **당시 결정 (Decision)**: 학습 계보는 archive/git history에 보존하고 활성 v120과 분리했다.
- **원문 근거 (Evidence)**:
  - `10484d9a:docs/current_status.md#L641`:
    > "| v120 활성 경로가 로드하는 yaml | **`configs/train_v98_p1_reverse_1536_1gpu.yaml` 단 1개** ([`scripts/node_env.sh`](../scripts/node_env.sh) `ICF_CONFIG` 기본값) |"
  - `10484d9a:docs/current_status.md#L670`:
    > "- **dangling 참조**: 코드/Living 문서에서 존재하지 않는 config 경로 참조 0건"
- **선행·후속 관계 (Relations)**: RU-61
- **확인 필요 사항 및 한계 (Uncertainties)**: 당시 BagPFN env 소실로 regression suite는 정적 검증으로 대체됐다.

---

### RU-63. 재현 기반 정비

- **일자 (Date)**: `2026-09-02 ~ 2026-09-03` (시작: 2026-09-02T16:38:49, 종료: 2026-09-03T12:34:34)
- **커밋 범위**: Index 415–425 (11개 커밋, `fd1095a6` ... `03d4e412`)
- **작업 유형**: `infrastructure`
- **질문 (Question)**: 잃어버린 환경과 단일 파일 구현 아래에서 비교를 재현할 수 있는가?
- **가설 (Hypothesis)**: venv·SSOT·모듈·계약 테스트가 기준선 비교를 반복 가능하게 한다.
- **실험 및 변경 (Experiment)**: 환경·handoff·archive를 정리하고 branch/common/aggregation 분리, YAML 계약 테스트와 회귀 실행기를 추가했다.
- **관측 결과 (Observations)**: 17 모듈/131 테스트가 18.9초에 통과했다.
- **당시 결정 (Decision)**: 성능 승격이 아닌 이후 비교의 재현 기반으로 채택했다.
- **원문 근거 (Evidence)**:
  - `1319aa05:docs/current_status.md#L27`:
    > "All 17 modules / 131 unit tests pass in 18.9s."
- **선행·후속 관계 (Relations)**: RU-64 이후 비교의 기반
- **확인 필요 사항 및 한계 (Uncertainties)**: 과거 CT 수치는 R02에서 정정

---

### RU-64. §209 CT 전사 오류와 단독 기준선

- **일자 (Date)**: `2026-09-03` (시작: 2026-09-03T14:30:37, 종료: 2026-09-03T14:30:37)
- **커밋 범위**: Index 426–426 (1개 커밋, `d74bb0c7` ... `d74bb0c7`)
- **작업 유형**: `benchmark/correction`
- **질문 (Question)**: CT가 단독 최상이며 SEAL 0.7197이라는 기록은 맞는가?
- **가설 (Hypothesis)**: 8개 단독 50-fold가 순위와 CT 병목을 확정한다.
- **실험 및 변경 (Experiment)**: 8개 branch Primary 7과 시간을 대조했다.
- **관측 결과 (Observations)**: DS .6265, QA .6209, CT .6147이고 CT SEAL은 .6882였다.
- **당시 결정 (Decision)**: CT를 단독 챔피언으로 취급하지 않고 비용을 기준 설정에 반영했다.
- **원문 근거 (Evidence)**:
  - `d74bb0c7:docs/current_status.md#L40`:
    > "- **CT Standalone Metric**: 과거 요약표의 `SEAL 10 0.7197`은 허위/오기록이며 실측치는 `0.6882`임. Primary 7 단독 1위는 CT(`0.6147`)가 아닌 DS(`0.6265`)임."
- **선행·후속 관계 (Relations)**: RU-68 CT-제외 공식 비교
- **확인 필요 사항 및 한계 (Uncertainties)**: SEAL은 선택 근거가 아닌 hold-out

---

### RU-65. §210 sub-bag/TTA의 과제별 상충

- **일자 (Date)**: `2026-09-03` (시작: 2026-09-03T17:39:19, 종료: 2026-09-03T17:39:19)
- **커밋 범위**: Index 427–427 (1개 커밋, `976fd5ab` ... `976fd5ab`)
- **작업 유형**: `experiment`
- **질문 (Question)**: sub-bag 증강이 형태학적 변이와 국소 변이를 함께 개선하는가?
- **가설 (Hypothesis)**: context 증강 또는 query TTA가 ARID1A·Grade를 개선한다.
- **실험 및 변경 (Experiment)**: 두 증강 방식을 Primary 7에서 비교했다.
- **관측 결과 (Observations)**: ARID1A .5471→.6179, Grade .6823→.7024였지만 KRAS .7295→.6395였다.
- **당시 결정 (Decision)**: 국소 변이를 보존한다는 anchor 가설을 이월했다.
- **원문 근거 (Evidence)**:
  - `976fd5ab:docs/current_status.md#L24`:
    > "  - **발견 2 (병리학적 한계 규명)**: 2~5% 면적의 국소 변이 세포에 의존하는 `KRAS` (0.7295 $\to$ 0.6395), `KEAP1`, `PBRM1`은 무작위 균일 샘플링 시 변이 패치가 누락되는 False-Negative Sub-bag 현상으로 양성 신호가 희석됨."
- **선행·후속 관계 (Relations)**: RU-68/RU-76/RU-77 anchor 계보
- **확인 필요 사항 및 한계 (Uncertainties)**: 국소 변이 누락은 당시 기전 가설

---

### RU-66. §211 LOO 차원 편향

- **일자 (Date)**: `2026-09-03` (시작: 2026-09-03T19:50:25, 종료: 2026-09-03T19:50:25)
- **커밋 범위**: Index 428–428 (1개 커밋, `401305b1` ... `401305b1`)
- **작업 유형**: `experiment/diagnosis`
- **질문 (Question)**: LOO가 서로 다른 branch/subsample 신뢰도를 고르는가?
- **가설 (Hypothesis)**: LOO가 episode 내 좋은 branch를 선택한다.
- **실험 및 변경 (Experiment)**: 16-branch dual pool과 PRESS LOO를 50-fold로 평가했다.
- **관측 결과 (Observations)**: CV 32640D는 DOF/N 88.1%에서 ctx LOO .94~1.000으로 팽창했다.
- **당시 결정 (Decision)**: 서로 다른 차원의 raw LOO를 신뢰도 비교에 쓰지 않는다.
- **원문 근거 (Evidence)**:
  - `401305b1:docs/current_status.md#L25`:
    > "    - 서로 다른 차원을 가진 브랜치(CV: $32,640\text{D}$ vs DS: $32\text{D}$)를 raw LOO로 비교 시, 고차원 브랜치는 $DOF/N = 88.1\%$의 극단적 암기(Overfitting)로 인해 Context LOO가 $0.94\sim 1.000$의 '가짜 천재' 점수를 얻음."
- **선행·후속 관계 (Relations)**: RU-67/RU-74 LOO 폐기
- **확인 필요 사항 및 한계 (Uncertainties)**: 같은 차원 LOO도 일반화 실패

---

### RU-67. §212 깨끗한 Context LOO 폐기

- **일자 (Date)**: `2026-09-03` (시작: 2026-09-03T20:35:22, 종료: 2026-09-03T20:35:22)
- **커밋 범위**: Index 429–429 (1개 커밋, `6a87a5d1` ... `6a87a5d1`)
- **작업 유형**: `experiment/decision`
- **질문 (Question)**: subsampling 없이 LOO 가중치만 더하면 성능이 개선되는가?
- **가설 (Hypothesis)**: S=1, f=1.0에서 LOO 가중치가 query AUROC를 높인다.
- **실험 및 변경 (Experiment)**: v120 6-branch 50-fold에서 LOO 가중치만 변경했다.
- **관측 결과 (Observations)**: .6265→.6125, 7개 중 5개 하락, ctx LOO/test rho=-.2679였다.
- **당시 결정 (Decision)**: Context LOO를 영구 폐기하고 trimmed mean을 유지했다.
- **원문 근거 (Evidence)**:
  - `6a87a5d1:docs/current_status.md#L24`:
    > "  - **원인 규명**: Context LOO 점수와 실제 Test AUROC 간 스피어만 순위 상관계수가 **음수 ($\rho = -0.2679$)**로 측정됨."
- **선행·후속 관계 (Relations)**: R12가 용량 보정 뒤에도 확정
- **확인 필요 사항 및 한계 (Uncertainties)**: v120은 v121 기준과 직접 비교하지 않음

---

### RU-68. §213 v121 기준선과 anchor 초기 측정

- **일자 (Date)**: `2026-09-03` (시작: 2026-09-03T22:49:19, 종료: 2026-09-03T22:49:19)
- **커밋 범위**: Index 430–430 (1개 커밋, `c3dd5352` ... `c3dd5352`)
- **작업 유형**: `benchmark/experiment`
- **질문 (Question)**: CT를 빼면 빠른 기준선을 만들고 anchor가 DS를 개선하는가?
- **가설 (Hypothesis)**: CT mute는 비용을 줄이고 anchor가 국소 변이를 보존한다.
- **실험 및 변경 (Experiment)**: CT weight=0 5-branch 50-fold와 salience-anchor DS를 실행했다.
- **관측 결과 (Observations)**: v121 macro .6171, 45분→12분. 당시 +7.65%p는 R14에서 +.43%p로 정정됐다.
- **당시 결정 (Decision)**: .6171 5-branch를 공식 비교 기준으로 삼고 anchor 판정은 보류했다.
- **원문 근거 (Evidence)**:
  - `c3dd5352:docs/current_status.md#L22`:
    > "  - 5-Branch (CV, BM, BD, QA, DS) Trimmed Mean Macro AUROC: **`0.6171`** (안정적 고속 평가 환경 확립)."
  - `cab8248c:docs/history/archive.md#L1058`:
    > "> ⚠️ **[§223 정정] 아래 "DS Standalone Full" 열은 DS 단독 성능이 아니라 §212의 v120 6-branch Trimmed Mean 값이 잘못 들어간 것입니다** (7개 과제·macro 전부 §212 표와 동일). 이 열을 기준으로 계산된 "+7.65%p 폭등", "−9.67%p" 등 **모든 증감 수치가 무효**입니다. 통제 비교 실측은 [§223](#§223-salience-anchor-subsampling-효과-정정-및-신호-b-판정)을 보십시오 — 실제 효과는 ARID1A +0.43%p, macro +0.03%p입니다."
- **선행·후속 관계 (Relations)**: RU-76 정정, RU-77 종료 철회
- **확인 필요 사항 및 한계 (Uncertainties)**: SEAL은 기준선 선택에 미사용

---

### RU-69. §214–214-V 집계 탐색 정정

- **일자 (Date)**: `2026-09-03` (시작: 2026-09-03T23:07:25, 종료: 2026-09-03T23:33:52)
- **커밋 범위**: Index 431–432 (2개 커밋, `73601f44` ... `143690a3`)
- **작업 유형**: `experiment/correction`
- **질문 (Question)**: adaptive 집계가 max-drop을 해결하고 승격되는가?
- **가설 (Hypothesis)**: 저장 5-branch 마진의 재집계가 개선한다.
- **실험 및 변경 (Experiment)**: §213 저장 예측 350 fold를 오프라인 재집계하고 sign agreement를 감사했다.
- **관측 결과 (Observations)**: 수치는 재현됐으나 두 방식 4/7, 집계 8종·부분집합 31종에서 5/7은 없었다.
- **당시 결정 (Decision)**: trimmed mean 유지, 집계 축 종료, 보고 무결성 규칙 추가.
- **원문 근거 (Evidence)**:
  - `143690a3:docs/current_status.md#L25`:
    > "- **그러나 승격 기준에 미달합니다.** 두 방식 모두 **sign agreement 4/7** (승격 요건 ≥5/7)."
  - `143690a3:docs/current_status.md#L67`:
    > "→ 이는 개별 후보의 실패가 아니라 **Primary 7 벤치마크가 1%p 부근 차이를 분해하지 못한다**는 뜻입니다. `agent_handoff.md` 불변식 3에 **분해능 하한** 조항으로 등록했고, **집계 함수 변형 탐색은 Closed Axis로 종료**했습니다."
- **선행·후속 관계 (Relations)**: R08부터 구조/게이트 우선
- **확인 필요 사항 및 한계 (Uncertainties)**: 당시 Oracle은 R15에서 판정 기준 폐기

---

### RU-70. §215–217 중복성·RM 기각

- **일자 (Date)**: `2026-09-03 ~ 2026-09-04` (시작: 2026-09-03T23:44:20, 종료: 2026-09-04T00:11:27)
- **커밋 범위**: Index 433–435 (3개 커밋, `a2f0dc6d` ... `e1bb3009`)
- **작업 유형**: `diagnosis/admission`
- **질문 (Question)**: 5개 branch가 독립 신호이고 RM은 새 축인가?
- **가설 (Hypothesis)**: rank와 상관으로 라벨 없이 중복·RM 직교성을 선별한다.
- **실험 및 변경 (Experiment)**: 세 실행 rank와 RM SCREEN_ONLY 350-fold를 대조했다.
- **관측 결과 (Observations)**: rank 2.26/5 반복, RM–CV max |r|=.690, 효율 45→43%.
- **당시 결정 (Decision)**: 성능 미조회로 RM을 기각하고 형상 축만 탐색한다.
- **원문 근거 (Evidence)**:
  - `e1bb3009:docs/current_status.md#L121`:
    > "- **결과: 기각.** max |r| = **0.690 (RM–CV)** > 0.6. 규칙에 따라 **성능은 조회하지 않았습니다**(`rm_screen.py`가 절차적으로 차단). 랭크 효율도 45% → 43%로 하락합니다."
- **선행·후속 관계 (Relations)**: RU-71 게이트 선례; BD 하나뿐은 R16에서 시점 한정 정정
- **확인 필요 사항 및 한계 (Uncertainties)**: rank는 R09에서 필터로 축소

---

### RU-71. §218 SH 채택·BS 기각

- **일자 (Date)**: `2026-09-04` (시작: 2026-09-04T11:47:33, 종료: 2026-09-04T12:07:16)
- **커밋 범위**: Index 436–437 (2개 커밋, `5c2881b5` ... `0b6e5611`)
- **작업 유형**: `admission/decision`
- **질문 (Question)**: 형상 통계가 새 정보를 제공하는가?
- **가설 (Hypothesis)**: SH/BS가 위치 계열과 직교해 rank와 앙상블을 개선한다.
- **실험 및 변경 (Experiment)**: SH·BS 상관/효율, 단독 정보량, 5-branch 확장을 평가했다.
- **관측 결과 (Observations)**: SH max |r|=.418, 7/7>.5; BS 4/7. rank 45→50%여도 macro/oracle 개선 없음.
- **당시 결정 (Decision)**: SH 채택·승격 보류, 직교성+정보량 2단계 게이트 확정.
- **원문 근거 (Evidence)**:
  - `0b6e5611:docs/current_status.md#L154`:
    > "**직교성은 필요조건이지 충분조건이 아니며, 랭크 효율은 성능의 예측자가 아니라 필터입니다.**"
- **선행·후속 관계 (Relations)**: RU-72 SH 변형, RU-78 gate 보완
- **확인 필요 사항 및 한계 (Uncertainties)**: hold-out 미검증

---

### RU-72. §219 SH 변형 반증과 SHJ 당시 보류

- **일자 (Date)**: `2026-09-04` (시작: 2026-09-04T12:14:17, 종료: 2026-09-04T12:40:47)
- **커밋 범위**: Index 438–440 (3개 커밋, `c3476dab` ... `ba9f6a43`)
- **작업 유형**: `experiment`
- **질문 (Question)**: SH 차원·로버스트화·모멘트 분리 또는 SHJ가 정보량을 높이는가?
- **가설 (Hypothesis)**: 강화 SH가 미회수 정보를 회수한다.
- **실험 및 변경 (Experiment)**: 7개 변형을 한 실행(949초)에 별도 마진으로 기록했다.
- **관측 결과 (Observations)**: 세 가설 반증; SHJ ARID1A .6363, 48/50, max |r|≤.227이나 4/7.
- **당시 결정 (Decision)**: 변형 축을 닫고 SHJ 정보는 게이트 개정 판단으로 이월했다.
- **원문 근거 (Evidence)**:
  - `ba9f6a43:docs/current_status.md#L154`:
    > "SHJ는 KRAS `0.4823`, Prog `0.4889`, PBRM1 `0.4797`로 **3개 과제에서 우연 이하**(단독>0.5가 4/7)이며, §218에서 사전 선언한 게이트 ②를 통과하지 못합니다."
- **선행·후속 관계 (Relations)**: RU-73 과제 특화 채택, RU-77 oracle 폐기
- **확인 필요 사항 및 한계 (Uncertainties)**: oracle 이동은 이후 판정 기준 아님

---

### RU-73. §220 과제 특화 gate②b와 SHJ 채택

- **일자 (Date)**: `2026-09-04` (시작: 2026-09-04T14:05:05, 종료: 2026-09-04T14:05:05)
- **커밋 범위**: Index 441–441 (1개 커밋, `749cbff4` ... `749cbff4`)
- **작업 유형**: `policy/admission`
- **질문 (Question)**: 전 과제 >.5 규칙이 과제 특화 직교 신호를 부당 배제하는가?
- **가설 (Hypothesis)**: 기존 branch 대칭성이 과제 특화 통로를 정당화한다.
- **실험 및 변경 (Experiment)**: ②a를 기존 5개에 소급하고 SHJ 재현성과 상관을 대조했다.
- **관측 결과 (Observations)**: 기존 5개도 ②a 전부 불통과; SHJ는 48/50, |r|≤.132.
- **당시 결정 (Decision)**: ②b로 SHJ 채택, 3/7이라 공식 구성 승격은 안 했다.
- **원문 근거 (Evidence)**:
  - `749cbff4:docs/history/archive.md#L1542`:
    > "**현행 모델의 5개 브랜치 전부가 ②a를 통과하지 못한다.**"
  - `749cbff4:docs/agent_handoff.md#L66`:
    > "> ②b로 통과한 후보는 archive에 `과제 특화 채택`으로 명시하고, 승격(≥5/7)은 별개 요건으로 남는다."
- **선행·후속 관계 (Relations)**: R15가 oracle 조건을 fold 재현성으로 교체
- **확인 필요 사항 및 한계 (Uncertainties)**: 활용 임계값 미검증

---

### RU-74. §221 용량 보정 뒤 LOO 활용 반증

- **일자 (Date)**: `2026-09-04` (시작: 2026-09-04T14:29:31, 종료: 2026-09-04T14:29:31)
- **커밋 범위**: Index 442–442 (1개 커밋, `a21c227a` ... `a21c227a`)
- **작업 유형**: `experiment/decision`
- **질문 (Question)**: LOO 차원 팽창을 없애면 context 점수가 query 성능을 예측하는가?
- **가설 (Hypothesis)**: 무편향 8D SHJ에서는 context LOO가 예측한다.
- **실험 및 변경 (Experiment)**: context LOO 마진을 저장하고 상관을 검사했다.
- **관측 결과 (Observations)**: 팽창 rho=.667은 확인됐지만 SHJ fold rho=-.091, 차원-ctx/qry rho=.154.
- **당시 결정 (Decision)**: context→query 분포 이동 문제이므로 §212 폐기 확정.
- **원문 근거 (Evidence)**:
  - `a21c227a:docs/current_status.md#L161`:
    > "**편향을 없애도 예측력이 돌아오지 않습니다.** 즉 문제는 낙관 편향이 아니라 **context↔query 분포 이동**입니다(§215 SMAD4 역전과 같은 축). §212 폐기는 용량과 무관하게 확정됩니다."
- **선행·후속 관계 (Relations)**: RU-67 강화, RU-77 선택 신호 부재와 같은 교착
- **확인 필요 사항 및 한계 (Uncertainties)**: 임계값을 평가셋에서 고를 수 없음

---

### RU-75. §222 SHJ 통합과 bf16 수정

- **일자 (Date)**: `2026-09-04` (시작: 2026-09-04T15:24:20, 종료: 2026-09-04T15:24:20)
- **커밋 범위**: Index 443–443 (1개 커밋, `d6b0a9ce` ... `d6b0a9ce`)
- **작업 유형**: `implementation/verification`
- **질문 (Question)**: SHJ를 안전하게 정식 통합할 수 있는가?
- **가설 (Hypothesis)**: fp32 백색화가 bf16 오차 증폭 없이 §220을 보존한다.
- **실험 및 변경 (Experiment)**: shj.py·config·집계·테스트를 연결하고 autocast를 대조했다.
- **관측 결과 (Observations)**: bf16 오차 약 100배, fp32 ARID1A .6403(48/50).
- **당시 결정 (Decision)**: fp32 백색화를 강제하고 채택을 유지했다.
- **원문 근거 (Evidence)**:
  - `d6b0a9ce:docs/history/archive.md#L1648`:
    > "- 원인은 평가 파이프라인이 **bf16 autocast** 안에서 실행된다는 점이다. 투영 행렬곱이 bf16(상대오차 ~1e-3)으로 계산되고, **백색화의 `eigvals.clamp_min(1e-8).rsqrt()`가 이를 약 100배 증폭**한다."
  - `d6b0a9ce:docs/history/archive.md#L1651`:
    > "**수정**: `shj_slide_features()` 내부에서 `torch.autocast(enabled=False)`로 float32를 강제한다. 주변 autocast 상태와 무관하게 재현된다."
- **선행·후속 관계 (Relations)**: RU-73 채택 구현
- **확인 필요 사항 및 한계 (Uncertainties)**: 초기 SHJ 수치는 bf16 경로

---

### RU-76. §223 §213 baseline 오류와 당시 축 종료

- **일자 (Date)**: `2026-09-04` (시작: 2026-09-04T15:36:58, 종료: 2026-09-04T15:36:58)
- **커밋 범위**: Index 444–444 (1개 커밋, `cab8248c` ... `cab8248c`)
- **작업 유형**: `correction/decision`
- **질문 (Question)**: §213 anchor +7.65%p가 동일 조건 DS full 비교인가?
- **가설 (Hypothesis)**: 통제 baseline을 바로잡으면 효과와 축 지속 여부를 판정할 수 있다.
- **실험 및 변경 (Experiment)**: §213 DS Standalone Full 열 계보를 감사했다.
- **관측 결과 (Observations)**: 그 열은 §212 v120 6-branch, 실제 효과 +.43%p.
- **당시 결정 (Decision)**: 당시 anchor 축 종료와 통제 비교 의무를 기록했다.
- **원문 근거 (Evidence)**:
  - `cab8248c:docs/history/archive.md#L1058`:
    > "> ⚠️ **[§223 정정] 아래 "DS Standalone Full" 열은 DS 단독 성능이 아니라 §212의 v120 6-branch Trimmed Mean 값이 잘못 들어간 것입니다**"
- **선행·후속 관계 (Relations)**: R15가 종료 철회
- **확인 필요 사항 및 한계 (Uncertainties)**: 종료는 R15에서 뒤집힘

---

### RU-77. §224 Oracle 폐기와 §225 dose-response

- **일자 (Date)**: `2026-09-04` (시작: 2026-09-04T18:39:41, 종료: 2026-09-04T19:18:00)
- **커밋 범위**: Index 445–446 (2개 커밋, `8c8bfc5d` ... `0e774f05`)
- **작업 유형**: `policy/experiment`
- **질문 (Question)**: oracle 비교를 쓰며 anchor 강도에 전역 이득이 있는가?
- **가설 (Hypothesis)**: oracle은 건전한 기준이고 §213 강도가 효과를 대표한다.
- **실험 및 변경 (Experiment)**: split-half oracle 편향 및 full/74.5/40.5/15% 강도를 대조했다.
- **관측 결과 (Observations)**: oracle +.0063 선택 편향; ARID1A +2.97%p와 PBRM1 -6.64%p 단조 상충.
- **당시 결정 (Decision)**: §223 종료는 철회하지만 전역 이득 없고 draw 안정성은 기각.
- **원문 근거 (Evidence)**:
  - `8c8bfc5d:docs/agent_handoff.md#L61`:
    > "이 비교는 **평가에 쓰는 데이터로 브랜치를 고르므로 승자의 저주로 상향 편향**되며(split-half 측정 결과 편향 +0.0063, 브랜치 성능이 근접한 ARID1A에서는 +0.0144), 애초에 상한도 아니다"
  - `0e774f05:docs/current_status.md#L161`:
    > "**4개 과제에서 완벽한 단조 용량반응**(|ρ|=1.000): ARID1A는 강할수록 상승(+2.97%p), KEAP1·KRAS·Prog는 단조 하락, PBRM1이 최대 손실(−6.64%p)."
- **선행·후속 관계 (Relations)**: RU-68/RU-76 anchor 해석 정정
- **확인 필요 사항 및 한계 (Uncertainties)**: 과제별 선택 신호 없음

---

### RU-78. §226 gate① 결함 수정과 진행 중 방향 라운드

- **일자 (Date)**: `2026-09-05` (시작: 2026-09-05T11:22:59, 종료: 2026-09-05T11:42:32)
- **커밋 범위**: Index 447–449 (3개 커밋, `5d2fa04e` ... `f3f0b201`)
- **작업 유형**: `도구 수정 및 진행 중 연구방향 라운드 (진행 중)`
- **질문 (Question)**: 후보가 채택 SH/SHJ와 상관 심사를 받고 다음 후보를 정할 수 있는가?
- **가설 (Hypothesis)**: adopted를 기준 집합에 넣고 형상 후보를 교차 검토하면 무단 통과를 막는다.
- **실험 및 변경 (Experiment)**: branch_screen에 adopted 기본값을 추가하고 20개 제안을 교차 채점했다; 모델 실험은 하지 않았다.
- **관측 결과 (Observations)**: 기존 gate①은 SH/SHJ를 재지 않았다. AKS·MDX·LID는 권고이나 동일 배치·독립 비교 부족 및 GPU 차단.
- **당시 결정 (Decision)**: gate①은 SH·SHJ 포함. 제안은 실험 결과가 아닌 진행 중 목록이며 규범/적대 검증 뒤에만 실행 판단.
- **원문 근거 (Evidence)**:
  - `f3f0b201:docs/current_status.md#L34`:
    > "- **게이트 ① 도구 결함 수정** (`scripts/analysis/branch_screen.py`). STEP 1의 기준 집합이 `BRANCHES`(공식 5-branch) + 형제 후보뿐이라, **이미 채택된 `SH`(§218)·`SHJ`(§220)와의 상관을 한 번도 재지 않았다.**"
  - `f3f0b201:docs/current_status.md#L40`:
    > "- **단, 세 건 모두 동일 에이전트(claude_shape) 한 배치 산출이고 자사 채점 제외 구조상 Claude 평가를 받지 않았다.** 제안 품질이 아니라 한 에이전트의 문서 작성 규약 숙련도가 점수에 반영됐을 가능성은 배제되지 않았다. 독립 비교군이 필요하다."
- **선행·후속 관계 (Relations)**: RU-70 screen 선례와 RU-71/RU-73 채택 형상 계열 연결
- **확인 필요 사항 및 한계 (Uncertainties)**: AKS·MDX·LID 미실험; GPU blocker와 fatal 판정 충돌

---

## 4. 종합 고찰 및 역사적 연구 교훈 (Epilogue & Synthesis)

ICF 프로젝트 450개 커밋의 역사적 궤적은 기계학습 모델 개발에서 매우 흔하게 발생하는 **'복잡성 과적합의 함정'을 엄격한 진단과 반증주의(falsificationism)를 통해 격파해 나간 모범적 과학적 여정**을 보여줍니다.

### 1. 가설 기각(Rejection)이 견인한 진정한 진보
- **외부 검색 계층(Retrieval) 폐기 (RU-08)**: 초기 거대 검색 풀 가설은 O(N) 비용 대비 통계적 혜택이 없음을 확인하고 조기 퇴출되었습니다.
- **소형 Bag 소멸 진단과 B1+B2 상호필수 (RU-17)**: Musk 전이 실패의 원인이 모델 용량이 아니라 centering에 의한 작은 bag의 계수 결핍(rank deficiency)임을 밝혀내고, `poolz_l2`와 Cardinality-faithful sampling을 결합해 도약했습니다.
- **§68 죽은 분기(Q-5) 적발과 CV-only 대전환 (RU-25)**: 복잡한 attention 및 rare instance 분기가 실제로는 상수값만을 출력하며 공분산 능선(ridge)의 성능을 깎아먹고 있음을 진단하여, 무려 11,000줄 이상의 불필요 코드를 일거에 정리했습니다.
- **딥러닝 학습에서 Training-Free 기저로의 도약 (RU-38, RU-40)**: 수백 에포크의 GPU 분산 학습보다 Within-context PCA와 closed-form ridge 판별기가 오히려 일반화와 OOD 평가에서 월등함을 확인하며 경량 고속 아키텍처로 안착했습니다.

### 2. 엄격한 평가 규범과 2단계 다면 게이트(Multi-faceted Gate)
- 단일 task의 우연한 향상이나 통계적 검정력 부족(t/p-value 오용)을 지양하고, **Primary 7 데이터셋 일관성(부호 일치 6/7 이상), 홀드아웃 검증, 그리고 코사인 직교성(0.20 미만) + 유효 랭크(1.50 이상)의 2단계 게이트**를 정립함으로써 연구의 재현 가능성을 극대화했습니다.

### RU-80. Primary 7 판정 설계의 표준오차와 MDE 산출

- **일자 (Date)**: `2026-09-06` ~ `2026-09-06`
- **커밋 범위**: - (`fed616f2` ... `fed616f2`)
- **작업 유형**: `reproduction_measurement`
- **질문 (Question)**: Primary 7(7과제 × 50fold = 350 표본) 대응 비교 설계에서 macro Δ의 표준오차와 최소검출효과(MDE)는 얼마인가? 현행 승격 컷오프 +1.0%p는 이 설계의 판별력과 어떤 관계에 있는가?
- **가설 (Hypothesis)**: 해당 없음 (측정 정밀도 산출). 경쟁 가설이 아니라 추정 대상은 'Primary 7이 대표하는 과제·fold 분포에서의 기대 macro AUROC'이며, 그 추정량의 변동성을 잰다.
- **판정 기준 (Criteria, 사전 고정)**: 완료 기준 — ① 5개 branch 각각의 350-fold AUROC 표준편차 sigma_b 산출, ② sigma = mean(sigma_b) 산출, ③ SE = sigma/sqrt(350)와 MDE = (z.975+z.80)*SE 산출, ④ 보조로 실제 후보-기준선 쌍의 per-fold Δ 표준편차에서 대응 SE·MDE 산출, ⑤ fold가 과제 내에서 독립이 아니므로 과제 군집(7 clusters) SE를 함께 보고. 수치가 산출되면 완료이며, 어떤 값이 나와야 성공인 것은 아니다.
- **예산 / 중단 조건**: CPU 노드 1 job, 30분 이내. GPU 0h. 기존 predictions/*.pt 재집계만 사용하고 새 추론을 하지 않는다. / predictions 파일의 fold 수나 과제 구성이 350 표본 가정과 다르면 즉시 중단하고 실제 구조를 먼저 보고한다.
- **실험 및 변경 (Experiment)**: scripts/analysis/decision_precision.py 신설. predictions/pathobench_<task>_<tag>_official50_bf16.pt 를 오프라인 재집계해 (1) 5-branch 각각의 350-fold AUROC 산포, (2) 후보-기준선 per-fold Δ의 표준편차, (3) 과제 군집 SE를 산출했다. slurm job 131295 (node3, CPU 8, 13초, GPU 0h).
- **관측 결과 (Observations)**: N=350 확인 (7 과제 × 50 fold). 산출1: σ_b = 0.1587/0.1567/0.1347/0.1677/0.1568 (cv/bm/bd/qa/ds), σ = mean = 0.1549, SE = 0.828%p, MDE = 2.32%p. 다만 이 σ는 과제 난이도 차이를 대부분 포함한다 (fold AUROC 범위 0.12~1.00). 산출2: 비교 태그 5건 중 shape_screen·sh_variants·rm_screen 3건은 Δ가 정확히 0 — SCREEN_ONLY가 앙상블을 건드리지 않은 것으로 §3.4 오염 검사가 설계대로 작동한 증거이며 판별력 추정에는 정보가 없어 제외했다. 정보 있는 2건(ds_sweep s_d=0.0043, loo_capacity s_d=0.0023)으로 s_d = 0.0033 (σ의 1/47), 대응 SE 독립 0.018%p / 과제 군집 0.022%p, MDE(군집 t, df=6) = 0.072%p. 산출3: 절대 macro SE는 독립 가정 0.849%p 대 과제 군집 4.342%p로 설계 효과 5.12배. Δ에서는 설계 효과가 약 1.2배로 줄어든다.
- **결정 (Decision)**: 증거 판정: 지지 (측정 완료). 운영 결정: 채택 — 사용자 결정으로 PROJECT.md §4·§4.1을 재작성했다. 현행 컷오프 +1.0%p는 실측 MDE의 14~20배였다. 판정 규칙을 'macro +1.0%p AND sign agreement ≥5/7'에서 'per-fold Δ의 과제 군집 95% 구간 하한 > δ_min = +0.3%p'로 교체하고 sign agreement를 보조 지표로 강등했다. D-003은 폐기하지 않고 '같은 실행 반복으로 얻은 동일 점수에 표본 추론 금지'로 적용 범위를 좁혀 공존시켰다 — fold는 시드 반복이 아니라 데이터 표집 단위다. 결정 레코드 D-018.
- **결과별 후속 행동**: MDE < 1.0%p: 현행 컷오프가 판별력보다 높다는 뜻 — 사용자와 새 컷오프를 정하고 PROJECT.md §4.1을 재작성한다. MDE > 1.0%p: 현 설계로는 1.0%p 효과도 안정 검출이 어렵다는 뜻 — 후보 탐색보다 평가 설계 개선을 우선순위로 올리고 사용자 판단을 받는다. 판별 불가/실행 무효: 데이터 구조 문제를 먼저 해소한다.
- **원문 근거 (Evidence)**:
  - slurm job 131295 · node3 · 2026-09-06
  - slurm_outputs/2026-09-06/1752/ru80_precision-131295.out
  - scripts/analysis/decision_precision.py
  - docs/decisions.md D-018
- **선행·후속 관계 (Relations)**: D-017(보편 규범 채택)이 촉발했다. D-003의 적용 범위를 좁히고 D-016의 소급 적용 금지 원칙을 따른다. RU-79(문서 체계 재구성)와 같은 세션.
- **확인 필요 사항 및 한계 (Uncertainties)**: s_d 추정이 실제 후보 비교 2건에만 기반한다 — 더 큰 변경을 가하는 후보에서는 s_d가 커질 수 있다. 과제 군집이 7개뿐이라 t(df=6) 구간이 넓고, 군집 SE 추정 자체의 불확실성이 크다. MDE는 정규 근사 계획값이며 작은 표본·다중 비교에는 그대로 적용되지 않는다. v115~v120은 당시 구간 추정을 하지 않았으므로 새 규칙으로 소급 재판정하지 않았다 — 현행 구성의 검증 필요성은 남는다.

---

### RU-79. 문서 체계 전면 재구성 및 RU 프로세스 도입

- **일자 (Date)**: `2026-09-06` ~ `2026-09-06`
- **커밋 범위**: - (`f3f0b201` ... `ca2c0a85`)
- **작업 유형**: `research_infrastructure`
- **질문 (Question)**: 진행이 반복 작업이 아니라 방향성 있는 축적이 되도록, 문서·진행 체계를 어떻게 바꿔야 하는가?
- **가설 (Hypothesis)**: 정본 분산·닫힌 축 경계 부재·정정의 주석 누적 세 가지가 반복의 구조적 원인이며, 정본 분리 + 경계 레지스트리 + RU 카드 + 정합성 테스트로 차단할 수 있다.
- **판정 기준 (Criteria, 사전 고정)**: ① 기준 수치가 PROJECT.md 밖에서 선언되면 테스트가 실패할 것. ② closed_axes.md의 모든 닫힌 축이 기각 기전·경계 안·경계 밖·재개 조건 네 필드를 갖출 것. ③ current_status.md가 80줄 이하일 것. ④ 137개 기존 회귀 테스트가 그대로 통과할 것.
- **예산 / 중단 조건**: GPU 0h. 1개 세션. / 기존 137개 회귀 테스트가 문서 변경으로 깨지면 즉시 중단하고 원인을 먼저 해소한다.
- **실험 및 변경 (Experiment)**: 정본 문서 체계·닫힌 축 경계·RU 프로세스·문서 정합성 검사를 정비했다. 최종 단계에서 사용자 결정 D-021·D-022를 PROJECT.md, closed_axes.md, 연구 방향, 인수인계, 발표 메모에 반영하고 역사 원문에는 현행 기준 안내를 추가했다. GPU 실험 없이 nexgem-s1에서 CPU 회귀 테스트를 실행했다.
- **관측 결과 (Observations)**: CUDA_VISIBLE_DEVICES 빈 값으로 bash scripts/run_tests.sh 실행: 147 tests, 23.348s, OK. 원래 137개 회귀에 문서 정합성 등이 포함된 현행 스위트 통과. closed_axes.md의 모든 CLOSED 축에 필수 경계 필드가 존재하며 경계 미확정 4건은 모두 비저촉으로 확정. current_status.md 80줄 이하. git diff --check 통과. 신규 성능 수치 산출 없음.
- **결정 (Decision)**: 검증 범위의 증거 판정: 지지 — 문서·프로세스 완료 조건 충족. 운영 결정: 정비 체계 채택, RU-79 종료. 반복 실패를 실제로 감소시키는 장기 효과와 기전은 미확인. 다음 연구 방향은 사용자가 설정한다.
- **결과별 후속 행동**: pass: 사용자가 다음 연구 방향(현재 목표)을 설정하고 그 RU 카드를 연다. fail: 실패한 정합성 항목만 좁혀서 재작업하고, 연구 방향 설정은 그대로 진행한다.
- **원문 근거 (Evidence)**:
  - docs/decisions.md D-015·D-018·D-019·D-020·D-021·D-022
  - docs/PROJECT.md §2~§5
  - docs/closed_axes.md §2~§3
  - tests/test_docs_consistency.py
  - logs/ru79/docs_alignment_tests.log (nexgem-s1, CPU, 2026-09-06; 로컬 로그, git 비추적)
- **선행·후속 관계 (Relations)**: RU-80의 판정 설계 정정을 반영. 후속 연구 RU는 사용자 방향 설정 후 등록.
- **확인 필요 사항 및 한계 (Uncertainties)**: 문서 정합성 통과는 연구 성능·기전의 입증이 아니다. 후보 성능과 hold-out은 미검증. 사전 카드의 research_infrastructure 유형·137 tests 기준·1개 세션 예산은 원문 보존: 문서 인프라 작업으로 여러 세션에 걸쳐 종료했고 GPU 사용은 0h였다. 종료 시 HEAD ca2c0a8 위 미커밋 문서 변경 포함; 원문 사전 기준을 소급 수정하지 않았다. 현재 호스트는 nexgem-s1이며 원격 Slurm 상태는 미확인.

---

### RU-81. 기존 Ridge 브랜치의 정규화 강도와 예측 민감도 진단

- **일자 (Date)**: `2026-09-06` ~ `2026-09-07`
- **커밋 범위**: - (`489dc7f7` ... `489dc7f7`)
- **작업 유형**: `diagnostic`
- **질문 (Question)**: 고정 λ=1이 CV/BM/QA/DS에 서로 다른 수축을 만들어 예측 순위 또는 마진 결합을 제한하는가? readout 개선과 특징 연구의 우선순위를 정한다.
- **가설 (Hypothesis)**: H1: 정규화 변화가 단독 예측 순위와 앙상블을 함께 바꾼다. H2: 단독 순위는 거의 그대로지만 마진 크기 변화가 앙상블을 바꾼다. H3: 정규화 변화에도 출력·성능이 둔감하다. 성능 효과와 기전은 별도로 판정한다.
- **판정 기준 (Criteria, 사전 고정)**: Primary 7 공식 50-fold 전량, 같은 특징·기저·fold·집계에서 Ridge 브랜치 하나만 교체. 각 branch의 weighted centered Gram 양의 고유값 평균을 s로 정의하고 λ=clip(s*[0.1,1,10],1e-4,1e4) 3점과 λ=1 기준을 성능 조회 전 고정. 기준 λ replay max margin abs error<=1e-5, baseline ensemble replay<=1e-6 및 slide/label/fold 일치 필수. branch별 spectrum, effective df, margin RMS, 순위 상관, 단독 AUROC 및 앙상블 대응 Δ·과제 군집 95% t 구간·sign agreement·모든 회귀 보고. 강도별 추세와 순위/크기 반응으로 경쟁 설명을 진단하며 12개 비교의 최댓값을 확증·승격으로 판정하지 않는다. CI는 기술적 진단용이다.
- **예산 / 중단 조건**: 구현·검증 작업 2시간 이내, GPU 합산 최대 2h. 사용자 승인: nexgem-s1에서 Slurm 없이 GPU 0~7 직접 사용. 8-worker 실행 wall time 최대 900초(2 GPU-h 예약 상한), 초과 시 중단. / 기준 λ replay 불일치, 누수·fold/label 불일치, NaN/Inf, GPU 오류 또는 예산 소진 시 중단하고 실행 무효/미완료를 기록한다. 점수를 보고 강도를 추가하지 않는다.
- **실험 및 변경 (Experiment)**: 공식 Primary7의 350 folds를 14 shards로 나눠 s1 GPU0~7에서 직접 실행했다. 각 fold에서 기존 특징을 공유하고 CV/BM/QA/DS 중 하나의 λ만 Context Gram 규모 기반 3점으로 변경했다. 같은 λ replay·원래 경로·merged 산출물 대응을 검증하고, 저장 예측을 독립 pairwise AUROC로 재집계했다. 상세: docs/reports/RU-81_regularization_diagnosis.md.
- **관측 결과 (Observations)**: 350 folds·12개 비교 완료. 680.58s wall, worker 합산 1.2934 GPU-h. 기준 λ=1의 df/rank 평균 CV 0.982 / BM 0.493 / QA 0.294 / DS 0.464. CV 중간 강도: 단독 Δ+1.1972%p·5/7, 앙상블 Δ+0.1304%p·3/7, 군집95%CI[-0.2489,+0.5096]%p. 앙상블 회귀 ARID1A·KEAP1·SMAD4·Progression; 단독 회귀 KEAP1·SMAD4. 12개 앙상블 구간 전부0 포함. 전량 결과·모든 회귀는 보고서 §3. replay/path margin error0, 독립 집계 확률오차1.7882e-7, 원시 재계산 Δ/구간 최대오차3.47e-18. manifest 코드·카드·fold hash 검증 통과.
- **결정 (Decision)**: 증거 판정: 서로 다른 상대적 수축과 정규화에 따른 순위·마진 반응은 지지. 일관된 앙상블 성능 개선은 판별 불가. 순위·크기 변화의 개별 기전 귀속은 미확인. 운영 결정: 기준선 유지, 이3점 진단 종료, λ 처방 보류, 마진 순위·크기 분리 진단을 후속 후보로 제안(D-024). 연구 축은 닫지 않음. hold-out 미검증.
- **결과별 후속 행동**: H1 단서: 고정한 정규화 처방과 별도 검증 계획 제안. H2 단서: 마진 스케일/집계 상호작용 추가 진단. H3 또는 판별 불가: 이 제한된 정규화 경로 보류하고 Context→Query 전이/특징 진단 검토. 실행 무효: 원인을 좁혀 수정하고 예산 내 재실행; 초과 시 사용자에게 보고.
- **원문 근거 (Evidence)**:
  - docs/reports/RU-81_regularization_diagnosis.md
  - docs/decisions.md D-023·D-024
  - predictions/ru81_reg_20260907_r1/manifest.json
  - predictions/ru81_reg_20260907_r1/plan.json (실행 전 카드 원문)
  - predictions/ru81_reg_20260907_r1/summary.json
  - predictions/ru81_reg_20260907_r1/audit.json
  - logs/ru81_regression.log
  - logs/ru81_audit.log
- **선행·후속 관계 (Relations)**: RU-80은 정밀도 기준의 선행. D-022의 P3-LIMIT-CURVE 경로를 좁힌 진단. 후속은 저장 마진에서 순위·크기 효과 분리 진단 초안이며 아직 착수하지 않음.
- **확인 필요 사항 및 한계 (Uncertainties)**: 초기 shell rc2 실패 후 새 tag r1로 재실행; 실패시 GPU 평가/점수 조회 없음. 한도 중단으로 날짜가 바뀌었고 활동 작업시간2h 준수는 미계측; GPU2h 상한은 충족. 순위/크기 개입이 분리되지 않아 H1/H2의 독점적 귀속 불가. 12개 구간은 다중비교 보정 없는 기술적 구간; Primary7 사전 접근과 이번 결과 선택 이력이 있어 승격 확증으로 사용하지 않음. bf16경로·3점 범위·공식fold 조건 한정, hold-out 미검증. 종료시 HEAD489dc7f 위 미커밋 진단 코드·문서 변경 포함; 실행 원본 해시는 manifest에 보존.

---

### RU-82. 람다 앙상블 오프라인 재집계 진단

- **일자 (Date)**: `2026-09-07` ~ `2026-09-07`
- **커밋 범위**: - (`eda748ec` ... `eda748ec`)
- **작업 유형**: `diagnostic`
- **질문 (Question)**: 브랜치별 정규화 강도 λ를 하나로 고르는 대신 여러 λ의 마진을 함께 결합하면, Primary 7 고정 fold(N=350) 대응 비교에서 현행 5브랜치 Trimmed Mean 앙상블 대비 Δ AUROC가 개선 방향으로 나타나는가? RU-81이 이미 산출한 3점 격자 마진만으로 답한다.
- **가설 (Hypothesis)**: H1(주): λ 평균은 브랜치 단독 예측의 λ 민감성을 상쇄해, 앙상블 Δ의 과제 간 분산(7개 과제 평균 Δ의 SD)을 RU-81의 단일 λ 개입 12건 대비 줄인다. 평균 Δ의 부호는 사전에 예측하지 않는다. H2: 현행 sigmoid 결합(B1)에서는 강수축 멤버(RMS 비 0.225~0.357)의 확률이 0.5 근처로 뭉쳐 자동 저가중되므로, 풀 통합(A2)+B1 구성의 평균 Δ는 |Δ| < 0.2%p로 기준선과 거의 구별되지 않는다. H3: 마진 표준화(B2)는 그 자동 저가중을 제거하므로 B1보다 큰 |Δ|와 큰 과제 간 분산을 낸다. 기전 미확인 — 어느 가설이 지지되어도 그것이 λ 앙상블의 인과 기전을 입증하지 않는다.
- **판정 기준 (Criteria, 사전 고정)**: [추정 대상] Primary 7 고정 fold 분포에서 후보와 기준선의 대응 차이 Δ AUROC의 기댓값(PROJECT.md §4.1). Δ는 fold마다 계산해 과제 내 평균 → 과제 7개의 t 구간(df=6). 절대 macro는 판정에 쓰지 않는다.
[기준선] RU-81 실행 ru81_reg_20260907_r1에 저장된 동일 fold의 5브랜치(cv,bm,bd,qa,ds) 기준선 마진. 새 실행이 아니라 같은 자료 내부의 대응 비교다.
[λ 격자] 브랜치당 4점 — 기준선 λ=1(절대) + Context Gram 규모 s의 ×0.1 / ×1 / ×10. bd는 ridge 개입 대상이 아니므로 항상 단일 멤버로 유지한다. 격자를 사후에 늘리거나 줄이지 않는다.
[사전 선언 4구성 — 전량 보고]
  A1×B1 브랜치 내부 λ 4점 평균(확률 평균) 후 5멤버 Trimmed Mean · 현행 sigmoid
  A1×B2 동일하되 각 멤버 마진을 fold 내 query RMS로 나눈 뒤 sigmoid
  A2×B1 (cv,bm,qa,ds)×4점 + bd = 17멤버 단일 Trimmed Mean · 현행 sigmoid
  A2×B2 동일하되 마진 RMS 표준화 적용
[증거 판정 — 구성별로 독립 기록]
  지지: 평균 Δ > 0 이고 95% 구간 하한 > 0 이고 sign agreement ≥ 6/7
  반박: 95% 구간 상한 < 0
  판별 불가: 구간이 0을 포함
  실행 무효: 아래 kill 1~2 발생
다중 비교 보정은 하지 않는다. 4구성을 전부 사전 선언하고 전부 보고하므로 최댓값 선택을 하지 않으며, 어떤 구성도 이 자료에서의 결과를 확증(confirmatory)으로 부르지 않는다. RU-81의 12개 비교 결과를 이미 열람한 상태에서 설계했으므로 이 RU는 전체가 탐색적 진단이다(R1·R4).
- **예산 / 중단 조건**: GPU 0. CPU 오프라인 재집계만 수행하며 새 특징 추출·새 fold·hold-out 접근은 없다. 구현+실행+독립 검산 벽시계 상한 2시간. 실행 자체는 350 folds × 4구성으로 수 분 이내를 예상한다. 상한 초과 시 kill 3. / 1. 저장 마진으로 RU-81 기준선 과제별 AUROC와 12개 비교 중 사전 지정 3건(CV×1, BM×10, DS×0.1)의 앙상블 Δ를 재현하지 못하면(절대오차 > 1e-6) 즉시 중단한다 — 계측 무결성 실패.
2. 비유한 값 발생, 또는 fold/slide ID·label 대응 불일치.
3. 벽시계 2시간 초과.
4. 사전 선언한 4구성 외의 결합 규칙을 추가하고 싶어지면 그 자리에서 실행을 멈추고 카드 개정으로 기록한 뒤에만 진행한다 — 사후 구성 탐색 방지.
- **실험 및 변경 (Experiment)**: RU-81 실행 ru81_reg_20260907_r1의 저장 마진(350 folds, 7과제 × 50 official folds)을 CPU에서 오프라인 재집계했다. 새 특징 추출·새 fold·GPU 사용·hold-out 접근 없음. 구현: scripts/analysis/ru82_lambda_ensemble.py. 사전 선언한 4구성(A1/A2 × B1/B2)만 계산했고 λ 격자는 브랜치당 4점(기준선 λ=1 + s×0.1/×1/×10), bd는 단일 멤버로 유지했다. 무결성 게이트: 기준선 과제별 AUROC 7개와 사전 지정 3개 비교(CV×1, BM×10, DS×0.1)의 평균 Δ 재현. 재집계 wall 2.79초, 스크립트 총 7.7초.
- **관측 결과 (Observations)**: 무결성 게이트 최대 절대오차 0.0 (기준선·3개 비교 모두, 허용치 1e-6). 비유한 값 0건, slide/label 대응 불일치 0건. 전 쌍 비교 방식 AUROC 교차검산 최대 차이 0.0(동일 스크립트 내부).
4구성 결과 — A1×B1 Δ +0.2083%p [-0.2077, +0.6243] sign 5/7 과제SD 0.4498; A1×B2 Δ -0.0405%p [-0.7264, +0.6455] sign 4/7 과제SD 0.7417; A2×B1 Δ -0.0418%p [-1.3917, +1.3080] sign 2/7 과제SD 1.4595; A2×B2 Δ -0.1341%p [-0.9366, +0.6684] sign 3/7 과제SD 0.8677. 4구성 모두 구간이 0을 포함하여 증거 판정은 전부 판별 불가다.
H2는 선언한 수치(|Δ|<0.2%p)는 맞았으나 A2×B1의 과제SD가 4구성 중 최대이고 sign 2/7, ARID1A -2.01%p·KRAS +2.79%p로 진폭이 가장 커서 '변화가 거의 없다'는 함의는 관측과 맞지 않는다. H3는 |Δ|와 과제SD의 방향이 결합 축마다 뒤집혀 지지되지 않았다. H1은 사전 정량 임계를 두지 않았으므로 사후 임계를 만들지 않고 관측만 기술했다 — RU-81 12건의 과제SD 중앙값 0.6371%p 대비 A1 계열은 그 부근, A2 계열은 상위권이다.
- **결정 (Decision)**: 증거 판정: 4구성 모두 판별 불가. 운영 결정: λ 앙상블에 추가 GPU 예산을 투입하지 않고 종료하며 정규화·결합 축은 닫지 않는다. D-024가 지정한 마진 순위·크기 분리 진단으로 복귀한다. 경계 판정은 CA-08 경계 밖(D-022)으로 확인했고 closed_axes.md는 변경하지 않는다. 결정 레코드 D-025.
- **결과별 후속 행동**: [지지 — 1개 이상 구성이 지지] 채택하지 않는다. 같은 자료에서 선택했으므로 확증이 아니다. 조밀한 λ 격자와 브랜치 동시 개입을 포함한 별도 RU를 제안하고, 결합 규칙을 결과 보기 전에 하나로 고정한 뒤 GPU 예산을 사용자 승인으로 확보한다. decisions.md에 탐색 결과와 선택 이력을 남긴다.
[판별 불가 — 4구성 모두] λ 앙상블에 추가 GPU 예산을 투입하지 않고 종료한다. 축은 닫지 않는다(현행 분해능에서 판별 못 한 것이지 무효가 입증된 것이 아니다). D-024의 마진 순위·크기 분리 진단으로 복귀한다.
[반박 — 기준선보다 유의하게 악화] λ 평균이 강수축 멤버를 섞어 기준선을 훼손한다는 관측을 기록하고, closed_axes.md 신규 축 후보로 §3에 올려 사용자 판정을 받는다. 자동으로 축을 닫지 않는다.
[실행 무효] 재현 실패 원인(저장물 손상·집계 코드 불일치)을 먼저 진단한다. 이 경우 RU-81 종료 보고의 수치 신뢰성도 함께 재검토 대상이 된다.
- **원문 근거 (Evidence)**:
  - docs/reports/RU-82_lambda_ensemble.md
  - predictions/ru82_lambda_ens/summary.json (git 비추적)
  - scripts/analysis/ru82_lambda_ensemble.py
  - 출처: predictions/ru81_reg_20260907_r1/*_diag.pt, summary.json (git 비추적)
- **선행·후속 관계 (Relations)**: 선행: RU-81(3점 λ 진단, 저장 마진 제공) · D-024(기준선 유지·λ 처방 보류). 경계 판정: CA-08 경계 밖 — D-022 '브랜치 내부 변형을 결합한다는 형식만으로 폐쇄 축을 적용하지 않는다'. CA-09의 FC 판례(여러 강도의 결합)와 같은 구조. CA-14 비저촉(닫힌 형태 해, 학습 파라미터 0 유지). 동기: CA-R1·§221 교착(라벨 없이 과제별 강도를 고를 선택 신호 부재)을 선택 대신 평균으로 우회하려는 시도.
- **확인 필요 사항 및 한계 (Uncertainties)**: B2의 마진 RMS는 fold의 query 슬라이드 통계로 계산하므로 transductive다. 라벨은 쓰지 않으나 배치 통계 의존이며, 이 성질을 운영 구성으로 승격할 때 별도 검토가 필요하다. 저장 마진은 bf16-mixed 특징 경로 산출값이며 fp32와 최대 0.5%p 차이 가능성이 남아 있다(기존 기술 부채). 교차검산은 동일 스크립트 내부 계산이며 RU-81 수준의 독립 검산이 아니다. 조밀한 λ 격자·브랜치 동시 개입·가중 결합·다른 집계 함수는 미검증으로 남았다. A2×B1처럼 평균이 0에 가까우면서 과제별 진폭이 최대인 구성의 원인이 순위 재배열인지 크기 재가중인지 현재 자료로 구별되지 않는다 — 이는 D-024 후속 진단의 질문과 동일하다.

---

### RU-83. 순위 효과와 크기 효과의 분리 진단

- **일자 (Date)**: `2026-09-07` ~ `2026-09-07`
- **커밋 범위**: - (`14f0c4b1` ... `14f0c4b1`)
- **작업 유형**: `diagnostic`
- **질문 (Question)**: RU-81/RU-82에서 관측된 앙상블 Δ의 변화는 단독 브랜치 예측의 순위 재배열 때문인가, 마진 크기(스케일) 변화로 앙상블 내 상대 가중이 바뀐 것 때문인가? 저장 마진으로 두 효과를 분리해 각각의 앙상블 Δ를 측정한다.
- **가설 (Hypothesis)**: H1(주, 크기 지배): 12개 (브랜치×배율) cell 중 과반(≥7)에서 |Δ_size| > |Δ_rank|다. 근거는 RU-81의 강수축에서 단독 Δ와 앙상블 Δ의 방향이 갈렸다는 관측과, RU-82에서 마진 표준화가 결과를 바꿨다는 관측이다.
H2(가법성): 12 cell 중 과반에서 |Δ_probe - (Δ_size + Δ_rank)| <= 0.5 * max(|Δ_size|, |Δ_rank|)다. 성립하면 두 효과가 근사적으로 분리되고, 붕괴하면 상호작용이 지배한다는 뜻이다.
H3(강수축 강화): 배율 ×10의 4개 cell 전부에서 |Δ_size| > |Δ_rank|다. RMS 비가 0.225~0.357로 가장 작은 구간이기 때문이다.
기전 미확인 — 어느 쪽이 지지되어도 그것이 앙상블 결합 규칙의 인과 기전을 입증하지 않는다.
- **판정 기준 (Criteria, 사전 고정)**: [대조 정의 — 결과 보기 전에 고정]
  기준선 마진 m_base, RU-81의 λ 개입 마진 m_probe (브랜치 b, 배율 f)에 대해
  크기 대조  m_size = m_base * RMS(m_probe)/RMS(m_base)   — 순위 보존, 크기만 m_probe에 맞춘다
  순위 대조  m_rank[i] = sort(m_base)[ rank_of(m_probe[i]) ] — 값 다중집합 보존, 순위만 m_probe를 따른다
[불변식 — 위반 시 kill]
  Spearman(m_base, m_size) = 1 ± 1e-9 이고 RMS(m_size)/RMS(m_probe) = 1 ± 1e-6
  Spearman(m_probe, m_rank) = 1 ± 1e-9 이고 sort(m_rank) == sort(m_base) 완전 일치
  단독 AUROC(m_size) == AUROC(m_base) 그리고 AUROC(m_rank) == AUROC(m_probe), 각각 ± 1e-9
  → 두 대조는 단독 브랜치 성능을 바꾸지 않는다. 변하는 것은 앙상블뿐이다.
[부작용 — 사전 기록]
  m_rank는 슬라이드별 마진 부호를 바꿀 수 있다(순위 변화의 의도된 결과이며 결정 임계 의미가 달라진다).
  마진 동값(tie)이 있으면 순위 배정이 모호하므로 stable argsort를 쓰고 tie 발생 fold 수를 보고한다.
  RMS(m_base)는 1e-12로 clamp한다.
[측정] 12 cell × 3 arm(probe/size/rank) = 36개 앙상블 Δ를 전량 보고한다. Δ는 fold별 계산 → 과제 내 평균 → 과제 7개 t 구간(df=6). 기준선은 같은 fold의 현행 5브랜치 앙상블.
[판정]
  H1 지지: |Δ_size| > |Δ_rank| 인 cell이 >= 7/12. 반박: <= 5/12. 판별 불가: 6/12.
  H2 지지: 가법성 조건을 만족하는 cell이 >= 7/12. 반박: <= 5/12. 판별 불가: 6/12.
  H3 지지: ×10 4개 cell 전부. 반박: 2개 이하. 판별 불가: 3개.
cell 단위 부등호 비교는 점추정 비교이므로 각 Δ의 구간을 함께 보고하고, 구간이 겹치는 cell 수를 명시한다. 다중 비교 보정은 하지 않으며 36개를 전부 사전 선언하고 전부 보고한다. RU-81/RU-82의 결과를 이미 열람했으므로 이 RU는 전체가 탐색적 진단이다(R1·R4).
[카드 개정 · 2026-09-07 · 착수 직후, 결과 관측 전]
원래 불변식 'Spearman(m_probe, m_rank) = 1 ± 1e-9'는 **구성상 달성 불가능했다** — base 마진에 동값이 있으면 순위 대조는 서로 다른 probe 값에 같은 값을 배정하므로 ρ < 1이 된다. 첫 실행이 ρ = 0.99943으로 kill 2에 걸려 중단됐고 **결과는 관측하지 않았다**. 실측 tie는 (fold,브랜치) 1400개 조합 중 718개에서 발생하고 초과 원소 합은 1329개다(전체의 약 1.9%). cv에는 없고 bm·qa·ds에 분포한다. bf16 해상도는 원인이 아니다(bf16 round-trip 오차 비영, 저장 dtype float32).
개정 내용: 순위 대조의 불변식을 **정확한 진술로 교체한다** — probe로 정렬했을 때 m_rank가 비감소이고, 값 다중집합이 base와 완전 일치할 것. 이 둘은 tie와 무관하게 정확히 성립해야 하며 위반은 kill이다. tie 때문에 생기는 Spearman 편차와 |AUROC(m_rank) − AUROC(m_probe)|는 **kill이 아니라 부작용 측정치로 전량 보고한다** (원래 카드가 이미 tie 보고를 요구했다). **H1·H2·H3의 판정 임계는 바꾸지 않았다.** 크기 대조의 불변식도 그대로 유지한다.
- **예산 / 중단 조건**: GPU 0. CPU 오프라인 재집계만 수행하며 새 특징 추출·새 fold·hold-out 접근은 없다. 구현+실행+검산 벽시계 상한 2시간. 실행은 350 folds × 12 cell × 3 arm으로 수 분 이내를 예상한다. / 1. RU-82와 동일한 무결성 게이트 — 기준선 과제별 AUROC 7개와 사전 지정 3개 비교(CV×1, BM×10, DS×0.1)의 평균 Δ를 재현하지 못하면(절대오차 > 1e-6) 즉시 중단.
2. 위 불변식 중 하나라도 허용치를 벗어나면 즉시 중단 — 대조 구성이 잘못됐다는 뜻이다.
3. 비유한 값 발생, 또는 fold/slide ID·label 대응 불일치.
4. 벽시계 2시간 초과.
5. 사전 정의한 2개 대조 외의 arm을 추가하고 싶어지면 실행을 멈추고 카드 개정으로 기록한 뒤에만 진행한다.
- **실험 및 변경 (Experiment)**: RU-81 저장 마진(350 folds)을 CPU에서 재집계해 12개 (브랜치×배율) cell마다 probe/크기/순위 3개 arm의 앙상블 Δ를 계산했다. GPU 0, 실행 wall 15.05초. 구현: scripts/analysis/ru83_rank_size.py. 무결성 게이트는 RU-82와 동일(기준선 AUROC 7개 + 사전 지정 3개 비교 재현). 첫 실행은 kill 2로 중단됐고 불변식을 정확한 진술로 교체한 뒤 재실행했다(카드 개정, 결과 관측 전).
- **관측 결과 (Observations)**: 무결성 게이트 최대 절대오차 0.0. 불변식 전량 통과, 비유한 값 0건.
H1(크기 지배, 임계 >=7/12) 반박 — 크기 우세 4/12, 순위 우세 8/12. QA 3개 cell 전부가 크기 우세이고 CV·BM·DS는 대부분 순위 우세다.
H2(가법성, 임계 >=7/12) 지지 — 7/12. 잔차 |평균| 0.0611%p로 두 효과보다 작다.
H3(×10에서 크기 지배, 임계 4/4) 반박 — 2/4.
|Δ| 평균: probe 0.1374%p, 크기 0.0911%p, 순위 0.1044%p. 전부 승격선 아래다.
**12개 cell 전부에서 Δsize와 Δrank의 95% 구간이 겹친다.** 36개 arm 중 구간이 0을 배제한 것은 DS×1 순위 대조 1건([-0.154, -0.018]%p)뿐이며 다중 비교 보정 없는 값이다. 따라서 H1 반박은 사전 계수 규칙의 적용이고 '순위 지배'의 통계적 확증이 아니다.
부작용: 기준선 마진 동값으로 (fold×브랜치×배율) 4200회 중 2154회에서 순위 대조가 근사가 됐다. probe 기준 Spearman 최솟값 0.99859, |AUROC(m_rank)-AUROC(m_probe)| 최댓값 0.855%p. 동값 원인은 bf16 해상도가 아니다(저장 dtype float32, round-trip 오차 비영). CV에는 없고 BM·QA·DS에 분포한다.
- **결정 (Decision)**: 증거 판정: H1 반박, H2 지지, H3 반박. 단 cell 단위 부등호는 12/12에서 구간이 겹쳐 판별력이 약하다. 운영 결정: 사전 next_actions의 'H1 반박' 분기를 적용해 D-024 계열 정규화·결합 진단을 종료하고 연구 자원을 신규 브랜치 후보(research_directions.md Tier 1)로 돌린다. 축은 닫지 않는다. 결정 레코드 D-028.
- **결과별 후속 행동**: [H1 지지 — 크기 지배] 스케일을 명시적으로 다루는 결합 규칙을 후속 후보로 기록한다. 마진 스케일은 라벨 없이 계산되므로 CA-R1·§221의 선택 신호 부재를 우회할 수 있는 드문 경로다. 결합 규칙을 결과 보기 전에 하나로 고정한 별도 RU를 제안하고, 승격은 PROJECT.md §4로 판정한다.
[H1 반박 — 순위 지배] 정규화·결합 축으로는 얻을 것이 적고 브랜치 자체의 판별력 개선이 필요하다고 기록하고 D-024 계열 진단을 종료한다. 연구 자원을 신규 브랜치(research_directions.md Tier 1)로 돌린다.
[H2 반박 — 가법성 붕괴] 순위·크기 분리 자체가 부적절한 프레임이라고 기록하고, D-024가 지정한 질문을 '상호작용이 지배한다'로 정정한다. 이 경우 H1의 부등호 결과는 해석하지 않는다.
[전부 판별 불가] 추가 예산을 투입하지 않고 종료한다. 축은 닫지 않는다.
- **원문 근거 (Evidence)**:
  - docs/reports/RU-83_rank_size_separation.md
  - predictions/ru83_rank_size/summary.json (git 비추적)
  - scripts/analysis/ru83_rank_size.py
  - 출처: predictions/ru81_reg_20260907_r1/*_diag.pt, summary.json (git 비추적)
- **선행·후속 관계 (Relations)**: 선행: RU-81(3점 λ 진단, 저장 마진) · RU-82(λ 앙상블 재집계, 평균≈0인데 과제별 진폭 최대인 구성 발견) · D-024(후속 질문 지정) · D-025. 백로그 등록: research_directions.md Phase 0-7. 경계: CA-08 경계 밖(D-022) — 결합 규칙의 진단이며 탐색 반복이 아니다. CA-07 비저촉 — LOO를 쓰지 않는다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 저장 마진은 bf16-mixed 특징 경로 산출값이며 fp32와 최대 0.5%p 차이 가능성이 남는다(기존 기술 부채). 순위 대조의 tie 근사가 Δ 표에 남긴 영향은 정량화하지 않았다. QA만 크기 우세로 갈린 이유는 기전 미확인이다. 다른 개입(브랜치 교체·신규 특징·결합 함수 변경)에서 같은 분해가 성립하는지는 미검증이다.

---

### RU-84. RBF Gram 대각 지배의 현행 규모 실측

- **일자 (Date)**: `2026-09-07` ~ `2026-09-07`
- **커밋 범위**: - (`14f0c4b1` ... `14f0c4b1`)
- **작업 유형**: `reproduction_measurement`
- **질문 (Question)**: §199가 CA-04의 기각 기전으로 제시한 'Few-Shot 환경에서 RBF Gram이 대각선 근처만 1이고 나머지는 0에 수렴해 표본 노이즈를 암기한다'가 현행 구현·현행 context 규모에서 성립하는가?
- **가설 (Hypothesis)**: H1: 성립하지 않는다. 현행 solver는 context 통계로 피처별 표준화(ctx_std)를 하고 기본 대역폭이 gamma = 1/dims이므로, 표준화된 쌍거리의 평균은 E||x_i - x_j||^2 = 2d이고 gamma를 곱하면 차원·상관·N_ctx와 무관하게 평균 2가 된다. Jensen에 의해 off-diagonal 커널의 평균은 exp(-2) = 0.1353 이상이며, 행별 off-diagonal 합의 하한은 (N_ctx - 1) * 0.1353이다. N_ctx 90에서 12.0, 235에서 31.6이므로 대각(=1)을 압도한다. **대각 지배가 아니라 off-diagonal 지배**를 예측한다.
H2: 이 결론은 N_ctx = 40에서도 성립한다 — 39 * 0.1353 = 5.3 > 1. 즉 기전 주장의 오류는 표본 수 전제와 무관하다.
기전 미확인 — 대각 지배가 아니라는 것이 §199의 성능 하락을 설명하지 않는다. 하락의 원인은 별개 문제다.
- **판정 기준 (Criteria, 사전 고정)**: [측정] 저장 geometry(고유값·rank·dimension·n_context)로 다음을 산출한다. GPU 0.
  trace = sum(eigenvalues) — 클래스 가중 중심화 design Gram의 대각합. 가중 합계는 2이므로
  클래스 균형 분산 v = trace / 2, 표준화 검증비 rho_std = v / dimension (기대 ~1)
  gamma * E||x_i-x_j||^2 = 2 * rho_std, off-diagonal 커널 하한 K_lo = exp(-2 * rho_std)
  행별 off-diagonal 합 하한 S_lo = (n_context - 1) * K_lo
[판정 — 결과 보기 전에 고정]
  대각 지배 불성립: 모든 (fold, 브랜치)에서 S_lo > 1
  대각 지배 성립 가능: S_lo <= 1인 (fold, 브랜치)가 하나라도 있으면 그 비율을 보고하고
    실제 디스크립터로 재측정이 필요하다고 적는다
  판별 무효: rho_std가 [0.5, 2.0]을 벗어나면 표준화 전제 해석이 틀렸다는 뜻이므로 kill
[범위] KRR 경로 브랜치 bm(32D)·qa(128D)·ds(32D)가 판정 대상이다. cv는 정규화 경로가 달라 (_normalize_descriptors) 별도로 보고만 하고 판정에 쓰지 않는다. **cosine 커널은 이 RU의 범위가 아니다** — 저장 고유값으로 판정할 수 없으므로 미측정으로 남긴다.
[근사] E||x_i-x_j||^2 = 2v는 클래스 가중 분산을 쓴 근사다. solver의 표준화는 비가중이므로 비가중 쌍거리 평균은 정확히 2d이며, rho_std는 그 전제의 독립 확인 역할이다.
- **예산 / 중단 조건**: GPU 0. 저장 geometry만 읽는 CPU 계산. 구현+실행+보고 벽시계 상한 1시간. / 1. rho_std가 판정 대상 브랜치에서 [0.5, 2.0]을 벗어나면 중단한다 — 표준화 전제 해석이 틀렸다.
2. 비유한 값, 또는 저장 geometry의 rank/dimension/n_context 결측.
3. 벽시계 1시간 초과.
4. 실제 디스크립터를 다시 계산하고 싶어지면(=GPU 필요) 실행을 멈추고 예산 승인을 받는다.
- **실험 및 변경 (Experiment)**: 저장 geometry(고유값·rank·dimension·n_context) 350 folds 전량을 CPU에서 읽어 rho_std = (trace/2)/dimension, off-diagonal 커널 하한 exp(-2*rho_std), 행별 off-diagonal 합 하한 (n_context-1)*하한을 산출했다. 코드 확인: solvers.solve_kernel_ridge의 피처별 표준화와 solvers.kernel_matrix의 gamma=1/dims 기본값, config.krr_gamma=None, §199 실험 스크립트(kernel_ridge_probe.py)도 gamma 미지정. GPU 0, wall 0.51초. 구현: scripts/analysis/ru84_gram_diagonal.py.
- **관측 결과 (Observations)**: rho_std 평균 — bm 0.9905, qa 0.9924, ds 0.9918, cv 0.9696. 전부 사전 허용 [0.5, 2.0] 안이고 1에 근접해 표준화 전제가 확인됐다.
off-diagonal 커널 하한 평균 0.1377~0.1451. 행별 off-diagonal 합 하한 평균 28.6, 최소 11.2, 최대 50.4. 대각은 1이다.
판정 대상 1050건(350 folds × bm/qa/ds) 전부에서 S_lo > 1 — **대각 지배 가능 fold 0건**. H1 지지.
N_ctx=40에서도 39*exp(-2)=5.28 > 1이므로 표본 수 전제와 무관하게 성립한다. H2 지지.
논거는 차원·상관·N_ctx와 무관하다 — 피처별 표준화로 E||xi-xj||^2 = 2d이고 gamma=1/d이므로 E[gamma*d^2]=2, Jensen으로 평균 off-diagonal >= exp(-2). 대각 지배는 N_ctx <= 8에서만 가능하다.
- **결정 (Decision)**: 증거 판정: H1·H2 모두 지지. §199가 CA-04의 기각 기전으로 제시한 RBF Gram 대각 지배는 반박됐다. 운영 결정: CA-04의 '활용 전 해소 항목 ①'을 해소로 기록하고, 비선형 커널 후보 재설계 시 대각 지배를 근거로 쓰지 않는다. §199의 성능 하락 원인은 기전 미확인으로 남긴다. CA-04 재개(D-027)는 유지한다. 결정 레코드 D-029.
- **결과별 후속 행동**: [H1 지지 — 대각 지배 불성립] CA-04의 '활용 전 해소 항목 ①'이 해소된 것으로 기록하고, §199의 기각 기전을 `반박`으로 내린다. 성능 하락의 원인은 미확인으로 남기고, 비선형 커널 후보를 재설계할 때 대각 지배를 근거로 쓰지 않는다. closed_axes.md CA-04와 decisions.md에 반영한다.
[H1 반박 — 대각 지배 성립] §199 기전이 현행 규모에서도 지지된다고 기록하고, CA-04 활용 시 대역폭 선택을 필수 설계 항목으로 올린다. 재개(D-027)는 유지한다.
[판별 무효] 표준화·대역폭 규칙을 다시 읽고 원인을 진단한다. 이 경우 저장 geometry 기반 진단 전반(RU-81의 df/rank 포함)의 해석을 재검토 대상으로 올린다.
- **원문 근거 (Evidence)**:
  - docs/reports/RU-84_gram_diagonal_dominance.md
  - predictions/ru84_gram_diagonal/summary.json (git 비추적)
  - scripts/analysis/ru84_gram_diagonal.py
  - src/models/common/solvers.py (표준화·gamma 규칙), src/models/config.py krr_gamma
- **선행·후속 관계 (Relations)**: 선행: D-026(N_ctx 전제 정정) · D-027(CA-04 재개, 활용 전 해소 항목 ①). 백로그 등록: research_directions.md Phase 0-8. 대상 축: CA-04. 출처 실행: predictions/ru81_reg_20260907_r1의 저장 geometry.
- **확인 필요 사항 및 한계 (Uncertainties)**: cosine 커널과 poly 커널은 미측정이다. cv 브랜치는 정규화 경로가 달라 판정에서 제외했다. 이 RU는 Gram 구조만 다루며 §199의 성능 하락 원인은 설명하지 않는다. 이 측정은 Gram 구조만 다루며 §199의 성능 하락(-0.0140)을 설명하지 않는다. E||xi-xj||^2 = 2v의 v는 클래스 가중 분산이며 비가중 표준화와의 차이는 rho_std로만 확인했다.

---

### RU-85. Tier 1 3건(AKS·MDX·LID) 라벨 무관 게이트 1 점검

- **일자 (Date)**: `2026-09-07` ~ `2026-09-07`
- **커밋 범위**: - (`732f1aab` ... `43296ecf`)
- **작업 유형**: `diagnostic`
- **질문 (Question)**: Tier 1 신규 브랜치 후보 3건(AKS 각도·4차 텐서 스펙트럼 / MDX 다봉성 깊이 프로파일 / LID 국소 이웃 스케일링)은 (a) 제안서가 사전 선언한 라벨 무관 조기 사망 조건에 걸리는가, (b) 걸리지 않은 후보는 게이트 ① 직교성(max |r| <= 0.6, 랭크 효율 비하락)을 통과하는가?
- **가설 (Hypothesis)**: H1: 세 후보 모두 조기 사망 조건을 통과한다(=퇴화하지 않는다). 근거는 세 특징이 서로 다른 자유도를 읽는다는 구조적 논거이며, 실제 데이터에서의 퇴화 여부는 미측정이다.
H2: AKS가 게이트 ①을 통과한다. 백색화가 2차 모멘트를 항등행렬로 만든 뒤 계산하므로 BD가 읽는 성분이 정의상 제거되고, 구면대칭 분포족에서 AKS 8D가 항등적으로 상수가 되는 반면 SHJ는 자유롭게 변한다. **단 이는 '특징이 서로의 함수가 아니다'까지만 보장하며 'Dual Ridge 마진의 |r| <= 0.6'과는 논리적 간극이 있다.**
H3: MDX·LID의 게이트 ① 통과는 구조적 보장이 아니라 데이터 조건부 기대다. MDX는 다봉이 희소하면 특징이 상수에 가까워져 |r|은 낮아지되 정보량이 없을 수 있고(§218 BS와 같은 '직교하나 무정보' 유형), LID는 토큰 수(슬라이드 크기)에 의한 대리 변수화 위험이 있다.
기전 미확인 — 통과·탈락 어느 쪽도 그 자체로 성능이나 기전을 입증하지 않는다.
- **판정 기준 (Criteria, 사전 고정)**: [단계 A — 라벨 무관 조기 사망 판정, 1과제 50 fold]
  과제는 `cptac_luad/KRAS_mutation` 하나로 사전 고정한다. 선정 근거는 후보 선호가 아니라   **Primary 7 중 context 슬라이드 수가 가장 많아(240~261장) '전체 슬라이드의 x% 이상' 형태의   조기 사망 조건 판정이 가장 안정적**이라는 것이다. 세 후보에 같은 과제를 쓴다.
  각 조건은 제안서가 사전 선언한 것을 그대로 쓴다.
  AKS 사망: 각도 산포 스펙트럼 엔트로피 H_q가 전체 슬라이드의 90% 이상에서 > 0.99 이고     슬라이드 간 표준편차 < 0.01. 부수: m_akd(각도 4D)와 m_akf(4차 4D)의 |r| > 0.9이면 8D가 중복이므로     4D로 축소해 재판정한다(사망이 아니라 설계 축소).
  MDX 사망: b = 0.25에서 (슬라이드 x 8방향) 쌍의 95% 이상이 K = 1(단봉).     부수 사망: 계곡 깊이 특징의 슬라이드 간 표준편차 < 0.02.     부수 사망: 격자 대조군 G = 129 마진과 본 마진(G = 257)의 |r| < 0.9 (격자 의존).
  LID 사망: 슬라이드 간 ID_TwoNN 표준편차 < 0.2차원.     또는 전체 슬라이드의 95% 이상에서 |ID_20 - ID_5| < 0.3 (다중 스케일 구조 부재).     부수 사망: m = 4096 마진과 m = 1024 마진의 |r| < 0.8 (부분추출 비강건).     부수 처리: 특징과 log N(토큰 수)의 |r| > 0.5인 특징은 제거한다(크기 대리변수화).     제거 후 남는 특징이 3개 미만이면 사망으로 본다.
  단계 A에서 처리량을 실측한다 — LID는 두 채점자가 비용 신고를 신뢰할 수 없다고 명시했으므로   1과제 실측 없이 전량 스윕을 집행하지 않는다.
[단계 B — 게이트 ① 직교성, 단계 A 생존 후보만, Primary 7 전량]
  `scripts/analysis/branch_screen.py --adopted m_sh,m_shj`로 수행한다. 기준 집합은 공식 5-branch  (CV·BM·BD·QA·DS)와 채택된 SH·SHJ를 모두 포함하며, 형제 후보들끼리도 상관을 잰다.
  기각: 어느 과제·어느 기준 브랜치에서든 max |r| > 0.6. **이때 성능을 조회하지 않고 종료한다.**
  기각: 랭크 효율(eff.rank / 브랜치 수)이 기존 구성보다 낮아지는 경우. 6-branch 통과선은 eff.rank >= 2.72.
  통과: 위 둘을 모두 만족. 통과는 **채택 후보 자격**이며 게이트 ②(정보량)와 승격은 별개다.
[불변식 — 위반 시 kill]
  오염 검사: SCREEN_ONLY 실행의 fold-mean AUROC가 기준선과 4자리까지 일치할 것   (후보 마진이 앙상블에 들어가지 않았음의 확인).
  라벨 반대칭: 후보 마진이 y -> 1-y에서 정확히 반전될 것(허용 오차 1e-4).
  결정론: 같은 입력 재실행 시 마진이 완전히 동일할 것. autocast를 끄고 fp32로 계산한다.
  eigh 부호 모호성: AKS는 고윳값만 쓰므로 부호에 무관해야 하고, MDX는 상위 8 고유벡터의   인접 고윳값 비 < 1.01 발생 빈도를 로깅한다(5% 초과 시 4방향으로 강등해 재판정).
  LID의 topk 동점과 r_1 = 0(중복 토큰) 처리 규칙을 구현 전에 선언한다 —   동점은 인덱스 오름차순으로 깨고, r_1 = 0은 1e-12로 클램프하며 중복 비율을 로깅한다.
[보고] 세 후보의 모든 조건 판정과 상관 표를 전량 보고한다. 통과한 후보만 골라 보고하지 않는다. 이 RU는 성능 승격을 판정하지 않으며 단계 B의 STEP 3 성능 표는 기록용이다.
[카드 개정 · 2026-09-07 · 구현 단계, 실제 데이터·라벨 열람 전]
① **MDX 봉우리 규칙에 prominence 하한을 추가한다.** 제안서의 '국소 최대 판정'에는 하한이 없었고, 합성 단봉 가우시안에서 꼬리 잡음 봉우리(전역 최대의 0.4~0.6% 높이)가 p2로 선택되어 D = 1 - v/p2가 0.811까지 올라갔다. 즉 단봉 구름에서 깊은 계곡이 보고됐다. 고정 상수 `_PEAK_FLOOR = 0.05`(전역 최대 대비)를 넘는 국소 최대만 봉우리로 센다. 데이터 의존 선택이 아니며 결정론을 유지한다. 수정 후 합성 검증: 단봉 0.015 / 균등 0.033 / 이봉 0.587 / 삼봉 0.604로, **제안서의 핵심 논거(초과첨도가 균등과 이봉을 구분하지 못한다)가 합성 데이터에서 재현됐다.**
② **AKS 8D의 구성 모호성을 해소한다.** 제안서는 5개 이름(정규화 엔트로피·최대값·participation ratio·조건수·Frobenius 이방성)으로 8D를 서술해 개수가 맞지 않았다. **조건수를 제외하고** 스펙트럼당 4개(엔트로피·최대값·participation ratio·이방성)로 8D를 구성한다. 근거는 제안서 자신이 지적한 것이다 — `log(q_max/q_min)`은 상계가 없어 '전부 유계' 설계 목표와 모순된다. 제외 후 8개 특징 전부가 [0,1]에 든다.
③ **LID의 안정성 한계를 기록한다.** 합성 측정에서 군집 구름은 시드·토큰수·부분추출에 걸쳐 안정적이나(군집 14~17 대 등방 22~25), 국소 선형(무잡음 1D 곡선)에서는 k-NN 거리비가 1로 붕괴해 ID = 1/mean(log ratio)가 주변 차원 32를 한 자릿수 넘게 초과한다. 이 성질은 사전 선언된 부분추출 강건성 사망 조건(m=4096 대 1024 |r| < 0.8)이 실질적 판정력을 갖는 이유다. **판정 기준·사망 조건·마진 목록은 바꾸지 않았다.**
- **예산 / 중단 조건**: GPU 합산 최대 2h — 단계 A 0.5h, 단계 B 1.5h. `nexgem-s1`에서 Slurm 없이 직접 실행한다(호스트가 GPU 8장을 갖고 sinfo/squeue가 없다). 구현+검증 벽시계 상한 3h. 단계 A의 실측 처리량이 단계 B 예산을 넘길 것으로 나오면 단계 B를 집행하지 않고 사용자 승인을 받는다. / 1. 오염 검사 실패 — SCREEN_ONLY 실행의 fold-mean AUROC가 기준선과 4자리 불일치.
2. 라벨 반대칭 위반(> 1e-4) 또는 재실행 비결정성.
3. 비유한 특징·마진 발생. nan_to_num으로 덮지 않고 원인을 먼저 진단한다.
4. GPU 합산 2h 또는 벽시계 3h 초과.
5. 단계 A에서 사망한 후보는 단계 B로 보내지 않는다.
6. 사전 선언한 7개 마진(m_aks, m_akd, m_akf, m_mdx, m_mdx129, m_lid, m_lid1024) 외의 변형을 추가하고 싶어지면 실행을 멈추고 카드 개정으로 기록한 뒤에만 진행한다.
- **실험 및 변경 (Experiment)**: 구현: src/models/branches/{aks,mdx,lid}.py, 배선은 scripts/test_pathobench.py의 sh_all에 ICF_TIER1 환경변수로 게이트했다(비어 있으면 기존 동작과 동일). 7개 후보 마진(m_aks, m_akd, m_akf, m_mdx, m_mdx129, m_lid, m_lid1024)과 3개 진단 텐서(d_aks, d_mdx, d_lid)를 기록한다.
단계 A: cptac_luad/KRAS_mutation 50 fold, 3후보 전부, GPU 1장, wall 1335초(0.371 GPU-h). 판정은 scripts/analysis/ru85_gate1.py.
단계 B: 생존자 MDX만 Primary 7 전량 50 fold, 7 GPU 병렬, wall 1207초. 게이트 ①은 scripts/analysis/branch_screen.py --adopted m_sh,m_shj.
회귀 테스트 tests/test_tier1_branches.py 10건 추가.
- **관측 결과 (Observations)**: 오염 검사: 단계 A·B 모두 fold별 AUROC가 v121_baseline과 최대차 0.00e+00 (7과제 × 50 fold).
단계 A (query 슬라이드-fold 쌍 3061개) — AKS 사망: H_q > 0.99인 슬라이드 비율 1.000(임계 ≥0.90), H_q 슬라이드 간 SD 0.0017(임계 <0.01), H_q 평균 0.9963. 백색화 후 32D 각도 산포가 사실상 등방이다. 부수 조건 |r|(m_akd, m_akf)=0.651로 8D 중복은 아니었다.
LID 사망: 부분추출 |r|(m=4096, m=1024)=0.314(임계 <0.8). 나머지는 통과 — ID_TwoNN SD 1.285, 다중 스케일 평탄 비율 0.000, 중복 토큰율 0.00002. 크기 대리변수화 미발생 |r|(m_lid, log N)=0.051.
MDX 생존: 단봉 쌍 비율 0.818(임계 ≥0.95), 계곡 깊이 SD 0.2403(임계 <0.02, 평균 0.3863), 격자 |r|(257 vs 129)=0.953(임계 <0.9), 고윳값 축퇴율 0.0015(강등 임계 0.05).
단계 B — MDX 게이트 ① 통과: max |r| = 0.267 (기각선 0.6, 과제·브랜치 전량 중 최대는 ARID1A의 BD 0.267). 랭크 효율 2.26/5 → 2.96/6, 45% → 49% 상승(통과선 eff.rank >= 2.72).
게이트 ① 통과 후 기록한 성능(판정 아님): MDX 단독 AUROC가 7과제 중 5과제에서 0.5 미만 (Grade 0.4270 최저, KRAS 0.5826 최고). 앙상블 macro -0.0016, sign agreement 3/7. 사전 등록 실패 — ②b 1순위로 지목한 Grade가 최저이고 지목하지 않은 KRAS가 최고다.
가설: H1 반박(3건 중 2건 사망). H2 판정 불가(AKS가 게이트 ① 도달 전 사망) — 단 구면대칭 구름에서 AKS 8D가 항등 상수라는 구조적 보장은 합성 검증에서 성립했고, 문제는 실제 데이터가 그 구면대칭에 가까웠다는 것이다. H3 지지(MDX는 직교하나 단독 무정보, LID는 예측한 두 위험 중 부분추출 비강건으로 사망).
- **결정 (Decision)**: 증거 판정: AKS·LID는 사전 선언한 라벨 무관 조기 사망 조건으로 성능 조회 전 종료. MDX는 게이트 ① 통과(max |r| 0.267, 랭크 효율 상승). 운영 결정: MDX를 채택 후보로 기록하고 게이트 ②(정보량) 심사를 별도 RU로 넘긴다. AKS·LID의 하위축을 닫을지는 closed_axes.md §3에 올려 사용자 판정을 받으며 자동으로 닫지 않는다. 3건이 동일 에이전트 한 배치 산출이라 독립 비교군이 없다는 결손은 그대로 남는다. RU 종료 자체로는 결정 레코드를 만들지 않는다(D-033).
- **결과별 후속 행동**: [3건 모두 게이트 ① 통과] 채택 후보로 기록하고 게이트 ②(정보량) 심사를 별도 RU로 제안한다. **3건이 동일 에이전트 한 배치 산출이며 독립 비교군이 없으므로 셋을 함께 보고 다음 단계를 정한다.** 성능·승격은 판정하지 않는다.
[일부 통과] 통과 후보만 게이트 ②로 보내고, 탈락 후보의 탈락 사유(상관 상대 브랜치와 |r|)를 기록한다. 상관이 높게 나온 상대가 SH·SHJ라면 형상 계열의 정보량 병목(§218)과 같은 축인지 별도로 적는다.
[조기 사망] 사망 조건에 걸린 후보의 하위 축을 닫는 근거로 쓴다 — MDX가 단봉 축퇴로 죽으면 '모드 수' 하위축을 닫고, 동시에 CT가 k-means로 얻으려던 정보가 애초에 희박했다는 독립 증거가 된다. closed_axes.md 신규 축 후보로 §3에 올려 사용자 판정을 받는다. 자동으로 축을 닫지 않는다.
[3건 모두 사망 또는 탈락] 형상 계열 확장 경로 전반의 재검토를 사용자에게 제안하고, 후보 큐의 P4·P5 항목으로 우선순위를 옮긴다. 형상 계열 일반의 불가능으로 확대하지 않는다.
- **원문 근거 (Evidence)**:
  - docs/reports/RU-85_tier1_gate1.md
  - predictions/ru85_stageA_gate1/summary.json (git 비추적)
  - predictions/*_ru85_stageA_official50_bf16.pt, *_ru85_stageB_official50_bf16.pt (git 비추적)
  - src/models/branches/{aks,mdx,lid}.py, scripts/analysis/ru85_gate1.py, scripts/run_ru85_tier1_screen.sh
  - tests/test_tier1_branches.py (10 tests)
- **선행·후속 관계 (Relations)**: 후보 출처: §226 라운드 Tier 1 3건(research_directions.md §2.1~§2.3). 후보 큐 P1-1·P1-2·P1-3. 선행 결정: 정규화·결합 진단 종료로 자원을 신규 브랜치로 돌린 것(D-028). 게이트 정본: PROJECT.md §5. 관련 축: CA-09(평균 사영 브랜치는 닫혔고 형상 계열은 경계 밖), CA-10(SH 모멘트 변주는 닫혔고 shj.py가 버리는 성분은 경계 밖 — Tier 1 3건이 그 영역이다). CA-11(draw 안정성 ICC)와 CA-07(context LOO)은 선택 신호로 쓰지 않는다.
- **확인 필요 사항 및 한계 (Uncertainties)**: AKS의 'trace 제거 = SHJ 성분의 명시적 제거' 주장은 근사다 — shj.py의 첨도 특징은 표준화 초과첨도이고 원시 4차 모멘트가 아니다. AKS 조건수 특징 log(q_max/q_min)은 상계가 없어 '전부 유계' 주장은 성립하지 않는다. 세 후보의 GPU 비용은 채점자 재산출이 없고 LID는 비용 신고에 명시적 불신이 붙었다. 적대적 검증 2건이 §226에서 미실행이며 어느 제안이 빠졌는지 기록이 없다. LID의 8개 특징별 log N 상관은 특징을 저장하지 않아 측정하지 못했다 — 마진(0.051)과 ID_TwoNN(0.425)만 기록했다. AKS·LID의 사망은 이 구성에서의 사망이며 다른 대역폭·추정기·부분추출 규칙에서 같은 결론이 나오는지는 미검증이다. MDX의 정보량은 게이트 ②의 판정 대상이다.

---

### RU-86. MDX 게이트 ② 정보량 심사 — 사전 등재 과제 Grade의 fold 재현성

- **일자 (Date)**: `2026-09-08` ~ `2026-09-08`
- **커밋 범위**: - (`c1b6f18b` ... `c1b6f18b`)
- **작업 유형**: `confirmatory`
- **질문 (Question)**: 게이트 ①을 통과한 유일한 Tier 1 후보 MDX가, RU-85가 결과를 보기 전에 사전 등재한 과제 `Grade`에서 게이트 ②b의 fold 재현성 기준을 만족하는가? 줄이는 불확실성: MDX가 §218 `BS`와 같은 '직교하나 무정보' 후보인가. 쓰이는 결정: 이 판정이 PROJECT.md §2 현재 목표의 완료 조건이며, 결과에 따라 §226 Tier 1 배치 전체를 닫고 다음 목표 슬롯을 연다.
- **가설 (Hypothesis)**: 주장: MDX는 `Grade`에서 게이트 ②b를 통과하지 못한다. 주장의 범위는 `Grade` 과제 · PathoBench official 50 fold · v121 구성 · bf16 경로에 한정된다. 통과하더라도 승격(§4)을 뜻하지 않으며 '과제 특화 채택'까지만이다.
이미 본 평가 자료(사전 공개): RU-85 보고의 7과제 fold-mean 단독 AUROC 표(Grade 0.4270 · KRAS 0.5826 등 7개 값)와 게이트 ① 상관값을 보았다. **fold별 단독 AUROC 분포는 보지 않았다** — 그것이 이 RU의 측정 대상이다.
- **판정 기준 (Criteria, 사전 고정)**: [1차 · 확증] 판정 대상은 사전 등재된 `Grade` 단 하나다.
  · 지지(통과): 50 fold 중 MDX 단독 AUROC > 0.5인 fold가 40개 이상이고, 이항검정(p0=0.5, 단측) p < 0.01.
  · 반박(실패): 위를 만족하지 못함.
  · 판별 불가: 양성 또는 음성 부재로 AUROC가 정의되지 않는 fold가 10개를 넘는 경우. 이때 유효 fold만으로 재계산해 판정하지 않는다.
  · 실행 무효: §3.4 오염 검사에서 baseline 대비 fold별 AUROC 최대차가 0을 넘는 경우.
[2차 · 탐색 — 결과 보기 전에 선언] 나머지 6과제의 fold별 단독 AUROC 분포도 함께 산출해 과제 내 fold 간 분산과 과제 간 분산을 비교한다. 목적은 MDX의 정보가 '없는' 것인지 '과제에 따라 갈리는' 것인지 구별하는 것이며, 후자면 CA-R1과 같은 벽이다.
**이 2차 자료는 어떤 경우에도 MDX 채택의 근거로 쓰지 않는다.** KRAS가 최고값이라는 것을 이미 보았으므로 사후에 1순위를 바꾸는 것은 §214 실패 양식이다. 탐색으로만 기록한다.
- **예산 / 중단 조건**: 상한 2.5 GPU-h · wall 1시간. 저장된 fold별 MDX 마진이 있으면 CPU 재집계만으로 끝난다(GPU 0). 없으면 RU-85 단계 B를 재실행한다 — 실측 wall 1,207초 · 7과제 7 GPU 병렬 ≈ 2.35 GPU-h. 호스트 nexgem-s1은 직접 실행 유형이며 사용자가 계산 자원 사용을 승인했다. / ① 오염 검사가 baseline과 불일치하면 즉시 중단하고 `실행 무효`로 기록한다. 판정하지 않는다.
② `Grade`의 유효 fold가 40개 미만이면 통과가 산술적으로 불가능하므로 즉시 종료한다.
③ 저장 마진이 없고 단계 B 재실행이 2.5 GPU-h 또는 wall 1시간을 넘기면 중단하고 비용 초과를 기록한다.
④ 구현 중 사양 결함을 발견하면 고치되 판정 기준·1순위 과제는 바꾸지 않는다.
- **실험 및 변경 (Experiment)**: 설계: MDX 단독 마진으로 과제별·fold별 AUROC를 산출한다. 표집 단위는 fold(과제당 50개), 비교군은 없다 — ②b는 절대 기준(AUROC > 0.5)에 대한 이항 재현성 검정이다. 고정 조건: PathoBench official 분할, v121 구성, SCREEN_ONLY(앙상블 미개입).
closed_axes_check: MDX는 게이트 ①을 통과했고(max |r| = 0.267) RU-85 시점에 닫힌 축 대조를 마쳤다. 이 RU는 그 후속 게이트이며 새로 저촉되는 축은 없다.
provenance: start_sha c1b6f18 · nexgem-s1 · .venv Python 3.12.11 / torch 2.14.0+cu130 · 구현 src/models/branches/mdx.py · 배선 scripts/test_pathobench.py(ICF_TIER1) · RU-85 산출물 predictions/ru85_stageA_gate1/summary.json.
- **관측 결과 (Observations)**: 오염 검사: 7과제 × 50 fold 앙상블 fold-AUROC가 v121_baseline 대비 최대차 0.000e+00 → 실행 유효.
1차(확증) 사전 등재 과제 Grade: 유효 fold 50/50, 단독 AUROC > 0.5인 fold k = 10, 이항검정 단측 p = 0.99999720. 기준(k>=40 그리고 p<0.01) 불충족.
2차(탐색) 7과제 fold별 단독 AUROC 중앙값: Grade 0.4179(k=10) · ARID1A 0.4495(15) · KEAP1 0.4627(19) · Progression 0.4744(17) · PBRM1 0.4820(24) · SMAD4 0.5778(39) · KRAS 0.5861(41). Grade는 50 중 40 fold가 0.5 미만, KRAS는 41 fold가 0.5 초과이며 |중앙값−0.5|가 각각 0.0821·0.0861로 크기가 거의 같고 방향이 반대다. 과제 간 분산 0.00413 대 과제 내 평균 분산 0.01081(비율 0.382). 저장 마진 재집계로 GPU 0 · CPU만 사용했다.
- **결정 (Decision)**: 증거 판정: 반박. 운영 결정: 종료.
사전 등재 과제 Grade에서 게이트 ②b 기준에 크게 미달했다(k=10/50). 게이트 ①을 통과한 유일한 Tier 1 후보 MDX를 종료하며, 이로써 §226 Tier 1 3건이 전부 종료됐다. 카드 next_actions의 '반박' 경로를 그대로 실행한다 — 연구 자원을 CA-R1 교착으로 돌린다(RU-88).
2차 관측(부호 반전)은 MDX를 되살리지 않는다. 부호를 뒤집으면 Grade가 k=40으로 문턱에 닿지만 그 부호 선택은 결과를 본 뒤의 것이고, 어느 방향인지 정하려면 라벨이 필요하다. 사후 부호 선택으로 후보를 되살리는 것은 §214의 실패 양식이다.
- **결과별 후속 행동**: 지지(통과): MDX를 `과제 특화 채택(Grade)`으로 기록한다. 승격(§4)은 별건으로 남기고 큐 §0에 'MDX 승격 심사'를 P2로 넣는다. closed_axes.md §3-G·§3-H는 계속 대기시킨다.
반박(실패): MDX를 종료한다. Tier 1 3건이 전부 종료되므로 §226 배치를 닫고, closed_axes.md §3에 '형상 계열 고차 통계' 하위축의 폐쇄 여부를 올린다. 연구 자원을 CA-R1 교착으로 돌린다.
판별 불가: MDX를 보류하고, fold별 단독 AUROC를 산출할 수 있는 평가 설계를 별도 RU로 분리한다. 보류 사유를 반박으로 적지 않는다.
실행 무효: 오염 원인을 고쳐 재실행한다. 재실행 없이 판정하지 않는다.
- **원문 근거 (Evidence)**:
  - docs/reports/RU-86_mdx_gate2.md
  - predictions/ru86_gate2/summary.json
  - scripts/analysis/ru86_gate2.py
- **선행·후속 관계 (Relations)**: 선행 RU-85(게이트 ① 통과, 사전 등재 Grade). PROJECT.md §5 게이트 ②b, §5.1 신설 경위. closed_axes.md CA-R1(라벨 무관 선택 신호 부재) — 2차 관측이 같은 벽에 귀속된다. research_directions.md §0 P2-10. 후속 RU-88(CA-R1 탐색).
- **확인 필요 사항 및 한계 (Uncertainties)**: 기전 미확인. MDX의 과제별 부호 반전에 대한 경쟁 설명 셋(실제 반전 신호 / 과제별 교란변수 결합 / 우연)을 구별하지 못했다. 우연은 Grade 40/50·KRAS 41/50에 대해 약한 설명이다.
게이트 ②의 통계량이 단측 'AUROC > 0.5'이므로 '정보량'이 아니라 '양의 방향 정보량'을 잰다. 부호 반전 후보는 정보를 담고도 구조적으로 탈락한다. 다만 양측 |AUROC−0.5|로 바꾸면 사후 부호 선택의 문이 열리므로 결과를 본 이 시점에 기준을 바꾸지 않는다 — 게이트 설계 재검토는 큐에 올린다.
범위: Grade 과제 · official 50 fold · v121 구성 · bf16 경로. bf16의 수치 개입은 RU-87에서 별도 측정.
§226 배치의 구조적 결손(3건이 동일 에이전트 한 배치 산출, 독립 비교군 없음)은 3건 전부 종료로도 해소되지 않는다. 미검증으로 남는다.
MDX 하위축(형상 다봉성 통계)의 폐쇄 여부는 closed_axes.md §3에 올린다. 부호 반전 관측이 있으므로 '무정보라서 닫는다'는 근거로는 닫을 수 없다.

---

### RU-87. bf16 대 fp32 — 절대 AUROC가 아니라 대응 Δ가 흔들리는가

- **일자 (Date)**: `2026-09-08` ~ `2026-09-08`
- **커밋 범위**: - (`c1b6f18b` ... `c1b6f18b`)
- **작업 유형**: `reproduction_measurement`
- **질문 (Question)**: bf16-mixed 추론 경로의 수치 오차가 승격 판정의 추정 대상인 **대응 per-fold Δ**에 얼마나 남는가? 절대 AUROC에서 관측된 편차가 짝짓기로 상쇄되는가, 아니면 Δ에도 그대로 남는가?
줄이는 불확실성: 문서가 상한으로 쓰는 `최대 0.5%p`는 §222에서 ARID1A 한 과제 · SHJ 한 브랜치의 1회 측정(bf16 0.6363 → fp32 0.6403, 차 0.40%p)을 일반화한 값이고 실측 범위가 아니다. 그리고 그 값은 승격 문턱 `δ_min = +0.3%p`(PROJECT.md §4)보다 크다.
쓰이는 결정: Δ에서도 0.3%p 규모가 남으면 승격 규칙이 수치 잡음 위에서 판정하고 있는 것이므로 δ_min과 MDE 0.072%p의 해석을 함께 재검토해야 한다. 상쇄되면 이 기술 부채를 종료한다.
- **가설 (Hypothesis)**: 경쟁 설명 두 가지를 구별한다.
  E1 (상쇄): 두 arm이 같은 정밀도 경로를 쓰므로 bf16 오차는 대부분 공통 성분이고, 짝지은 Δ에서 상쇄된다. 절대 AUROC 편차 ≫ Δ 편차.
  E2 (잔존): 오차가 arm의 특징 분포에 따라 다르게 작용해 Δ에도 남는다. 절대 AUROC 편차 ≈ Δ 편차.
사전 예상은 E1이지만 근거는 이론적 기대뿐이며 **실측된 바 없다**. 이 RU는 예상 확인이 아니라 두 설명을 구별하는 계측이다.
이미 본 평가 자료(사전 공개): §222의 ARID1A SHJ bf16/fp32 대조 1건, PROJECT.md §3.2 기준선 macro 0.6171, §4.1 RU-80 정밀도 표(s_d 0.0033 · MDE 0.072%p). fp32 재실행 결과는 보지 않았다.
- **판정 기준 (Criteria, 사전 고정)**: 재현·계측 유형이므로 pass/fail이 아니라 **측정 완료 기준과 해석 구간을 결과 전에 고정**한다.
[측정 완료] 7과제 × 50 fold에 대해 네 arm(A_bf16 · A_fp32 · B_bf16 · B_fp32)의 fold별 AUROC를 확보하고, 아래 두 양의 분포를 산출한다.
  · 절대 편차 D_abs = |AUROC_fp32 − AUROC_bf16| (arm별로 따로)
  · Δ 편차 D_delta = |(B_fp32 − A_fp32) − (B_bf16 − A_bf16)| (같은 fold에서)
각각 최대 · 95 백분위 · 과제별 fold-mean 차를 보고한다.
[해석 구간 — 판정 대상은 D_delta의 과제별 fold-mean 차의 최대값]
  · `상쇄`(E1 지지): 최대 D_delta < 0.072%p (RU-80 MDE 미만). 이 축의 기술 부채를 종료한다.
  · `부분 잔존`: 0.072%p 이상 0.3%p 미만. δ_min은 유지되나 MDE 아래 Δ 해석은 좁아진다.
  · `잔존`(E2 지지): 0.3%p 이상. δ_min이 수치 정밀도 아래이므로 승격 규칙 재검토를 큐에 올린다.
[판별 불가] 네 arm 중 하나라도 7과제를 완주하지 못한 경우. 완주 과제만으로 상한을 주장하지 않는다.
[실행 무효] fp32 arm의 fold 구성 · context 규모 · max_tiles가 bf16 arm과 다른 경우. 대응 비교가 깨지므로 수치를 쓰지 않는다.
**결과를 본 뒤 위 세 구간의 경계를 바꾸지 않는다.**
- **예산 / 중단 조건**: 상한 4 GPU-h · wall 2시간. bf16 350-fold 스윕 1회의 실측 앵커는 0.9~1.0 GPU-h(research_directions.md 단가 근거)이고, fp32는 더 느릴 것으로 **추정**한다 — 배율은 미측정이며 이번에 실측한다. 신규 실행은 A_fp32 · B_fp32 두 arm이고, B_bf16이 저장돼 있지 않으면 한 arm을 추가로 돌린다(합 3 arm). 호스트 nexgem-s1은 직접 실행 유형이며 사용자가 자원을 승인했다. / ① fp32 arm이 OOM으로 `max_tiles`·배치 구성을 바꿔야 하면 **즉시 중단하고 Main에 올린다.** 구성을 바꾸면 bf16 arm과의 대응 비교가 깨져 이 RU의 추정 대상 자체가 사라진다. 임의로 구성을 조정해 완주시키지 않는다.
② wall 2시간 또는 4 GPU-h를 넘기면 중단하고 실측 비용을 기록한다.
③ A_fp32와 저장된 A_bf16의 fold 구성이 일치하지 않으면 즉시 중단하고 `실행 무효`로 적는다.
④ 구현 결함을 발견하면 고치되 해석 구간의 경계는 바꾸지 않는다.
- **실험 및 변경 (Experiment)**: 설계: 2 arm × 2 정밀도의 대응 비교.
  · Arm A = 공식 기준 구성 `v121_baseline` (CV,BM,BD,QA,DS · CT 가중 0 · Trimmed Mean · Primary 7 macro 0.6171, PROJECT.md §3.2)
  · Arm B = A + `SHJ` (§220에서 채택된 구성. `최대 0.5%p` 단서가 붙은 바로 그 수치 계열)
정밀도 전환은 `--precision 32-true`(src/utils/utils.py `add_eval_precision_argument`)로만 하고 다른 인자는 arm 간 동일하게 고정한다. 표집 단위는 fold(7과제 × 50 = 350).
고정 조건: PathoBench official 분할, fold 재분할 금지(PROJECT.md §3.1), 동일 seed, 동일 max_tiles·context 규모.
주의: §222에서 `shj_slide_features()` 내부는 이미 `autocast(enabled=False)`로 fp32가 강제돼 있다. 따라서 이 RU가 재는 것은 SHJ 내부가 아니라 **나머지 경로**의 bf16 노출이다.
closed_axes_check: 계측 정밀도 측정이며 닫힌 축에 저촉되지 않는다. research_directions.md §0 P2-9에 등재된 후보다.
provenance: nexgem-s1 · .venv Python 3.12.11 / torch 2.14.0+cu130 · 저장 bf16 arm `predictions/pathobench_*_v121_baseline_official50_bf16.pt` · 0.5%p 표기의 출처는 docs/history/archive.md §222.
- **관측 결과 (Observations)**: fold 구성 일치: 7과제 350 fold 전수(fold_indices·slide_id 순서·label·context_label) → 실행 유효. ru87_bf16 확률이 v121_baseline과 7과제 전부 max|diff| = 0.0이고 §3.4 오염 검사 상수(SMAD4 0.4421 · PBRM1 0.5553)를 재현했다.
D_delta(판정 대상) 과제별 fold-mean 차: SMAD4 0.0674%p(최대) · PBRM1 0.0586 · KEAP1 0.0414 · Progression 0.0342 · Grade 0.0138 · ARID1A 0.0124 · KRAS 0.0113.
D_abs 과제별 fold-mean 차 Arm A: PBRM1 0.0837%p(최대) · SMAD4 0.0455 · KEAP1 0.0328 · ARID1A 0.0184 · Progression 0.0171 · Grade 0.0103 · KRAS 0.0008. Arm B 최대는 Progression 0.0513.
350 fold 풀링: D_delta 최대 2.579%p · 95백분위 0.638 · 평균 0.157. D_abs Arm A 최대 3.373 · 95백분위 0.375 · 평균 0.099.
Main 독립 재계산으로 Arm A D_abs 4개 과제를 소수 4자리까지 대조 확인했다.
실측 wall: bf16 pass 2,074초 · fp32 pass 1,095초(GPU 6장). 분석은 CPU 30초.
Arm B는 ICF_SHAPE_SCREEN_ONLY=1로 저장된 m_shj 마진을 §3.2 Trimmed Mean 규약으로 오프라인 재집계해 만들었다.
- **결정 (Decision)**: 증거 판정: 상쇄(E1 지지). 운영 결정: 채택 — bf16 기술 부채를 범위 한정으로 종료한다.
D_delta 과제별 fold-mean 최대 0.0674%p로 카드가 사전 고정한 상쇄 구간(< 0.072%p)에 든다. bf16 오차는 두 arm의 공통 성분으로 작용해 짝지은 Δ에서 대부분 상쇄되므로, δ_min = +0.3%p가 수치 잡음 아래라는 우려는 성립하지 않는다.
카드 next_actions의 '상쇄' 경로를 실행한다 — 문서의 '최대 0.5%p' 표기를 실측값으로 정정하고, 그것이 절대 AUROC fold-mean에 적용되며 대응 Δ에는 적용되지 않음을 명시한다. 기준 수치의 정정이므로 archive.md 결정 이력에 레코드를 남긴다.
- **결과별 후속 행동**: 상쇄(E1): current_status.md의 bf16 기술 부채 항목을 **실측 범위로 정정하고 종료**한다. §217~§221 수치에 붙은 `최대 0.5%p` 표기를 실측값으로 바꾸고, 그 값이 절대 AUROC에만 적용되고 Δ에는 적용되지 않음을 명시한다. archive.md 결정 이력에 레코드를 남긴다(규범이 아니라 기준 수치의 정정이므로).
부분 잔존: δ_min은 유지하되, MDE 0.072%p 아래의 Δ를 근거로 한 서술을 `수치 정밀도 한계 이내`로 표시할 대상 목록을 만들어 큐에 올린다.
잔존(E2): **승격 규칙 재검토를 P1으로 큐에 올린다.** δ_min = 0.3%p가 수치 정밀도 아래라면 그 문턱으로 내린 판정의 해석 범위가 좁아진다. 소급 재판정은 하지 않되(§0 원칙) 현행 규칙의 유효성 점검을 별도 RU로 분리한다. 동시에 fp32를 기본 경로로 승격할지 비용과 함께 사용자에게 올린다.
판별 불가: 완주하지 못한 원인(자원·구현)을 기록하고 보류한다. 부분 자료로 상한을 주장하지 않는다.
실행 무효: 대응 구성을 맞춰 재실행한다. 재실행 없이 판정하지 않는다.
- **원문 근거 (Evidence)**:
  - docs/reports/RU-87_precision_bf16_fp32.md
  - predictions/ru87_precision/summary.json
  - scripts/analysis/ru87_precision.py
  - scripts/run_ru87_precision.sh
  - logs/ru87/driver.log
- **선행·후속 관계 (Relations)**: history/archive.md §222(0.5%p 표기의 출처)·§217~§221(SHJ 계열 수치). PROJECT.md §4 승격 기준 δ_min=+0.3%p, §4.1 RU-80 정밀도(MDE 0.072%p). research_directions.md §0 P2-9. RU-82·RU-83 보고의 동일 단서.
- **확인 필요 사항 및 한계 (Uncertainties)**: ① 판정값이 문턱에 근접했다 — 0.0674 대 0.072, 여유 0.0046%p(문턱의 6.4%). 구간은 결과를 본 뒤 바꾸지 않았다. 다만 문턱으로 쓴 0.072%p는 RU-80의 MDE이고 PROJECT.md §4.1이 '모든 후보에 공통인 검출 한계가 아니다'라고 명시한 값이다. 카드 설계 시 이를 경계로 삼은 것은 편의였고, 근접이 우연인지 구조적인지는 미확인이다. '상쇄'를 '정밀도 문제 부재'로 확대해 읽지 않는다.
② 상쇄는 fold-mean에서만 일어난다. per-fold D_delta는 95백분위 0.638%p · 최대 2.579%p다. 단일 fold 수치를 근거로 쓰는 서술에는 상쇄가 적용되지 않으며 약 0.6%p 규모의 잡음을 안는다. 승격 판정은 350 fold 평균 위이므로 영향받지 않는다.
③ 정밀도별 순수 속도 배율은 미측정이다. fp32 pass가 1.89배 빨랐으나 페이지 캐시 교란으로 보이며 기전 미확인이다. 재실행은 예산 밖이다.
④ 원 출처 §222의 0.40%p에는 shj_slide_features() autocast 결함의 수정분이 섞여 있어 순수 정밀도 효과가 아니다. 그 경로는 현재 존재하지 않는다.
⑤ 범위: Primary 7 · official 50 fold · v121 · Arm A/B. 다른 브랜치 집합·집계 함수에서 같은 상쇄가 성립하는지는 미검증. SEAL hold-out 미검증.
⑥ logs/ru87/driver.log의 fp32 elapsed=3169s는 누적값 오기다(실제 1,095초). 관측 기록이라 고치지 않았다.

---

### RU-88. CA-R1 탐색 — fold 수준 라벨 무관 지문이 과제 특화 효과를 예측하는가

- **일자 (Date)**: `2026-09-08` ~ `2026-09-08`
- **커밋 범위**: - (`c1b6f18b` ... `c1b6f18b`)
- **작업 유형**: `exploratory`
- **질문 (Question)**: 저장된 자료에서 **fold 수준으로 계산되는 라벨 무관 지문** 중, 같은 fold의 과제 특화 개입 효과를 예측하는 것이 있는가?
무엇을 발견하려는가: CA-R1은 '과제 특화 이득은 실재하나 라벨 없이 과제를 판별할 수단이 없다'는 교착이다. 지금까지의 시도(§212 신호 A · §225 신호 B/C · CA-11)는 모두 **과제 수준**의 판별을 노렸다. 이 RU는 해상도를 **fold 수준**으로 내려 표본을 7에서 350으로 늘리고, 예측 대상도 '과제 정체'가 아니라 '개입의 효과 크기'로 바꾼다.
후보 범위를 이렇게 고른 이유: docs/proposals/2026-09-08_car1-label-free-task-identification.md의 후보 1·3·4에서 유래한 지문 3종만 쓴다. 후보 2·5는 닫힌 축 경계 판정이 갈려 closed_axes.md §3에 올렸고 이 RU에 넣지 않는다.
결과가 무엇을 알려주는가: 어떤 지문도 예측력을 보이지 않으면 fold 수준 해상도라는 접근 자체의 가치가 낮다는 단서가 된다(부재의 입증은 아니다). 하나라도 보이면 사전 등록 확증 RU의 대상이 된다.
- **가설 (Hypothesis)**: 미정 (탐색 유형). 사전 기전 가설을 세우지 않는다.
다만 대조 기준 하나를 결과 전에 고정한다: proposals.md 개선안 2 `Task-Geometry`가 제안한 공분산 스펙트럼 지문(F3)을 **대조군**으로 함께 산출한다. 새 지문(F1·F2)이 F3보다 낫지 않으면 '새로움 없음'으로 기록한다.
이미 본 평가 자료(사전 공개): RU-86의 7과제 fold별 MDX 단독 AUROC 전량(T2의 원자료), §225 살리언스 서브샘플링의 과제별 요약, closed_axes.md의 모든 기각 기전. F1·F2·F3의 값은 아직 산출하지 않았고 어떤 상관도 보지 않았다.
- **판정 기준 (Criteria, 사전 고정)**: 탐색 유형이므로 pass/fail이 아니라 **완료 기준과 후속 진입 기준**을 결과 전에 고정한다.
[사전 등재 패널 — 이 목록에서 늘리거나 줄이지 않는다]
  지문(전부 fold 수준, 쿼리 라벨 미사용):
    F1 쿼리 슬라이드의 context 임베딩 분포 대비 이상치 근접도 (제안 후보 1)
    F2 context↔query 분포 이동량 MMD (제안 후보 4 · CA-07이 지목한 기전의 직접 계측)
    F3 context 공분산 스펙트럼 감쇠 지수와 participation ratio (proposals.md 개선안 2 · 대조군)
  표적(전부 fold 수준):
    T1 살리언스 서브샘플링의 대응 per-fold Δ (§225 저장 자료) — CA-R1이 실제로 막고 있는 이득
    T2 MDX 단독 AUROC − 0.5 (RU-86 산출) — 부호가 과제에 따라 뒤집히는 양
  조합 3 × 2 = **6개를 전부 산출하고 전량 보고한다.** 유리한 것만 고르지 않는다(agent_handoff.md §4 보고 무결성 계약).
[측정] 각 조합에 대해 fold 수준 상관 ρ를 과제 군집 t 구간(df=6, PROJECT.md §4.1 설계)으로 낸다. 표본은 350 fold(7과제 × 50)이며 과제 내 fold는 독립이 아니다.
[탐색 완료] 6개 조합의 ρ와 95% 구간을 모두 산출하면 완료다.
[후속 확증 진입 기준 — 결과 전 고정] 95% 구간이 0을 배제하고 |ρ| >= 0.3인 조합만 '확증 후보'로 큐에 등재한다. 그 미만은 `판별 불가`로 적는다.
[닫지 않는 것] 전부 미달이어도 **CA-R1을 닫지 않는다.** 지문 3종·표적 2종의 실패는 라벨 무관 신호 일반의 부재가 아니며, 과제 군집 7로는 부재를 입증할 검정력이 없다.
[실행 무효] 지문과 표적의 fold 정렬이 어긋나거나 fold 구성이 다른 실행에서 온 경우.
**결과를 본 뒤 패널·진입 기준·군집 구조를 바꾸지 않는다.**
- **예산 / 중단 조건**: **GPU 0 · wall 상한 2시간.** 저장된 임베딩·마진의 오프라인 재집계만 쓴다. 신규 파이프라인 실행을 하지 않는다 — 이 RU에서 새 수치를 만드는 것은 지문 계산뿐이다. / ① T1(살리언스 서브샘플링의 fold별 Δ)을 저장 자료로 복원할 수 없으면 **즉시 중단하고 Main에 올린다.** 신규 GPU 실행으로 범위를 넓히지 않는다 — 이 RU는 GPU 0 예산으로 설계됐다.
② 지문과 표적의 fold 인덱스가 정렬되지 않으면 즉시 중단하고 `실행 무효`로 적는다. 임의로 재정렬해 맞추지 않는다.
③ wall 2시간을 넘기면 중단한다.
④ 패널 밖의 지문을 추가하고 싶어지면 추가하지 말고 Main에 올린다. 사후 패널 확장은 §214의 실패 양식이다.
- **실험 및 변경 (Experiment)**: 설계: 350 fold(7과제 × 50)를 표본 단위로 하고 과제를 군집으로 보는 상관 추정. 비교군은 F3(기존 제안 Task-Geometry 지문). 고정 조건: PathoBench official 분할, fold 재분할 금지.
closed_axes_check — Main이 판정하고 근거를 남긴다:
  · F1 대 `CA-11`: 비저촉. CA-11이 닫은 것은 **draw 간 일치도(ICC)**를 선택 신호로 쓰는 것이고 기각 기전은 draw 평균 구조상 ICC가 0.9996~1.0000으로 포화한다는 것이다. F1은 draw 일치도가 아니라 context 분포 대비 이상치 근접도이며 포화 기전이 적용되지 않는다.
  · F2 대 `CA-07`: 비저촉. CA-07의 경계 안은 'context LOO 점수를 선택 신호나 가중치로 쓰는 것'이고 경계 밖은 'LOO 마진을 진단용으로 기록·분석하는 것'이다. F2는 LOO 점수가 아니라 분포 이동량이며, 이 RU에서 **선택이나 가중에 쓰지 않고 계측만 한다.** 오히려 CA-07의 기각 기전이 원인으로 지목한 바로 그 양(context↔query 분포 이동)을 직접 재는 것이다. 재개 조건('분포 이동을 직접 교정하는 수단 확립')을 충족했다고 주장하지 않는다 — 교정이 아니라 계측이다.
  · F3: proposals.md 개선안 2로 이미 제안된 것이며 저촉 축 없음.
  · 제안 후보 2(context 라벨 조건부 분리도, `CA-06` 경계)와 후보 5(다중 강도 DS 동시 브랜치, `P2-SELECTOR-CEILING`·`CA-09` 경계)는 **판단이 갈려 closed_axes.md §3에 올렸다.** 임의로 한쪽을 채택하지 않았고 이 RU에 넣지 않았다.
provenance: nexgem-s1 · .venv Python 3.12.11 · 제안서 docs/proposals/2026-09-08_car1-label-free-task-identification.md · T2 원자료 predictions/ru86_gate2/summary.json · T1 원자료는 §225 계열 저장 예측 파일.
- **관측 결과 (Observations)**: 진입 기준(구간 0 배제 그리고 |ρ|>=0.3) 충족 0/8. Pearson ρ와 과제 7군집 클러스터-로버스트 95% 구간: F1×T1 −0.016 [−0.186,+0.153] · F1×T2 −0.089 [−0.285,+0.108] · F2×T1 +0.060 [−0.094,+0.213] · F2×T2 −0.029 [−0.136,+0.078] · F3a×T1 −0.139 [−0.290,+0.011] · F3a×T2 −0.093 [−0.396,+0.209] · F3b×T1 +0.231 [−0.022,+0.483] · F3b×T2 +0.043 [−0.249,+0.335]. 최대 점추정은 F3b×T1의 +0.231이다.
T1 복원: v121_ds_sweep 파일의 같은 fold 안 m_ds_full 대 m_ds_f000. 한 실행에서 전 arm을 동시 산출한 것이므로 §4 통제 비교 의무를 충족한다. 독립 재계산이 §225 표와 일치했다.
fold 정렬 검증 350건 전수 통과. 과제 간/과제 내 분산비: F3a 125.9 · F3b 48.0 · F1 1.31 · F2 0.20 · T1 0.17 · T2 0.30.
실측: wall 695초 · GPU 0 · CPU float64 · 피크 RSS 약 34GB.
- **결정 (Decision)**: 증거 판정: 조합별로 나뉜다 — 반박 5건(F1×T1·F1×T2·F2×T1·F2×T2·F3a×T1, 95% 구간이 사전 등록한 관심 크기 |ρ|=0.3을 배제), 판별 불가 3건(F3a×T2·F3b×T1·F3b×T2, 구간이 0.3을 포함). '충족 0개'로 뭉뚱그리지 않는 것은 보편 규범 §4.3(효과 없음과 판별 불가의 구분) 때문이다.
운영 결정: 보류. **CA-R1을 닫지 않는다** — 카드의 [닫지 않는 것] 조항대로, 지문 3종·표적 2종의 실패는 라벨 무관 신호 일반의 부재가 아니며 과제 군집 7로는 부재를 입증할 검정력이 없다.
새 지문 F1·F2는 두 표적 모두에 대해 관심 크기의 예측력이 없다는 쪽으로 반박됐으므로 보류한다. 대조군 F3(Task-Geometry 계열)은 이 RU로 판정되지 않았으므로 큐에 남긴다 — 카드 next_actions의 'F3만 유의' 분기는 F3b 구간이 0을 포함하므로 발동하지 않는다.
- **결과별 후속 행동**: 확증 후보 발생(구간이 0 배제 그리고 |ρ|>=0.3): 해당 조합을 research_directions.md §0 큐에 P1으로 등재하고 **사전 등록 확증 RU로 분리**한다. **같은 자료로 확증하지 않는다** — 선택에 쓴 자료임을 명시하고 별도 설계를 만든다(§214 교훈). 이 RU 자체는 채택 근거가 아니다.
전부 판별 불가: CA-R1을 닫지 않고 `fold 수준 해상도 접근의 예측력 미확인`으로 기록한다. 제안 후보 2·5의 경계 판정을 사용자에게 올려 다음 경로를 정한다.
F3(대조군)만 유의: 새 지문의 기여가 없다는 뜻이므로 Task-Geometry를 큐에 올리고 F1·F2는 보류한다.
실행 무효: 정렬 문제를 고쳐 재실행한다. 부분 자료로 결론을 적지 않는다.
- **원문 근거 (Evidence)**:
  - docs/reports/RU-88_car1_fingerprint_panel.md
  - predictions/ru88_fingerprint/summary.json
  - scripts/analysis/ru88_fingerprint.py
  - logs/ru88_fingerprint.log
- **선행·후속 관계 (Relations)**: closed_axes.md CA-R1(닫지 않음)·CA-07·CA-11. 후보 출처는 docs/proposals/2026-09-08_car1-label-free-task-identification.md 후보 1·3·4. T2 원자료는 RU-86, T1 원자료는 §225 v121_ds_sweep. research_directions.md §0 P1-B 행 B1. 경계 미확정 후보 2·5는 closed_axes.md §3-I·§3-J.
- **확인 필요 사항 및 한계 (Uncertainties)**: 설계 전제가 부분적으로만 성립했다. '해상도를 fold로 내려 표본을 350으로 늘린다'는 전제와 달리 추론 구간은 과제 군집(df=6)이 지배한다. 지문이 과제 내에서 거의 변하지 않으면 실효 표본은 여전히 7이다. F1(분산비 1.31)·F2(0.20)에서는 fold 해상도가 실제로 정밀도를 샀고(F2×T2 구간 폭 0.21) 그래서 반박까지 갈 수 있었으나, F3a(125.9)·F3b(48.0)는 사실상 과제 수준 회귀변수여서 사지 못했다 — 판별 불가 3건이 전부 F3 쪽인 이유다.
기전 미확인. F1·F2가 왜 예측력이 없는지 설명하지 못했다.
제안서 후보 1의 자체 반증 조건 중 '이상치 근접도가 슬라이드 패치 수의 대리변수인가'는 미측정이다 — 카드 패널 밖이라 kill ④를 우선했다. GPU 0·수 초로 추가 가능하다.
패널 확장 1건 공개: 카드가 F3를 두 통계로 적으면서 표를 3×2=6으로 고정한 것은 카드 자체의 모호성이다. Coding은 하나를 버리지 않고 F3b를 둘째 성분으로 추가 산출·전량 보고했다. 8개 전부 미달이므로 이 추가가 선택 편향을 만들지 않는다.
카드가 '상관 ρ'로만 적어 추정량이 모호했다. Coding이 실행 전 스크립트 독스트링에 Pearson을 primary로 선언하고 Spearman을 기술 통계로 병기했다 — 사후 선택 여지는 없다.
범위: Primary 7 · official 50 fold · v121. SEAL hold-out 미검증.

---

### RU-89. 5-branch + SH + SJ 7-branch paired comparison and shape-family additivity

- **일자 (Date)**: `2026-09-09` ~ `2026-09-09`
- **커밋 범위**: - (`265fb05e` ... `25a58e67`)
- **작업 유형**: `experiment`
- **질문 (Question)**: SH(Shape Moments)와 SJ(자체백색화 반경 형상)를 동시에 투입한 7-branch(CV,BM,BD,QA,DS,SH,SJ)는 공식 5-branch 기준선(0.6171) 대비 승격 기준을 넘는가? 그리고 같은 형상 계열인 두 브랜치의 이득은 가산적인가?
- **가설 (Hypothesis)**: [사전 선언] SH(+0.29%p)와 SJ(+0.32%p)는 같은 형상 계열이라 중복되어, 동시 투입 이득이 단순합(+0.61%p)에 크게 못 미친다. 기전 미확인 — SH-SJ 상호 |r|이 미측정이었다. [Phase 0 결과에 의해 도전받음] 상관 실측 결과 SH-SJ max |r| = 0.177(과제별 -0.022~0.177)로 거의 직교했다. 즉 '중복' 가설의 전제가 상관 수준에서는 성립하지 않는다. 가설은 사후 수정하지 않고 그대로 두며, Phase 1의 판정으로 검증한다.
- **판정 기준 (Criteria, 사전 고정)**: [Phase 1 착수 전 고정] 대응 per-fold Δ(7과제×50fold=350)의 과제 군집 t 구간(df=6) 95%로 판정한다. Δ_i = (7-branch fold-i AUROC) − (5-branch fold-i AUROC), 같은 fold에서 짝지어 계산. 구간 하한 > +0.3%p(δ_min) → 지지 / 구간이 0을 포함 → 판별 불가 / 구간 상한 < +0.3%p → 반박. sign agreement(7과제 중 개선 수)는 보조 지표로만 보고하며 단독 판정 근거가 아니다. 악화된 과제는 전량 명시한다. 가산성은 부차 지표로 관측치 Δ(SH+SJ)와 Δ(SH)+Δ(SJ)의 차로 보고하되 이것으로 지지/반박을 판정하지 않는다.
- **예산 / 중단 조건**: Phase 0(SH-SJ 상관, 저장 마진 오프라인 재집계): GPU 0h — 집행 완료. Phase 1(7-branch 350 fold 평가, GPU 6·7만 사용): 상한 2.0 GPU-h. 기준 단가 스윕 1회 0.7~1.0 GPU-h(research_directions.md:20) 기준 대조 2 arm. / ① 오염 검사 불일치 — SJ 배선 추가 후 기본값 off 상태에서 SMAD4 0.4421 / PBRM1 0.5553이 소수 4자리로 재현되지 않으면 기준선 오염이므로 즉시 중단한다(PROJECT.md §3.4). ② [사전 선언, Phase 0 실행 전 고정] SH-SJ max |r| > 0.6이면 게이트 ① 위반이므로 Phase 1을 집행하지 않고 종료한다 → 실측 0.177로 미발동, Phase 1 진행. ③ 예산 2.0 GPU-h 초과 시 중단. ④ GPU 6·7 외 장치를 점유하게 되면 중단.
- **실험 및 변경 (Experiment)**: Phase 0: scripts/analysis/branch_screen.py --tag v121_sh_variants --candidate m_sh --adopted m_sj (GPU 0, 집행 완료). Phase 1: SJ 로짓 융합 배선(ICF_FIXED_HEAD_SJ_WEIGHT) 추가 후 5-branch arm과 7-branch arm을 동일 조건 Primary 7 × 50 fold로 실행하고 저장 마진에서 대응 Δ를 산출한다. GPU는 6·7만 사용한다.
- **관측 결과 (Observations)**: [Phase 0 · GPU 0] SH 대 기존 브랜치 max |r| = 0.418 → ADMIT. SH-SJ 상호 |r| = 0.141/-0.022/0.177/0.146/0.171/0.075/0.103 (max 0.177) — 거의 직교. eff.rank 2.26/5 → 2.81/6 (효율 45% → 47%) 유지. [Phase 1 · GPU 0 · 저장 마진 오프라인 재집계, tag=v121_sh_variants] 충실성 검증: BASE macro 0.6171(공식 기준선 일치), SMAD4 0.4421 / PBRM1 0.5553(오염 검사 상수 소수 4자리 일치). macro: BASE 0.6171 / +SH 0.6197 / +SJ 0.6202 / +SH+SJ(7-branch) 0.6231. 대응 per-fold Δ의 과제 군집 95% t 구간(df=6): +SH +0.26%p [-0.69, +1.21] sign 4/7 · +SJ +0.32%p [-0.79, +1.42] sign 3/7 · +SH+SJ +0.60%p [-1.05, +2.25] sign 4/7. 가산성 gap = +0.60%p − (+0.26%p + +0.32%p) = +0.03%p — 사실상 완전 가산. 악화 과제(전량): 7-branch에서 Histologic_Grade, progression_regression, PBRM1 3건. 이 3건은 +SH·+SJ 단독에서도 동일하게 악화되어 방향이 일관된다. 기전 미확인.
- **결정 (Decision)**: [증거 판정] 판별 불가. 7-branch의 대응 Δ 95% 구간 [-1.05%p, +2.25%p]가 0을 포함한다. 사전 고정 기준(하한 > +0.3%p → 지지 / 0 포함 → 판별 불가 / 상한 < +0.3%p → 반박)을 그대로 적용했다. 점추정 +0.60%p는 δ_min을 넘지만 구간이 넓어 확증되지 않는다. sign agreement 4/7로 보조 지표도 승격선(≥5/7) 미달이다. [사전 가설에 대한 판정] 반박. '같은 형상 계열이라 중복되어 단순합에 크게 못 미친다'는 가설은 가산성 gap +0.03%p로 반박됐다. 이득은 사실상 가산적이며 Phase 0의 직교성 실측(max |r| 0.177)과 정합한다. 가설을 사후 수정하지 않고 반박된 채로 남긴다. [운영 결정] 보류. 승격 심사에 상정하지 않는다. 축은 닫지 않는다 — 부재를 입증한 것이 아니라 과제 군집 7의 검정력으로 판별하지 못한 것이다. [부수 확정 사실] 7-branch 평가는 GPU를 전혀 쓰지 않고 수행 가능하다. 저장 마진 오프라인 재집계로 충분하며, 사전 배정한 2.0 GPU-h는 집행하지 않았다.
- **결과별 후속 행동**: 지지(구간 하한 > +0.3%p) → 7-branch를 공식 구성 교체 후보로 승격 심사에 상정하고, hold-out 미검증을 명시한 채 사용자 판단을 요청한다. 판별 불가(0 포함) → 관측 구간과 검정력을 기록하고 보류한다. 축은 닫지 않는다. 반박(상한 < +0.3%p) → 형상 계열 동시 투입으로 승격에 도달하는 경로를 종료하고 closed_axes.md 상정을 검토한다. 세 경우 모두 SH-SJ 직교성 실측(0.177)은 별도 사실로 기록해 이후 후보 심사의 게이트 ① 기준집합에 반영한다.
- **원문 근거 (Evidence)**:
  - scripts/analysis/ru89_shape_joint.py (신규, commit 25a58e6) — 오프라인 재집계·군집 t 구간 산출
  - scripts/analysis/branch_screen.py --tag v121_sh_variants --candidate m_sh --adopted m_sj — Phase 0 상관
  - predictions/pathobench_{PRIMARY7}_v121_sh_variants_official50_bf16.pt — m_sh·m_sj 저장 마진
  - 회귀 스위트 121 tests OK
- **선행·후속 관계 (Relations)**: §218(SH 채택, archive.md:1411-1451) · §219-§220(SJ 채택, archive.md:1484-1571) · D-005(CT 제외 5-branch 기준) · D-018/D-019/D-021/D-041(승격 기준·정밀도 — 2026-09-10 D-041 사용자 명확화에 따라 7-branch 평균 대응 Δ +0.60%p는 확정 개선량 기준 +0.3%p를 충족함. 당시 구간 하한 기준에 따른 판별 불가·보류 판정 기록은 보존) · D-038(SJ 명칭) · D-039(SH 통합). 선행 코드 작업: c050a13(SH를 src/models로 통합), 그리고 본 RU를 위한 SJ 융합 배선 추가.
- **확인 필요 사항 및 한계 (Uncertainties)**: ① 검정력이 결론을 지배한다. 과제 군집 SE가 +SH 0.39%p → +SJ 0.45%p → 7-branch 0.67%p로 커져 δ_min = 0.3%p를 구간으로 가르지 못한다. RU-80의 대응 SE 0.022%p는 거의 동일한 두 arm의 비교값이며 여기에 적용되지 않는다(PROJECT.md §4.1의 '공통 검출 한계가 아니다'). ② Trimmed Mean은 최저·최고 1개씩 절사하므로 브랜치 수가 5에서 7로 늘면 절사 비율이 바뀐다. 이 집계 규칙 변화가 Δ에 기여한 몫은 분리하지 않았다. ③ Histologic_Grade·progression_regression·PBRM1 3과제에서 형상 계열이 일관되게 해로운 이유는 기전 미확인이다. ④ 모든 판정은 hold-out 미검증이다(PROJECT.md §3.1). ⑤ 상관이 낮다는 것과 이득이 가산적이라는 것은 별개 사실이며, 여기서는 둘 다 관측됐을 뿐 전자가 후자를 함의한다고 주장하지 않는다.

---

### RU-90. Live-path equivalence of the corrected BS/SH/SJ wiring and single-generation re-measurement of the SH/SJ arms

- **일자 (Date)**: `2026-09-10` ~ `2026-09-10`
- **커밋 범위**: - (`c5a6c90d` ... `4a53636c`)
- **작업 유형**: `reproduction_measurement`
- **질문 (Question)**: 정정된 배선(c5a6c90)에서 공식 live 평가 경로가 저장 마진의 오프라인 재집계와 동일한 결과를 내는가? 그리고 BS·SH·SJ를 한 코드 세대·한 실행에서 동시에 스크리닝한 마진으로 재산출한 +SH·+SJ·+SH+SJ의 대응 per-fold Δ가 RU-89 값을 재현하는가? BS는 게이트 ② 기각 상태(31/50, p=0.059)이므로 사용자 결정에 따라 arm에서 제외하고 마진만 저장한다.
- **가설 (Hypothesis)**: [사전 선언] (1) 배선 등가성은 성립한다 — live 집계와 오프라인 재집계는 같은 마진에 같은 수식을 적용하므로 부동소수 오차 내에서 일치할 것이다. (2) Δ 재현은 성립한다 — RU-89의 Δ와 |차| ≤ 0.10%p로 일치할 것이다. [이 가설을 위협하는 사전 관측] v121_shape_screen(§218)과 v121_sh_variants(§219) 두 태그에서 5개 기본 브랜치(m_cv·m_bm·m_bd·m_qa·m_ds)와 label은 비트 단위 동일(max|diff| = 0)한데 m_sh만 max|diff| 2.5e-2~6.5e-2로 달랐다. 두 실행 스크립트의 ICF_SH_DIM=32·ICF_SH_WIDE=256은 동일하므로 저장 마진이 서로 다른 코드 세대의 산출물임을 뜻한다. 기전 미확인 — 투영 폭(32 대 256)에 따른 bf16 누적 차이인지 §219의 SH 정의 변경인지 판별되지 않았다. 따라서 (2)는 반증될 수 있고, 가설은 사후 수정하지 않는다.
- **판정 기준 (Criteria, 사전 고정)**: [Phase 실행 전 고정] 3층으로 나누어 판정한다. 층 1 계측 무결성(필수): BASE 5-branch(CV,BM,BD,QA,DS · trimmed_mean)의 Primary 7 macro가 0.6171이고 오염 검사 상수 SMAD4 0.4421 · PBRM1 0.5553이 모두 소수 4자리로 일치한다(PROJECT.md §3.2·§3.4). 불일치 → 판정 `실행 무효`, 즉시 중단. 층 2 배선 등가성(필수): 동일 fold·동일 마진에서 (a) ICF_SHAPE_SCREEN_ONLY=0으로 실행한 live 경로의 per-slide 확률과 (b) 저장 마진의 오프라인 재집계 확률이 max|diff| ≤ 1e-6이고 per-fold AUROC가 소수 6자리 일치한다. 대상 집계는 공식 trimmed_mean 1종, 대상 구성은 BASE·+SH·+SJ·+SH+SJ 4개 arm 전부. 불일치 → 판정 `배선 결함 잔존`, 원인 진단으로 전환. 층 3 RU-89 대조(진단, 필수 아님): 새 런 마진으로 산출한 대응 per-fold Δ의 과제 군집 평균이 RU-89 값(+SH +0.26%p · +SJ +0.32%p · +SH+SJ +0.60%p)과 |차| ≤ 0.10%p면 `코드 세대 무관`, 초과면 `세대 간 SH 정의 변화 확증`으로 적는다. 승격(§4 δ_min = +0.3%p, 대응 per-fold Δ의 과제 군집 95% t 구간 하한)은 이 RU에서 판정하지 않는다 — RU-89가 같은 arm 집합을 이미 `판별 불가`로 종료했고 본 RU의 유형은 재현·계측이다. 관측 후 이 기준을 바꾸지 않으며, 변경이 필요하면 원래 기준과 함께 드러낸다.
- **예산 / 중단 조건**: 상한 2.0 GPU-h. 실측 앵커: Primary 7 × 50 fold 1 arm = 0.9~1.0 GPU-h(§214-V 5-GPU 병렬 12분, §219 로그 과제별 소요 합 3,262 GPU-초). 본 RU는 screen-only 단일 런 1회로 4개 arm 전부를 오프라인 재집계하므로 1 arm 단가가 주 예산이고, 나머지 약 1.0 GPU-h는 층 2의 live 경로 확인 재실행 몫이다. 사용자 승인(2026-09-10): GPU는 실행 직전 nvidia-smi로 유휴 장치를 찾아 쓰고 개수 제한은 없다. / ① 층 1 오염 검사 3개 상수 중 하나라도 소수 4자리 불일치 → 기준선 오염이므로 즉시 중단한다. ② 예산 2.0 GPU-h 초과 → 중단한다. ③ 유휴 GPU가 없어 타 사용자 프로세스를 점유하게 되는 경우 → 중단한다. 실행 직전 nvidia-smi로 유휴 장치를 확인하고 그 목록만 쓴다. ④ 층 2에서 live 경로와 오프라인 재집계의 max|diff|가 1e-3을 초과 → 단순 부동소수 문제가 아니므로 추가 arm 실행을 멈추고 배선 진단으로 전환한다. ⑤ 저장된 m_sh·m_sj(또는 m_shj)·m_bs 중 하나라도 전 fold None으로 나오면 → 스크리닝 배선 실패이므로 중단하고 원인을 고친 뒤 재개한다.
- **실험 및 변경 (Experiment)**: Phase 1: scripts/run_ru90_shape_triple.sh — v121 arm에 BS·SH·SJ를 동시 screen-only(ICF_SHAPE_SCREEN_ONLY=1)로 켜고 Primary 7 × 50 fold를 유휴 GPU 6장(0,2,3,4,6,7)에 round-robin 팬아웃. tag=ru90_shape_triple. screen-only이므로 이 런의 확률 자체가 공식 5-branch BASE arm이다. Phase 2: scripts/analysis/ru90_wiring_equivalence.py --phase1 로 층 1과 층 3을 오프라인 산출(GPU 0h). Phase 3: scripts/run_ru90_live_arms.sh — ICF_SHAPE_SCREEN_ONLY=0으로 +SH·+SJ·+SH+SJ 3 arm을 ucla_lung/progression_regression 1과제에 live 실행(3 GPU 병렬, tag=ru90_live_{sh,sj,shsj})하고 --phase3 으로 층 2를 산출. 과제 선택 근거는 Phase 1 실측에서 가장 빠른 과제(151초)라는 것이며, 등가성은 정확 산술이라 과제 수가 증거력을 더하지 않는다. BS는 마진만 저장하고 어떤 arm에도 넣지 않았다.
- **관측 결과 (Observations)**: [층 1 계측 무결성 — 통과] Primary 7 macro 0.617054(소수 4자리 0.6171, PROJECT.md §3.2 기준 일치). 더 강한 결과로 7개 과제 fold-mean AUROC가 공식 v121_baseline과 전부 max|diff| = 0.000e+00으로 비트 단위 동일했다 — screen-only가 앙상블을 건드리지 않았을 뿐 아니라 배선 리팩터가 5-branch 경로를 정확히 보존했다. 오염 검사: SMAD4 0.442138(0.4421) MATCH · PBRM1 0.555330(0.5553) MATCH. 마진 존재(중단 조건 ⑤): m_sh 50/50 · m_sj 50/50 · m_bs 50/50, 7과제 전부. 미발동. [층 2 배선 등가성 — 통과] BASE arm: 저장 probability 대 오프라인 trimmed_mean 재집계 max|diff| = 1.788e-07, per-fold AUROC 소수 6자리 일치(7과제 전부). live 3 arm: ref 태그와 m_cv·m_bm·m_bd·m_qa·m_ds·m_sh·m_sj 전부 max|diff| = 0.000e+00(결정성 확인), live 확률 대 오프라인 재집계 max|diff| = 1.788e-07(3 arm 동일), per-fold AUROC max|diff| = 0.000e+00. 기준 1e-6 충족. [층 3 RU-89 대조 — 코드 세대 무관] macro BASE 0.6171 / +SH 0.6198 / +SJ 0.6202 / +SH+SJ 0.6227. 대응 per-fold Δ의 과제 군집 95% t 구간(df=6): +SH +0.27%p [-0.68, +1.23] sign 4/7 · +SJ +0.32%p [-0.79, +1.42] sign 3/7 · +SH+SJ +0.56%p [-1.15, +2.28] sign 4/7. RU-89 값과의 차: +SH 0.01%p · +SJ 0.00%p · +SH+SJ 0.04%p — 모두 사전 고정한 0.10%p 이내. sign agreement와 악화 과제 목록이 RU-89와 동일하다. 악화 과제(전량): +SH+SJ에서 Histologic_Grade · progression_regression · PBRM1 3건이며 +SH·+SJ 단독에서도 같다(+SJ는 KRAS 추가로 4건). 가산성 gap -0.03%p(RU-89 +0.03%p) — 부호는 뒤집혔으나 크기가 같아 사실상 가산이라는 결론은 동일하다. SH-SJ 상호 상관 max|r| = 0.180(RU-89 0.177). [부수 관측] test_pathobench 내부 AUROC와 branch_diagnostics.auroc는 동일 확률 벡터에서 +8.55e-05의 고정 차를 낸다(3 arm 전부 동일). 짝지은 Δ에서 상쇄되지만 절대 AUROC를 두 출처에서 섞어 인용하면 소수 4자리에서 어긋난다. 기전 미확인. [예산 집행] Phase 1 4,690 GPU-초 + Phase 3 296 GPU-초 = 4,986 GPU-초 = 1.385 GPU-h (상한 2.0).
- **결정 (Decision)**: [증거 판정] 층 1 통과 · 층 2 통과 · 층 3 `코드 세대 무관`. 사전 고정한 3층 기준을 그대로 적용했고 관측 후 변경하지 않았다. 사전 가설 (1)·(2) 모두 지지됐다 — 다만 (2)를 위협했던 관측(두 태그의 m_sh가 max|diff| 6.5e-2로 달랐다)은 사라지지 않았다. 세대 차는 실재하되 과제 군집 평균 Δ까지 전파되지 않는다는 것이 밝혀진 것이며, 개별 과제 Δ는 최대 0.0033(progression_regression의 +SH+SJ가 -0.0113 대 -0.0080)까지 달랐다. m_sh 세대 차의 기전은 여전히 미확인이다. [운영 결정] 배선 수정(c5a6c90·9633199)을 확정한다. RU-89의 증거 판정(`판별 불가`)과 운영 결정(`보류`)을 그대로 유지한다 — 본 RU는 재현·계측이며 승격을 판정하지 않았다. closed_axes.md는 변경하지 않는다. 배선 결함과 그 수정을 결정 이력 `D-040`으로 남긴다. [BS] 게이트 ② 기각 상태(31/50, p = 0.059)는 변하지 않는다. 마진 50/50 저장을 확인했을 뿐이며 어떤 arm에도 넣지 않았고, 어떤 구성에서도 승격 심사에 상정하지 않는다. 부호 반전 BS는 사용자 결정으로 등록하지 않았다. [부수 확정 사실] 정정된 live 경로는 오프라인 재집계와 1.788e-07 안에서 일치한다. 따라서 형상 계열의 오프라인 재집계 수치는 앞으로 live 경로의 대리값으로 쓸 수 있고, 그 근거는 본 RU다.
- **결과별 후속 행동**: · 층 1 통과 & 층 2 통과 & 층 3 `코드 세대 무관` → 배선 수정을 확정하고 RU-89의 판정(판별 불가)과 운영 결정(보류)을 그대로 유지한다. closed_axes.md는 건드리지 않는다. 배선 결함과 수정을 결정 이력에 레코드로 남긴다. · 층 1 통과 & 층 2 통과 & 층 3 `세대 간 변화 확증` → 배선 수정은 확정하되 RU-89·§218·§219의 SH 관련 수치에 `해당 코드 세대 한정`을 명기하고, 세대 간 SH 정의 차이의 기전 규명을 별도 진단 RU 후보로 큐에 올린다. RU-89의 운영 결정은 바꾸지 않는다. · 층 2 불통과 → 배선 수정이 미완이라는 뜻이므로 결함 위치를 진단해 고치고 본 RU를 재실행한다. 층 3은 판정하지 않는다. · 층 1 불통과 → `실행 무효`로 적고 기준선 오염 원인을 진단한다. 배선 수정 커밋을 되돌릴지 여부를 사용자에게 상정한다.
- **원문 근거 (Evidence)**:
  - scripts/analysis/ru90_wiring_equivalence.py (신규, commit 4a53636) — 3층 오프라인 산출. v121_sh_variants에서 RU-89를 |차| 0.00%p로 재현해 정확성을 검증했다
  - scripts/run_ru90_shape_triple.sh · scripts/run_ru90_live_arms.sh · scripts/lib/free_gpus.sh (commit fa8b2c8)
  - predictions/pathobench_{PRIMARY7}_ru90_shape_triple_official50_bf16.pt — m_sh·m_sj·m_bs 50/50 저장
  - predictions/pathobench_ucla_lung_progression_regression_ru90_live_{sh,sj,shsj}_official50_bf16.pt — 층 2 live arm
  - 배선 수정 commit c5a6c90 (5개 집계 배선·중복 제거), 건전성 commit 9633199 (context_loo 빈 pool 가드·import 호이스트·불변성 테스트)
  - 회귀 스위트 153 tests (test_bs_branch.py 18건 · test_shape_branch_invariance.py 12건 신설)
- **선행·후속 관계 (Relations)**: RU-89(동일 arm 집합을 오프라인 재집계로 `판별 불가` 종료 — 본 RU가 그 수치의 live 재현성을 확보했다) · §218(BS·SH 스크리닝, BS 게이트 ② 기각, archive.md:1402-1452) · §219-§220(SJ 채택) · D-038(SJ 명칭) · D-039(SH src 통합 및 context_loo에서 SJ 누락 결함 수정 — 본 RU의 결함은 같은 종류의 누락이 평가 경로에 남아 있던 것이다) · D-040(본 RU가 근거인 배선 결함 수정 레코드) · D-041(사용자 승격 기준 명확화 — 7-branch 평균 대응 Δ +0.56%p는 확정 개선량 기준 +0.3%p를 충족함) · PROJECT.md §3.2(기준선 0.6171) · §3.4(오염 검사 상수) · §4(승격 기준, 본 RU에서는 판정하지 않음) · §5(게이트, BS 기각 근거)
- **확인 필요 사항 및 한계 (Uncertainties)**: ① 층 2는 단일 과제(progression_regression) 50 fold, 공식 trimmed_mean 1종에서 확인했다. 등가성은 정확 산술이므로 과제 확대가 증거력을 더하지 않지만, 나머지 4개 집계(soft_voting·hard_gated·adaptive_trimmed·context_loo)에서 형상 계열이 pool에 들어가는 것은 회귀 테스트로만 고정됐고 실측 평가로는 확인되지 않았다. ② m_sh의 세대 간 차이(max|diff| 6.5e-2)의 기전은 미확인이다. 집계량에는 전파되지 않았으나 개별 과제 Δ는 최대 0.0033 달랐다. 상관이 낮다는 것과 집계량이 안정적이라는 것은 별개 사실이며 전자가 후자를 함의한다고 주장하지 않는다. ③ AUROC 추정기가 두 개다 — test_pathobench 내부와 branch_diagnostics.auroc가 동일 입력에서 +8.55e-05 차를 낸다. 짝지은 Δ에서 상쇄되므로 판정에는 영향이 없으나, 절대 AUROC를 두 출처에서 섞어 인용하면 소수 4자리에서 어긋난다. 어느 쪽이 옳은지는 판정하지 않았다. ④ 모든 판정은 hold-out 미검증이다(PROJECT.md §3.1). ⑤ Trimmed Mean은 최저·최고 1개씩 절사하므로 브랜치 수가 5에서 6·7로 늘면 절사 비율이 바뀐다. 이 집계 규칙 변화가 Δ에 기여한 몫은 RU-89와 마찬가지로 분리하지 않았다. ⑥ Histologic_Grade·progression_regression·PBRM1 3과제에서 형상 계열이 일관되게 해로운 이유는 기전 미확인이다. RU-89와 동일한 3과제이며 방향도 같다.

---
