# 최근 100개 커밋 연구 단위 감사 및 이력 정리 보고서 (Research Units 000–099)

> **역사 기록의 적용 범위**: 아래 과거 판정·인용은 당시 기록이다. 현행 승격·정밀도는
> [`PROJECT.md`](../PROJECT.md), 경계는 [`closed_axes.md`](../closed_axes.md)를 따른다.
> 옛 컷오프·통계 금지·경계 미확정의 정정은 [결정 이력](archive.md)의 `D-021`·`D-022`에 있다.

> **감사 기준선 및 개요**  
> - **고정 HEAD**: `f3f0b201f85d2bb7eafe1e63561e2bbda2b277c1` (Index 99)  
> - **시작 커밋**: `aac69013e4d3def23977cff98e13f5a0c1c58ddc` (Index 0)  
> - **대상 범위**: 100개 커밋 (`first-parent` 순서, 인덱스 0~99 전수 검증 완료)  
> - **식별된 연구 단위**: 총 38개 (`RU-01` ~ `RU-38`)  
> - **원문 인용 검증**: 총 73건 (고정 커밋 시점의 파일 라인 및 텍스트 100% 비트 매칭 확인)  
> - **작성 일자**: 2026-09-05 (KST)  

---

## 1. 개요 및 방법론

본 감사는 ICF(In-Context Foundation) 저장소의 최근 100개 커밋을 **연구 질문(Research Question)과 가설(Hypothesis)**을 기준으로 압축·구조화하고, 당시의 관측과 판단, 그리고 이후 밝혀진 정정 관계를 빠짐없이 추적하여 연구 자산으로 확립하는 것을 목적으로 합니다.

### 5대 감사 원칙 적용
1. **고정 커밋 범위 (Fixed 100 First-Parent SHAs)**: 시작 시점의 HEAD(`f3f0b201`)를 기준으로 100개 커밋(Index 0~99)을 고정하여 실행 도중 발생할 수 있는 범위 왜곡을 방지했습니다. 0번 이전부터 진행된 연구(`RU-01`)는 '시작 경계 밖에서 착수된 연구'로 명시하고, 99번 시점에 진행 중인 연구(`RU-38`)는 '진행 중'으로 보존했습니다.
2. **변경분(Diff Packet) 중심 순회**: 0번 커밋에서 당시 존재하는 대상 문서의 기준선을 파악한 후, 1~99번은 패킷화된 문서 diff와 전후 문맥을 검토하여 요약을 갱신했습니다. 특히 `docs/history.md`와 `docs/history/archive.md`를 모두 교차 추적하여 사후 정정 기록의 누락을 원천 차단했습니다.
3. **질문·가설 중심의 경계 분리**: 단순 기술 수정이나 버전 변경이 아닌, **'동일한 질문에 답하기 위한 구현·수정·실험인가'**를 기준으로 연구 단위를 결합했습니다(예: DD ordered×typicality 구현·버그 수정과 v112 승격을 `RU-09`로 통합). 새로운 가설이나 독립적 검증 질문이 등장할 때만 단위를 분리했습니다.
4. **커밋 순회 인덱스와 연구 시작점의 분리**: 커밋 순회 인덱스(`cursor`)와 연구 단위 시작점(`unit_start`)을 분리 관리하여 연속된 커밋군이 하나의 가설 검증 흐름을 이룰 때 실제 착수 지점을 정확히 보존했습니다.
5. **당시 기록 추출 및 정정 관계 연결**: 기록에 명시되지 않은 가설이나 결정은 임의 추정하지 않고 명시된 사실만을 추출했습니다. 후속 커밋에서 오류가 밝혀져 판정이 뒤집히더라도 과거 기록을 소급 삭제하지 않고, `선행·후속 관계`를 통해 상호 연결했습니다.

---

## 2. 연구 단위 요약 일람표 (RU-01 ~ RU-38)

| ID | 커밋 범위 | 작업 유형 | 제목 | 핵심 결정 | 후속 정정 / 연결 |
|:---|:---:|:---|:---|:---|:---|
| **RU-01** | 0–1 | 시작 경계 밖에서 착수된 연구 | CT 부분공간과 ridge readout의 결합 | raw 조건에 한정됐던 readout 결론을 정정하고 PCA.. | E05의 token 생성 개선 이전의 CT 기준선이.. |
| **RU-02** | 2 | 가설 검증 | DD의 가중치와 코호트 의존성 | 전역 weight는 고르지 않고 0.343을 유지했다. co.. | RU-03~E09의 DD readout 탐색으로 이.. |
| **RU-03** | 3–4 | 가설 검증 | DD LLR 보정과 상대 거리의 판정 가능성 | LLR 훅은 보정·pooled 평가용으로만 남기고 승격하지 .. | E02의 DD 문제를 전역 가중 대신 margin .. |
| **RU-04** | 5–11 | 가설 검증 | CV descriptor의 정보 분해 | v109에서 off-diagonal CV만 유지하고 √λᵢλ.. | E05의 v109 승격과 E01의 CV-CT 공유 .. |
| **RU-05** | 6–7 | 승격 결정 | CT token 생성 개선과 v109 | CV off-diagonal과 k-means CT weigh.. | E04의 CV 정리와 E06의 cells/token.. |
| **RU-06** | 8–12 | 승격 결정 | CT 표본 수·token 수·ridge λ의 분리 | cells_per_bag=64와 CT tokens=32를 v.. | E05를 확정하고 E08의 full-cell tok.. |
| **RU-07** | 13 | 인프라 기록 | 서버 이동용 실행 환경 정비 | 후속 CT/DD 실험은 중앙 node 설정과 갱신된 hand.. | E08의 CT 평가와 E09의 GPU worker .. |
| **RU-08** | 14–24 | 탐색 및 운영 선택 | 대체 CT dictionary와 full-cell hierarchical v111 | 사용자 결정으로 v111을 공식 baseline으로 채택하고.. | E06의 v110은 historical predic.. |
| **RU-09** | 25–30 | 가설 검증 및 승격 채택 | DD ordered×typicality 후보의 구현·수정 및 v112 승격 | 사용자 결정으로 v112를 활성 baseline으로 승격하고.. | E03의 LLR/상대거리 기각 후속 대안이며, M0.. |
| **RU-10** | 31 | feasibility_decision | CT fraction sampling feasibility | 예측 개선이 아니라 평가 가능성을 이유로 v113을 승격했다. | RU-09, RU-11 |
| **RU-11** | 32–33 | controlled_evaluation | unit fixed-head weight 통일 | 사용자 결정으로 v114를 승격했다. Primary 7·ho.. | RU-10, RU-15 |
| **RU-12** | 34 | rejection | CT kernel ridge와 top-k pooling 기각 | 세 변형을 모두 기각하고 재현 코드만 보존했다. | RU-11 |
| **RU-13** | 35–44 | research_infrastructure | 모듈 분해와 BM 구현·문서 SSOT | M05 이후 BM 성능 비교는 이 구현·문서 기반을 전제로 .. | RU-15, RU-21 |
| **RU-14** | 45 | evaluation_protocol | Primary 7과 SEAL hold-out 역할 전환 | 후속 branch 평가는 Primary 7로 선택하고 SEA.. | RU-13, RU-15 |
| **RU-15** | 46 | branch_promotion | Projected Bag-Mean(BM) 채택 | 사용자 승인으로 BM 포함 v115를 승격했다. | RU-13, RU-16 |
| **RU-16** | 47 | branch_promotion | Bag-dispersion spectral entropy(BD) 채택 | 당시 사용자 승인으로 BD 포함 v116으로 승격했다. | RU-15, RU-17 |
| **RU-17** | 48–50 | aggregation_promotion | DD 제거와 soft-vote v118 | v118을 활성으로 승격하고 SEAL/지도학습 표는 hold.. | RU-16, RU-18 |
| **RU-18** | 51–52 | branch_and_aggregation_promotion | QA branch와 trimmed-mean v119 | CV를 유지한 QA 5-branch trimmed-mean .. | RU-17, RU-20, RU-21 |
| **RU-19** | 53 | branch_promotion | DS salience-denoising과 v120 | 사용자 결정으로 DS 포함 v120을 당시 baseline으.. | RU-18, RU-20 |
| **RU-20** | 54–56 | postmortem_and_benchmark | v120 사후 실패 축과 CT 단독 | 네 축을 기각하고 CT는 단독 기준으로 보되 앙상블 대체로 .. | RU-18, RU-21 |
| **RU-21** | 57–62 | reproducibility | v120 harness·arm 모듈화와 결정성 경계 | tolerance 통과는 engineering check로만.. | RU-18, RU-22 |
| **RU-22** | 63–64 | configuration_cleanup | obsolete harness 제거와 학습 config 아카이브 | 학습 계보는 archive/git history에 보존하고 .. | RU-21 |
| **RU-23** | 65–75 | infrastructure | 재현 기반 정비 | 성능 승격이 아닌 이후 비교의 재현 기반으로 채택했다. | RU-24 이후 비교의 기반 |
| **RU-24** | 76 | benchmark/correction | §209 CT 전사 오류와 단독 기준선 | CT를 단독 챔피언으로 취급하지 않고 비용을 기준 설정에 반.. | RU-28 CT-제외 공식 비교 |
| **RU-25** | 77 | experiment | §210 sub-bag/TTA의 과제별 상충 | 국소 변이를 보존한다는 anchor 가설을 이월했다. | RU-28/RU-36/RU-37 anchor 계보 |
| **RU-26** | 78 | experiment/diagnosis | §211 LOO 차원 편향 | 서로 다른 차원의 raw LOO를 신뢰도 비교에 쓰지 않는다. | RU-27/RU-34 LOO 폐기 |
| **RU-27** | 79 | experiment/decision | §212 깨끗한 Context LOO 폐기 | Context LOO를 영구 폐기하고 trimmed mean.. | R12가 용량 보정 뒤에도 확정 |
| **RU-28** | 80 | benchmark/experiment | §213 v121 기준선과 anchor 초기 측정 | .6171 5-branch를 공식 비교 기준으로 삼고 anc.. | RU-36 정정, RU-37 종료 철회 |
| **RU-29** | 81–82 | experiment/correction | §214–214-V 집계 탐색 정정 | trimmed mean 유지, 집계 축 종료, 보고 무결성 .. | R08부터 구조/게이트 우선 |
| **RU-30** | 83–85 | diagnosis/admission | §215–217 중복성·RM 기각 | 성능 미조회로 RM을 기각하고 형상 축만 탐색한다. | RU-31 게이트 선례; BD 하나뿐은 R16에서 .. |
| **RU-31** | 86–87 | admission/decision | §218 SH 채택·BS 기각 | SH 채택·승격 보류, 직교성+정보량 2단계 게이트 확정. | RU-32 SH 변형, RU-38 gate 보완 |
| **RU-32** | 88–90 | experiment | §219 SH 변형 반증과 SHJ 당시 보류 | 변형 축을 닫고 SHJ 정보는 게이트 개정 판단으로 이월했다. | RU-33 과제 특화 채택, RU-37 oracle.. |
| **RU-33** | 91 | policy/admission | §220 과제 특화 gate②b와 SHJ 채택 | ②b로 SHJ 채택, 3/7이라 공식 구성 승격은 안 했다. | R15가 oracle 조건을 fold 재현성으로 교체 |
| **RU-34** | 92 | experiment/decision | §221 용량 보정 뒤 LOO 활용 반증 | context→query 분포 이동 문제이므로 §212 폐기.. | RU-27 강화, RU-37 선택 신호 부재와 같은.. |
| **RU-35** | 93 | implementation/verification | §222 SHJ 통합과 bf16 수정 | fp32 백색화를 강제하고 채택을 유지했다. | RU-33 채택 구현 |
| **RU-36** | 94 | correction/decision | §223 §213 baseline 오류와 당시 축 종료 | 당시 anchor 축 종료와 통제 비교 의무를 기록했다. | R15가 종료 철회 |
| **RU-37** | 95–96 | policy/experiment | §224 Oracle 폐기와 §225 dose-response | §223 종료는 철회하지만 전역 이득 없고 draw 안정성은.. | RU-28/RU-36 anchor 해석 정정 |
| **RU-38** | 97–99 | 도구 수정 및 진행 중 연구방향 라운드 (진행 중) | §226 gate① 결함 수정과 진행 중 방향 라운드 | gate①은 SH·SHJ 포함. 제안은 실험 결과가 아닌 진.. | RU-30 screen 선례와 RU-31/RU-33.. |

---

## 3. 연구 단위별 상세 감사 기록

### RU-01. CT 부분공간과 ridge readout의 결합

- **커밋 범위**: Index 0–1 (2개 커밋, `aac69013` ... `ebcdb835`)
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

### RU-02. DD의 가중치와 코호트 의존성

- **커밋 범위**: Index 2 (1개 커밋, `bd7647bb`)
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
- **선행·후속 관계 (Relations)**: RU-03~E09의 DD readout 탐색으로 이어진다.
- **확인 필요 사항 및 한계 (Uncertainties)**: 집단 차이가 신호 강도인지 코호트 차이인지는 분리되지 않았다.

---

### RU-03. DD LLR 보정과 상대 거리의 판정 가능성

- **커밋 범위**: Index 3–4 (2개 커밋, `c61a4100` ... `c85156b8`)
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

### RU-04. CV descriptor의 정보 분해

- **커밋 범위**: Index 5–11 (2개 커밋, `80e7045c` ... `9c1889d2`)
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

### RU-05. CT token 생성 개선과 v109

- **커밋 범위**: Index 6–7 (2개 커밋, `7ddb525a` ... `7456a22c`)
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

### RU-06. CT 표본 수·token 수·ridge λ의 분리

- **커밋 범위**: Index 8–12 (4개 커밋, `f2f29ccc` ... `fb91a069`)
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

### RU-07. 서버 이동용 실행 환경 정비

- **커밋 범위**: Index 13 (1개 커밋, `69360766`)
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

### RU-08. 대체 CT dictionary와 full-cell hierarchical v111

- **커밋 범위**: Index 14–24 (11개 커밋, `b7d2f784` ... `65a20d9c`)
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

### RU-09. DD ordered×typicality 후보의 구현·수정 및 v112 승격

- **커밋 범위**: Index 25–30 (6개 커밋, `80f17db7` ... `fc814a3c`)
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

### RU-10. CT fraction sampling feasibility

- **커밋 범위**: Index 31 (1개 커밋, `c727e103`)
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
- **선행·후속 관계 (Relations)**: RU-09, RU-11
- **확인 필요 사항 및 한계 (Uncertainties)**: v113 전체 17-task와 hold-out 7은 당시 재측정되지 않았다.

---

### RU-11. unit fixed-head weight 통일

- **커밋 범위**: Index 32–33 (2개 커밋, `7acfd24e` ... `a495a18c`)
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
- **선행·후속 관계 (Relations)**: RU-10, RU-15
- **확인 필요 사항 및 한계 (Uncertainties)**: unit-weight의 Primary 7·hold-out 7은 당시 미측정이다.

---

### RU-12. CT kernel ridge와 top-k pooling 기각

- **커밋 범위**: Index 34 (1개 커밋, `eee4650c`)
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
- **선행·후속 관계 (Relations)**: RU-11
- **확인 필요 사항 및 한계 (Uncertainties)**: 당시 SEAL 10 기반의 기각이며 현행 CT 제외 기준에는 쓰지 않는다.

---

### RU-13. 모듈 분해와 BM 구현·문서 SSOT

- **커밋 범위**: Index 35–44 (10개 커밋, `af004748` ... `41aff6e5`)
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
    > "| [`docs/current_status.md`](../current_status.md) | 개발 현황 **SSOT** — 최신 수치·커밋·Action Plan |"
- **선행·후속 관계 (Relations)**: RU-15, RU-21
- **확인 필요 사항 및 한계 (Uncertainties)**: 35–45에는 문서/리팩터링이 포함되므로 이를 독립 성능 주장으로 쓰지 않는다.

---

### RU-14. Primary 7과 SEAL hold-out 역할 전환

- **커밋 범위**: Index 45 (1개 커밋, `b60a8a03`)
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
- **선행·후속 관계 (Relations)**: RU-13, RU-15
- **확인 필요 사항 및 한계 (Uncertainties)**: 현재 규칙은 이 당시 protocol을 후속 정정해 SEAL을 선택에 쓰지 않으며 CT도 제외한다.

---

### RU-15. Projected Bag-Mean(BM) 채택

- **커밋 범위**: Index 46 (1개 커밋, `42078217`)
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
- **선행·후속 관계 (Relations)**: RU-13, RU-16
- **확인 필요 사항 및 한계 (Uncertainties)**: 현재 승격 규칙은 이 시점의 규칙과 다르다.

---

### RU-16. Bag-dispersion spectral entropy(BD) 채택

- **커밋 범위**: Index 47 (1개 커밋, `dfcc1c00`)
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
- **선행·후속 관계 (Relations)**: RU-15, RU-17
- **확인 필요 사항 및 한계 (Uncertainties)**: 4/7은 현재 5/7 요건을 못 채우며 당시 사용자 결정으로만 기록한다.

---

### RU-17. DD 제거와 soft-vote v118

- **커밋 범위**: Index 48–50 (3개 커밋, `3987dcc9` ... `6deb53ec`)
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
- **선행·후속 관계 (Relations)**: RU-16, RU-18
- **확인 필요 사항 및 한계 (Uncertainties)**: SEAL은 현행 선택에 쓰지 않는 hold-out이다.

---

### RU-18. QA branch와 trimmed-mean v119

- **커밋 범위**: Index 51–52 (2개 커밋, `9dce8c5e` ... `b0ae85b4`)
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
- **선행·후속 관계 (Relations)**: RU-17, RU-20, RU-21
- **확인 필요 사항 및 한계 (Uncertainties)**: v119와 v120은 branch set이 달라 단순 우열 근거가 아니다. 현재 공식 비교 기준은 CT 제외 v121이다.

---

### RU-19. DS salience-denoising과 v120

- **커밋 범위**: Index 53 (1개 커밋, `a42a0be0`)
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
- **선행·후속 관계 (Relations)**: RU-18, RU-20
- **확인 필요 사항 및 한계 (Uncertainties)**: 현재 CT 제외 기준과 직접 비교하지 않는다.

---

### RU-20. v120 사후 실패 축과 CT 단독

- **커밋 범위**: Index 54–56 (3개 커밋, `6faa1c3b` ... `1b17863d`)
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
- **선행·후속 관계 (Relations)**: RU-18, RU-21
- **확인 필요 사항 및 한계 (Uncertainties)**: 초기 CT SEAL 0.7197과 후속 상세표 0.6882는 불일치해 이 단위에 SEAL CT 수치를 채택하지 않았다. 현재 CT는 공식 기준선에서 제외된다.

---

### RU-21. v120 harness·arm 모듈화와 결정성 경계

- **커밋 범위**: Index 57–62 (6개 커밋, `d858deea` ... `dd06c931`)
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
- **선행·후속 관계 (Relations)**: RU-18, RU-22
- **확인 필요 사항 및 한계 (Uncertainties)**: DE/SW/LOO는 tooling 추가이며 결과 연구단위로 합치지 않았다.

---

### RU-22. obsolete harness 제거와 학습 config 아카이브

- **커밋 범위**: Index 63–64 (2개 커밋, `83c5664e` ... `10484d9a`)
- **작업 유형**: `configuration_cleanup`
- **질문 (Question)**: 0-parameter v120에서 남은 harness·학습 config가 활성 entry point를 혼동시키는가?
- **가설 (Hypothesis)**: 미참조 도구와 학습 계보를 archive로 옮기면 활성 경로가 명료해진다.
- **실험 및 변경 (Experiment)**: obsolete harness task를 제거하고 v77–v105 config를 archive로 이관, 미참조 group을 삭제하고 정적 검증했다.
- **관측 결과 (Observations)**: v120은 train_v98 config 하나를 로드한다고 기록되며 26개 config 이관, 23개 group·5개 harness yaml 삭제, dangling 참조 0건이 기록됐다.
- **당시 결정 (Decision)**: 학습 계보는 archive/git history에 보존하고 활성 v120과 분리했다.
- **원문 근거 (Evidence)**:
  - `10484d9a:docs/current_status.md#L641`:
    > "| v120 활성 경로가 로드하는 yaml | **`configs/train_v98_p1_reverse_1536_1gpu.yaml` 단 1개** ([`scripts/node_env.sh`](../../scripts/node_env.sh) `ICF_CONFIG` 기본값) |"
  - `10484d9a:docs/current_status.md#L670`:
    > "- **dangling 참조**: 코드/Living 문서에서 존재하지 않는 config 경로 참조 0건"
- **선행·후속 관계 (Relations)**: RU-21
- **확인 필요 사항 및 한계 (Uncertainties)**: 당시 BagPFN env 소실로 regression suite는 정적 검증으로 대체됐다.

---

### RU-23. 재현 기반 정비

- **커밋 범위**: Index 65–75 (11개 커밋, `fd1095a6` ... `03d4e412`)
- **작업 유형**: `infrastructure`
- **질문 (Question)**: 잃어버린 환경과 단일 파일 구현 아래에서 비교를 재현할 수 있는가?
- **가설 (Hypothesis)**: venv·SSOT·모듈·계약 테스트가 기준선 비교를 반복 가능하게 한다.
- **실험 및 변경 (Experiment)**: 환경·handoff·archive를 정리하고 branch/common/aggregation 분리, YAML 계약 테스트와 회귀 실행기를 추가했다.
- **관측 결과 (Observations)**: 17 모듈/131 테스트가 18.9초에 통과했다.
- **당시 결정 (Decision)**: 성능 승격이 아닌 이후 비교의 재현 기반으로 채택했다.
- **원문 근거 (Evidence)**:
  - `1319aa05:docs/current_status.md#L27`:
    > "All 17 modules / 131 unit tests pass in 18.9s."
- **선행·후속 관계 (Relations)**: RU-24 이후 비교의 기반
- **확인 필요 사항 및 한계 (Uncertainties)**: 과거 CT 수치는 R02에서 정정

---

### RU-24. §209 CT 전사 오류와 단독 기준선

- **커밋 범위**: Index 76 (1개 커밋, `d74bb0c7`)
- **작업 유형**: `benchmark/correction`
- **질문 (Question)**: CT가 단독 최상이며 SEAL 0.7197이라는 기록은 맞는가?
- **가설 (Hypothesis)**: 8개 단독 50-fold가 순위와 CT 병목을 확정한다.
- **실험 및 변경 (Experiment)**: 8개 branch Primary 7과 시간을 대조했다.
- **관측 결과 (Observations)**: DS .6265, QA .6209, CT .6147이고 CT SEAL은 .6882였다.
- **당시 결정 (Decision)**: CT를 단독 챔피언으로 취급하지 않고 비용을 기준 설정에 반영했다.
- **원문 근거 (Evidence)**:
  - `d74bb0c7:docs/current_status.md#L40`:
    > "- **CT Standalone Metric**: 과거 요약표의 `SEAL 10 0.7197`은 허위/오기록이며 실측치는 `0.6882`임. Primary 7 단독 1위는 CT(`0.6147`)가 아닌 DS(`0.6265`)임."
- **선행·후속 관계 (Relations)**: RU-28 CT-제외 공식 비교
- **확인 필요 사항 및 한계 (Uncertainties)**: SEAL은 선택 근거가 아닌 hold-out

---

### RU-25. §210 sub-bag/TTA의 과제별 상충

- **커밋 범위**: Index 77 (1개 커밋, `976fd5ab`)
- **작업 유형**: `experiment`
- **질문 (Question)**: sub-bag 증강이 형태학적 변이와 국소 변이를 함께 개선하는가?
- **가설 (Hypothesis)**: context 증강 또는 query TTA가 ARID1A·Grade를 개선한다.
- **실험 및 변경 (Experiment)**: 두 증강 방식을 Primary 7에서 비교했다.
- **관측 결과 (Observations)**: ARID1A .5471→.6179, Grade .6823→.7024였지만 KRAS .7295→.6395였다.
- **당시 결정 (Decision)**: 국소 변이를 보존한다는 anchor 가설을 이월했다.
- **원문 근거 (Evidence)**:
  - `976fd5ab:docs/current_status.md#L24`:
    > "  - **발견 2 (병리학적 한계 규명)**: 2~5% 면적의 국소 변이 세포에 의존하는 `KRAS` (0.7295 $\to$ 0.6395), `KEAP1`, `PBRM1`은 무작위 균일 샘플링 시 변이 패치가 누락되는 False-Negative Sub-bag 현상으로 양성 신호가 희석됨."
- **선행·후속 관계 (Relations)**: RU-28/RU-36/RU-37 anchor 계보
- **확인 필요 사항 및 한계 (Uncertainties)**: 국소 변이 누락은 당시 기전 가설

---

### RU-26. §211 LOO 차원 편향

- **커밋 범위**: Index 78 (1개 커밋, `401305b1`)
- **작업 유형**: `experiment/diagnosis`
- **질문 (Question)**: LOO가 서로 다른 branch/subsample 신뢰도를 고르는가?
- **가설 (Hypothesis)**: LOO가 episode 내 좋은 branch를 선택한다.
- **실험 및 변경 (Experiment)**: 16-branch dual pool과 PRESS LOO를 50-fold로 평가했다.
- **관측 결과 (Observations)**: CV 32640D는 DOF/N 88.1%에서 ctx LOO .94~1.000으로 팽창했다.
- **당시 결정 (Decision)**: 서로 다른 차원의 raw LOO를 신뢰도 비교에 쓰지 않는다.
- **원문 근거 (Evidence)**:
  - `401305b1:docs/current_status.md#L25`:
    > "    - 서로 다른 차원을 가진 브랜치(CV: $32,640\text{D}$ vs DS: $32\text{D}$)를 raw LOO로 비교 시, 고차원 브랜치는 $DOF/N = 88.1\%$의 극단적 암기(Overfitting)로 인해 Context LOO가 $0.94\sim 1.000$의 '가짜 천재' 점수를 얻음."
- **선행·후속 관계 (Relations)**: RU-27/RU-34 LOO 폐기
- **확인 필요 사항 및 한계 (Uncertainties)**: 같은 차원 LOO도 일반화 실패

---

### RU-27. §212 깨끗한 Context LOO 폐기

- **커밋 범위**: Index 79 (1개 커밋, `6a87a5d1`)
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

### RU-28. §213 v121 기준선과 anchor 초기 측정

- **커밋 범위**: Index 80 (1개 커밋, `c3dd5352`)
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
- **선행·후속 관계 (Relations)**: RU-36 정정, RU-37 종료 철회
- **확인 필요 사항 및 한계 (Uncertainties)**: SEAL은 기준선 선택에 미사용

---

### RU-29. §214–214-V 집계 탐색 정정

- **커밋 범위**: Index 81–82 (2개 커밋, `73601f44` ... `143690a3`)
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

### RU-30. §215–217 중복성·RM 기각

- **커밋 범위**: Index 83–85 (3개 커밋, `a2f0dc6d` ... `e1bb3009`)
- **작업 유형**: `diagnosis/admission`
- **질문 (Question)**: 5개 branch가 독립 신호이고 RM은 새 축인가?
- **가설 (Hypothesis)**: rank와 상관으로 라벨 없이 중복·RM 직교성을 선별한다.
- **실험 및 변경 (Experiment)**: 세 실행 rank와 RM SCREEN_ONLY 350-fold를 대조했다.
- **관측 결과 (Observations)**: rank 2.26/5 반복, RM–CV max |r|=.690, 효율 45→43%.
- **당시 결정 (Decision)**: 성능 미조회로 RM을 기각하고 형상 축만 탐색한다.
- **원문 근거 (Evidence)**:
  - `e1bb3009:docs/current_status.md#L121`:
    > "- **결과: 기각.** max |r| = **0.690 (RM–CV)** > 0.6. 규칙에 따라 **성능은 조회하지 않았습니다**(`rm_screen.py`가 절차적으로 차단). 랭크 효율도 45% → 43%로 하락합니다."
- **선행·후속 관계 (Relations)**: RU-31 게이트 선례; BD 하나뿐은 R16에서 시점 한정 정정
- **확인 필요 사항 및 한계 (Uncertainties)**: rank는 R09에서 필터로 축소

---

### RU-31. §218 SH 채택·BS 기각

- **커밋 범위**: Index 86–87 (2개 커밋, `5c2881b5` ... `0b6e5611`)
- **작업 유형**: `admission/decision`
- **질문 (Question)**: 형상 통계가 새 정보를 제공하는가?
- **가설 (Hypothesis)**: SH/BS가 위치 계열과 직교해 rank와 앙상블을 개선한다.
- **실험 및 변경 (Experiment)**: SH·BS 상관/효율, 단독 정보량, 5-branch 확장을 평가했다.
- **관측 결과 (Observations)**: SH max |r|=.418, 7/7>.5; BS 4/7. rank 45→50%여도 macro/oracle 개선 없음.
- **당시 결정 (Decision)**: SH 채택·승격 보류, 직교성+정보량 2단계 게이트 확정.
- **원문 근거 (Evidence)**:
  - `0b6e5611:docs/current_status.md#L154`:
    > "**직교성은 필요조건이지 충분조건이 아니며, 랭크 효율은 성능의 예측자가 아니라 필터입니다.**"
- **선행·후속 관계 (Relations)**: RU-32 SH 변형, RU-38 gate 보완
- **확인 필요 사항 및 한계 (Uncertainties)**: hold-out 미검증

---

### RU-32. §219 SH 변형 반증과 SHJ 당시 보류

- **커밋 범위**: Index 88–90 (3개 커밋, `c3476dab` ... `ba9f6a43`)
- **작업 유형**: `experiment`
- **질문 (Question)**: SH 차원·로버스트화·모멘트 분리 또는 SHJ가 정보량을 높이는가?
- **가설 (Hypothesis)**: 강화 SH가 미회수 정보를 회수한다.
- **실험 및 변경 (Experiment)**: 7개 변형을 한 실행(949초)에 별도 마진으로 기록했다.
- **관측 결과 (Observations)**: 세 가설 반증; SHJ ARID1A .6363, 48/50, max |r|≤.227이나 4/7.
- **당시 결정 (Decision)**: 변형 축을 닫고 SHJ 정보는 게이트 개정 판단으로 이월했다.
- **원문 근거 (Evidence)**:
  - `ba9f6a43:docs/current_status.md#L154`:
    > "SHJ는 KRAS `0.4823`, Prog `0.4889`, PBRM1 `0.4797`로 **3개 과제에서 우연 이하**(단독>0.5가 4/7)이며, §218에서 사전 선언한 게이트 ②를 통과하지 못합니다."
- **선행·후속 관계 (Relations)**: RU-33 과제 특화 채택, RU-37 oracle 폐기
- **확인 필요 사항 및 한계 (Uncertainties)**: oracle 이동은 이후 판정 기준 아님

---

### RU-33. §220 과제 특화 gate②b와 SHJ 채택

- **커밋 범위**: Index 91 (1개 커밋, `749cbff4`)
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

### RU-34. §221 용량 보정 뒤 LOO 활용 반증

- **커밋 범위**: Index 92 (1개 커밋, `a21c227a`)
- **작업 유형**: `experiment/decision`
- **질문 (Question)**: LOO 차원 팽창을 없애면 context 점수가 query 성능을 예측하는가?
- **가설 (Hypothesis)**: 무편향 8D SHJ에서는 context LOO가 예측한다.
- **실험 및 변경 (Experiment)**: context LOO 마진을 저장하고 상관을 검사했다.
- **관측 결과 (Observations)**: 팽창 rho=.667은 확인됐지만 SHJ fold rho=-.091, 차원-ctx/qry rho=.154.
- **당시 결정 (Decision)**: context→query 분포 이동 문제이므로 §212 폐기 확정.
- **원문 근거 (Evidence)**:
  - `a21c227a:docs/current_status.md#L161`:
    > "**편향을 없애도 예측력이 돌아오지 않습니다.** 즉 문제는 낙관 편향이 아니라 **context↔query 분포 이동**입니다(§215 SMAD4 역전과 같은 축). §212 폐기는 용량과 무관하게 확정됩니다."
- **선행·후속 관계 (Relations)**: RU-27 강화, RU-37 선택 신호 부재와 같은 교착
- **확인 필요 사항 및 한계 (Uncertainties)**: 임계값을 평가셋에서 고를 수 없음

---

### RU-35. §222 SHJ 통합과 bf16 수정

- **커밋 범위**: Index 93 (1개 커밋, `d6b0a9ce`)
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
- **선행·후속 관계 (Relations)**: RU-33 채택 구현
- **확인 필요 사항 및 한계 (Uncertainties)**: 초기 SHJ 수치는 bf16 경로

---

### RU-36. §223 §213 baseline 오류와 당시 축 종료

- **커밋 범위**: Index 94 (1개 커밋, `cab8248c`)
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

### RU-37. §224 Oracle 폐기와 §225 dose-response

- **커밋 범위**: Index 95–96 (2개 커밋, `8c8bfc5d` ... `0e774f05`)
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
- **선행·후속 관계 (Relations)**: RU-28/RU-36 anchor 해석 정정
- **확인 필요 사항 및 한계 (Uncertainties)**: 과제별 선택 신호 없음

---

### RU-38. §226 gate① 결함 수정과 진행 중 방향 라운드

- **커밋 범위**: Index 97–99 (3개 커밋, `5d2fa04e` ... `f3f0b201`)
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
- **선행·후속 관계 (Relations)**: RU-30 screen 선례와 RU-31/RU-33 채택 형상 계열 연결
- **확인 필요 사항 및 한계 (Uncertainties)**: AKS·MDX·LID 미실험; GPU blocker와 fatal 판정 충돌

---

## 4. 100커밋 종합 분석: 주요 연구 축의 진화와 방법론적 정정

### 4.1 모델 아키텍처 진화 계보
1. **CT (Cell-Type Abundance) 축의 부침과 뮤트**:
   - `RU-01`에서 raw 1536차원 대신 PCA32 투영과 ridge readout을 결합하여 v108로 채택되었고, `RU-05`에서 k-means token 생성을 통해 v109로 개선되었습니다.
   - `RU-08`에서 full-cell hierarchical k-means(v111), `RU-10`에서 fraction sampling(v113)으로 발전했으나, `RU-12`의 비선형/top-k 변형은 기각되었습니다.
   - 결국 `RU-23`~`RU-24`(§209)에서 CT의 실측 단독 성능(SEAL 0.6882)이 기존 기록(0.7197)의 전사 오류임이 밝혀지고, 런타임의 80%를 차지하는 k-means 병목 대비 앙상블 기여 부족으로 `RU-28`(v121, §213)에서 완전 뮤트(CT=0)되었습니다.
2. **DD (Direct Distance / LLR) 축의 기각**:
   - `RU-02`에서 DD의 task 집단 간 부호 상충(SEAL 감소 vs 홀드아웃 증가)이 발견되었고, `RU-03`의 LLR log-det 보정(AUROC 불변) 및 상대 마진(성능 저하) 시도가 실패했습니다.
   - `RU-09`에서 bounded ordered-coordinate와 typicality를 결합한 v112가 일시 승격되었으나, `RU-17`(v118)에서 DD를 완전히 제거하는 것이 오히려 일반화를 개선한다는 실측에 따라 DD 축은 영구 폐기되었습니다.
3. **위치(Location) vs 형상(Shape) 브랜치 분류 체계 확립**:
   - `RU-15`(BM, bag-mean 투영), `RU-18`(QA, 4분위수), `RU-19`(DS, 살리언스 디노이징)가 잇달아 추가되었으나, `RU-30`(§215–217) 분석을 통해 이들이 모두 평균의 사영/재가중 계열(상호 상관 0.25~0.93)로 높은 중복성(유효 랭크 2.26/5)을 보임이 입증되었습니다.
   - 반면 `RU-16`(BD, 공분산 스펙트럼 엔트로피)는 위치와 독립적인 퍼짐/이질성을 측정하여 프로젝트 역사상 유일한 직교 축(|r| <= 0.16)으로 자리잡았습니다.
   - 이후 `RU-31`(SH, 반경 분산 및 고차 모멘트)과 `RU-33`(SHJ, 자코비안 기하)가 신규 형상 브랜치로 채택되며 현행 형상 3대 축(`BD`, `SH`, `SHJ`)을 구축했습니다.

### 4.2 연구 규범 및 평가 계약의 진화
- **§151 (RU-01/RU-02) 결정론적 통계 규범**: 결정론적(deterministic) arm에 대해 seed 기반 t-통계량/p-값/CI 보고를 전면 철회하고, 독립 코호트 부호 일치와 재현성으로 판정 기준을 전환했습니다.
- **§193 (RU-14) 벤치마크 역할 분리**: Primary 7은 모델 탐색 및 하이퍼파라미터 튜닝용, SEAL 10은 동결된 최종 hold-out 검증용으로 역할을 엄격히 분리했습니다.
- **§212/§221 (RU-27/RU-34) In-Episode Context LOO 영구 폐기**: Context 표본으로 계산한 LOO가 고차원 브랜치의 암기 편향과 context↔query 간 분포 이동으로 인해 query AUROC를 예측하지 못함을 확인하고 폐기했습니다.
- **§214-V (RU-29) 보고 무결성 계약 및 분해능 하한**: 사후 집계 튜닝으로 macro 1위만을 주장하고 회귀를 누락하는 왜곡을 방지하기 위해 전체 결과 보고를 계약화하고, 1%p 미만의 미세 차이에 대한 집계 축 탐색을 종료했습니다.
- **§218/§220/§224 (RU-31/RU-33/RU-37) 신규 브랜치 2단계 게이트**: ① 라벨 무관 직교성(max |r| <= 0.6 및 랭크 효율 증가)과 ② 정보량(범용 ②a 또는 fold 재현성 >= 40/50의 과제 특화 ②b) 심사를 확립했습니다.
- **§223 (RU-36) 통제 비교 의무**: 서로 다른 실행 결과를 사후 대조하여 이득을 주장하는 것을 금지하고, 동일 실행·동일 basis·동일 fold에서 대상 기법만 토글하는 통제 비교(`ICF_DS_SUBSAMPLE_COMPARE=1`)를 의무화했습니다.
- **§224 (RU-37) Oracle 비교 금지**: 사후 최댓값으로 정의된 Oracle 상한은 선택 편향(승자의 저주)을 유발하므로 연구 방향이나 판정의 근거로 사용하는 것을 전면 금지했습니다.
- **§226 (RU-38) 게이트 ① 결함 수정**: 게이트 ① 심사 시 기채택 브랜치(SH, SHJ)가 누락되던 결함을 수정하고 `--adopted`를 반영했습니다.