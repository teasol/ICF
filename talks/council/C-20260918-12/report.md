# ICF Council — C-20260918-12

- **연구 질문**: 외부 문헌 4편이 처음으로 들어왔다. 첨부된 브리프만 근거로 삼는다.

가장 눈에 띄는 사실: ICMIL은 12개 MIL 벤치마크 평균에서 ABMIL(79.97)을 앞서지만(84.17), 그 중 유일한 병리 WSI 과제인 TCGA에서는 ABMIL 90.7이 ICMIL 87.9를 앞선다. 그리고 그 TCGA 과제는 UNI2 1536차원 patch feature를 PCA 25차원으로 줄인 LUAD vs LUSC 이진 분류로, 우리와 같은 특징 추출기를 쓴다. 다만 저들은 bag을 약 100개로 subsample한다.

우리 맥락: 목표는 SEAL 10에서 ABMIL을 넘는 것이고 개선 근거는 Primary 7 뿐이다. 학습 파라미터 제약은 폐기됐다. 현재 공식 7-branch의 유효 랭크는 3.52/7로 중복이 크고, 확률적 적합을 쓰면 적합 분산이 승격 기준의 최대 2.8배가 된다.

질문: (1) 병리 과제에서만 ICMIL이 지는 것은 무엇을 시사하는가? 저들의 실험 설정 중 무엇이 그 결과를 만들었을 가능성이 있는가. (2) 저들은 여러 prior를 섞어 상보성을 얻었다고 주장한다. 우리는 브랜치를 늘려도 유효 랭크가 오르지 않는다. 무엇이 다른가? 이 차이가 우리에게 옮겨올 수 있는 것인가. (3) ICMIL의 분산은 TCGA에서 ±0.1, ABMIL은 ±1.6이다. 우리의 적합 분산 문제에 이 구조가 시사하는 바가 있는가. (4) 이 네 논문에서 우리가 옮겨올 수 없는 것은 무엇이고 왜인가. (5) Primary 7에서 무엇을 재면 위 가설들이 갈리는가.

주의: 외부 결과는 주장이지 우리의 관측이 아니다. 어떤 외부 수치도 우리 게이트·승격의 근거가 될 수 없다. 과제 집합이 다르므로 절대값 직접 비교는 금지다. 브리프에 없는 논문·수치를 만들어내지 마라. 브리프 자체가 로컬 모델 요약이며 원문 대조가 되지 않았다는 점도 감안하라.
- **RU 유형**: 탐색
- **최대 권한 등급**: T1
- **소집자**: Claude Code (claude-opus-5)
- **좌석**: E1(E, temp 0.3, http://127.0.0.1:8000/v1/chat/completions), R1(R, temp 0.5, http://127.0.0.1:8001/v1/chat/completions), T1(T, temp 0.4, http://127.0.0.1:8003/v1/chat/completions), P1(P, temp 0.9, http://127.0.0.1:8000/v1/chat/completions), A1(A, temp 0.2, http://127.0.0.1:8001/v1/chat/completions), S1(S, temp 0.2, http://127.0.0.1:8003/v1/chat/completions)
- **호출**: 11/18 · **소요**: 965.1초

> 이 문서는 로컬 모델이 생성한 것이며 **검증되지 않았습니다.** 좌석이 인용한 수치는
> 정본 문서와 대조하기 전에는 결론에 넣지 마십시오. 좌석 수와 동의 수는 증거가 아닙니다.

## 편성 근거 (결과 전 기록)

E(외부) 좌석의 첫 투입이다. 외부 문헌이 들어온 회차이므로 E를 첫 좌석에 둔다. 가장 큰 위험은 새 논문에 휩쓸려 우리 제약을 잊는 것이므로 감사(A)가 '과제 집합이 다른 절대값 비교'와 '외부 수치의 우리 근거화'를 감시한다. 반증(R)은 'ICMIL을 따라하면 된다'는 손쉬운 결론과, 반대로 '병리에서 졌으니 무관하다'는 성급한 기각을 둘 다 공격한다. 이론(T)은 prior 혼합의 상보성과 우리 유효 랭크 문제의 관계를 맡는다. 제안(P)은 온도를 높여(0.9) 옮겨올 설계를 낸다. 일반화(G)는 제외했다 - 이번 질문 자체가 분포 밖 전이에 관한 것이라 E와 역할이 겹친다. 실측(M)도 제외: 아직 후보가 확정되지 않았다. 브리프는 로컬 모델 요약이고 원문 대조가 안 됐으므로 그 한계를 프롬프트에 명시했다.

---

## Phase 1 — 독립

### P1 (제안)

## 제안 1: Provenance 복구와 ICMIL-TCGA 설정 혼동자 프로브
- **바꾸는 것**: 먼저 `D-050`으로 무효화된 공식 7-branch 산출물의 귀속 가능성을 복원한다. 산출물에 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시를 기록하고, `SCREEN_ONLY`/게이트 ① 실행 경로가 이 provenance를 검사하도록 만든다. 그 위에 Primary 7에서 ICMIL TCGA와 겹치는 두 설정 변형을 탐색용으로 수행한다. ① UNI2 patch feature를 25차원으로 PCA, ② 공식 fold의 train context slide/bag를 100개로 고정 부분추출한다. ICMIL의 100-bag subsample이 context와 query를 어떻게 나눴는지는 문서에 없으므로, 이 제안은 train context만 100으로 제한하는 단순 변형으로 한정한다. PCA 투영 범위도 ICMIL 원문 규칙이 문서에 없으므로 fold test/SEAL 정보 없이 context-only PCA를 쓰는 변형으로 사전 고정한다. 공식 fold 경계, SEAL 10, test 라벨은 사용하지 않으며 결과는 탐색 기록으로만 둔다.
- **바꾸는 전제**: 지금 “저장 예측 파일을 보면 특정 branch 구성으로 귀속할 수 있다”고 두고 있고, ICMIL이 TCGA에서 ABMIL에 밀린 것을 병리 WSI in-context MIL의 본질적 열세 또는 고차원 UNI2 feature의 한계로 읽고 있다. 이 제안은 그 전제를 뒤집는다. provenance가 없으면 어떤 Primary 7 관찰도 공식 7-branch·PCA25·subsample arm으로 귀속할 수 없다. 또한 ICMIL TCGA 격차는 병리 과제 본질보다는 PCA25, 100-bag context 축소, 그리고 ICMIL의 ABMIL baseline이 early stopping·200 epoch까지 학습된 task-specific baseline이라는 설정 차이가 만든 것일 수 있다.
- **기대하는 차이**: provenance 복원이 되면 arm별 per-fold AUROC, branch margin, effective rank, 적합 분산을 같은 fold에서 짝짓고 `지지 / 반박 / 판별 불가 / 실행 무효`를 적을 수 있다. PCA25·100-bag arm이 공식/full arm 대비 paired per-fold Δ로 낮아지면 ICMIL TCGA 열세의 일부는 feature compression 또는 context 축소라는 설정 혼동자로 설명될 여지가 생긴다. 거의 변하지 않으면 ICMIL TCGA 열세의 원인은 병리 WSI in-context 적합성, label rule, bag 정의, ABMIL baseline 강도 등 다른 요인으로 이동한다. 우리 후보가 full에서 좋아도 PCA25에서 무너지면 고차원 feature 의존성이 승격 안정성 위험이 된다. 정량 예측 불가.
- **기전**: PCA25는 고차원 UNI2 variance를 제거하고, 100-bag context 축소 is context 정보량을 줄인다. 둘 다 신호를 줄이거나 bias-variance 균형을 바꿔 ABMIL 대비 격차를 만들 수 있다. ICMIL의 ABMIL baseline은 single stratified validation, 최대 200 epoch, early stopping patience 20, lowest validation CE를 쓰므로 SEAL baseline의 20 epoch 고정 ABMIL과 비교 강도가 다르다. 이 기전은 외부 브리프에서 입증된 것이 아니라 가설이다. 기전 미확인.
- **실패 조건**: `src/` 환경변수·branch control 결함이 남아 branch list·가중치·코드 해시를 기록해도 실행이 특정 구성과 연결되지 않으면 이 프로브는 실행 무효다. PCA/subsample 효과가 fold 노이즈 수준이면 ICMIL TCGA 혼동자 가설을 판별할 수 없다. ICMIL의 PCA normalization·fit scope·100-bag subsample 대상이 문서에 없어 동일 설정 재현이 불가능하면 외부 TCGA 결과와의 연결은 주장 수준으로만 남는다. ICMIL의 learned Perceiver pretraining, WSI-scale attention cost, high-dim future work는 우리 Primary 7 탐색에서 직접 옮기는 대상이 아니다.

## 제안 2: PSW — prediction-variance weighted readout
- **바꾸는 것**: 공식 7-branch의 trimmed mean 집계를 기준선으로 두고, 후보 readout으로 fold-wise prediction-variance weighted combination(PSW)을 사전 고정한다. 각 fold에서 각 branch score를 여러 stochastic fit 또는 context resample로 재추산해 branch별 score variance를 추정하고, 가중치는 variance 역수 + ridge 형태를 쓴다. ridge 상수·resample 수·가중치 정규화 규칙은 Primary 7 결과 확인 전에 고정하고, AUROC를 보고 가중치 상수나 branch 부분집합을 조정하지 않는다. `D-050` provenance 복구 이후 공식 baseline 대비 paired per-fold Δ_macro, 최종 score 적합 분산, 과제별 Δ, sign agreement, 악화 과제를 Primary 7 50-fold에서 재측정한다.
- **바꾸는 전제**: 지금 적합 분산이 승격 기준의 최대 2.8배라는 문제를 “보고만 하면 되는 nuisance”로 두고 있고, ICMIL의 variance 감소는 learned PFN의 Bayesian model averaging에만 해당한다고 읽고 있다. 이 제안은 그 전제를 뒤집는다. 분산을 readout 가중에 쓸 수 있는 정보로 쓰고, closed-form branch ensemble에서도 variance-aware averaging이 승격 후보가 될 수 있다. ICMIL의 BMA 주장은 우리 closed-form ridge·branch margin에 전이된다는 직접 증거가 없으므로, 전이가 아니라 PSW의 동기일 뿐이다.
- **기대하는 차이**: branch variance가 fold AUROC 열세를 예측하면 PSW가 공식 baseline 대비 paired Δ_macro `+0.3%p`를 넘을 수 있고, 최종 score의 적합 분산이 감소한다. variance와 fold AUROC가 무관하면 Δ는 0 부근에서 잡히고, 저분산 branch가 bias만 큰 branch를 과대 가중하면 일부 과제가 악화될 수 있다. Primary 7에서 갈리는 측정은 branch variance와 per-fold Δ의 관계, PSW 최종 score variance, 과제별 paired Δ, 악화 과제 전량이다. 정량 예측 불가.
- **기전**: 역분산 가중은 일반 model averaging에서 ensemble variance를 줄이는 구조다. ICMIL은 labeled context와 일관된 가설에 대한 Bayesian model averaging을 근사해 scarce label에서 단일 aggregator fitting variance를 줄인다고 주장한다. 그러나 그 주장이 우리 7-branch margin, effective rank 3.52/7, stochastic closed-form fit에 전이된다는 증거는 브리프에 없다. 기전 미확인.
- **실패 조건**: stochastic fit variance가 seed/implementation noise를 반영할 뿐 task signal을 반영하지 못하면 AUROC와 상관이 없고 PSW는 효과 없다. 가중치가 fold-specific noise에 overfit하거나, variance가 낮은 branch의 bias를 줄이지 못하면 Primary 7 paired Δ가 0 이하가 될 수 있다. `CA-08`의 기존 branch margin combination 반복으로 보이면 실행 무효다. 따라서 PSW는 prediction-variance라는 경계 밖 new info를 쓰고, branch 부분집합 탐색이나 AUROC 후행 선택이 아님을 사전 기록해야 한다. `D-050` 복구 전에는 승격 주장이 아니라 탐색 기록만 가능하다.

## 제안 3: QJF context-geometry rank probe
- **바꾸는 것**: 새 branch 후보로 에피소드 내부 context slide의 linear Gram을 읽는 QJF를 하나 사전 고정한다. position provenance가 확인된 경우 LSAK를 대안으로 둘 수 있다. 이 후보는 기존 7-branch의 mean/location/shape statistic 변형이 아니라, context 간 선형 관계 또는 position×shape interaction이라는 다른 정보원을 읽는다. `D-050` 복구 이후 `SCREEN_ONLY` label-free gate ①을 먼저 수행한다. 기준은 공식 7-branch 대비 max |r| ≤ 0.6, 8-branch effective rank ≥ 4.02, 즉 현재 3.52/7의 50.3% 효율 비하락이다. gate ①에서 실패하면 성능을 조회하지 않고 종료한다. 통과 시 gate ②, 이후 Primary 7 paired promotion 비교를 진행한다.
- **바꾸는 전제**: 지금 ICMIL의 prior mixture 상보성을 “브랜치를 늘리면 얻는다”는 식으로 읽고, readout/branch 수 증가가 effective rank를 올릴 것이라고 두고 있다. 이 제안은 그 전제를 뒤집는다. ICMIL의 mixture는 서로 다른 synthetic prior와 inductive bias를 섞는 것이며, 우리 7-branch는 같은 slide-level statistic의 변형에 가까워서 effective rank가 3.52/7으로 낮을 수 있다. 따라서 상보성을 옮긴다면 branch 수 증가가 아니라 정보원 차원과 rank 증가를 전제 조건으로 삼아야 한다.
- **기대하는 차이**: QJF가 기존 branch와 직교하면 8-branch effective rank가 4.02 이상으로 오르고 gate ①을 통과한다. 그 뒤 Primary 7 paired Δ_macro `+0.3%p` 달성 여부가 승격 판정 대상이다. 직교하지 않으면 max |r| > 0.6 또는 rank 효율 하락으로 gate ①에서 종료된다. Primary 7에서 갈리는 측정은 max |r|, effective rank, gate ② 정보량, 통과 시 paired per-fold Δ, 과제별 악화 전량이다. 정량 예측 불가.
- **기전**: ICMIL 브리프의 inter-bag attention disabled 45.8 vs enabled 83.3은 cross-bag context mixing이 중요했다는 외부 주장이다. QJF는 그 idea를 nonlinear learned attention이 아니라 closed-form linear Gram으로 근사한 것이다. 그러나 이것이 ICMIL의 learned prior mixture와 같은 기전인지, 염색·스캐너·batch nuisance를 제거한 context geometry를 읽는지, effective rank를 실제로 올리는지는 미확인이다. 기전 미확인.
- **실패 조건**: Gram이 염색 톤, 스캐너, batch, context 크기 같은 nuisance와 상관하거나, CV/BD가 이미 같은 context geometry를 읽고 있으면 max |r| > 0.6으로 gate ①에서 종료된다. rank가 오르지 않으면 ICMIL의 prior mixture 상보성을 우리 7-branch에 단순히 branch 추가 방식으로 옮길 수 없다는 근거가 된다. `CA-04`의 과거 성능 악화 기전이 linear Gram에도 적용될 수 있으므로 성능은 Primary 7 paired 비교로만 판단해야 한다. 직접 전이 불가 항목: ICMIL learned pretraining/Perceiver attention, PFN 불균형의 thresholding(AUROC 지표상 직접 무의미), MMPFN modality projector/fine-tuning, PFN foundations의 tabular bias-variance bound는 우리 WSI bag MIL·AUROC 승격·effective rank·closed-form ridge와 문맥이 다르다. 이 제안은 linear closed-form probe만 옮긴다.

### E1 (외부)

## 외부 항목 1: ICMIL
- **저들의 주장**: 브리프(로컬 모델 요약, 원문 미검증)에 따르면 ICMIL은 PFN을 bag-structured MIL에 적용해 synthetic pretraining만으로 low-label MIL task를 single forward pass로 풀고, task-specific training 없이 12개 MIL benchmark에서 average AUROC/rank를 개선한다고 주장한다. mixture prior는 Joint 0.70, Factorized(cont, MLP) 0.15, Factorized(disc, lookup) 0.15이며, scaled ICMIL average AUROC 84.17, ABMIL 79.97이다. TCGA는 UNI2 1536차원 patch feature를 PCA 25차원으로 줄인 LUAD vs LUSC 이진 분류이고 bag을 100개로 subsample하며, 보고값은 ICMIL 87.9±0.1, ABMIL 90.7±1.6이다. ICMIL은 Bayesian model averaging으로 scarce labels에서 단일 aggregator fitting의 variance를 줄인다고 주장하고, PCA 25차원 평가와 high-dimensional foundation-model embeddings는 한계/미래 작업으로 둔다 (출처: icmil_digest §1, §3, §4, Table 1, Appendix E).
- **우리와 다른 전제**: 저들의 TCGA는 우리와 같은 UNI2 특징 추출기를 쓰지만 PCA 25차원·100 bag subsample·LUAD/LUSC 이진 과제이고, 저들의 ABMIL 기준선은 10% validation, lr/weight-decay grid, 최대 200 epoch, early stopping patience 20을 쓴 task-specific supervised baseline이다. 우리의 SEAL ABMIL baseline은 고정 20 epoch·validation/early stopping 없는 50-fold protocol이고, Primary 7/SEAL 10 과제 집합·fold protocol·고차원 patch embedding 환경은 저들과 다르다. ICMIL은 learned Perceiver PFN과 대규모 synthetic prior를 전제하지만, 우리는 현재 공식 7-branch closed-form ridge 계열이고 유효 랭크 3.52/7, 적합 분산 문제를 안고 있다. 학습 파라미터 제약은 폐기됐지만, 저들의 pretraining 규모·synthetic prior·PCA 25 전제는 그대로 전이되지 않는다.
- **우리에게 옮겨올 수 있는 것**: (가설) in-context PFN/label-aware inter-bag attention은 low-label fold에서 task-specific fitting 없이 bag label을 예측하는 구조이므로, 우리의 closed-form ridge in-context 분류와 같은 문제 구조를 공유할 수 있고, fit variance를 줄이는 방향일 수 있다. (가설) Joint/Factorized prior mixture는 '동일 feature 위에서 상관 높은 readout을 늘리는 것'이 아니라, 서로 다른 data-generating/aggregation 가설을 섞어 상보성을 얻는 설계 원칙으로 읽을 수 있다. (가설) TCGA 결과만으로는 pathology WSI에서 in-context MIL이 작동할 수 있다는 sanity check가 되지만, 우리 SEAL 10/Primary 7 전이는 보장되지 않는다. (가설) variance 보고 구조는 fold 분산과 적합 분산을 함께 보고해야 한다는 우리의 의무와 일치한다.
- **옮길 수 없는 것과 이유**: 84.17/79.97/87.9/90.7, mixture weights 0.70/0.15/0.15, prior ablation 수치는 다른 과제·평가·baseline 설정의 외부 주장이므로 우리 게이트·승격·채택 근거가 될 수 없다. TCGA에서 ABMIL이 ICMIL을 앞선다는 보고값은 '병리 WSI에서는 task-specific tuning이 이길 수 있다'는 가설만 줄 뿐, 기전은 미확인이다. ±0.1/±1.6의 산출 출처(시드/리샘플/폴드)는 브리프에 명시되지 않아, 우리의 적합 분산과 동일하다고 가정할 수 없다. PCA 25차원 결과는 고차원 UNI2 1536 환경으로 전이할 수 없고, 브리프도 high-dimensional embeddings를 future work로 둔다. 100 bag subsample은 WSI-scale patch 수·slide 수·런타임이 브리프에 없어 우리 fold protocol로 옮길 수 없다. ICMIL이 branch effective rank 3.52/7 중복을 줄인다는 보고는 브리프에 없고, 우리의 확률적 ridge 적합 분산 최대 2.8배 문제를 직접 비교한 결과도 브리프에 없다.
- **우리 자료로 검증하는 최소 실험**: D-050으로 공식 기준선·gate ① 계측이 unverified-hold/무효이므로, 먼저 provenance(branch list, 가중치, 코드 해시, 데이터 버전, fold manifest 해시)와 SCREEN_ONLY 경로를 복구하지 않으면 실행 무효. 후보가 label-aware context mixing 또는 learned in-context MIL이라면, 성능 조회 전에 gate ①로 공식 7-branch 대비 max |r|과 effective rank를 측정하고, 8-branch라면 eff.rank ≥4.02(50.3% 비하락)를 요구한다. 탈락 시 성능을 조회하지 않고 종료. 통과 시 Primary 7 고정 50-fold에서 공식 7-branch와 대응 per-fold Δ_macro를 계산하고, task cluster 95% CI·SE·sign agreement·악화 과제를 보고한다. 승격은 Δ_macro ≥ +0.3%p일 때만 지지. variance 가설은 동일 후보를 여러 seed/fit으로 돌려 fit-level macro AUROC std와 per-fold Δ 산포를 보고하고, baseline 적합 분산과 비교한다. fit variance가 줄어도 Δ < +0.3%p 또는 CI가 0을 포함하면 판별 불가/지지 부족이고, Δ ≥ +0.3%p이면서 fit variance가 허용 범위면 지지. 병리 TCGA 열세 가설은 Primary 7에서 고정 20 epoch ABMIL과 task-specific tuning을 하는 ABMIL 변형(저들의 validation/early stopping/hyperparameter grid를 참고한 사전 고정 grid)을 같은 fold로 대응 비교해 per-task Δ를 진단한다. tuned ABMIL이 이진 subtype 계열에서 우세하면 task-specific tuning 설명에 지지, 그렇지 않으면 약화. Primary 7에 TCGA와 동일한 LUAD/LUSC 과제가 없으면 이 가설은 직접 판별 불가이고, per-task Δ 패턴만 간접 진단이다. 이 진단은 승격 근거가 아니고, 외부 TCGA 수치는 쓰지 않는다.

## 외부 항목 2: PFN imbalance
- **저들의 주장**: 브리프(로컬 모델 요약, 원문 미검증)에 따르면 PFN은 클래스 불균형에서 희귀 클래스 성능이 떨어지고, TabPFN은 balanced context에서는 잘 calibration되지만 imbalanced context에서는 majority class bias를 보인다. thresholding은 calibration 특성 때문에 잘 작동하고, binary에서 최대 balanced accuracy는 대략 τ=π1에서 나타난다. downsampling은 성능이 비슷하고 추론 계산 비용을 줄이며, oversampling과 TabPFGen synthetic upsampling은 base model보다 나쁘다고 주장한다 (출처: pfn_imbalance_digest §1, §3, Table II, Fig. 4).
- **우리와 다른 전제**: 저들은 OpenML-CC18 tabular row-level binary classification, TabPFN-2.5, class imbalance, per-class accuracy/balanced accuracy/WCA를 다룬다. WSI patch/bag MIL, slide-level label, spatial context, ABMIL, Primary 7 fold, branch effective rank, 확률적 적합 분산, 승격 기준은 본문에 없다. 우리의 평가 지표는 AUROC이고, thresholding은 AUROC를 바꾸지 않는다.
- **우리에게 옮겨올 수 있는 것**: (가설) 우리의 soft score가 prior-driven majority bias를 보인다면, task class prior에 맞춘 thresholding은 WCA/balanced accuracy 같은 운영 지표를 개선할 수 있다. (가설) majority context를 deterministically downsample하면 context 중복·추론 비용을 줄일 수 있고, branch 중복이 majority context 반복에서 온다면 effective rank에 영향을 줄 수 있다. (가설) WCA·calibration·ROC diagnostics를 Primary 7 진단에 추가하면 희귀 클래스 성능 저하를 더 잘 볼 수 있다.
- **옮길 수 없는 것과 이유**: τ=π1 규칙은 binary tabular PFN의 결과이고, multi-class/hierarchical WSI label에 직접 적용할 근거가 브리프에 없다. Table II의 balanced/WCA 수치는 다른 데이터·모델·지표의 외부 주장이므로 우리 게이트·승격 근거가 될 수 없다. 이 논문은 ABMIL, WSI MIL, effective rank, fitting variance를 다루지 않으므로, 우리 적합 분산 최대 2.8배 문제나 7-branch rank 3.52/7 문제를 해결한다는 직접 증거가 없다. TabPFN의 calibration 성질이 closed-form ridge in-context classifier에도 성립한다고 단정할 수 없다.
- **우리 자료로 검증하는 최소 실험**: D-050 복구 전 실행 무효. thresholding 가설은 Primary 7에서 현재 soft score의 calibration, WCA, balanced accuracy를 재고, τ=0.5와 τ=task prior를 비교한다. AUROC는 threshold-invariant이므로 승격 판정에 쓰지 않고, WCA/balanced accuracy 개선만 있으면 calibration/운영 지표 문제 지지, AUROC 변화가 없으면 승격 무관. downsampling 가설은 majority context를 라벨 균형을 위해 deterministic stratified subsample하는 변형을 만든다. 먼저 context/branch 표현의 effective rank와 공식 7-branch 대비 상관을 라벨 무관으로 측정하고, rank가 하락하면 기각. 이후 Primary 7 대응 per-fold Δ_macro, task cluster CI·SE·sign agreement·악화 과제를 보고한다. Δ_macro ≥ +0.3%p이고 rank 비하락이면 지지, Δ가 0 근처면 판별 불가, rank 하락 또는 provenance 결함이면 실행 무효/기각. random subsampling을 쓰면 seed별 fit variance를 함께 보고한다.

## 외부 항목 3: PFN foundations
- **저들의 주장**: 브리프(로컬 모델 요약, 원문 미검증)에 따르면 PFN은 사전학습된 고정 네트워크가 추론 시 파라미터 업데이트 없이 단일 forward pass로 예측하고, 고정 네트워크의 분산은 개별 학습 샘플에 대한 민감도가 사라지면 사라진다. 편향이 사라지려면 예측자가 테스트 feature 주변으로 충분히 localized 되어야 하며, 현재 transformer PFN은 diminishing sensitivity로 vanishing variance를 보장하지만 localization을 보장하지 않는다. 1-layer transformer PFN의 variance는 n^{-1/2}로 사라진다고 주장하고, localized TabPFN은 bias를 줄이지만 variance가 약간 커질 수 있다고 보고한다 (출처: pfn_foundations_digest §1, §5, §6).
- **우리와 다른 전제**: 저들은 tabular/simulation, transformer PFN, bias/variance 이론을 다룬다. WSI patch/bag MIL, pathology, ABMIL, Primary 7 fold, branch effective rank, 확률적 ridge 적합 분산, 승격 기준은 본문에 없다. 저들의 정량 검증은 5차원 tabular 시뮬레이션과 TabPFN bias/variance에 한정된다.
- **우리에게 옮겨올 수 있는 것**: (가설) PFN의 bias/variance 분해는 우리의 closed-form ridge in-context 분류기를 진단하는 틀로 쓸 수 있다. (가설) permutation symmetrization, diminishing sensitivity, 개별 patch 민감도 평균화는 확률적 적합의 fit variance를 줄이는 구조적 후보다. (가설) query patch/slide 주변 spatial neighborhood만 context로 제한하는 post-hoc localization은 bias를 줄일 수 있지만, context 축소로 variance가 커지는 trade-off가 있을 수 있다. (가설) sample-size prior는 fold context 크기별 regularizer 설계 아이디어로 쓸 수 있다.
- **옮길 수 없는 것과 이유**: Theorem 5.2/5.4/6.2 같은 결과는 고정 transformer PFN의 조건 하의 주장이고, 우리의 closed-form ridge에 자동 적용된다고 단정할 수 없다. WSI 규모 patch 수, MIL bag 구조, pathology prior, ABMIL 비교, effective rank, fitting variance에 대한 직접 수치는 브리프에 없다. localization이 bias를 줄인다는 외부 서술은 우리 fold context에서 variance 증가와 함께 나타날 수 있으므로, 성능 전이를 보장하지 않는다.
- **우리 자료로 검증하는 최소 실험**: D-050 복구 전 실행 무효. variance 가설은 Primary 7에서 현재 확률적 ridge와 symmetrized/permutation-averaged 변형을 같은 fold에서 비교한다. 측정 항목은 (1) fit-level macro AUROC std(여러 seed/fit), (2) fold-level AUROC std, (3) 대응 per-fold Δ_macro, (4) task cluster CI·SE·sign agreement, (5) 악화 과제다. 변형이 fit variance를 줄이고 Δ_macro ≥ +0.3%p면 지지, fit variance만 줄고 Δ < +0.3%p면 variance reduction alone insufficient/판별 불가, Δ가 0 근처면 판별 불가. localization 가설은 spatial neighborhood context 변형을 만든다. 새 브랜치라면 gate ①에서 max |r|과 effective rank를 먼저 재고, 8-branch라면 eff.rank ≥4.02를 요구한다. 통과 시 Primary 7 대응 Δ와 fit variance를 보고한다. Δ 개선과 fit variance 허용이면 bias reduction 지지, Δ 악화와 fit variance 증가면 반박, Δ 변화가 작으면 판별 불가. branch rank 가설은 branch 출력의 pairwise correlation과 effective rank를 재고, symmetrization/averaging이 rank를 올리는지 확인한다. rank 개선만으로는 승격이 안 되고, Δ_macro ≥ +0.3%p가 필요하다.

## 외부 항목 4: MMPFN
- **저들의 주장**: 브리프(로컬 모델 요약, 원문 미검증)에 따르면 MMPFN은 TabPFN을 tabular/image/text 통합 처리하도록 확장하고, modality projector가 non-tabular 임베딩을 tabular-compatible token으로 변환한다. MGM과 CAP은 overcompressed non-tabular embeddings와 token-count-induced attention imbalance를 완화한다고 주장한다. medical/general-purpose multimodal dataset에서 경쟁 SOTA보다 우수하고, low-data regime에서 robust하며, modality가 추가될수록 양의 스케일링된다고 주장한다 (출처: multimodal_pfn_digest §1, §3, §4).
- **우리와 다른 전제**: 저들은 row-level multimodal tabular classification이고, WSI patch/bag MIL, slide-level label, spatial context, ABMIL, Primary 7 fold, branch effective rank, 확률적 적합 분산, 승격 기준은 본문에 없다. 지표는 accuracy/rank이고, DINOv2/ELECTRA, modality projector, TabPFN backbone, 100-step fine-tuning을 쓴다. patch-token 비교는 PAD-UFES-20 이미지 576 token 규모이며, WSI-scale patch 수·런타임·메모리는 브리프에 없다.
- **우리에게 옮겨올 수 있는 것**: (가설) CAP의 learnable query pooling은 fold context에서 branch output token/feature 수를 compact하게 조절하는 장치로 옮길 수 있다. (가설) MGM의 multi-head + GLU는 단일 embedding의 overcompression을 완화하고 출력 orthogonality를 높였으므로, branch 중복 감소 후보가 될 수 있다. (가설) token count 불균형이 attention mass를 왜곡한다는 관점은 branch/token 수 균형 설계 아이디어가 된다. (가설) low-data robustness 결과는 작은 fold context에서 pretrained prior가 도움이 될 수 있다는 가설을 준다.
- **옮길 수 없는 것과 이유**: Table 2/3/4/5/6/7의 accuracy/rank, MGM/CAP ablation, token count K/N, fine-tuning 100 steps/lr 1e-5는 다른 과제·모델·지표의 외부 주장이므로 우리 게이트·승격 근거가 될 수 없다. 이 논문은 ABMIL, WSI MIL, effective rank, fitting variance, promotion criterion을 다루지 않으므로, 7-branch rank 3.52/7이나 적합 분산 최대 2.8배 문제를 직접 해결한다는 증거가 없다. patch-token storage/performance 수치는 WSI-scale로 전이할 근거가 브리프에 없다. learning parameters가 허용됐다고 해도, 저들의 projector/backbone/fine-tuning 구성을 그대로 옮기는 것은 Primary 7 검증 전까지 가설일 뿐이다.
- **우리 자료로 검증하는 최소 실험**: D-050 복구 전 실행 무효. CAP/MGM-like 후보를 만들면, 먼저 gate ①로 공식 7-branch 대비 max |r|과 effective rank를 라벨 무관으로 측정한다. 8-branch라면 eff.rank ≥4.02를 요구하며, max |r| > 0.6 또는 rank 효율 하락이면 성능 조회 없이 기각. 통과 시 Primary 7 대응 per-fold Δ_macro, task cluster CI·SE·sign agreement·악화 과제를 보고하고, Δ_macro ≥ +0.3%p일 때만 지지. token count/orthogonality 가설은 branch 출력 orthogonality, effective rank, attention mass(구현 시)를 측정한다. rank/orthogonality 개선만으로는 승격이 안 되고, Δ_macro ≥ +0.3%p가 필요하다. fit variance는 여러 seed/fit으로 macro AUROC std를 보고한다. fit variance가 줄고 Δ ≥ +0.3%p면 지지, fit variance만 줄면 판별 불가, Δ가 0 근처면 판별 불가, provenance 결함이면 실행 무효.

### R1 (반증)

## 반론 1
- **공격 대상**: "ICMIL이 12개 MIL 벤치마크 평균 AUROC 84.17로 ABMIL 79.97을 넘지만, 유일한 병리 WSI 과제 TCGA에서는 ABMIL 90.7±1.6이 ICMIL 87.9±0.1을 앞선다는 관측은 병리 WSI에서 in-context MIL의 구조적 열세 또는 우리 전이 가능성을 시사한다"는 주장.
- **더 단순한 설명**: TCGA는 UNI2 1536차원 patch feature를 PCA 25차원으로 줄인 LUAD vs LUSC 이진 분류이고, bag을 약 100개로 subsample한 특수 설정이다. ICMIL 브리프의 ABMIL 기준선은 learning rate·weight decay grid, single stratified 10% held-out validation, early stopping patience 20을 쓴 per-task 학습 모델이다. 반면 ICMIL은 pretrained 고정 모델의 single forward pass다. 또한 우리 SEAL baseline ABMIL은 20 epoch 고정·validation 없는 프로토콜이므로 ICMIL 표의 ABMIL과 동일하지 않다. 따라서 ABMIL 우위는 "병리 WSI 구조"보다 "작은 TCGA subsample에서 ABMIL이 validation 선택·early stopping으로 유리하게 적합됐거나, PCA25 이진 과제가 ABMIL에 유리했다"는 더 단순한 설명으로도 성립한다. 게다가 12개 중 병리 WSI로 지목된 과제가 TCGA 1개뿐이므로 "병리에서만 진다"는 일반화는 n=1 관측이다. 기전 미확인.
- **무엇이 참이면 이 주장이 무너지는가**: (a) 같은 TCGA subsample에서 ABMIL을 early stopping 없이 고정 epoch 프로토콜로 재계산했을 때 ABMIL 우위가 사라지거나, (b) PCA25가 아닌 고차원 UNI2 또는 다른 병리 WSI 과제에서 ICMIL이 ABMIL을 넘는다면 "병리 WSI 구조적 열세" 설명은 약해진다. 반대로 고정 프로토콜에서도 ABMIL이 일관되게 이기면 이 반론의 단순 설명은 약해진다. 단, 외부 결과 재현이 우리 Primary 7에서 확인되기 전까지는 가설일 뿐이다.
- **이 반론의 한계**: 이 브리프는 로컬 모델 요약이고 원문 대조가 되지 않았다. Table 1의 ±0.1/±1.6이 어떤 resample·seed·fold 프로토콜에서 나온 것인지 문서에 없음. TCGA는 우리 Primary 7·SEAL 10과 과제·fold·bag 규모가 다르므로, TCGA의 실험 설정 인과를 우리 벤치마크로 일반화할 수 없다. ABMIL early stopping이 실제 우위의 원인인지 문서에 근거가 없다.

## 반론 2
- **공격 대상**: "ICMIL이 joint/factorized prior mixture로 상보성을 얻었으므로, 우리도 branch를 늘리면 유효 랭크가 오를 수 있다"는 주장.
- **더 단순한 설명**: ICMIL의 mixture는 Joint 0.70, Factorized(cont, MLP) 0.15, Factorized(disc, lookup) 0.15처럼 서로 다른 생성 기원과 learned nonlinear transform을 섞는 것이다. 우리 7-branch는 같은 UNI2/slide 통계에서 나온 margin/readout 변형이며, 기록된 공식 7-branch 유효 랭크 3.52/7(D-050 이후 unverified-hold)은 출력이 이미 공유 정보를 읽고 있음을 보여준다. branch 수 증가가 아니라 출력 공분산의 독립 자유도가 문제다. 같은 입력과 선형·닫힌 형태 readout 위에서 branch를 늘리면 랭크가 오르지 않는 것이 더 단순한 설명이다.
- **무엇이 참이면 이 주장이 무너지는가**: 후보 branch가 gate ①에서 max |r| ≤ 0.6이고, 8-branch 기준 eff.rank ≥ 4.02(현행 50.3% 비하락)를 만족하면서 Primary 7 paired Δ_macro ≥ +0.3%p를 사전 고정한 설계로 보여주면 "prior mixture/branch 증가가 랭크를 올릴 수 있다"는 가설이 살아남는다. 반대로 새 branch가 기존 7개와 max |r| > 0.6이거나 eff.rank가 4.02 미만이면, ICMIL의 mixture 상보성은 우리 branch 중복에 전이되지 못한다.
- **이 반론의 한계**: ICMIL은 effective rank를 측정·보고하지 않는다(문서에 없음). 따라서 ICMIL mixture가 실제로 출력 공간의 유효 랭크를 올렸는지, 단순히 성능만 올렸는지는 알 수 없다. 우리 gate ① 실행 경로가 D-050으로 현재 무효이므로, 랭크 측정 전에는 판별 불가다.

## 반론 3
- **공격 대상**: "ICMIL의 TCGA 분산 ±0.1과 ABMIL ±1.6의 격차는, 확률적 적합의 적합 분산(승격 기준 최대 2.8배) 문제를 amortized in-context prediction/PFN으로 줄일 수 있음을 시사한다"는 주장.
- **더 단순한 설명**: ABMIL은 task-specific 학습·early stopping을 하므로 training set·seed에 따른 분산이 클 수 있다. ICMIL은 pretrained 고정 모델의 single forward pass이므로 per-task fitting 분산이 구조적으로 작을 수 있다. 이 격차는 "PFN이 적합 분산을 줄인다"는 기전 주장보다 "per-task 학습을 안 하면 fitting variance가 작아지지만 bias가 커질 수 있다"는 더 단순한 설명으로도 성립한다. 브리프는 Table 1의 ±0.1/±1.6이 어떤 resample·seed·fold에서 나온 것인지 명시하지 않아 문서에 없음.
- **무엇이 참이면 이 주장이 무너지는가**: Primary 7에서 같은 fold·과제에 대해 (a) 확률적 적합의 seed/fold 분산과 (b) 고정·amortized 예측자의 seed/fold 분산을 paired Δ 기준으로 비교했을 때, 고정 예측자가 적합 분산과 paired Δ 불확실성을 낮추고 paired Δ_macro ≥ +0.3%p를 유지하면 가설이 살아남는다. 만약 분산만 줄고 paired Δ가 +0.3%p 미만이면, 또는 특정 과제 악화로 sign agreement가 나빠지면 무너진다.
- **이 반론의 한계**: ICMIL의 ±0.1이 low variance의 증거인지, 과제 난이도·PCA25·100-bag subsample·평가 프로토콜의 산물인지는 문서로 판별 불가다. PFN foundations의 variance 정리도 transformer PFN의 특정 조건이므로 우리 closed-form ridge에 자동 적용된다고 단정할 수 없다.

## 반론 4
- **공격 대상**: "네 논문에서 thresholding/downsampling, PFN variance 이론, MGM/CAP, ICMIL mixture prior를 옮겨와 우리 ABMIL 초과에 쓸 수 있다"는 일반적 전이 주장.
- **더 단순한 설명**: 이 논문들은 우리 문제의 핵심 변수를 직접 검증하지 않는다. pfn_imbalance는 tabular binary classification에서 balanced accuracy/WCA를 다루고, thresholding의 이득은 balanced accuracy/WCA로 보고된다. AUROC는 임계값 선택에 무관한 순위 지표이므로, thresholding만으로 AUROC 승격 기준을 넘긴다는 주장은 이 브리프에는 근거가 없다. pfn_foundations는 transformer PFN의 bias/variance 이론과 5차원 시뮬레이션에 한정된다. multimodal_pfn은 row-level multimodal tabular classification이고 WSI patch/bag MIL, effective rank, stochastic fit variance를 다루지 않는다. icmil은 PCA 25차원 MIL benchmark에서 학습된 Perceiver와 synthetic prior를 쓰며, 고차원 foundation embedding·WSI-scale·우리 fold protocol은 future work/미검증이다. 따라서 "성능이 좋았으므로 전이된다"는 주장보다 "과제·지표·구조가 달라 전이가 보장되지 않는다"가 더 단순하다.
- **무엇이 참이면 이 주장이 무너지는가**: 특정 전이 구성(예: context downsampling, CAP-style compact query pooling, learned prior mixture, fixed in-context predictor)이 Primary 7에서 (1) D-050 이후 provenance가 검증된 산출물로, (2) gate ① max |r| ≤ 0.6 및 eff.rank 비하락을 통과하고, (3) 공식 7-branch 대비 paired Δ_macro ≥ +0.3%p를 사전 고정한 설계로 보여주면, 해당 구성은 전이 가능하다고 판정할 수 있다. 이 셋 중 하나라도 없으면 전이 주장은 근거 없다.
- **이 반론의 한계**: "문서에 없음"은 원문 전체를 대조한 반증이 아니다. 학습 파라미터 제약 폐기(D-048)로 learned component가 허용됐으므로, MMPFN/ICMIL식 learned pooling/prior가 완전히 불가능하다고 주장하는 것이 아니다. Primary 7에서 실행 가능한 후보로 전환되면 재평가해야 한다.

## 반론 5
- **공격 대상**: "Primary 7에서 후보의 절대 AUROC 또는 외부 논문 수치와 비교만 하면 위 가설들이 갈린다"는 주장.
- **더 단순한 설명**: 우리 승격 기준은 절대 macro가 아니라 공식 7-branch 대비 대응 per-fold Δ_macro ≥ +0.3%p이고, fold는 과제 내에서 독립이 아니므로 과제 군집 불확실성과 적합 분산을 함께 봐야 한다. 따라서 절대값, SEAL 10, 외부 TCGA 수치, 단일 seed, 또는 gate ① 이전 성능 조회만으로는 가설을 갈리지 않는다. 더 단순한 설명은 "측정 설계가 다르면 같은 관측을 낳는다"이다.
- **무엇이 참이면 이 주장이 무너지는가**: 다음을 모두 재야 가설이 갈린다. (a) D-050 복구 후 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시가 기록된 산출물; (b) 후보 vs 공식 7-branch의 paired per-fold Δ, 과제별 Δ, 95% CI(df=6), SE, sign agreement, 악화 과제 전량; (c) 성능 조회 전 gate ① max |r|과 eff.rank; (d) 확률적 적합의 seed/fold 적합 분산과 paired Δ 분산; (e) TCGA 인과 가설을 살리려면 Primary 7 내 PCA25/고차원, bag subsample, ABMIL 고정/early-stopping 프로토콜을 사전 고정한 진단용 ablation. 이 중 (a)가 없으면 실행 무효, (b)만 있으면 성능만 갈리고 기전은 미확인, (c)가 없으면 랭크 가설은 판별 불가, (d)가 없으면 분산 가설은 판별 불가, (e)가 없으면 TCGA 설정 인과는 판별 불가다.
- **이 반론의 한계**: Primary 7은 TCGA·SEAL 10과 과제 집합이 다르므로, Primary 7 결과가 ICMIL의 TCGA 실험 설정 인과를 직접 입증하지는 못한다. 또한 D-050 계측 복구와 예산 없이는 (a)~(e)를 수행할 수 없어 현재는 실행 무효에 가깝다.

### T1 (이론)

이론적 병목 1
- 가정: ICMIL의 12개 MIL 평균 우위(84.17 vs ABMIL 79.97)가 병리 WSI·UNI2 고차원 patch·우리 SEAL ABMIL 프로토콜에도 동일한 estimand로 성립한다고 두고 있다.
- 이 가정이 깨지는 조건: TCGA처럼 UNI2 1536→PCA 25, bag 100 subsample, LUAD/LUSC 이진 과제, 그리고 ABMIL의 10% validation·early stopping·lr/weight-decay grid가 ICMIL의 고정 pretrained single forward pass와 다른 편향/분산을 만들 때. 특히 외부 ABMIL과 우리 SEAL ABMIL(고정 50-fold, validation/early stopping 없음, 20 epoch)이 동일 기준이 아닐 때.
- 깨질 때 나타나는 관측 가능한 증상: 평균에서는 ICMIL 우위, 병리 TCGA에서는 ABMIL 우위(90.7±1.6 vs 87.9±0.1); ABMIL 분산이 크고 ICMIL 분산이 작음; PCA 차원·bag subsample·ABMIL 학습 프로토콜을 바꾸면 순위가 바뀌는 것. 이 수치는 외부 주장이지 우리 관측이 아니며 절대값 비교 대상이 아니다.
- 문서의 어떤 관측이 이 가정과 충돌하는가: icmil_digest §3/§4의 TCGA·ABMIL 설정과 수치는 외부 주장으로만 존재한다. 우리 문서의 직접 관측은 해당 관측 없음. 다만 PROJECT.md §1의 SEAL ABMIL 프로토콜(고정 20 epoch, validation/early stopping 없음)과 icmil_digest §3의 ABMIL 프로토콜(10% validation, early stopping patience 20, max 200 epochs)이 다르므로, 동일 estimand 전제는 문서 수준에서 지지되지 않는다.

이론적 병목 2
- 가정: 여러 prior를 섞으면 상보성이 생기고, 우리처럼 브랜치를 늘리면 유효 랭크와 정보가 함께 오른다고 두고 있다.
- 이 가정이 깨지는 조건: ICMIL의 prior mixture는 synthetic data 생성 분포의 혼합이고, 우리 브랜치는 고정 UNI2 특징 위에서 산출된 마진/디스크립터의 readout 혼합이라는 점이 다르면. 브랜치 출력이 같은 저차원 방향을 읽거나 ridge readout이 중복을 압축하면 prior 다양성과 유효 랭크는 분리된다. 따라서 ICMIL의 prior mixture가 우리 rank 중복을 해결한다는 전이는, Primary 7에서 새 브랜치가 max |r|≤0.6·rank efficiency 비하락·paired Δ +0.3%p를 동시에 내기 전에는 미확정이다.
- 깨질 때 나타나는 관측 가능한 증상: 브랜치 추가 시 eff.rank가 오르지 않거나 rank efficiency가 하락(문서에 기록된 3.52/7), 게이트 ① max |r| 초과, ensemble paired Δ가 +0.3%p 미달, 과제별 악화. ICMIL prior ablation에서도 단일 prior와 mixture의 평균 차이가 제한적으로 기록되며(외부 주장), rank 지표는 없다.
- 문서의 어떤 관측이 이 가정과 충돌하는가: PROJECT.md §3.2 공식 7-branch eff.rank 3.52/7; closed_axes CA-09에서 BM/QA/DS의 eff.rank 1.29/3, 위치 계열 상관 0.25~0.93; CA-10에서 형상 병목이 절사가 아니라 정보량으로 기록됨. D-050으로 이 7-branch 기준·게이트 실적은 verified로 인용할 수 없다.

이론적 병목 3
- 가정: ICMIL TCGA ±0.1과 ABMIL ±1.6의 차이는 Bayesian model averaging/amortized inference가 확률적 적합의 적합 분산을 줄이는 일반 기전이라는 전제.
- 이 가정이 깨지는 조건: ICMIL의 ±가 시드/적합 분산이 아니라 resampling·fold·benchmark 재표본의 표준편차이거나, 고정 pretrained 파라미터로 per-task fitting이 없어 적합 분산 자체가 0에 가깝거나, PCA 25·100-bag subsample이 변동원을 줄인 것일 때. 브리프는 Table 1 ±의 산출 단위를 명시하지 않아 기전 미확인.
- 깨질 때 나타나는 관측 가능한 증상: 우리 측에서는 seed/초기화/적합만 바꿔도 macro 폭 0.0200, 적합 분산이 승격 기준 +0.3%p의 최대 2.8배(제시된 맥락), paired Δ CI가 0 포함. ICMIL식 고정 forward pass를 도입해도 fold variance는 줄지 않고 fit variance만 0이 되어 bias/estimator 문제가 남을 수 있다.
- 문서의 어떤 관측이 이 가정과 충돌하는가: CA-14 D-048에서 RU-97 §8은 동일 구성 seed만 바꿔 macro 폭 0.0200, 대응 Δ가 양방향으로 0을 배제하는 것을 실측했다. PROJECT.md §3.2는 cluster CI [-1.15%p, +2.28%p], SE 0.70%p, sign 4/7을 보고한다. icmil_digest의 TCGA ±는 외부 주장이고, ± 산출 프로토콜에 대한 우리 관측은 해당 관측 없음.

이론적 병목 4
- 가정: 네 논문의 기법·수치가 우리 WSI bag MIL, UNI2 고차원 patch, Primary 7 paired AUROC 승격 문제로 직접 이전된다고 두고 있다.
- 이 가정이 깨지는 조건: ICMIL은 learned Perceiver PFN·synthetic pretraining·PCA 25·100-bag subsample이고, pfn_imbalance는 tabular row-level balanced accuracy/WCA이며, pfn_foundations는 tabular PFN bias/variance 이론이고, MMPFN은 tabular+image/text row-level accuracy이며, 우리 문제는 slide/bag-level AUROC·effective rank·fit variance·ABMIL/MeanMIL 목표일 때.
- 깨질 때 나타나는 관측 가능한 증상: thresholding은 AUROC를 바꾸지 못하므로 승격 지표에 무효; downsampling·CAP·MGM·prior mixture는 Primary 7에서 paired Δ +0.3%p, rank efficiency 비하락, fit variance 통제를 동시에 못 냄; ABMIL 대비 우위가 pathology task에서만 사라짐.
- 문서의 어떤 관측이 이 가정과 충돌하는가: 네 digest의 §5/§6 모두 WSI MIL·ABMIL·effective rank·fit variance·closed-form ridge·promotion criterion에 대해 “본문에서 확인 못함”을 기록한다. PROJECT.md D-047은 외부 수치가 게이트·승격 근거가 될 수 없음을 명시한다. pfn_imbalance_digest의 지표는 balanced/WCA이고 AUROC 승격과 동일하지 않다.

이론적 병목 5
- 가정: 현재 Primary 7 관측(공식 7-branch macro, eff.rank, 적합 분산 보고)만으로 (1)~(4)의 가설을 갈라놓을 수 있다고 두고 있다.
- 이 가정이 깨지는 조건: 외부 TCGA의 PCA 25·100-bag·tuned ABMIL, prior mixture의 출력 다양성, ICMIL ±의 분산 원천, 그리고 우리 fit variance의 fold/fit 분해가 분리되지 않으면. D-050으로 branch list·가중치·코드 해시 provenance가 없으면 게이트 ① 실행 자체가 무효다.
- 깨질 때 나타나는 관측 가능한 증상: 다음 Primary 7 측정이 필요하고, 없으면 판별 불가: (a) 동일 fold paired Δ, 과제별 Δ, sign agreement, cluster CI; (b) 성능 조회 전 max |r|, eff.rank, rank efficiency, component occupancy; (c) seed/fit 반복으로 fit variance와 fold variance 분해, δ_min=0.003 대비 크기; (d) PCA 25 vs 고차원 UNI2(1536 또는 현재 고정 특징) paired Δ, bag subsample 100 vs 현재 고정 bag 규모 paired Δ; (e) 단일 prior/branch vs mixture의 paired Δ와 rank 변화; (f) ABMIL 고정 20 epoch 프로토콜과 tuned 프로토콜의 Primary 7 paired Δ. (d)의 현재 bag 규모는 문서에 없으므로 새로 지어내지 말고 현재 프로토콜 고정값을 사용한다.
- 문서의 어떤 관측이 이 가정과 충돌하는가: PROJECT.md §4/§4.1은 paired Δ·CI·sign agreement·MDE를, §5는 max |r|·eff.rank·rank efficiency를 정의한다. CA-14는 seed macro 폭 0.0200을 실측했다. 그러나 PCA 25·100-bag·prior mixture·fit variance 분해에 대한 Primary 7 직접 관측은 해당 관측 없음. D-050은 계측 결함으로 공식 기준선·게이트 실적을 verified로 인용할 수 없게 한다. 현재 판정은 (1)~(3) 외부 주장 기반 판별 불가, (4) 문서 근거로 이전 불가, D-050 관련 게이트 실행은 실행 무효로 구분한다.

### A1 (감사)

## 점검 1: 외부 TCGA 수치의 게이트·승격 오용 경로
- **확인하려는 것**: ICMIL/ABMIL의 TCGA 수치가 우리 Primary 7 승격이나 SEAL 10 목표 판정에 쓰이는 문서 경로가 없는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: TCGA에서 ABMIL 우위를 근거로 ICMIL 계열을 기각하거나, SEAL 10까지의 잔여 거리를 외부 수치로 축소해 주장하는 증상.
- **확인 방법**: `docs/PROJECT.md` §1·§4, `talks/lit/icmil_digest.md` 헤더; `grep -Rn "외부 주장\|게이트\|승격\|SEAL 10" docs/PROJECT.md talks/lit/icmil_digest.md`.

## 점검 2: TCGA PCA 25의 fit 범위
- **확인하려는 것**: TCGA UNI2 1536차원 → PCA 25차원 변환이 train bag/fold에만 fit되는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: test bag 포함 fit이면 TCGA AUROC가 데이터 새기로 과대해질 수 있다. ICMIL과 ABMIL이 같은 PCA fit을 쓰는지 불분명하면 모델 비교가 아니라 전처리 새기 비교가 된다. 현재 문서만으로는 판별 불가.
- **확인 방법**: `talks/lit/icmil_digest.md` §3·Appendix E; `grep -Rn "PCA\|1,536\|25\|TCGA" talks/lit/icmil_digest.md`. 원문/코드 확인 필요.

## 점검 3: TCGA 100-bag subsample의 고정 manifest
- **확인하려는 것**: TCGA 100-bag subsample이 고정 manifest·seed·라벨 무관 선택으로 수행됐는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: subsample이 라벨·과제 성능·test bag에 의존하면 TCGA 결과가 새기 된다. seed가 다르면 ±0.1/±1.6의 분산 해석도 불안정해진다.
- **확인 방법**: `talks/lit/icmil_digest.md` §3·Appendix E; `grep -Rn "100 bags\|subsample\|TCGA" talks/lit/icmil_digest.md`. 원문 data manifest 필요.

## 점검 4: ABMIL과 ICMIL의 TCGA 입력 동일성
- **확인하려는 것**: ABMIL TCGA 90.7이 ICMIL과 동일한 PCA 25·100-bag subsample 입력에서 나온가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: ABMIL이 다른 feature dimension, 다른 subsample, 다른 PCA fit을 썼다면 TCGA 차이는 모델 차이가 아니라 입력 설계 차이일 수 있다.
- **확인 방법**: `talks/lit/icmil_digest.md` §3; `grep -Rn "ABMIL\|PCA\|100 bags\|TCGA" talks/lit/icmil_digest.md`. 원문 코드/설정 필요.

## 점검 5: ABMIL 기준선의 validation 라벨 사용 비대칭
- **확인하려는 것**: ABMIL이 10% validation split·early stopping·hyperparameter 선택을 쓰고, ICMIL은 task-specific training/tuning을 쓰지 않는 비대칭이 문서에서 확인되는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: ABMIL의 TCGA 우위가 task-specific tuning·validation 선택에서 나온 것일 수 있다. ICMIL은 고정 사전학습 모델이므로 라벨 접근 방식이 다르다. TCGA 역전의 기전은 현재 미확인.
- **확인 방법**: `talks/lit/icmil_digest.md` §3; `grep -Rn "ABMIL\|validation\|early stopping\|learning rate\|weight decay" talks/lit/icmil_digest.md`.

## 점검 6: ICMIL 추론의 test bag 간 attention 차단
- **확인하려는 것**: ICMIL 추론에서 test bag이 다른 test bag에 attend하지 않고 training context bag에만 attend하는 평가 독립성이 문서에서 확인되는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: test bag 간 attention이 허용되면 test bag 특징·라벨 정보가 새어 AUROC가 과대해질 수 있다. 현재 브리프는 차단된 구조라고 요약한다.
- **확인 방법**: `talks/lit/icmil_digest.md` §2; `grep -Rn "test bag\|training context\|attend" talks/lit/icmil_digest.md`.

## 점검 7: Table 1의 ±0.1 / ±1.6 분산 원천
- **확인하려는 것**: ICMIL TCGA ±0.1, ABMIL TCGA ±1.6이 어떤 resampling·fit·training 변동인지 문서에서 확인되는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: ICMIL ±가 context resampling만 반영하고 fit randomness를 제외하면, 우리 적합 분산 문제와 비교 불가하다. ABMIL ±가 training/early stopping을 포함하면 두 분산의 구조가 다르다.
- **확인 방법**: `talks/lit/icmil_digest.md` §4·Fig. 1; `grep -Rn "±\|resampled\|variance\|AUROC" talks/lit/icmil_digest.md`. 원문 Table 1 caption/코드 필요.

## 점검 8: 우리 프로젝트의 적합 분산 보고 의무
- **확인하려는 것**: D-048 이후 모든 비교가 fold 분산과 적합 분산을 함께 다루어야 하고, RU-97 §8에서 seed만 바꾼 macro 폭 0.0200이 실측됐다는 점이 문서에서 확인되는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 단일 적합으로 per-fold Δ만 보고 승격하면, 적합 분산이 승격 기준보다 클 때 실제 불확실성을 과소평가한다.
- **확인 방법**: `docs/PROJECT.md` §1, `docs/closed_axes.md` CA-14; `grep -Rn "적합 분산\|seed std\|0.0200\|D-048" docs/PROJECT.md docs/closed_axes.md`.

## 점검 9: “적합 분산이 승격 기준 최대 2.8배” 수치의 문서 근거
- **확인하려는 것**: 첨부 문서에 적합 분산이 승격 기준 최대 2.8배라는 수치가 명시되어 있는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 사용자 맥락의 2.8배를 문서 근거처럼 인용하면 근거 범위를 넘는다. 승격 판단에 쓰려면 Primary 7에서 직접 재계산이 필요하다.
- **확인 방법**: `grep -Rn "2.8\|승격 기준\|적합 분산" docs/PROJECT.md docs/closed_axes.md talks/lit/`.

## 점검 10: ICMIL prior mixture의 effective rank 효과
- **확인하려는 것**: ICMIL의 joint/factorized prior mixture가 branch effective rank를 직접 측정하거나 개선한다는 보고가 문서에서 확인되는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: prior mixture의 상보성을 우리 branch rank 중복 해결로 오독할 수 있다. ICMIL은 effective rank metric을 다루지 않는다고 브리프가 명시한다.
- **확인 방법**: `talks/lit/icmil_digest.md` §1·§4·§6; `grep -Rn "effective rank\|rank\|mixture\|prior" talks/lit/icmil_digest.md`.

## 점검 11: ICMIL mixture weight 선택 절차
- **확인하려는 것**: 최종 mixture weight 0.70/0.15/0.15가 12개 benchmark 결과 조회 후 선택됐는지 문서에서 확인되는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: weight가 같은 12개 benchmark에서 선택됐다면 average AUROC 84.17에 선택 편향이 섞일 수 있다. TCGA 개별 악화가 평균에 숨길 수 있다.
- **확인 방법**: `talks/lit/icmil_digest.md` §2·§4 Table 3; `grep -Rn "0.70\|0.15\|mixture\|Table 3" talks/lit/icmil_digest.md`. 원문 선택 절차 확인 필요.

## 점검 12: 우리 7-branch 유효 랭크와 ICMIL rank 부재
- **확인하려는 것**: 우리 공식 7-branch 유효 랭크 3.52/7이 문서에 있고, ICMIL은 effective rank metric을 제공하지 않는다는 점이 문서에서 확인되는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: ICMIL mixture가 rank를 올린다는 전제 없이 우리 rank 병목에 적용하면, 관측되지 않은 기전을 성능 개선 근거로 바꾸게 된다.
- **확인 방법**: `docs/PROJECT.md` §3.2·§5, `talks/lit/icmil_digest.md` §6; `grep -Rn "3.52\|유효 랭크\|effective rank" docs/PROJECT.md talks/lit/icmil_digest.md`.

## 점검 13: PCA 25 TCGA 결과의 UNI2 1536차원 전이
- **확인하려는 것**: ICMIL TCGA가 PCA 25차원 평가이고, 우리 맥락의 UNI2 1536차원 환경과 차원이 다르다는 점이 문서에서 확인되는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: PCA 25 TCGA 결과를 full UNI2 1536 WSI fold에 전이하면, 평가 설계가 다른 결과를 동일 환경 결과로 오독한다.
- **확인 방법**: `talks/lit/icmil_digest.md` §3·§5·§6, `docs/closed_axes.md` CA-06; `grep -Rn "PCA\|1,536\|25\|UNI2" talks/lit/icmil_digest.md docs/closed_axes.md`.

## 점검 14: ICMIL learned pretraining 전이 가능성
- **확인하려는 것**: ICMIL이 synthetic pretraining과 learned parameters를 요구하며, CA-14 폐기 후에도 Primary 7 새 증거가 필요하다는 점이 문서에서 확인되는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: D-048의 학습 파라미터 제약 폐기만으로 ICMIL 채택 근거로 사용하면, 승격 기준을 무시한 채택이 된다.
- **확인 방법**: `talks/lit/icmil_digest.md` §5, `docs/closed_axes.md` CA-14; `grep -Rn "synthetic\|learned\|CA-14\|D-048" talks/lit/icmil_digest.md docs/closed_axes.md`.

## 점검 15: pfn_imbalance의 thresholding/downsampling 지표 불일치
- **확인하려는 것**: pfn_imbalance의 thresholding/downsampling 결과가 balanced accuracy/WCA 중심이고, 우리 승격 지표는 AUROC라는 점이 문서에서 확인되는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: τ=π1 thresholding을 AUROC 승격에 적용하면, hard threshold 변경만으로는 AUROC가 변하지 않을 수 있다. validation에서 threshold를 고르면 선택 편향이 생긴다.
- **확인 방법**: `talks/lit/pfn_imbalance_digest.md` §3·§4, `docs/PROJECT.md` §4; `grep -Rn "balanced accuracy\|WCA\|AUROC\|threshold" talks/lit/pfn_imbalance_digest.md docs/PROJECT.md`.

## 점검 16: pfn_imbalance downsampling의 effective rank 효과
- **확인하려는 것**: pfn_imbalance에서 context downsampling이 effective rank를 줄인 정량 결과가 문서에서 확인되는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: majority context downsampling을 branch rank 중복 완화로 주장하면, 측정되지 않은 rank 효과를 성능 근거로 오용한다.
- **확인 방법**: `talks/lit/pfn_imbalance_digest.md` §5·§6; `grep -Rn "downsampling\|effective rank\|rank" talks/lit/pfn_imbalance_digest.md`.

## 점검 17: pfn_foundations variance 정리의 ridge 전이
- **확인하려는 것**: pfn_foundations의 variance 감소 정리가 고정 transformer PFN·tabular 시뮬레이션 조건에 한정되고, 우리 stochastic ridge/WSI에 자동 적용된다는 근거가 문서에서 확인되는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: diminishing sensitivity·symmetrization을 근거로 적합 분산 감소를 단정하면, 기전 미확인을 기전 입증으로 바꾸게 된다.
- **확인 방법**: `talks/lit/pfn_foundations_digest.md` §5·§6; `grep -Rn "variance\|ridge\|WSI\|transformer" talks/lit/pfn_foundations_digest.md`.

## 점검 18: MMPFN orthogonality 수치의 rank·분산 오독
- **확인하려는 것**: MMPFN의 orthogonality 수치 0.0565/0.0913이 effective rank 또는 적합 분산 개선으로 보고되는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: orthogonality를 branch effective rank 개선으로 오독하면, 다른 지표의 수치를 rank 병목 해결 근거로 사용하게 된다. MMPFN은 WSI MIL·ABMIL·fit variance를 다루지 않는다.
- **확인 방법**: `talks/lit/multimodal_pfn_digest.md` §4·§6; `grep -Rn "orthogonality\|effective rank\|variance\|MIL" talks/lit/multimodal_pfn_digest.md`.

## 점검 19: D-050으로 인한 현재 계측·provenance 무효
- **확인하려는 것**: D-050으로 현재 공식 기준선이 unverified-hold이고, `src/` 환경변수 경로 무효, 저장 예측 파일에 branch list·가중치·코드 해시·fold manifest 해시가 없어 provenance가 무효라는 점이 문서에서 확인되는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 현재 저장 예측으로 Primary 7 가설을 판정하면 실행 무효다. SCREEN_ONLY 오염 검사 통과를 주장하는 실행도 무효다.
- **확인 방법**: `docs/PROJECT.md` §3.2·§3.4, `talks/reports/2026-09-18_ru90_baseline_regen.md`; `grep -Rn "D-050\|unverified-hold\|provenance\|contamination" docs/PROJECT.md`.

## 점검 20: Primary 7에서 가설을 갈리는 사전 고정 측정
- **확인하려는 것**: 외부 가설을 Primary 7에서 갈리는 사전 고정 측정 항목이 문서에 있는가: 대응 per-fold Δ, effective rank, 적합 분산, PCA 25/full 1536 민감도, bag subsample 100 민감도, calibration/WCA, provenance.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 결과를 보고 측정 항목·임계값·subsample·PCA 차원을 고르면 선택 편향이 생긴다. 승격 근거가 Primary 7 대응 비교가 아니라 탐색 후 확증이 된다.
- **확인 방법**: `docs/PROJECT.md` §4·§5, `scripts/analysis/branch_screen.py`, `tests/test_docs_consistency.py`; 새 RU 카드에 위 항목을 사전 고정해야 한다.

---

## Phase 2 — 교차

### E1 (외부)

## 제안 1 — Provenance 복구와 ICMIL-TCGA 설정 혼동자 프로브

- **외부 근거가 실제로 말하는 것**
  - ICMIL 브리프는 TCGA가 UNI2 1,536차원 patch feature를 PCA 25차원으로 줄인 LUAD vs LUSC 이진 과제라고 적는다. 큰 벤치마크는 약 100 bags로 subsample된다고 적고, TCGA도 100 bags로 subsample된다고 적는다.
  - ICMIL TCGA AUROC 87.9±0.1, ABMIL TCGA AUROC 90.7±1.6은 외부 주장이다. 이 수치는 ICMIL이 12개 MIL 벤치마크 평균에서는 ABMIL을 앞서지만, 유일한 병리 WSI 과제인 TCGA에서는 ABMIL에 밀린다는 사실만 제공한다.
  - ICMIL의 ABMIL baseline은 single stratified 10% held-out validation split, 최대 200 epochs, early stopping patience 20, lowest validation cross-entropy 설정이다. 이는 SEAL baseline의 고정 20 epoch ABMIL 프로토콜과 다르다.

- **외부 근거가 말하지 않는 것**
  - PCA 25차원, 100-bag subsample, ABMIL baseline 강도 중 어느 것이 TCGA 격차를 만들었는지를 분리하는 ablation은 문서에 없음.
  - ICMIL TCGA의 PCA fit scope, normalization, bag/slide 대응, patch 수, context/query 분할, 100 bags가 train context인지 전체 benchmark인지 문서에 없음.
  - Table 1의 ±가 fold 분산, seed 분산, resampling 분산 중 무엇인지 문서에 없음. 따라서 이 ±를 우리 적합 분산 문제와 동일 개념으로 읽을 근거가 없다.
  - ICMIL이 고차원 foundation-model embedding을 그대로 쓰는 설정의 TCGA 성능은 문서에 없음. 고차원 embedding은 future work로만 언급된다.

- **제안의 전이 오류**
  - P1은 “ICMIL TCGA 열세의 일부는 PCA25/100-bag”를 설정 혼동자 가설로 제시한다. 이는 브리프가 뒷받침하지 않는 추측이다. 브리프는 TCGA에서 ICMIL이 ABMIL에 지는 사실과 축소된 평가 설정만 제공하지, 그 격차의 원인을 분리하지 않는다.
  - 우리가 Primary 7에서 PCA25/100-bag arm을 재더라도, 그 결과는 “우리 설정이 그 변형에 민감하다”는 것이지 ICMIL TCGA 격차의 원인을 입증하지 않는다. 반대로 우리에서 효과가 없더라도 ICMIL 측 혼동자를 반박하지 않는다.
  - ICMIL의 100-bag subsample 대상이 문서에 없으므로, P1의 “train context만 100으로 제한”은 ICMIL 재현이 아니라 우리 설정의 민감도 프로브일 뿐이다.
  - ICMIL의 ABMIL이 early stopping·200 epoch까지 학습된 task-specific baseline이라는 점은, ICMIL TCGA 수치를 우리 SEAL ABMIL 목표와 직접 비교할 수 없음을 강화할 뿐이다.
  - Provenance 복구는 내부 실행 무효를 막는 필수 조건이지만, 외부 ICMIL 결과를 설명하는 근거는 아니다. `D-050`으로 provenance가 무효인 상태에서 어떤 arm도 공식 7-branch에 귀속할 수 없다.

- **Primary 7에서 갈리는 최소 측정**
  - 사전 고정 arm:  
    A. 공식 full  
    B. context-only PCA25  
    C. train context 100-bag subsample  
    D. B+C  
  - 같은 fold에서 paired per-fold Δ_macro, 적합 분산, branch margin effective rank를 재고 branch list·가중치·코드 해시·fold manifest 해시를 기록한다.
  - 판정:
    - B/C/D가 A 대비 일관된 paired Δ와 rank/variance 변화를 주면, “우리 설정의 혼동자”로만 지지.
    - ICMIL TCGA 원인에 대해서는 여전히 판별 불가.
    - 변화가 fold 노이즈 수준이면 판별 불가.
    - provenance가 복구되지 않으면 실행 무효.
  - 이 프로브의 외부 결론은 “ICMIL TCGA를 설명한다”가 아니라 “우리 후보의 PCA25/100-bag 민감성을 측정한다”로 한정해야 한다.

---

## 제안 2 — PSW: prediction-variance weighted readout

- **외부 근거가 실제로 말하는 것**
  - ICMIL 브리프는 ICMIL이 labeled context와 일관된 가설에 대한 Bayesian model averaging을 근사하여, scarce labels에서 단일 aggregator fitting의 variance를 줄인다고 주장한다.
  - PFN foundations 브리프는 고정 PFN의 variance가 개별 학습 샘플에 대한 민감도가 사라지면 사라진다고 주장하고, 1-layer transformer PFN의 variance가 `n^{-1/2}`로 사라진다고 주장한다.
  - pfn_imbalance 브리프는 thresholding/downsampling이 PFN 불균형 문제를 완화할 수 있다고 주장하지만, fitting variance를 측정하지 않는다.
  - MMPFN 브리프는 multimodal token balancing과 low-data robustness를 주장하지만, fitting variance나 branch ensemble variance를 다루지 않는다.

- **외부 근거가 말하지 않는 것**
  - closed-form ridge branch ensemble에서 branch score variance를 역수 가중으로 쓰면 AUROC가 개선된다는 결과는 문서에 없음.
  - ICMIL의 variance 감소가 branch-level score variance에 해당한다는 근거는 문서에 없음. ICMIL의 주장은 learned PFN의 aggregator fitting variance에 대한 것이다.
  - PFN foundations의 variance bound가 우리 closed-form ridge, 7-branch margin, effective rank 3.52/7, 적합 분산 최대 2.8배 문제에 자동 적용된다는 근거는 문서에 없음.
  - pfn_imbalance의 thresholding/downsampling이 AUROC 기반 승격 기준에 도움이 된다는 근거는 문서에 없음.
  - MMPFN의 token count balancing이 effective rank를 높인다는 근거는 문서에 없음.

- **제안의 전이 오류**
  - PSW의 핵심 전이는 “ICMIL의 BMA variance reduction”에서 “closed-form branch ensemble의 inverse-variance weighting”으로 넘어가는 것이다. 이 전이는 기전 미확인이다.
  - ICMIL의 variance 감소가 learned amortized model averaging에서 오는 것이라면, 우리 closed-form ridge branch margin에 같은 구조가 존재한다고 단정할 수 없다.
  - PFN foundations의 variance 감소는 고정 transformer PFN의 diminishing sensitivity 조건 하의 주장이며, 우리 ridge readout의 적합 분산 감소로 직접 전이되지 않는다.
  - PSW는 fold-wise variance를 가중치로 쓴다. 이 variance가 task signal을 반영하지 않고 seed/implementation noise를 반영하면, 가중치는 노이즈에 overfit할 수 있다.
  - `CA-08` 경계에서 prediction-variance는 new info로 허용될 수 있으나, ridge 상수·resample 수·가중치 정규화 규칙을 Primary 7 결과를 보고 조정하면 선택 편향으로 실행 무효가 된다. 사전 고정만으로는 부족하고, 선택 이력 전체가 사전에 고정되어야 한다.
  - ICMIL Table 1의 ±0.1 같은 외부 분산 수치는 PSW의 설계 근거가 될 수 없다. 그 ±의 통계적 의미도 문서에 없으며, 우리 적합 분산과 동일하지 않다.

- **Primary 7에서 갈리는 최소 측정**
  - 사전 고정:
    - stochastic fit 또는 context resample 수
    - variance 추정 대상: branch score, final score, 또는 both
    - inverse-variance ridge 상수
    - 가중치 정규화 규칙
    - AUROC를 보고 상수·branch 부분집합을 조정하지 않음
  - 측정:
    - branch별 fold-wise score variance
    - branch score variance와 paired per-fold Δ 또는 fold AUROC의 관계
    - PSW final score의 적합 분산
    - 공식 7-branch trimmed mean 대비 paired per-fold Δ_macro
    - 과제별 Δ, sign agreement, 악화 과제 전량
  - 통제:
    - label-permuted 또는 context-permuted fit에서 variance가 같은 구조를 보이는지 확인
    - variance가 context size, fold size, implementation noise와만 상관되는지 확인
  - 판정:
    - variance가 paired Δ를 예측하고 final score 적합 분산을 줄이며, 사전 고정 규칙만으로 paired Δ_macro +0.3%p를 넘으면 지지.
    - variance와 paired Δ가 무관하면 반박.
    - 상수·resample 수·가중치 규칙을 Primary 7 결과 확인 후 조정해야만 효과가 나타나면 실행 무효.
    - 효과가 fold 노이즈 수준이면 판별 불가.

---

## 제안 3 — QJF context-geometry rank probe

- **외부 근거가 실제로 말하는 것**
  - ICMIL 브리프는 inter-bag attention을 비활성화하면 mean AUROC 45.8, 활성화하면 83.3이라고 주장한다.
  - ICMIL 브리프는 factorized prior와 joint prior가 서로 다른 inductive bias를 가지며, mixture가 per-task 강점을 상속한다고 주장한다.
  - ICMIL의 prior ablation에서 Mixed 83.38, Joint 82.56, Fact.(cont, MLP) 82.76 등은 외부 주장이다.
  - MMPFN 브리프는 token count 불균형이 attention mass를 왜곡한다고 주장하지만, effective rank나 branch redundancy를 측정하지 않는다.

- **외부 근거가 말하지 않는 것**
  - ICMIL의 inter-bag attention이 closed-form linear Gram으로 대체 가능하다는 근거는 문서에 없음.
  - ICMIL의 prior mixture가 branch output의 effective rank를 높인다는 근거는 문서에 없음.
  - ICMIL은 7-branch effective rank 3.52/7 같은 branch redundancy 문제를 다루지 않는다.
  - ICMIL TCGA는 병리 WSI 관련 유일한 과제이지만, ABMIL 90.7±1.6에 ICMIL 87.9±0.1로 지는 외부 결과다. 따라서 ICMIL의 context mixing이 병리 WSI에서 ABMIL을 넘는다는 긍정적 전이 증거가 아니다.
  - PFN foundations의 multi-head attention/submodel ensemble 논의가 우리 ridge branch ensemble의 effective rank 문제를 해결한다는 근거는 문서에 없음.
  - MMPFN의 MGM/CAP token balancing이 우리 branch 중복을 줄인다는 근거는 문서에 없음.

- **제안의 전이 오류**
  - P3는 ICMIL의 learned cross-bag attention과 prior mixture를 closed-form linear Gram branch 추가 방식으로 옮긴다. 이 전이는 기전 미확인이다.
  - ICMIL의 mixture 상보성은 서로 다른 synthetic prior와 inductive bias의 혼합에 대한 주장이다. 우리 7-branch의 effective rank 3.52/7은 branch output의 중복에 대한 내부 관측이다. 둘은 같은 자유도가 아니다.
  - “브랜치를 늘리면 effective rank가 오른다”는 전제는 외부 근거가 없다. ICMIL은 branch 수 증가가 아니라 prior mixture와 learned Perceiver attention을 사용한다.
  - QJF가 `CA-04`의 linear Gram 경계 밖이라는 사실은 내부 경계 판단일 뿐, ICMIL이 QJF의 성능을 보증하는 외부 근거가 된다.
  - ICMIL의 TCGA 결과가 ABMIL에 지므로, ICMIL의 context geometry를 병리 WSI에서 ABMIL 대비 이득으로 전이하는 근거는 더 약해진다.
  - QJF가 effective rank를 올려도, 그것이 AUROC paired Δ_macro +0.3%p를 보장하지 않는다. rank 개선은 승격 기준이 아니다.

- **Primary 7에서 갈리는 최소 측정**
  - `D-050` provenance 복구 후 `SCREEN_ONLY` label-free gate ①을 먼저 수행한다.
  - gate ① 측정:
    - 공식 7-branch 대비 max |r|
    - 8-branch effective rank
    - 통과선: max |r| ≤ 0.6, effective rank ≥ 4.02
  - gate ① 실패 시 성능을 조회하지 않고 종료한다.
  - gate ① 통과 시:
    - gate ② 정보량 측정
    - Primary 7 50-fold paired per-fold Δ_macro
    - 과제별 Δ, sign agreement, 악화 과제 전량
    - 적합 분산
  - 통제:
    - QJF margin이 context size, context slide 수, 또는 단순 Gram norm과만 상관되는지 확인
    - context 순서/구성 permuted Gram 대비 QJF margin의 안정성 확인
  - 판정:
    - max |r| > 0.6 또는 effective rank < 4.02이면 gate ① 반박.
    - rank가 오르지만 paired Δ_macro +0.3%p를 넘지 못하면 승격 반박.
    - rank가 오르고 paired Δ_macro +0.3%p를 넘으면 지지.
    - rank 변화와 Δ가 fold 노이즈 수준이면 판별 불가.
    - provenance가 없으면 실행 무효.

### R1 (반증)

## 반론 1
- **공격 대상**: 제안 1의 “provenance 복구가 `D-050`으로 무효화된 공식 7-branch 산출물의 귀속 가능성을 복원한다”는 주장과, ICMIL TCGA의 PCA25·100-bag 설정을 Primary 7에서 재현하면 그 열세의 혼동자를 판별할 수 있다는 주장.
- **더 단순한 설명**: 현재 산출물이 특정 구성으로 귀속되지 않는 이유는 metadata가 없어서가 아니라 `src/`가 환경변수를 읽지 않고 `SH`/`SJ`를 켜는 문서화 수단이 동작하지 않는 등 branch control 경로 자체가 무효이기 때문일 수 있다. ICMIL TCGA 열세는 PCA25·100-bag가 아니라 저들의 ABMIL baseline이 early stopping·최대 200 epoch까지 학습된 task-specific baseline이라는 점, 또는 TCGA라는 단일 외부 과제의 분산·설정 차이로 설명될 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 저장된 예측 파일이 branch list·가중치·코드 해시·데이터 버전·fold manifest와 연결될 수 없다면, 향후 실행에 provenance를 기록하는 것은 과거 산출물의 귀속을 복원하지 못하고 새 실행만 귀속 가능하게 할 뿐이다. ICMIL의 PCA normalization·fit scope·100-bag subsample 대상이 브리프에 없다면, 제안 1의 context-only PCA와 train-context-only 100-bag는 ICMIL 설정과 동일하지 않으므로 Primary 7에서 관찰된 차이를 ICMIL TCGA 열세의 혼동자로 귀속할 수 없다. 우리 closed-form ridge와 ICMIL의 learned Perceiver가 PCA25·100-bag에 다르게 반응한다면, 이 프로브는 ICMIL을 설명하는 것이 아니라 우리 모델의 민감도 테스트에 그친다. 또한 “공식 fold 경계, SEAL 10, test 라벨은 사용하지 않으며”라는 문장이 Primary 7 고정 fold manifest를 쓰지 않는다는 뜻이면 Primary 7 비교가 성립하지 않고, 쓴다는 뜻이면 제안서 문장이 자기모순이 되어 결과는 탐색 기록으로만 남는다. 이 경우 판별 불가 또는 실행 무효에 가깝다.
- **이 반론의 한계**: 원본 산출물·코드·데이터·fold manifest가 복구 가능하고, ICMIL 원문에서 PCA와 subsample 프로토콜이 정확히 특정될 수 있다면 provenance 복구와 설정 프로브는 유효해질 수 있다. Primary 7에서 PCA25 또는 100-bag context가 안정적으로 성능을 떨어뜨린다면, 적어도 우리 모델이 feature compression·context 축소 민감하다는 국지적 사실은 얻는다. 다만 그것이 ICMIL TCGA 열세의 기전이라는 증거는 여전히 아니다.

## 반론 2
- **공격 대상**: 제안 1이 전제하는 “ICMIL이 TCGA에서 ABMIL에 밀린 것은 병리 WSI in-context MIL의 본질적 열세 또는 고차원 UNI2 feature의 한계로 읽힐 수 있다”는 해석.
- **더 단순한 설명**: TCGA는 UNI2 1536차원을 PCA 25차원으로 줄인 LUAD vs LUSC 이진 분류이고, bag을 약 100개로 subsample한 단일 외부 과제다. ABMIL 90.7 vs ICMIL 87.9는 브리프의 외부 주장이며 우리의 관측이 아니다. 이 격차는 ABMIL baseline의 학습 강도, TCGA 특유의 feature 분포, subsample 크기, 또는 외부 벤치마크의 분산으로 설명될 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: ICMIL TCGA 결과가 저들의 ABMIL hyperparameter search·early stopping·PCA 프로토콜·bag subsample·label rule에 의해 주도된다면, “병리 WSI in-context MIL이 약하다” 또는 “고차원 UNI2가 문제다”로 일반화할 근거가 없다. 우리 Primary 7·SEAL 10 과제는 TCGA와 과제 집합·bag 규모·context 규모·feature 차원·라벨 구조가 다르므로, 외부 절대값으로 우리 모델의 병리 WSI 적합성을 판단할 수 없다. 브리프가 로컬 모델 요약이고 원문 대조가 되지 않았다면, 이 수치 격차 자체도 아직 검증되지 않은 주장이다. 따라서 이 전제는 반박되지 않았지만 증거 범위 밖으로 확장된 것이다.
- **이 반론의 한계**: ICMIL 원문에서 TCGA 열세가 ABMIL 설정 변화, PCA 차원 변화, bag subsample 변화, pathology-specific prior 추가에도 지속된다면 병리 WSI 특화 약세 가설은 강해진다. 우리 Primary 7에서 고차원 UNI2 또는 작은 context가 안정적으로 열세를 만든다면, ICMIL TCGA와 동일한 기전이라고 단정하지 않더라도 우리 모델에 국지적으로 유의미한 위험 요인으로 남는다.

## 반론 3
- **공격 대상**: 제안 2 PSW의 “기존 7-branch의 prediction variance를 역수 가중으로 쓰면 적합 분산이 줄고 승격 기준을 넘을 수 있다”는 주장.
- **더 단순한 설명**: branch score variance는 seed·implementation noise, context resample noise, 또는 branch 복잡도의 산물일 수 있고, task signal의 신뢰도를 반영하지 않을 수 있다. 공식 7-branch의 유효 랭크가 3.52/7로 중복이 크다면, PSW는 새 정보를 추가하는 것이 아니라 기존 중복 branch를 재가중하는 것에 가까울 수 있다. 적합 분산 감소는 bias 증가와 교환될 수 있고, 승격 기준은 paired Δ_macro `+0.3%p`이지 최종 score variance 감소 자체다.
- **무엇이 참이면 이 주장이 무너지는가**: branch variance가 fold-specific noise와 상관하고 AUROC 개선과 무관하면 PSW는 noise에 overfit한다. low-variance branch가 bias가 크고 high-variance branch가 유용한 신호를 가진다면 역분산 가중은 성능을 떨어뜨린다. variance 추정에 context resampling을 쓰면 고정 fold의 candidate-baseline 대응 비교와 estimand가 달라질 수 있고, stochastic fit만 쓰면 variance가 구현·seed 변동일 뿐이다. `D-050`으로 branch control과 provenance가 복구되기 전에는 baseline과 candidate를 특정 구성에 귀속할 수 없으므로 승격 비교는 실행 무효다. `CA-08` 경계에서 prediction variance를 new info로 허용하더라도, 그 variance가 실제로 task-reliable information인지 Primary 7에서 검증되지 않으면 PSW는 기존 margin combination의 변형으로 읽힐 수 있다.
- **이 반론의 한계**: Primary 7에서 branch variance가 label-free이고 안정적으로 branch 신뢰도를 예측하며, 역분산 가중이 paired Δ_macro `+0.3%p`를 넘기고 적합 분산을 줄인다면 PSW는 지지될 수 있다. ICMIL의 BMA 주장을 전제로 하지 않더라도, 일반 ensemble에서 variance-aware averaging이 우리 closed-form branch ensemble에 유효한 것으로 실측된다면 제안 2의 기전은 국지적으로 성립할 수 있다.

## 반론 4
- **공격 대상**: 제안 3 QJF의 “context slide의 linear Gram은 기존 7-branch와 다른 정보원을 읽고 effective rank를 올릴 수 있으며, ICMIL prior mixture의 상보성을 우리 설정으로 옮기는 방향이다”라는 주장.
- **더 단순한 설명**: QJF는 같은 UNI2 patch feature에서 만든 slide-level context statistic의 또 다른 변형일 수 있다. 기존 `CV`·`BD` 등이 이미 context covariance·distance·shape 정보를 읽고 있으므로, linear Gram은 그것들과 상관하거나 염색·스캐너·batch·context size 같은 nuisance와 상관할 수 있다. ICMIL의 상보성은 joint/factorized synthetic prior mixture와 learned Perceiver attention에서 나온 외부 주장이며, linear Gram branch가 effective rank 중복을 해결한다는 보고는 브리프에 없다.
- **무엇이 참이면 이 주장이 무너지는가**: 공식 7-branch 대비 max |r| > 0.6이거나 8-branch effective rank 효율이 50.3% 아래로 떨어지면 gate ①에서 실패하고 성능을 조회하지 않아야 한다. Gram이 stain·scanner·batch·context size와 상관하면 task-relevant geometry가 아니라 nuisance를 읽는 것이다. QJF가 label-free geometry만 읽어서 gate ② 정보량에 도달하지 못하면 후보는 기각된다. rank가 올라가도 paired Δ_macro가 `+0.3%p` 미만이면 승격이 아니다. `CA-04`의 과거 성능 악화 기전이 미확인이라도, linear Gram의 성능은 Primary 7 paired 비교로만 판단해야 하며, 비저촉 판정만으로는 성공을 보장하지 않는다.
- **이 반론의 한계**: QJF가 기존 branch와 직교하고 gate ①·②를 통과하며, Primary 7 paired Δ_macro를 `+0.3%p` 이상 개선하면 “새 정보원”이라는 주장은 국지적으로 지지된다. Gram이 context 간 관계에서 기존 slide-level statistic이 읽지 않는 구조를 안정적으로 포착하고 nuisance에 강하다면, effective rank 증가와 성능 개선은 동시에 가능할 수 있다.

### T1 (이론)

## 이론적 병목 1
- **가정**: 제안 1은 ICMIL TCGA의 ABMIL 우위를 PCA25·100-bag·task-specific ABMIL baseline이라는 설정 차이로 분해하고, Primary 7에서 context-only PCA25와 train-context 100-bag subsample을 재면 그 설정 효과가 paired per-fold Δ로 재현되어 ICMIL 격차의 인과 요인을 판별할 수 있다고 두고 있다. 또한 provenance 기록만 추가하면 D-050으로 무효화된 공식 7-branch 산출물을 특정 구성에 귀속시킬 수 있다고 두고 있다. 이 설정 혼동자 기전은 외부 브리프에서 입증된 것이 아니라 미확인이다.
- **이 가정이 깨지는 조건**: ICMIL 브리프에 PCA fit scope·normalization·100-bag의 train/test 대상이 없거나, Primary 7 fold의 context slide 수가 100 미만인 fold가 있어 100 고정 subsample을 정의할 수 없거나, provenance 기록을 위한 코드 변경이 baseline 수치 자체를 바꾸거나, SH/SJ branch control 결함이 복구되지 않거나, arm별 paired Δ가 적합 분산·fold 노이즈 안에 머물거나, 탐색 arm을 나중에 승격 후보 선별에 쓰면.
- **깨질 때 나타나는 관측 가능한 증상**: 100 미만 fold에서 subsample 불가·보간·fold별 arm 정의 변화로 실행 무효. provenance가 있어도 branch list·가중치·코드 해시가 실제 실행 경로와 연결되지 않으면 산출물 귀속 불가. 코드 변경 전후 baseline drift로 paired Δ 귀속 불가. PCA25·100-bag arm의 paired Δ가 0 부근, 과제별 부호 불일치, 적합 분산이 Δ를 덮으면 ICMIL TCGA 열세를 설정 혼동자로 설명하지 못하고 판별 불가. ICMIL 규칙과 다른 context-only 변형만 측정하면 외부 결과와의 연결은 주장 수준으로 남는다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: PROJECT.md D-050은 src/가 env를 읽지 않아 ICF_* export가 무효, SH/SJ를 켜는 문서화 수단이 동작하지 않음, 저장 예측에 branch list·가중치·코드 해시가 없음을 기록한다. closed_axes CA-04·CA-06은 현행 context 슬라이드 수가 90~261장이라고 기록한다. icmil_digest는 TCGA가 UNI2 1536→PCA25, 100 bags, LUAD vs LUSC binary, ABMIL baseline이 max 200 epochs/early stopping이라고 주장하지만 PCA fit scope와 100-bag의 train/test 분할은 문서에 없다. 이 외부 수치는 우리 게이트·승격 근거가 될 수 없다(외부 주장, 원문 미검증).

## 이론적 병목 2
- **가정**: 제안 2는 branch score의 stochastic fit/context resample 분산이 예측 불확실성의 유효한 대리변수이고, 역분산 가중이 closed-form 7-branch ensemble의 AUROC를 개선하며 적합 분산을 줄인다고 두고 있다. ICMIL의 BMA/variance 감소 주장과 TCGA ±0.1 대 ABMIL ±1.6 같은 외부 분산 차이는 이 closed-form readout에도 같은 구조가 적용된다는 신호라고 두고 있다. ICMIL BMA가 closed-form ridge에 전이된다는 기전은 미확인이다.
- **이 가정이 깨지는 조건**: 분산 추정이 seed/implementation noise, context resample 표본 변동, fold 내 label leakage에 지배되거나, low-variance branch가 high-bias branch를 과대 가중하거나, fold-wise 가중치가 Primary 7 outcome에 의존해 선택 편향과 구별되지 않거나, D-050으로 baseline이 unverified-hold인 상태에서 paired Δ를 계산할 수 없거나, ICMIL ±의 산출 정의가 문서에 없어 우리 적합 분산과 비교할 수 없으면.
- **깨질 때 나타나는 관측 가능한 증상**: branch variance와 per-fold Δ/AUROC의 상관이 0 부근, 가중치의 fold 간 불안정, low-variance high-bias branch로 일부 과제 악화, 최종 score fit variance는 줄지만 paired Δ_macro +0.3%p 미달, CA-08 경계에서 기존 margin combination으로 판정되어 실행 무효. 적합 분산 보고만 개선되고 승격 근거는 미달. 외부 ±가 다른 estimand이면 분산 구조 비교는 판별 불가.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: PROJECT.md §1과 closed_axes CA-14는 적합 분산 보고를 필수로 하고 RU-97 §8에서 seed만 바꿔도 macro 폭 0.0200, 대응 Δ의 0 배제 가능성을 실측했다고 기록한다. PROJECT.md §4는 승격이 paired per-fold Δ_macro +0.3%p이며 fit variance는 보고 항목이지 승격 기준이 아님을 기록한다. closed_axes CA-08는 기존 branch margin combination/부분집합 탐색의 선택 편향을 폐기했고, PSW는 new info라 비저촉으로 여겨졌지만 가중치가 Primary 7 outcome에 의존하면 선택 편향 재도입 가능성이 있다. icmil_digest는 ICMIL variance 감소가 MNIST-Pos/Neg와 12 MIL benchmark에서 보고되고 probabilistic ridge/branch rank와 직접 비교되지 않았으며, TCGA ICMIL 87.9±0.1, ABMIL 90.7±1.6을 주장하지만 ± 산출 정의는 문서에 없다고 기록한다(외부 주장).

## 이론적 병목 3
- **가정**: 제안 3은 ICMIL의 prior mixture 상보성이 “서로 다른 정보원을 읽는 branch를 추가하면 effective rank가 오른다”는 형태로 전이되며, QJF linear Gram은 기존 7-branch가 읽지 않는 context geometry를 제공해 8-branch effective rank를 4.02 이상으로 올릴 수 있다고 두고 있다. 또한 D-050 복구 후 gate ①을 유효하게 실행할 수 있다고 두고 있다. QJF가 context geometry를 읽는지 nuisance를 읽는지는 기전 미확인이다.
- **이 가정이 깨지는 조건**: QJF가 기존 branch와 max |r| > 0.6으로 상관하거나, effective rank 효율이 50.3%보다 낮아지거나, Gram이 염색·스캐너·batch·context 크기 nuisance를 읽거나, D-050의 branch control/provenance/오염 검사 재설계가 완료되지 않아 SCREEN_ONLY 실행이 무효이거나, rank가 올라도 Primary 7 paired Δ가 +0.3%p에 미달하면.
- **깨질 때 나타나는 관측 가능한 증상**: gate ①에서 max |r| > 0.6 또는 eff.rank < 4.02로 성능 조회 전 종료. 통과해도 paired Δ 0 부근, 과제별 악화, rank 상승과 AUROC 개선 분리. Gram의 fold 간 변동이 nuisance와 상관하면 branch는 정보원이 아니라 nuisance reader. rank는 오르지만 paired Δ가 미달이면 rank 자체가 승격 충분조건이 아님. D-050 복구 전 실행은 실행 무효.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: PROJECT.md §3.2는 공식 7-branch effective rank 3.52/7(50.3%)을 기록하지만 D-050 이후 unverified-hold다. PROJECT.md §5는 gate ①의 8-branch 통과선을 eff.rank ≥ 4.02로 둔다. closed_axes CA-05는 원천 패치 내적/유사도가 염색 톤·배경 노이즈로 오선별될 수 있다고 기록하며, QJF linear Gram은 같은 내적/유사도 영역에 가깝다. CA-04는 과거 범용 비선형 커널 readout의 성능 하락 기전이 미확인이고 linear Gram에도 적용 근거가 불충분하다고 기록한다. D-050은 SCREEN_ONLY/오염 검사 재설계 전 오염 검사 통과를 주장하는 실행을 무효로 하고 branch control 경로를 무효로 기록한다. icmil_digest는 inter-bag attention disabled 45.8 vs enabled 83.3을 외부 주장으로만 기록하고, QJF가 같은 기전인지, ICMIL이 branch rank 중복을 다루는지 문서에 없다고 기록한다.

### A1 (감사)

## 점검 1: P1 — 과거 공식 7-branch 산출물의 provenance 귀속
- **확인하려는 것**: D-050으로 무효화된 과거 공식 7-branch 예측 파일을 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시로 특정 구성에 귀속시킬 수 있는지.
- **문서에서 확인되는가**: 모순됨
- **샌다면 나타날 증상**: 과거 `0.6227`, `+0.56%p`, 게이트 ①/② 실적을 `verified`로 인용하거나, 재현되지 않는 baseline과 새 arm의 paired Δ를 계산해 승격·반박·판별 불가 판정을 실행 무효로 만들 수 있다.
- **확인 방법**: `docs/PROJECT.md` §3.2·§3.4 D-050; 저장 예측 파일의 메타데이터 필드 검사; 새 실행에서 provenance writer가 branch list·가중치·코드 해시·fold manifest 해시를 기록하는지 단위 테스트; `tests/test_docs_consistency.py` 또는 provenance 스키마 검사.

## 점검 2: P1 — PCA 25의 fit 범위와 fold 누수
- **확인하려는 것**: PCA 25가 fold별 context/train patch feature에서만 fit되고, query/test/SEAL feature를 fit에 쓰지 않는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: test/query feature가 PCA 적재 벡터에 들어가면 fold AUROC가 인플레되고, ICMIL TCGA와의 설정 비교가 아니라 누수된 내부 평가가 된다. fold 간 PCA가 전역 fit이면 fold 독립성이 깨진다.
- **확인 방법**: `talks/lit/icmil_digest.md` §3·Appendix E에서 PCA fit scope 확인(없음); PCA 코드에서 fit 입력 ID가 fold context/train만인지 검사; fold manifest와 fit ID 교차; test ID를 fit에 넣었을 때 적재 벡터가 바뀌는지 민감도 실험.

## 점검 3: P1 — PCA 대상이 patch feature인지 slide/bag embedding인지
- **확인하려는 것**: PCA 25가 UNI2 patch feature에 먼저 적용된 뒤 bag/slide aggregate하는지, 아니면 slide/bag embedding을 25차원으로 줄이는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: ICMIL TCGA는 “UNI2 patch features 1,536차원을 PCA 25차원으로 줄인” task로 요약되지만, bag/slide 단위 feature인지 patch 단위인지가 같으면 우리 probe와 동일하지 않을 수 있다. 대상이 다르면 TCGA 열세 설명이 설정 혼동자가 아니라 feature-level mismatch가 된다.
- **확인 방법**: `talks/lit/icmil_digest.md` §3·Appendix E; 우리 pipeline에서 PCA 적용 위치(patch 전처리 vs slide descriptor) 코드 확인; ICMIL 원문 대조 필요(현재 브리프만으로는 판별 불가).

## 점검 4: P1 — 100-bag subsample의 대상과 fold manifest
- **확인하려는 것**: 100 subsample이 train context bag/slide만인지, context+query total인지, class별 균형을 쓰는지, fold test 집합을 건드리지 않는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: ICMIL은 “큰 벤치마크는 약 100 bags로 subsample”만 있고 context/query 분할이 없다. train context만 100으로 줄이면 ICMIL 재현이 아니라 우리 변형이다. subsample이 test/query를 포함하면 누수이고, class label로 균형을 맞추면 라벨 사용 선택이 된다. fold context가 100 미만인 fold의 처리가 없으면 fold manifest가 고정되지 않는다.
- **확인 방법**: `talks/lit/icmil_digest.md` §3·Appendix E; subsample 코드에서 선택 ID가 context/train만인지, replacement 여부, class 사용 여부 확인; fold별 selected context ID와 test ID 교차; fold manifest 해시 기록.

## 점검 5: P1 — 공식 fold 경계 유지
- **확인하려는 것**: P1의 “공식 fold 경계, SEAL 10, test 라벨은 사용하지 않으며”가 Primary 7 공식 fold를 재분할하지 않는다는 뜻인지, 아니면 fold 경계 자체를 쓰지 않는다는 뜻인지.
- **문서에서 확인되는가**: 모순됨
- **샌다면 나타날 증상**: `docs/PROJECT.md` §3.1은 fold를 PathoBench 공식 분할 그대로 고정하고 재분할을 금지한다. 공식 fold를 쓰지 않으면 baseline과 candidate가 다른 fold에서 평가되어 paired Δ가 무효이고, SEAL 10에 접하면 hold-out 오염이다.
- **확인 방법**: `docs/PROJECT.md` §3.1; 실행 설정의 fold ID/manifest가 PathoBench 공식 fold와 동일한지 해시 대조; SEAL 10 파일 접근 로그 확인.

## 점검 6: P1 — ICMIL ABMIL baseline과 우리 baseline 혼동
- **확인하려는 것**: ICMIL TCGA의 ABMIL `90.7±1.6`를 우리 SEAL ABMIL `0.7266` 또는 공식 7-branch baseline과 동일하게 취급하지 않는지.
- **문서에서 확인되는가**: 모순됨
- **샌다면 나타날 증상**: ICMIL ABMIL은 single stratified 10% validation, 최대 200 epoch, early stopping patience 20, lowest validation CE를 쓴 task-specific baseline이다. 우리 SEAL baseline은 slide-level 50-fold, 20 epoch 고정, validation/early stopping 없는 프로토콜이다. 절대값 직접 비교는 금지이고, 외부 수치는 게이트/승격 근거가 될 수 없다.
- **확인 방법**: `talks/lit/icmil_digest.md` §3 ABMIL 설정; `docs/PROJECT.md` §1 SEAL Baseline 측정 프로토콜; 제안서에서 ICMIL ABMIL 수치가 승격/게이트 문장에 쓰이지 않는지 `grep`.

## 점검 7: P1 — PCA/subsample 변형의 선택 편향
- **확인하려는 것**: PCA fit scope, PCA 대상, subsample 방법, 100 미만 fold 처리가 Primary 7 AUROC 조회 전에 고정되는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 여러 변형 중 Primary 7에서 좋은 arm을 고른 뒤 “탐색”이라 부르면 §214식 선택 편향이 된다. 같은 Primary 7에서 후속 승격에 쓰면 승격 판정이 실행 무효 또는 과대평가된다.
- **확인 방법**: 사전 등록 문서/설정 파일의 timestamp와 AUROC 산출 로그 순서; 변형 목록이 결과 조회 전에 고정됐는지; 선택 이력 기록; `current_status.md` 또는 RU 카드에 자료 접근 이력.

## 점검 8: P2 — branch score variance의 추정 데이터 범위
- **확인하려는 것**: PSW의 branch score variance가 test/query 예측, test 라벨, fold AUROC를 쓰지 않고 context/train stochastic fit에서만 추정되는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: test 예측의 분산으로 가중치를 정하면 평가 자료를 모델 선택에 쓴 adaptive evaluation이 된다. fold AUROC를 variance 계산에 쓰면 라벨이 가중치에 새고 paired Δ가 인플레된다.
- **확인 방법**: PSW 코드에서 variance 계산 함수의 입력이 context/train predictions인지, test predictions인지 확인; test feature/label을 제거했을 때 가중치가 동일하게 유지되는지 실험; 가중치 산출 시점과 test 평가 시점 로그.

## 점검 9: P2 — context resample의 paired 설계
- **확인하려는 것**: context resample을 쓸 경우 baseline과 candidate가 같은 fold, 같은 context ID, 같은 seed에서 비교되는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: candidate만 context를 resample하면 Δ가 모델 차이가 아니라 context 집합 차이가 된다. branch별 resample seed가 다르면 variance 추정이 branch 간 비교 불가능해지고, seed noise가 적합 분산 2.8배 문제에 섞인다.
- **확인 방법**: resample seed와 context ID를 branch·fold별로 기록; baseline/candidate context set 동일성 검사; 동일 seed에서 baseline/candidate paired 실행; seed std 보고.

## 점검 10: P2 — ridge/resample/normalization 상수 선택
- **확인하려는 것**: inverse-variance + ridge의 ridge 상수, resample 수, 가중치 정규화 규칙이 Primary 7 AUROC 조회 전에 고정되는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: Primary 7에서 상수를 튜닝한 뒤 승격 비교하면 CA-08 경계의 “선택에 쓴 평가자료의 최댓값을 확증으로 쓰는” 양식에 접근한다. variance가 fold noise를 반영하면 ridge 상수가 noise fitting을 결정한다.
- **확인 방법**: 상수 값과 선택 근거가 AUROC 로그 이전인지; grid search 로그가 있으면 실행 무효로 처리; 사전 등록 문서와 코드 상수 대조.

## 점검 11: P2 — 기준선 재현 가능성
- **확인하려는 것**: PSW의 paired Δ_macro 기준선이 D-050 이후 재현 가능한 공식 7-branch trimmed mean인지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 현재 공식 7-branch baseline은 `unverified-hold`이고, `src/`가 `ICF_*` 환경변수를 읽지 않아 SH/SJ 통제가 무효이며, 저장 예측에 branch list·가중치·코드 해시가 없다. 이 상태에서 PSW Δ를 계산하면 기준선 귀속이 안 돼 승격·반박·판별 불가 모두 실행 무효다.
- **확인 방법**: `docs/PROJECT.md` §3.2 D-050; baseline 재현 스크립트가 branch list·가중치·코드 해시를 기록하는지; SH/SJ 활성화 경로가 실제로 동작하는지 단위 테스트; 재현된 baseline의 fold AUROC와 저장값 대조.

## 점검 12: P2 — 적합 분산 감소와 승격 기준 분리
- **확인하려는 것**: PSW가 최종 score 적합 분산을 줄였다는 사실만으로 승격 또는 채택을 주장하지 않는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: `docs/PROJECT.md` §4의 승격은 Primary 7 대응 Δ_macro ≥ `+0.3%p`이고, 적합 분산은 함께 보고할 불확실성이다. 분산 감소만으로는 승격이 안 되고, 분산 감소와 AUROC 개선이 다른 fold에서 갈리면 “판별 불가”를 “지지”로 바꾸지 않아야 한다.
- **확인 방법**: 보고서에 paired Δ_macro, SE/CI, sign agreement, 악화 과제, 적합 분산을 별도 필드로 두는지; 승결 문장이 Δ_macro 기준에만 연결되는지.

## 점검 13: P3 — QJF Gram의 데이터 범위
- **확인하려는 것**: QJF linear Gram이 fold context slides만 사용하고 query/test/SEAL slides를 Gram 행렬에 넣지 않는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: query/test slide가 Gram에 들어가면 label-free gate ①이라도 test feature가 후보 선택에 쓰여 평가 독립성이 깨진다. SEAL slide가 들어가면 hold-out 오염이다.
- **확인 방법**: QJF 코드에서 Gram 행렬의 slide ID가 fold context만인지; test/SEAL ID와 교차; context ID manifest 해시; SEAL 파일 접근 로그.

## 점검 14: P3 — Gram 전처리/정규화 fit 범위
- **확인하려는 것**: Gram 계산 전 표준화·whitening·PCA·스케일링이 fold context에서만 fit되는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 전역 fit이면 fold 간 정보가 새고, effective rank나 max |r|이 전처리 산출물에 의해 인공적으로 변한다. branch 간 스케일 차이가 Gram trace에 반영되면 rank 효율 50.3% 기준과 비교 불가능해진다.
- **확인 방법**: 전처리 코드에서 fit 입력 ID 확인; fold별 전처리 파라미터 기록; 스케일 불변성 검사(브랜치 마진을 상수배缩放했을 때 rank/max|r|가 변하는지).

## 점검 15: P3 — gate ① 실행 경로 유효성
- **확인하려는 것**: D-050 이후 SCREEN_ONLY·`branch_screen.py`·오염 검사 경로가 실제로 복구되어 max |r|·eff.rank 결과를 구성에 귀속할 수 있는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 현재 `src/`가 환경변수를 읽지 않아 branch control이 무효이고, 값 기반 오염 검사는 폐지됐다. 복구 전 max |r| ≤ 0.6 또는 eff.rank ≥ 4.02를 통과로 쓰면 구성 귀속이 안 돼 실행 무효다.
- **확인 방법**: `docs/PROJECT.md` §3.4 D-050; `docs/closed_axes.md` §3-I; `scripts/analysis/branch_screen.py`가 branch list·가중치·코드 해시를 기록하는지; SCREEN_ONLY가 앙상블에 영향을 주지 않는 provenance 검사.

## 점검 16: P3 — effective rank 계산 범위
- **확인하려는 것**: 8-branch effective rank ≥ 4.02가 현행 3.52/7과 같은 fold, 같은 margin, 같은 전처리, 같은 fit 범위에서 계산되는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: rank가 다른 fold 수, 다른 branch margin 스케일, 다른 stochastic fit에서 계산되면 4.02 통과가 실제 정보량 증가가 아니라 계측 차이일 수 있다. rank 효율 50.3% 비하락 기준이 무의미해진다.
- **확인 방법**: effective rank 코드와 입력 margin 파일 확인; 50-fold Primary 7 전 fold에서 baseline 7-branch와 candidate 8-branch를 같은 fold에서 계산; rank 계산에 쓰인 seed/context/전처리 manifest 기록.

## 점검 17: P3 — QJF/LSAK 후보 선택
- **확인하려는 것**: QJF와 LSAK 중 하나를 고르는 규칙이 position provenance 확인과 rank/AUROC 조회 전에 고정되는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: “position provenance가 확인된 경우 LSAK”가 모호하면, rank나 AUROC를 보고 LSAK로 바꾸는 선택이 가능하다. 이는 gate ① 이전 후보 선택 편향이 되고, CA-04/CA-09 경계 심사도 무력화된다.
- **확인 방법**: 후보 선택 규칙이 사전 문서에 있는지; position provenance 확인이 label-free이고 성능/랭크 조회 이전인지; QJF/LSAK 실행 로그 순서.

## 점검 18: P3 — ICMIL inter-bag attention의 기전 전이
- **확인하려는 것**: ICMIL의 inter-bag attention disabled 45.8 vs enabled 83.3를 QJF linear Gram이 effective rank를 올리거나 AUROC를 개선할 것이라는 기전 증거로 사용하지 않는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 외부 성능 관측을 기전 입증으로 바꾸면, learned Perceiver attention의 효과가 closed-form linear Gram으로 전이된다는 미확인 주장을 세운다. QJF의 유효성은 Primary 7 gate ①/②와 paired Δ로만 판정해야 한다.
- **확인 방법**: 제안서에서 ICMIL Table 2 수치가 “기전” 문장에 쓰이지 않는지; QJF 기전 필드에 “기전 미확인”이 유지되는지; Primary 7 측정 전 외부 수치 인용 금지.

## 점검 19: 공통 — Primary 7 탐색과 승격 확인 분리
- **확인하려는 것**: P1의 PCA/subsample 탐색, P2의 PSW, P3의 QJF가 같은 Primary 7에서 후보를 고르고 같은 Primary 7에서 승격을 확정하지 않는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 탐색에서 Primary 7 결과를 보고 살아남은 후보만 승격 비교에 넣으면 §214의 선택 편향이 반복된다. 승격은 사전 고정한 규칙과 선택·적응을 반영한 설계로 분리해야 한다.
- **확인 방법**: RU 카드에 자료 접근 이력과 선택 이력 기록; 탐색용 arm과 승격용 candidate가 다른지; 승격 비교 전에 후보가 고정됐는지; `docs/PROJECT.md` §4.1 준수 여부.

## 점검 20: P1 — TCGA와 Primary 7의 과제 동일성
- **확인하려는 것**: ICMIL TCGA LUAD vs LUSC가 우리 Primary 7의 특정 과제와 동일하거나 같은 fold protocol을 공유하는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: TCGA가 Primary 7에 없거나 fold가 다르면 “병리 WSI 과제”라는 공통점만으로 ICMIL TCGA 결과를 우리 과제 설정의 혼동자 증거로 쓰는 것은 과대추론이다. 절대값 직접 비교도 금지된다.
- **확인 방법**: Primary 7 과제 목록과 fold manifest가 TCGA LUAD/LUSC와 일치하는지 `docs/current_status.md` 또는 fold manifest 확인; `talks/lit/icmil_digest.md` Appendix E의 TCGA 정의와 우리 task 정의 대조; 일치하지 않으면 “판별 불가”로만 기록.

## 점검 21: P2 — PSW의 “새 정보” 성립 여부
- **확인하려는 것**: prediction-variance가 기존 branch margin·score에서 파생된 재가중 신호인지, CA-08 경계 밖의 독립적 정보인지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: variance가 stochastic fit의 seed noise나 context resample noise만 반영하면 PSW는 기존 margin 위의 결합 함수 탐색으로 퇴화한다. CA-08의 “새 정보(예측분산)” 예외가 적용되지 않아 실행 무효가 될 수 있다.
- **확인 방법**: variance 추정량이 branch margin과 독립적인지; 동일 context에서 seed만 바꿨을 때 variance가 task signal을 반영하는지; `docs/closed_axes.md` CA-08 경계 문구 대조.

---

## Phase 2b — 제안 좌석 응답

### P1 (제안)

### E1 — 제안 1
- **결정**: 수용·수정
- **근거·수정**: `icmil_digest.md`는 TCGA가 UNI2 1,536차원 patch feature를 PCA 25차원으로 줄인 100-bag subsample 과제이며, ABMIL baseline이 single stratified validation·최대 200 epoch·early stopping을 썼다고만 주장한다. PCA fit scope·normalization, 100-bag의 train/test 대상, Table 1 ±의 산출 정의는 문서에 없으므로 ICMIL TCGA 열세의 인과 요인은 **판별 불가**로 둔다. 제안 1에서 “ICMIL TCGA 열세의 일부는 PCA25/100-bag”라는 인과 문구를 제거하고, 프로브 결론을 “우리 Primary 7 후보의 PCA25·context 축소 민감성 측정”으로 한정한다. `D-050`은 저장 예측에 branch list·가중치·코드 해시가 없으므로, 과거 산출물의 verified 귀속 복원이 아니라 새 실행의 provenance writer 복구로 목표를 수정한다.

### E1 — 제안 2
- **결정**: 수용·수정
- **근거·수정**: ICMIL의 Bayesian model averaging은 learned PFN의 aggregator fitting variance에 대한 외부 주장이고, closed-form 7-branch margin의 inverse-variance weighting으로 전이된다는 브리프 근거는 없다. PFN foundations의 variance bound도 고정 transformer PFN의 조건 하의 주장이다. 따라서 PSW의 승격 문장을 “적합 분산 감소”에서 Primary 7 paired Δ_macro `+0.3%p`로만 연결한다. branch variance가 seed·implementation noise인지 판별하기 위해 label/context permuted 대조와 seed-only noise 대조를 사전 통제에 넣고, variance와 paired Δ가 무관하면 **반박**, fold 노이즈 수준이면 **판별 불가**, 결과 보고 후 상수·resample·branch 부분집합을 조정하면 **실행 무효**로 처리한다.

### E1 — 제안 3
- **결정**: 수용·수정
- **근거·수정**: ICMIL의 prior mixture와 inter-bag attention이 closed-form linear Gram branch의 effective rank 개선이나 AUROC 이득을 보증한다는 브리프 근거는 없다. ICMIL Table 2의 `inter-bag attention disabled 45.8 / enabled 83.3`은 성능 관측이지 QJF 기전의 입증이 아니다. 또한 ICMIL이 TCGA에서 ABMIL에 지므로, 이 결과로 QJF의 병리 WSI 전이를 강하게 주장하기 어렵다. 제안 3에서 ICMIL prior mixture 전이 주장을 제거하고, QJF를 내부 label-free rank probe로만 둔다. gate ① 실패 시 성능 미조회, rank 상승만으로는 승격 불충분, paired Δ_macro `+0.3%p` 미달이면 **반박**으로 처리한다.

### R1 — 반론 1
- **결정**: 수용·수정
- **근거·수정**: `PROJECT.md` §3.2·§3.4의 `D-050`은 `src/`가 `ICF_*` 환경변수를 읽지 않고, `SH`/`SJ`를 켜는 문서화 수단이 동작하지 않으며, 저장 예측 파일에 branch list·가중치·코드 해시가 없음을 기록한다. 따라서 provenance 기록은 과거 산출물의 verified 복원이 아니라 새 실행의 구성 귀속만 보장한다. 제안 1의 “과거 공식 7-branch 산출물의 귀속 가능성을 복원한다”는 문구를 “새 실행에서 특정 구성에 귀속되도록 provenance writer와 branch control을 복구한다”로 수정한다. ICMIL PCA/subsample 프로토콜이 브리프에 없으므로 ICMIL TCGA 혼동자 귀속은 **판별 불가**로 남긴다. “공식 fold 경계, SEAL 10, test 라벨은 사용하지 않는다”는 문장은 “PathoBench 공식 fold 경계와 fold manifest 해시를 사용하고, SEAL 10 파일·test 라벨 접근만 금지한다”로 명확화한다.

### R1 — 반론 2
- **결정**: 수용·수정
- **근거·수정**: ICMIL TCGA의 `87.9±0.1`과 ABMIL `90.7±1.6`은 로컬 모델 요약 기반 외부 주장이고, TCGA는 UNI2 1536차원을 PCA 25차원으로 줄인 LUAD vs LUSC 단일 이진 과제다. 이 과제는 우리 Primary 7의 과제 집합·bag 규모·context 규모·feature 차원·라벨 구조와 동일하다는 문서 근거가 없다. 따라서 “병리 WSI in-context MIL의 본질적 열세” 또는 “고차원 UNI2 feature의 한계”로 확장하지 않는다. 제안 1에서 이 해석을 제거하고, ICMIL TCGA는 외부 동기만 제공한다고 둔다.

### R1 — 반론 3
- **결정**: 수용·수정
- **근거·수정**: branch score variance는 seed·implementation noise, context resample noise, branch 복잡도의 산물일 수 있고 task signal의 신뢰도를 반영하지 않을 수 있다. 역분산 가중이 저분산 high-bias branch를 과대 가중하면 AUROC가 악화될 수 있다. PSW의 승격 문장은 paired Δ_macro `+0.3%p`로만 연결하고, 적합 분산 감소는 보조 보고로 둔다. label-permuted 또는 context-permuted fit에서 variance 구조가 유지되는지, seed-only noise에서 variance가 task signal과 분리되는지를 사전 통제에 넣는다. `D-050` 복구 전에는 PSW paired Δ를 승격·반박·판별 불가의 정식 근거로 쓰지 않고 탐색 기록으로만 둔다.

### R1 — 반론 4
- **결정**: 수용·수정
- **근거·수정**: QJF linear Gram은 기존 `CV`·`BD` 등 context covariance·distance·shape statistic과 같은 UNI2 patch feature에서 나온 slide-level statistic 변형일 수 있다. `CA-05`는 원천 패치 유사도가 염색 톤·배경 노이즈로 오선별될 수 있음을 기록하므로, QJF도 염색·스캐너·batch·context size nuisance를 읽을 가능성을 사전 통제해야 한다. ICMIL prior mixture 전이 주장을 제거하고, QJF의 유효성은 gate ①의 max |r|·effective rank, gate ② 정보량, Primary 7 paired Δ로만 판정한다. rank가 올라가도 paired Δ_macro `+0.3%p` 미만이면 승격이 되지 않는다.

### T1 — 이론적 병목 1
- **결정**: 수용·수정
- **근거·수정**: `closed_axes.md` CA-04·CA-06은 현행 context slide 수를 90~261장으로 기록하므로, context slide가 100 미만인 fold에서 100 고정 subsample은 불안정하다. 제안 1의 100-bag arm을 “context slide ≥100이면 무작위 100개, 100 미만이면 전체 context를 사용. replacement 없음, class label 미사용, test/query 미포함”으로 사전 고정한다. 이는 ICMIL 재현이 아니라 우리 모델의 민감도 프로브다. ICMIL PCA fit scope·normalization·100-bag 대상이 문서에 없으므로 ICMIL TCGA 원인은 **판별 불가**로 남는다.

### T1 — 이론적 병목 2
- **결정**: 수용·수정
- **근거·수정**: `PROJECT.md` §4는 승격 기준을 Primary 7 대응 Δ_macro `+0.3%p`로 두고, 적합 분산은 함께 보고할 불확실성으로 다룬다. `CA-14`와 `RU-97` §8은 seed만 바꿔도 macro 폭 `0.0200`이 실측됐음을 기록하므로, variance 추정이 seed noise를 포함할 수 있다. PSW의 승격 문장을 적합 분산 감소가 아니라 paired Δ_macro `+0.3%p`로만 연결한다. ICMIL Table 1의 ±0.1/±1.6은 산출 정의가 문서에 없으므로 우리 적합 분산과 동일 개념으로 비교하지 않는다.

### T1 — 이론적 병목 3
- **결정**: 수용·수정
- **근거·수정**: `D-050` 이후 SCREEN_ONLY·`branch_screen.py`·오염 검사 재설계가 복구되지 않으면 QJF gate ① 결과도 특정 구성에 귀속할 수 없다. 따라서 QJF 실행은 provenance·branch control 복구 전에는 **실행 무효**다. `CA-05`는 원천 패치 유사도가 염색 톤·배경 노이즈로 오선별될 수 있음을 기록하므로, QJF Gram도 같은 nuisance 영역의 가능성을 통제한다. rank가 4.02를 넘더라도 paired Δ_macro `+0.3%p` 미달이면 **반박**으로 처리한다.

### A1 — 점검 1
- **결정**: 수용·수정
- **근거·수정**: `PROJECT.md` §3.2·§3.4 `D-050`은 과거 공식 7-branch 값 `0.6227`, 승격 근거 `+0.56%p`, gate ①/② 실적을 `verified`로 인용하지 않도록 기록한다. 제안 1의 목표는 과거 산출물 재확인이 아니라, 새 실행에서 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시를 기록하고, 이 writer가 실제로 동작하는지 단위 테스트로 확인하는 것으로 수정한다.

### A1 — 점검 2
- **결정**: 수용·수정
- **근거·수정**: `icmil_digest.md`에는 PCA 25의 fit scope가 없다. 우리 probe의 PCA는 fold별 context/train patch feature에서만 fit하고, query/test/SEAL feature를 fit 입력에 넣지 않는 것을 사전 고정한다. fit에 쓴 slide/patch ID, fold manifest 해시, 적재 벡터 해시를 provenance에 기록한다.

### A1 — 점검 3
- **결정**: 수용·수정
- **근거·수정**: ICMIL 원문 대조가 안 되므로, 브리프의 “UNI2 patch features 1,536차원을 PCA 25차원으로 줄인” 문장을 따라 우리 probe는 patch feature PCA로만 수행한다. slide/bag embedding PCA는 별도 arm으로 두고, ICMIL TCGA 재현이라고 부르지 않는다.

### A1 — 점검 4
- **결정**: 수용·수정
- **근거·수정**: `icmil_digest.md`는 “큰 벤치마크는 약 100 bags로 subsample”만 기록하고 100-bag의 train/test 대상을 특정하지 않는다. 우리 probe는 train/context만 subsample하고, replacement·class balance·test/query 포함을 금지한다. context slide 수 100 미만 fold는 전체 context 사용으로 사전 고정한다. 이 arm은 ICMIL 재현이 아니라 우리 민감도 프로브임을 명시한다.

### A1 — 점검 5
- **결정**: 수용·수정
- **근거·수정**: `PROJECT.md` §3.1은 fold를 PathoBench 공식 분할 그대로 고정하고 재분할을 금지한다. 제안 1의 “공식 fold 경계, SEAL 10, test 라벨은 사용하지 않는다”는 문장은 “PathoBench 공식 fold 경계와 fold manifest 해시를 사용하고, SEAL 10 파일·test 라벨 접근만 금지한다”로 수정한다.

### A1 — 점검 6
- **결정**: 수용·수정
- **근거·수정**: ICMIL ABMIL `90.7±1.6`은 single stratified 10% validation, 최대 200 epoch, early stopping patience 20, lowest validation CE를 쓴 외부 baseline이다. 우리 SEAL ABMIL `0.7266`은 slide-level 50-fold CV, 20 epoch 고정, validation/early stopping 없는 프로토콜이다. 두 수치는 절대값 직접 비교 금지이고, ICMIL ABMIL 수치를 우리 게이트·승격 문장에 쓰지 않는다.

### A1 — 점검 7
- **결정**: 수용·수정
- **근거·수정**: PCA fit scope, PCA 대상, subsample 방법, 100 미만 fold 처리는 Primary 7 AUROC 조회 전에 고정한다. 여러 arm 중 Primary 7에서 좋은 arm을 보고 고르면 §214식 선택 편향으로 **실행 무효**가 된다. arm 목록, 선택 이력, 자료 접근 이력을 사전 기록하고 결과는 탐색 기록으로만 둔다.

### A1 — 점검 8
- **결정**: 수용·수정
- **근거·수정**: PSW의 branch score variance는 context/train fit의 score 변동에서만 추정한다. test/query 예측, test 라벨, fold AUROC를 가중치 산출에 쓰지 않는다. test feature/label을 제거해도 가중치가 동일하게 유지되는지 확인한다.

### A1 — 점검 9
- **결정**: 수용·수정
- **근거·수정**: context resampling을 쓰면 baseline과 candidate가 같은 fold, 같은 context ID, 같은 seed에서 paired로 실행된다. candidate만 context를 resample하면 Δ가 모델 차이가 아니라 context 집합 차이가 된다. branch별 seed, context ID, fold manifest를 기록하고, 동일 seed에서 paired 실행을 확인한다.

### A1 — 점검 10
- **결정**: 수용·수정
- **근거·수정**: inverse-variance ridge 상수, resample 수, 가중치 정규화 규칙은 Primary 7 AUROC 조회 전에 고정한다. 결과 보고 후 상수·resample 수·branch 부분집합을 조정하면 `CA-08`의 선택 편향 양식으로 **실행 무효** 처리한다.

### A1 — 점검 11
- **결정**: 수용·수정
- **근거·수정**: `D-050` 이후 공식 7-branch baseline은 `unverified-hold`이고, `src/`의 branch control 경로가 무효다. 이 상태에서 PSW paired Δ를 계산해도 기준선 귀속이 안 되므로, 복구 전에는 승격·반박·판별 불가 판정 대상이 아니라 탐색 기록으로만 둔다.

### A1 — 점검 12
- **결정**: 수용·수정
- **근거·수정**: `PROJECT.md` §4는 승격 기준을 paired Δ_macro `+0.3%p`로 두고, 적합 분산은 함께 보고할 불확실성으로 명시한다. PSW 보고서에서 paired Δ_macro, SE/CI, sign agreement, 악화 과제, 적합 분산을 별도 필드로 두고, 승결 문장을 적합 분산 감소가 아니라 paired Δ_macro 기준에만 연결하도록 수정을 재확인한다.

### A1 — 점검 13
- **결정**: 수용·수정
- **근거·수정**: QJF linear Gram 행렬은 fold context slide만 사용하고, query/test/SEAL slide ID를 Gram 행렬에 넣지 않는다. context ID manifest 해시, SEAL 파일 접근 로그를 provenance에 기록한다.

### A1 — 점검 14
- **결정**: 수용·수정
- **근거·수정**: Gram 계산 전 표준화·whitening·PCA·스케일링은 fold context에서만 fit한다. 전처리 파라미터, fit 입력 ID, fold별 전처리 해시를 기록한다. branch margin을 상수배 스케일링했을 때 max |r|·effective rank가 변하지 않는지 계측 불변성 검사를 넣는다.

### A1 — 점검 15
- **결정**: 수용·수정
- **근거·수정**: `D-050` 이후 SCREEN_ONLY·`branch_screen.py`·오염 검사 재설계가 복구되기 전에는 QJF gate ①의 max |r|·effective rank 결과를 특정 구성에 귀속할 수 없다. 복구 전 실행은 **실행 무효**다.

### A1 — 점검 16
- **결정**: 수용·수정
- **근거·수정**: 8-branch effective rank `≥ 4.02`는 baseline 7-branch와 같은 fold, 같은 margin 스케일, 같은 전처리, 같은 fit 범위에서 계산한다. 다른 fold 수, 다른 seed, 다른 전처리에서 계산된 rank로 통과를 주장하지 않는다.

### A1 — 점검 17
- **결정**: 수용·수정
- **근거·수정**: QJF/LSAK 선택 규칙이 모호하면 rank/AUROC 보고 후 후보 변경 가능성이 있다. QJF를 먼저 고정하고, LSAK는 position provenance가 label-free로 확인된 뒤에 별도 후보로만 진행한다. 후보 변경 규칙은 rank/AUROC 조회 전에 문서화한다.

### A1 — 점검 18
- **결정**: 수용·수정
- **근거·수정**: ICMIL Table 2의 `inter-bag attention disabled 45.8 / enabled 83.3`은 성능 관측이지 closed-form linear Gram의 기전 입증이 아니다. QJF 기전 필드는 “기전 미확인”으로 유지하고, QJF 유효성은 Primary 7 gate ①/②와 paired Δ로만 판정한다.

### A1 — 점검 19
- **결정**: 수용·수정
- **근거·수정**: P1 PCA/subsample arm, P2 PSW, P3 QJF는 Primary 7 탐색으로만 수행한다. 탐색에서 Primary 7 결과를 보고 살아남은 arm을 승격 candidate로 자동 격상하지 않는다. 탐색 arm과 승격 candidate를 RU 카드에서 분리하고, 자료 접근 이력과 선택 이력을 기록한다.

### A1 — 점검 20
- **결정**: 수용·수정
- **근거·수정**: ICMIL TCGA LUAD vs LUSC가 우리 Primary 7 과목과 동일하거나 같은 fold protocol을 공유하는지는 문서에 없다. “병리 WSI 과제”라는 공통점만으로 ICMIL TCGA 결과를 우리 과제 설정의 혼동자 증거로 쓰지 않는다. ICMIL TCGA 원인은 **판별 불가**로 둔다.

### A1 — 점검 21
- **결정**: 수용·수정
- **근거·수정**: PSW variance가 기존 branch margin에서 파생된 재가중 신호로 퇴화할 수 있다. `CA-08`의 “새 정보(예측분산)” 예외를 주장하기 전에, seed-only noise·label/context permuted 대조로 variance가 margin과 독립적인 task-reliable 신호인지 확인한다. 확인 전에는 PSW를 기존 margin combination 변형으로 보고 승격을 주장하지 않는다.

---

## Phase 3 — 종합

### S1 (종합)

## 지지 (증거가 뒷받침함)
- **ICMIL TCGA는 우리 SEAL/Primary 7과 다른 평가 설정이다.**  
  근거: `icmil_digest.md` §3·Appendix E는 TCGA가 UNI2 1,536차원 patch feature를 PCA 25차원으로 줄인 LUAD vs LUSC 이진 과제이고, 큰 벤치마크는 약 100 bags로 subsample된다고 기록한다. 같은 digest의 ABMIL baseline은 single stratified 10% validation, learning-rate/weight-decay grid, 최대 200 epochs, early stopping patience 20, lowest validation CE를 쓴 task-specific supervised baseline이다. 반면 `PROJECT.md` §1의 SEAL ABMIL baseline은 slide-level 50-fold CV, validation/early stopping 없이 20 epoch 고정 학습이다. 따라서 두 ABMIL 수치는 동일 estimand가 아니다.

- **ICMIL TCGA의 ABMIL 우위는 우리 SEAL 10 목표의 직접 근거가 될 수 없다.**  
  근거: 네 digest 헤더와 `PROJECT.md` §1 `D-047`은 외부 문헌 수치가 우리 게이트·승격의 근거가 될 수 없고, 모델 개선 근거는 Primary 7뿐이라고 명시한다. TCGA는 우리 Primary 7·SEAL 10과 과제 집합·fold protocol·feature 차원·bag 규모가 다르며, 절대값 직접 비교는 금지된다.

- **ICMIL의 prior mixture는 “브랜치 수 증가”가 아니라 synthetic data-generating prior의 혼합이다.**  
  근거: `icmil_digest.md` §1·§2는 factorized prior와 joint prior가 서로 다른 inductive bias를 가지며, 최종 mixture가 Joint 0.70, Factorized(cont, MLP) 0.15, Factorized(disc, lookup) 0.15라고 기록한다. 그러나 같은 digest §5·§6은 ICMIL이 branch effective rank, branch redundancy, probabilistic ridge, 우리 7-branch rank 문제를 직접 다루지 않는다고 기록한다.

- **우리 내부 문서에는 브랜치 출력이 이미 중복된다는 관측이 있으나, 현재 verified 승격 근거는 아니다.**  
  근거: `PROJECT.md` §3.2는 공식 7-branch 유효 랭크 `3.52/7`을 기록하지만, 같은 절의 `D-050`은 이 값을 `unverified-hold`로 둔다. `closed_axes.md` CA-09는 `BM`·`QA`·`DS`의 유효 랭크 `1.29/3`과 위치 계열 상관을, CA-10은 형상 계열 병목이 절사가 아니라 정보량이라고 기록한다. 이 관측들은 “브랜치를 늘리면 유효 랭크가 자동으로 오른다”는 전제를 지지하지는 않지만, D-050 때문에 verified 승격 근거로는 쓸 수 없다.

- **D-048 이후 적합 분산 보고는 필수이며, 단일 적합의 per-fold 구간만으로는 불확실성을 과소평가한다.**  
  근거: `PROJECT.md` §1은 학습 파라미터 0개 제약 폐기 후 모든 비교가 fold 분산과 적합 분산을 함께 다루어야 한다고 명시한다. `closed_axes.md` CA-14는 `RU-97` §8에서 동일 구성에서 seed만 바꿔도 macro 폭 `0.0200`이 실측됐다고 기록한다.

- **pfn_imbalance의 thresholding 결과는 우리 AUROC 승격 지표와 직접 연결되지 않는다.**  
  근거: `pfn_imbalance_digest.md` §3·§4는 thresholding/downsampling의 이득을 balanced accuracy, worst-class accuracy, calibration 중심으로 보고한다. AUROC는 임계값 변경 자체에 불변인 순위 지표이므로, soft score만 같은 상태에서 threshold를 바꾸는 것은 AUROC 승격 기준을 넘게 만들 근거가 되지 않는다.

- **MMPFN의 orthogonality·token balancing 결과는 우리 WSI bag MIL의 effective rank나 적합 분산 문제를 직접 다루지 않는다.**  
  근거: `multimodal_pfn_digest.md` §5·§6은 MMPFN이 row-level multimodal tabular classification이고, WSI patch/bag MIL, ABMIL, effective rank, stochastic fit variance, promotion criterion을 본문에서 확인 못 했다고 기록한다.

## 반박 (증거가 반박함)
- **“ICMIL이 TCGA에서 ABMIL에 졌으므로 병리 WSI in-context MIL이 구조적으로 약하다”는 일반화는 증거 범위를 넘는다.**  
  근거: TCGA는 UNI2 1,536→PCA 25, LUAD vs LUSC 이진, 약 100-bag subsample이라는 특수 설정이고, ICMIL 표의 ABMIL은 validation·early stopping·hyperparameter grid를 쓴 task-specific baseline이다. `icmil_digest.md`에는 PCA fit scope, 100-bag의 train/test 대상, Table 1 ±의 산출 정의, PCA25/100-bag/ABMIL 학습 강도를 분리하는 ablation이 없다. 따라서 TCGA 단일 결과만으로 “병리 WSI 본질적 열세” 또는 “고차원 UNI2 한계”를 결론짓는 것은 반박된다.

- **ICMIL 12개 MIL 평균 우위나 TCGA 수치를 우리 SEAL 10 ABMIL 초과 근거로 쓰는 것은 반박된다.**  
  근거: `PROJECT.md` §1·§3.1과 digest 헤더는 외부 수치가 우리 게이트·승격 근거가 될 수 없고, SEAL 10은 최종 hold-out으로 선택·튜닝에 사용 금지라고 명시한다. TCGA와 우리 SEAL 10은 과제 집합·fold protocol·baseline 프로토콜이 다르므로 절대값 직접 비교는 금지된다.

- **ICMIL prior mixture 또는 inter-bag attention이 우리 closed-form branch ensemble의 effective rank를 직접 올린다는 전이는 반박된다.**  
  근거: `icmil_digest.md` §5·§6은 ICMIL이 branch effective rank, branch rank metric, probabilistic ridge, 우리 7-branch 중복 문제를 다루지 않는다고 기록한다. Table 2의 `inter-bag attention disabled 45.8 / enabled 83.3`은 learned Perceiver PFN의 성능 관측이지, closed-form linear Gram 또는 우리 ridge readout의 기전 입증이 아니다.

- **ICMIL TCGA의 ±0.1/±1.6을 우리 확률적 ridge 적합 분산 문제의 직접 해결 근거로 쓰는 것은 반박된다.**  
  근거: `icmil_digest.md`에는 Table 1의 ±가 seed, fold, resampling, training, early stopping 중 무엇에서 나온 것인지 명시되지 않는다. 같은 digest §5·§6은 ICMIL의 variance 결과가 MNIST-Pos/Neg와 12개 MIL benchmark에서 보고됐지만, probabilistic ridge나 branch rank와 직접 비교하지 않았다고 기록한다. 따라서 외부 ±를 우리 적합 분산과 동일 개념으로 취급하는 것은 근거가 없다.

- **pfn_imbalance의 thresholding만으로 우리 AUROC 승격 기준을 넘을 수 있다는 주장은 반박된다.**  
  근거: `pfn_imbalance_digest.md`의 thresholding 이득은 balanced accuracy/WCA/calibration 중심이고, 우리 승격 기준은 Primary 7 대응 per-fold Δ_macro AUROC이다. soft score 분포가 그대로라면 threshold 변경은 AUROC를 바꾸지 않는다. downsampling은 입력 context를 바꾸므로 별개 후보가 될 수 있으나, 이 논문은 effective rank나 WSI MIL AUROC 효과를 측정하지 않았다.

- **MMPFN의 orthogonality 수치나 MGM/CAP token balancing을 우리 branch effective rank 개선 근거로 쓰는 것은 반박된다.**  
  근거: `multimodal_pfn_digest.md` §4·§6은 orthogonality, accuracy, token count ablation을 보고하지만, effective rank, WSI bag MIL, ABMIL, stochastic fit variance, promotion criterion은 다루지 않는다. 다른 지표의 수치를 rank 병목 해결 근거로 쓰는 것은 증거 범위를 넘는다.

- **PFN foundations의 variance 정리를 우리 closed-form ridge에 자동 적용하는 것은 반박된다.**  
  근거: `pfn_foundations_digest.md` §5·§6은 variance 감소 주장이 고정 transformer PFN, tabular/simulation 조건에 한정되고, WSI patch/bag MIL, closed-form ridge, effective rank, fitting variance에 대한 직접 수치가 없다고 기록한다. 따라서 theorem을 우리 ridge 적합 분산 감소의 입명으로 쓰는 것은 기전 미확인을 기전 입증으로 바꾸는 것이다.

## 판별 불가 (증거가 가르지 못함)
- **ICMIL TCGA에서 ABMIL이 ICMIL을 앞선 구체적 원인은 판별 불가하다.**  
  PCA 25, 100-bag subsample, task-specific ABMIL tuning, LUAD/LUSC 분포, bag 정의, label rule, ± 산출 정의 중 어느 것이 격차를 만들었는지 분리하는 ablation이 문서에 없다.

- **ICMIL의 in-context MIL이 우리 Primary 7/SEAL 10 고차원 UNI2 환경에서 ABMIL을 넘을지는 판별 불가하다.**  
  TCGA는 PCA 25차원 평가이고, `icmil_digest.md` §5·§6은 high-dimensional foundation-model embeddings를 future work/미검증으로 둔다. 우리 fold protocol, bag 규모, context 규모, 과제 집합에 대한 직접 결과는 없다.

- **ICMIL prior mixture가 출력 공간의 유효 랭크를 실제로 올렸는지는 판별 불가하다.**  
  ICMIL은 effective rank, branch redundancy, output covariance를 측정·보고하지 않는다. 성능 평균 개선만으로는 rank 증가를 추론할 수 없다.

- **ICMIL의 fixed pretrained forward pass가 우리 확률적 closed-form ridge의 적합 분산을 줄일지는 판별 불가하다.**  
  ICMIL은 per-task fitting을 하지 않는 고정 모델이고, ABMIL은 task-specific 학습 모델이다. 그러나 그 ±의 산출 정의가 문서에 없고, 우리 seed/fit/fold 분산 구조와 동일 estimand인지 확인되지 않는다.

- **PSW(prediction-variance weighted readout)가 AUROC를 개선하거나 적합 분산을 줄일지는 판별 불가하다.**  
  branch score variance가 task signal을 반영하는지, seed/implementation noise인지, low-variance branch가 high-bias인지 Primary 7에서 재기 전에는 알 수 없다. ICMIL BMA 주장은 learned PFN aggregator fitting variance에 대한 외부 주장이지, closed-form 7-branch margin의 inverse-variance weighting으로 전이된다는 직접 증거가 아니다.

- **QJF/LSAK 같은 context-geometry branch가 기존 7-branch와 직교하고 effective rank를 올릴지는 판별 불가하다.**  
  Gram이 task-relevant context geometry를 읽는지, 염색·스캐너·batch·context size nuisance를 읽는지는 `D-050` 복구 후 label-free gate ①과 Primary 7 paired 비교 전에는 알 수 없다.

- **context downsampling이 effective rank를 줄이거나 AUROC를 개선할지는 판별 불가하다.**  
  `pfn_imbalance_digest.md`는 downsampling이 balanced accuracy/WCA와 계산 비용에 영향을 줄 수 있다고만 하고, effective rank, WSI MIL, ABMIL, AUROC 승격 기준을 측정하지 않는다.

- **spatial localization이 bias를 줄이면서 variance를 허용 범위 내로 유지할지는 판별 불가하다.**  
  `pfn_foundations_digest.md`는 localization이 bias를 줄일 수 있지만 sample size 감소로 variance가 약간 커질 수 있다고 기록한다. 우리 fold context에서의 trade-off는 Primary 7에서 재야 한다.

- **D-048 이후 학습 파라미터를 허용한다고 해서 ICMIL/MMPFN식 learned 구성이 우리 SEAL 10에서 ABMIL을 넘을지는 판별 불가하다.**  
  `closed_axes.md` CA-14는 학습 파라미터 축이 폐기됐지만, 과거 RU-38 시점의 관측은 소급 취소되지 않는다고 기록한다. 따라서 learned component를 활성 경로에 두려면 Primary 7에서 새 증거가 필요하다.

- **첨부 문서의 “적합 분산이 승격 기준 최대 2.8배” 수치는 문서 근거로 확인되지 않는다.**  
  `PROJECT.md`·`closed_axes.md`·네 digest에서 해당 수치를 확인하지 못했으므로, 승격 판단에 쓰려면 Primary 7에서 직접 재계산해야 한다.

## 실행 무효 (실행·계측 결함으로 판단 불가)
- **`D-050` 이후 현재 저장된 공식 7-branch 산출물로 gate/promotion 판단을 하는 것은 실행 무효다.**  
  근거: `PROJECT.md` §3.2·§3.4는 `src/`가 `ICF_*` 환경변수를 읽지 않아 실행 스크립트 export가 무효이고, `SH`/`SJ`를 켜는 문서화 수단이 동작하지 않으며, 저장 예측 파일에 branch list·가중치·코드 해시가 없어 특정 구성에 귀속할 수 없다고 기록한다. 따라서 `0.6227`, `+0.56%p`, 게이트 ①/② 실적을 `verified`로 인용하는 비교는 실행 무효다.

- **provenance writer와 branch control 복구 전의 QJF/PSW/PCA/subsample 탐색은 구성 귀속이 안 되어 gate/promotion 판단에 쓸 수 없다.**  
  근거: `PROJECT.md` §3.4는 재설계가 완료될 때까지 오염 검사 통과를 주장하는 실행은 무효라고 명시한다. `closed_axes.md` §3-I도 게이트 ① 실행 경로가 현재 무효라고 기록한다.

- **PCA fit에 query/test/SEAL feature를 넣거나, subsample에 test/query를 포함하거나, class label을 사용해 context를 고르는 실행은 실행 무효다.**  
  근거: `PROJECT.md` §3.1은 fold를 PathoBench 공식 분할 그대로 고정하고, SEAL 10은 최종 hold-out으로 선택·튜닝에 어떤 형태로도 사용 금지라고 명시한다.

- **100-bag subsample arm을 context slide 수 100 미만 fold에 사전 규칙 없이 적용하면 실행 무효다.**  
  근거: `closed_axes.md` CA-04·CA-06은 현행 context 슬라이드 수가 90~261장이라고 기록한다. 100 미만 fold에서 100 고정 subsample을 어떻게 처리할지 사전 고정하지 않으면 fold별 arm 정의가 달라져 paired 비교가 무효가 된다.

- **PSW에서 test 예측·test 라벨·fold AUROC를 variance 가중치 산출에 쓰거나, Primary 7 결과를 보고 ridge 상수·resample 수·branch 부분집합을 조정하면 실행 무효다.**  
  근거: `PROJECT.md` §4.1과 `closed_axes.md` CA-08은 탐색·후보 선별과 성능 확증을 분리하고, 선택에 쓴 평가자료의 최댓값을 확증 성과로 쓰는 것을 금지한다.

- **QJF/LSAK 후보를 rank 또는 AUROC를 보고事後 변경하면 실행 무효다.**  
  근거: `PROJECT.md` §5는 게이트 ①에서 탈락 시 성능을 조회하지 않고 종료하도록 하고, `closed_axes.md` CA-08·§3-J는 선택 시점과 자료 접근 이력을 요구한다.

## 좌석 공백 (기권·실패로 빠진 관점)
- 없음. 입력에 “기권한 좌석: 없음”으로 명시되어 있다.

## 다음 행동 (권한 등급과 함께)
- **[사용자 결정 필요] `D-050` 계측 복구 범위를 승인하라.**  
  복구 대상: branch list·가중치·코드 해시·데이터 버전·fold manifest 해시를 기록하는 provenance writer, `SH`/`SJ` 포함 branch control 경로, `SCREEN_ONLY` 오염 검사 재설계, 단위 테스트. 복구 전에는 gate ①·gate ②·승격 비교를 `verified`로 수행할 수 없다.

- **[사용자 결정 필요] 현재 공식 목표(RU-97 GF 단독 Fisher Vector 탐색)에서 벗어나 외부 문헌에서 파생한 후보를 공식 후보로 만들지 결정하라.**  
  근거: `PROJECT.md` §2는 현재 목표가 RU-97 GF 단독 탐색이고, 공식 7-branch 비교·채택 주장은 범위 밖이라고 명시한다. PSW·QJF·PCA/subsample 프로브를 공식 후보로 올리려면 새 사용자 결정과 사전 설계가 필요하다.

- **[연구자 자율, 문서화] Primary 7 탐색 계획을 사전 등록하라.**  
  최소 고정 항목:  
  1. PCA arm: patch feature PCA 25, fold별 context/train feature에서만 fit, query/test/SEAL 미포함, 적재 벡터 해시 기록.  
  2. subsample arm: context slide ≥100이면 무작위 100개, <100이면 전체 context, replacement 없음, class label 미사용, test/query 미포함.  
  3. PSW arm: variance 추정 대상, stochastic fit/context resample 수, inverse-variance ridge 상수, 가중치 정규화 규칙, label/context permuted 대조, seed-only noise 대조.  
  4. QJF/LSAK arm: Gram 입력이 fold context slide만인지, 전처리 fit 범위, QJF 우선 고정, LSAK는 position provenance가 label-free로 확인된 뒤 별도 후보.  
  5. 모든 arm의 자료 접근 이력과 선택 이력 기록.

- **[연구자 자율, `D-050` 복구 후] QJF/LSAK는 label-free gate ①을 먼저 수행하라.**  
  측정: 공식 7-branch 대비 max |r|, 8-branch effective rank, 같은 fold·같은 margin 스케일·같은 전처리·같은 fit 범위. 통과선: max |r| ≤ 0.6, effective rank ≥ 4.02. 실패 시 성능을 조회하지 않고 종료한다.

- **[연구자 자율, `D-050` 복구 후] PSW는 승격 문장을 paired Δ_macro `+0.3%p`로만 연결하라.**  
  측정: branch별 fold-wise score variance, variance와 paired per-fold Δ의 관계, 최종 score 적합 분산, 공식 7-branch 대비 paired per-fold Δ_macro, 과제별 Δ, sign agreement, 악화 과제 전량. 적합 분산 감소만으로는 승격이 되지 않는다.

- **[연구자 자율, `D-050` 복구 후] PCA/subsample 프로브의 결론을 “ICMIL TCGA 원인 설명”이 아니라 “우리 Primary 7 후보의 PCA25·context 축소 민감성 측정”으로 한정하라.**  
  측정: full, PCA25, context 100-bag, PCA25+100-bag arm의 paired per-fold Δ_macro, 과제별 Δ, sign agreement, task cluster CI/SE, 적합 분산, effective rank. ICMIL TCGA 원인은 여전히 판별 불가로 둔다.

- **[즉시 금지] SEAL 10을 개선 방향 선택에 사용하지 마라.**  
  근거: `PROJECT.md` §1·§3.1은 SEAL 10을 최종 hold-out으로 두고, 선택·튜닝에 어떤 형태로도 사용 금지라고 명시한다.

## 하지 말 것 (이유와 함께)
- **ICMIL/ABMIL TCGA 절대값(87.9, 90.7, 84.17, 79.97 등)을 우리 게이트·승격·SEAL 10 잔여 거리 근거로 쓰지 마라.**  
  이유: 외부 주장이고, 로컬 모델 요약이며 원문 대조가 되지 않았다. 과제·fold·baseline 프로토콜이 다르므로 절대값 직접 비교는 금지된다.

- **TCGA 단일 결과로 “병리 WSI in-context MIL이 구조적으로 약하다” 또는 “고차원 UNI2가 문제다”라고 결론짓지 마라.**  
  이유: TCGA는 PCA 25, 100-bag, LUAD/LUSC 이진, tuned ABMIL이라는 특수 설정이고, 병리 WSI로 지목된 과제는 1개뿐이다.

- **ICMIL Table 1의 ±0.1/±1.6을 우리 적합 분산과 동일하게 취급하지 마라.**  
  이유: ±의 산출 정의가 문서에 없고, ICMIL은 probabilistic ridge·branch rank·우리 fold protocol과 직접 비교하지 않았다.

- **ICMIL prior mixture 또는 inter-bag attention 수치를 closed-form linear Gram의 기전 입명으로 쓰지 마라.**  
  이유: learned Perceiver PFN의 성능 관측을 closed-form ridge의 기전으로 전이하는 것은 기전 미확인을 기전 입증으로 바꾸는 것이다.

- **브랜치를 늘리면 유효 랭크가 자동으로 오른다고 가정하지 마라.**  
  이유: 문서에 기록된 7-branch 유효 랭크 `3.52/7`은 D-050 이후 `unverified-hold`이고, CA-09·CA-10은 같은 계열 statistic의 중복과 정보량 병목을 기록한다. ICMIL은 effective rank를 측정하지 않았다.

- **thresholding만으로 AUROC 승격을 주장하지 마라.**  
  이유: AUROC는 임계값 변경에 불변이다. pfn_imbalance의 thresholding 이득은 balanced accuracy/WCA/calibration 중심이다.

- **MMPFN orthogonality 수치를 effective rank 개선으로, 또는 MMPFN accuracy를 WSI MIL 성능 전이로 쓰지 마라.**  
  이유: MMPFN은 row-level multimodal tabular classification이고, WSI bag MIL·ABMIL·effective rank·fit variance를 다루지 않는다.

- **PFN foundations의 variance theorem을 closed-form ridge에 자동 적용하지 마라.**  
  이유: theorem은 고정 transformer PFN과 tabular/simulation 조건에 한정된다. 우리 ridge·WSI·fold protocol에 대한 직접 증거가 없다.

- **`D-050` 복구 전 저장된 공식 7-branch 예측으로 승격·반박·판별 불가 판정을 내리지 마라.**  
  이유: branch list·가중치·코드 해시가 없어 특정 구성에 귀속할 수 없고, 공식 기준선은 `unverified-hold`다.

- **Primary 7에서 여러 arm을 탐색한 뒤 같은 Primary 7에서 살아남은 arm을 승격 후보로 자동 격상하지 마라.**  
  이유: §214식 선택 편향과 CA-08의 경계에 저촉된다. 탐색과 승격 확증은 사전 고정한 규칙과 선택 이력 기록으로 분리해야 한다.
