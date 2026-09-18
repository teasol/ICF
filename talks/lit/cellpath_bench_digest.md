# 문헌 브리프 — cellpath_bench (로컬 모델 요약)

```
url: https://arxiv.org/abs/2608.21060
fetched_from: https://arxiv.org/html/2608.21060
fetched_at: 2026-09-18 10:50:32 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8000` · 2026-09-18 10:54 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- 기존 PFM 평가는 patch/region/WSI downstream utility에 치우쳐, frozen PFM 자체의 cell-level representation capability를 직접 측정하지 못한다 (출처: Introduction).
- CellPath-Bench는 frozen PFM을 측정 대상으로 삼고, registered nuclear coordinates에서 WSI feature map을 샘플링한 뒤 unified multiclass linear probe로 평가한다 (출처: Abstract, Introduction, Benchmark Overview).
- 52개 Xenium 후보를 QC하여 25개 spatially aligned H&E–Xenium tissue sections, 11 organs, 7,079,283 cells 패널을 만들고 fine/coarse taxonomies로 조화했다 (출처: Abstract, Benchmark Data and Task Construction).
- CRA는 section 내부에서 nucleus-anchored representation이 patch-level mean pooling보다 cell-type 정보를 더 잘 linearly decodable하게 하는 우위를 측정한다 (출처: Abstract, Introduction, Results).
- CRT는 IOCD/MOCV/LOOO cross-domain protocols에서 cell-type decodability의 panel-relative transferability를 요약한다 (출처: Abstract, Introduction, Results).
- 30개 pathology-specific/general-purpose foundation models를 3 magnifications, 2 taxonomies, 7 representation modes, 4 protocols로 304,920회 linear-probe 실행했다 (출처: Abstract, Experimental scope, Table 3).
- IS에서 Nuc readout은 30개 모델 모두, 두 taxonomy에서 Mean보다 consistently 높았다 (출처: Results, Table 4).
- Positive CRA across all 30 models는 nucleus-anchored sampling이 patch-level spatial averaging 이후에는 덜 접근 가능한 cell-type 정보를 보존한다는 것을 시사한다 (출처: Introduction).
- within-section CRA와 cross-domain CRT는 fully aligned되지 않아 frozen PFM representation의 complementary properties를 보여준다 (출처: Introduction, Results).
- 20x는 40x에 비해 성능 유지가 크고, 10x는 더 크게 떨어진다; IS retention은 20x 99.6%, 10x 92.9% (출처: Results, Fig. 3).
- H-Optimus-1은 최고 CRA 10.14±6.05로 29개 경쟁 모델을 significantly outperformed (출처: Results, Table 4).
- SEAL(UNI2)와 UNI2가 최고 CRT 26.33, H-Optimus-1 25.33, H-Optimus-0 24.33; Virchow2는 CRA 2위지만 CRT 16.33 (출처: Results, Table 5).
- 벤치마크는 frozen linear probing과 discrete cell-type classification에 제한되고, CRT는 평가 패널과 statistical power에 의존한다 (출처: Discussion).

## 2. 방법의 핵심
- 구조: registered cellular reference, coordinate-aligned representation interface, protocol-specific linear probing의 3단계 워크플로우 (출처: Benchmark Overview, Fig. 2).
- 각 tissue section은 H&E image, registered nuclear coordinates, harmonized Xenium-derived cell labels를 제공한다 (출처: Benchmark Overview).
- Benchmark Overview에서는 fixed readout operators가 “Nuc, Mean, Cat representations”를 만든다고만 쓰고, WSI-to-Cell Representation Framework에서는 base readouts Nuc/Mean/Cls와 addition/concatenation fusions로 최대 7 modes를 정의한다 (출처: Benchmark Overview, WSI-to-Cell Representation Framework).
- WSI는 encoder native input size에 맞는 non-overlapping patches로 나누고, boundary patches는 padding한다 (출처: WSI-to-Cell Representation Framework).
- 각 cell은 registered nuclear center가 포함된 patch에 할당되며, center가 patch boundary에 정확히 있으면 nuclear area의 가장 큰 부분이 포함된 인접 patch를 선택한다 (출처: WSI-to-Cell Representation Framework).
- frozen encoder는 spatial token grid \(F \in \mathbb{R}^{H_m \times W_m \times D_m}\)와, architecture가 지원하면 class token \(c \in \mathbb{R}^{D_m}\)를 생성한다 (출처: WSI-to-Cell Representation Framework).
- Nuc readout은 registered nuclear center에서 spatial token grid를 bilinear sampling한다: \(z_{si}^{m,r,\mathrm{Nuc}} = \operatorname{Bilinear}(F, u)\) (출처: WSI-to-Cell Representation Framework, Eq. 2).
- Mean readout은 patch spatial tokens의 mean pooling이고, Cls readout은 native class token이다 (출처: WSI-to-Cell Representation Framework, Eq. 2).
- Nuc와 Mean/Cls는 normalization, rescaling, projection 없이 element-wise addition 또는 feature concatenation으로 결합한다 (출처: WSI-to-Cell Representation Framework).
- 최대 7 representation modes: Nuc, Mean, Cls, Nuc+Mean, Nuc+Cls, Nuc|Mean, Nuc|Cls (출처: WSI-to-Cell Representation Framework).
- 학습 방식: encoder는 전체 절차에서 frozen이고, 각 evaluation unit마다 새로운 multiclass linear probe를 training representations에 fitting하며, validation partition에서 model selection을 한 뒤 test partition에서 한 번 평가한다 (출처: WSI-to-Cell Representation Framework, Experimental scope).
- random seeds는 probe initialization 반복만이 아니라 data partitioning에 관여한다 (출처: Experimental scope).
- 모든 probes는 동일한 training procedure를 따르지만, optimization details와 computational configurations은 Supplementary Material로만 언급되어 본문에서 확인 못함 (출처: Experimental scope).
- 추론 방식: frozen encoder → coordinate-aligned readout → linear probe classification (출처: Benchmark Overview, WSI-to-Cell Representation Framework).
- 재현 설정: 3 magnifications(40x, 20x, 10x), 2 taxonomies(T_fine 9-class, T_coarse 3-class), 7 representation modes, 4 protocols(IS, IOCD, MOCV, LOOO) (출처: Experimental scope, Table 3).
- Table 3 run counts: IS 25 sections × 5 seeds = 157,500; IOCD 23 folds(9 organs) × 3 seeds = 86,940; MOCV 5 folds × 3 seeds = 18,900; LOOO 11 folds(11 organs) × 3 seeds = 41,580; total 304,920 (출처: Table 3).
- 데이터 구축: 14개 labeled scRNA-seq references(1 normal-tissue atlas + 13 cancer-specific), 68 molecular cell types, marker knowledge base(>300 publications), SpCAST로 Xenium sections에 label transfer, manual review (출처: Benchmark Data and Task Construction, Table 1).
- 68 molecular cell types는 deterministic mapping으로 9-class T_fine과 3-class T_coarse로 변환된다 (출처: Benchmark Data and Task Construction).
- QC: valid classes가 2개 미만인 section 제외; provider-supplied spatial alignment은 20 sampled H&E regions overlay로만 점검하고 image re-registration은 수행하지 않음 (출처: Benchmark Data and Task Construction).
- CellViT++는 H&E에서 nuclear morphology를 독립적으로 예측하고, Xenium-derived labels와 mutual nearest neighbors로 matching한 뒤 shared three-class space에서 section-level balanced accuracy를 계산한다; 이 점수는 section ranking에만 사용되고 label 생성/수정에는 사용되지 않는다 (출처: Benchmark Data and Task Construction).
- 25 highest-scoring sections이 primary panel S25이고, nested subsets S10⊂S15⊂S20⊂S25로 robustness를 평가한다 (출처: Benchmark Data and Task Construction, Appendix C).
- class eligibility: T_fine은 class당 최소 500 training cells와 200 test cells, T_coarse는 각각 100 cells; Macro-F1은 이 gate를 통과한 classes 평균 (출처: Appendix C).
- Table 6의 WSI-equivalent pretraining counts는 patch 또는 image–text pair counts를 1,000 patches/pairs per WSI로 환산한 값 (출처: Experimental Setup).

## 3. 평가 설정
- 데이터셋: 25개 spatially registered H&E–Xenium tissue sections, 11 organs, 7,079,283 spatially indexed cells (출처: Abstract, Benchmark Data and Task Construction).
- 후보 풀: 52개 Xenium slides; Table 7에서 rank 1–25가 CellPath-Bench test panel이고, rank 52는 coarse class가 1개로 collapse되어 ranking에서 제외됨 (출처: Appendix B, Table 7).
- nested panels: S25(25 sections, 11 organs), S20(20, 10), S15(15, 9), S10(10, 6) (출처: Appendix C).
- annotation references: 14개 scRNA-seq references; Table 1에 tissue/disease state별 reference cell counts와 section counts가 있음 (출처: Benchmark Data and Task Construction, Table 1).
- 비교 기준선: Table 2/6의 30개 foundation models — H-Optimus-1, H-Optimus-0, Prov-GigaPath, Virchow2, Virchow, UNI2, GPFM, Hibou-L, UNI, Phikon-v2, Kaiko ViT-L/14, Hibou-B, Kaiko ViT-B/8, Kaiko ViT-B/16, H0-mini, Phikon, CTransPath(CHIEF), Kaiko ViT-S/8, Kaiko ViT-S/16, Lunit ViT-S/8, MUSK, CONCH v1.5, mSTAR, CONCH, PLIP, SEAL(UNI2), OmiCLIP, SEAL(CONCH), DINOv3 ViT-L/16, DINOv3 ViT-B/16 (출처: Table 2, Appendix A Table 6).
- pretraining paradigms: pathology vision(PV), pathology vision–language(PVL), pathology vision–omics(PVO), general-purpose vision(GV) (출처: Experimental Setup, Table 6).
- ABMIL: 본문에서 확인 못함(없음).
- 평가 protocols: IS, IOCD, MOCV, LOOO (출처: Evaluation Protocols).
- IS: 단일 section 내 spatially separated regions; ROI 내 cells로 train/validation, ROI 밖 cells로 test (출처: Evaluation Protocols).
- IOCD: 동일 organ 내 한 section을 완전히 hold-out하고, 같은 organ의 나머지 sections으로 train/validation (출처: Evaluation Protocols).
- MOCV: organ-aware fixed cross-validation manifest에서 complete sections을 fold 단위로 hold-out (출처: Evaluation Protocols).
- LOOO: target organ의 모든 sections을 train/validation에서 완전히 제외하는 zero-shot cross-organ transfer (출처: Evaluation Protocols).
- primary metric: Macro-F1; secondary metric: Macro-AUROC (출처: Evaluation Protocols).
- Appendix C complete IS tables에는 Macro-F1, Macro-AUROC, Macro-AUPRC가 포함됨 (출처: Appendix C).
- aggregation: IS는 split–seed scores를 section 내 평균 후 동일 organ의 sections 간 평균; IOCD는 organ 내 seeds와 held-out sections 평균; MOCV는 test fold 내 seeds 평균; LOOO는 held-out organ 내 seeds 평균 (출처: Evaluation Protocols).
- statistical comparison: model pair별 two-sided paired Wilcoxon signed-rank test; p<0.05이고 median paired difference가 해당 모델을 favor하면 significant win (출처: Evaluation Protocols).
- CRA: 20x IS에서 matched Nuc−Mean Macro-F1 difference를 organ별로 평균한 뒤 organs 간 평균 (출처: Results, Eq. 4).
- CRT: Nuc readout, 20x, T_fine 기준 IOCD/MOCV/LOOO significant-win counts의 평균; 범위 0–29 (출처: Results, Eq. 5).

## 4. 정량 결과
- 실행 규모: IS 157,500, IOCD 86,940, MOCV 18,900, LOOO 41,580, total 304,920 (출처: Table 3).
- 패널 QC: Table 7에서 52 slides 중 rank 1 AGR-BAcc 0.704, rank 25 AGR-BAcc 0.433; rank 52는 coarse class collapse로 AGR-BAcc undefined (출처: Appendix B, Table 7).
- taxonomy rank concordance: fine vs coarse Spearman’s ρ = 0.994(IS), 0.956(IOCD), 0.949(MOCV), 0.933(LOOO) (출처: Results, Fig. 3).
- magnification retention: IS에서 40x 대비 20x 99.6%, 10x 92.9% (출처: Results, Fig. 3).
- Table 4, 20x IS, organ-balanced Macro-F1: H-Optimus-1 T_fine Nuc 36.73±8.51, Mean 26.95±6.52; T_coarse Nuc 56.88±8.96, Mean 47.81±5.78; CRA 10.14±6.05; Sig. wins 29 (출처: Table 4).
- Table 4: Virchow2 CRA 10.12±5.82, wins 25; H-Optimus-0 CRA 9.96±5.89, wins 25; UNI2 CRA 9.94±6.03, wins 25; SEAL(UNI2) CRA 9.89±6.03, wins 25 (출처: Table 4).
- Table 4: OmiCLIP T_fine Nuc 29.20±7.13, Mean 25.03±6.08; T_coarse Nuc 49.91±6.57, Mean 45.65±4.81; CRA 4.61±3.75; wins 0 (출처: Table 4).
- Table 5, 20x cross-domain, Nuc: UNI2 IOCD T_fine F1 28.85±8.28, AUROC 68.14±10.18; SEAL(UNI2) MOCV T_fine F1 29.14±5.30, AUROC 74.91±4.38; H-Optimus-0 LOOO T_fine F1 23.96±6.90, AUROC 69.28±7.97 (출처: Table 5).
- Table 5, coarse: H-Optimus-0 IOCD F1 49.57±9.67; H-Optimus-1 MOCV F1 60.68±7.67; H-Optimus-1 LOOO F1 54.59±10.32 (출처: Table 5).
- Table 5: Virchow2 IOCD T_fine F1 26.90±8.14, MOCV 27.28±5.40, LOOO 22.34±7.41 (출처: Table 5).
- CRT: SEAL(UNI2) 26.33, UNI2 26.33, H-Optimus-1 25.33, H-Optimus-0 24.33, Virchow2 16.33 (출처: Results, Table 5).
- Appendix C, Table 9, S25 40x T_fine Macro-F1: UNI2 Nuc 36.94±12.68, Mean 29.29±8.25, CRA 7.90±5.72, wins 27; H-Optimus-1 Nuc 36.74±12.66, Mean 29.25±8.02, CRA 7.78±6.00, wins 27 (출처: Appendix C, Table 9).
- Appendix C, Table 15, S25 20x T_fine Macro-F1: H-Optimus-1 Nuc 36.73±12.19, Mean 26.95±7.63, CRA 10.10±6.43, wins 28; UNI2 Nuc 36.41±12.08, Mean 26.60±7.32, CRA 9.95±6.32, wins 26 (출처: Appendix C, Table 15).
- Appendix C, Table 16, S25 20x T_coarse Macro-F1: H-Optimus-1 Nuc 56.88±13.19, Mean 47.81±10.11, CRA 10.18±5.63, wins 29; Virchow2 Nuc 55.90±13.14, Mean 45.97±9.86, CRA 10.27±5.47, wins 24 (출처: Appendix C, Table 16).
- Appendix C, Table 21, S25 10x T_fine Macro-F1: CONCH v1.5 Nuc 33.63±10.87, Mean 23.42±6.78, CRA 10.41±6.48, wins 28; Virchow2 Nuc 33.71±10.06, Mean 24.04±6.72, CRA 10.01±6.09, wins 27 (출처: Appendix C, Table 21).
- Appendix C, Table 22, S25 10x T_coarse Macro-F1: CONCH v1.5 Nuc 54.16±12.90, Mean 42.97±9.25, CRA 11.67±7.23, wins 27; CONCH Nuc 53.72±12.60, Mean 42.95±8.85, CRA 11.37±6.71, wins 26 (출처: Appendix C, Table 22).
- Table 4와 Appendix C의 동일 모델 수치는 집계 단위가 다를 수 있다: Table 4 caption은 organ-balanced Macro-F1(mean±SD across 11 organs)이라 하고, Appendix C Aggregation은 section-level sample standard deviation이라 한다 (출처: Table 4 caption, Appendix C Aggregation).

## 5. 우리 프로젝트와의 관련 (가설)
- 우리가 옮겨올 수 있는 것(가설): registered nuclear coordinates에서 bilinear sampling하는 Nuc readout을 우리 WSI in-context 분류기에 적용하면, patch mean pooling보다 cell-level signal을 보존하여 branch 간 중복을 줄일 수 있을지 검증해 볼 수 있다. 근거는 저자들이 Nuc가 Mean보다 consistently 높고 positive CRA라고 주장한 것(출처: Results, Table 4)이지만, 이는 frozen linear probe의 cell-type decodability이지 WSI MIL 성능을 보장하는 증거는 아니다.
- 우리가 옮겨올 수 있는 것(가설): IS/IOCD/MOCV/LOOO와 organ/fold-balanced aggregation, paired Wilcoxon significant wins를 우리 fold context 평가에 차용하면, ABMIL 대비 승격 기준의 안정성을 높일 수 있을지 가설로 테스트할 수 있다. 근거는 Evaluation Protocols의 leakage control과 unit-balanced aggregation(출처: Evaluation Protocols)이다.
- 우리가 옮겨올 수 있는 것(가설): CRA처럼 matched evaluation units에서 baseline 대비 paired difference를 먼저 계산한 뒤 organ/fold로 평균하는 방식이, 확률적 적합 분산을 승격 기준으로 평가할 때 유사한 paired metric으로 쓰일 수 있을지 가설이다. 단, CRA는 nucleus-anchored vs patch mean difference이지 ridge fit variance가 아니다(출처: Results, Eq. 4).
- 우리가 옮겨올 수 있는 것(가설): Nuc+Mean/Nuc|Mean fusion이 consistent improvement를 주지 않았으므로, 우리 branch 설계에 patch context를 추가하는 것이 linear decodability를 반드시 높이지는 않을 수 있다는 가설을 세울 수 있다. 근거는 Results의 “combining Nuc with Mean or Cls provided no consistent improvement”(출처: Results).
- 우리가 옮겨올 수 있는 것(가설): T_fine/T_coarse class eligibility thresholds(500/200, 100/100)를 우리 evaluation에 적용하면 rare class로 인한 evaluation variance를 줄일 수 있을지 가설이다. 근거는 Appendix C의 class gate(출처: Appendix C).
- 우리가 옮기기 어려운 것: 이 논문은 ABMIL 또는 학습 기반 MIL 기준선을 포함하지 않으므로, “ABMIL을 넘는 WSI 벤치마크” 문제에 직접 답하지 못한다. ABMIL은 본문에서 확인 못함(없음)(출처: Table 2, Table 6, 전체 본문).
- 우리가 옮기기 어려운 것: 이 논문은 frozen encoder + linear probe를 전제로 하므로, 우리처럼 학습 파라미터를 허용하는 in-context classifier의 최적화, 정규화, 승격, stochastic fit variance에 대한 직접 증거를 제공하지 않는다(출처: WSI-to-Cell Representation Framework, Experimental scope).
- 우리가 옮기기 어려운 것: branch effective rank, 7개 branch 중복, closed-form ridge, 확률적 적합 분산이 승격 기준의 최대 2.8배로 커지는 문제에 대한 분석은 본문에서 확인 못함.
- 우리가 옮기기 어려운 것: Nuc readout은 registered nuclear coordinates가 전제되므로, 우리 WSI benchmark에 해당 좌표 또는 동등한 cell anchor가 없으면 직접 이전이 어렵다(출처: WSI-to-Cell Representation Framework).
- 우리가 옮기기 어려운 것: CRA/CRT는 30개 모델 패널, valid evaluation units, statistical power에 의존하는 panel-relative metric이므로, 절대 성능이나 ABMIL 대비 이득을 직접 보장하지 않는다(출처: Results, Eq. 5, Discussion).

## 6. 이 논문이 답하지 않는 것
- ABMIL 또는 다른 학습 기반 MIL 기준선과 CellPath-Bench 모델들의 비교: 본문에서 확인 못함.
- 우리 closed-form ridge in-context classifier의 성능, 적합 분산, 승격 기준: 본문에서 확인 못함.
- 7개 branch의 유효 랭크 3.52/7 또는 branch 중복 감소 방법: 본문에서 확인 못함.
- 확률적 적합을 썼을 때 적합 분산이 승격 기준의 최대 2.8배로 커지는 문제의 완화 방법: 본문에서 확인 못함.
- cell-level linear probe 성능이 WSI-level classification 성능으로 어떻게 전이되는지: 본문에서 확인 못함.
- linear probe의 optimization details, hyperparameters, regularization, solver, computational configurations: 본문에서는 Supplementary Material로만 언급되어 본문에서 확인 못함(출처: Experimental scope).
- nuclear coordinates가 없는 일반 WSI 벤치마크에서 Nuc readout을 어떤 대체 readout으로 사용할지: 본문에서 확인 못함.
- CRA/CRT가 downstream WSI task performance 또는 ABMIL 대비 이득과 어떤 상관이 있는지: 본문에서 확인 못함.
