# Current development status & multi-location sync SSOT

**Last updated**: `2026-08-12` (§102 — configs 루트 67 → 2개, v78 기각)

**한 줄**: 활성 baseline은 v77 Hard orthogonal(SEAL macro **0.6873**)이고, 앞으로 arm 판정은 점추정 macro 차이가 아니라 **fold-paired Δ + bootstrap CI**로 한다(§99).

**Status**: **활성 baseline v77 Hard orthogonal 0.6873. 실행 중인 학습·평가 없음 — v78은 Δ −0.0004 [−0.0021,+0.0013]로 기각(§102-5). 역사적 전체 최고는 v41_K128 0.6940.**

> [!IMPORTANT]
> **읽는 순서 (2026-08-12)**: 판정 방식이 §99에서 바뀌었다. arm을 비교하려면 §99를 먼저 읽고
> `scripts/compare_arms_paired.py`를 쓸 것. §98 판정표 4건은 §99-1에서 fold-paired CI로
> 재검증되어 전부 유지됐다. **v78**(§100)은 Δ −0.0004로 기각됐다(§102-5).
> §2~§97 본문은 [`history.md`](history.md) §20–§23으로 아카이빙됐다(§101).

* **계보 A = CV-only** (`src/models/baseline.py`, 학습 파라미터 **229개**).
  현행 최고 **v41_K128 = SEAL 10개 0.6940** (ABMIL 0.727에 −0.033).
  §73에서 죽은 5개 분기를 소스에서 삭제해 `baseline.py`가 5,685 → **2,224줄**이 됐다.
  ⚠️ **prune 이전 ckpt는 현재 트리로 로드 불가** — `8caa96c` 고정 worktree
  (`/NHNHOME/BASE/kimds/ICF_pre_prune`)를 쓸 것.
* **계보 B = Encoder+Ridge** (`src/models/set_transformer_ridge.py`, **5.01M개**).
  §69가 확인한 "label-free 사영은 전부 0.68 천장"을 우회하는 유일한 축 —
  **라벨을 보는(학습되는) 사영**. 첫 판본(v50~v52)은 내 설계 오류로 기술자가 256차원에
  묶였다(0.6047/0.6619). 재설계(v53/v54)로 세포 간 attention과 16,384차원 기술자를 얻어
  합성 val_auroc가 0.784 → **0.849**로 올랐으나 **SEAL은 0.6619 → 0.6526으로 내려갔다**
  (§79-6). **현재 형태로는 기각.** 문제는 용량이 아니라 일반화다.
* **CV-2는 더 파지 말 것** — margin activation(−0.017), subspace_rank(±0.001),
  head 구조(−0.0003) 셋 다 10개 평균을 못 움직였다. 병목이 아니다.
* **판정은 SEAL 10개 macro 평균만** (§71-4). 합성 val_ce·val_AUROC는 신뢰하지 않는다.
* **GPU 정책**: ICF는 GPU 0–3만 사용한다. GPU 4–7은 사용하지 않는다.

현행 아키텍처 명세는 [`current_architecture.md`](current_architecture.md),
실험 절차·결과표·금지사항은 [`current_experiments.md`](current_experiments.md).

**지금 돌아가는 것 (2026-08-12)**: 없음. v78까지 완료·기각됐다(§102-5).

결과 재확인:
```bash
for tag in v53_enc v54_enc; do
  printf "%-10s " $tag
  grep -hoP 'fold-mean AUROC: \K[0-9.]+' logs/official50/*_${tag}.log \
    | awk '{s+=$1;k++} END{printf "%.4f (%d개)\n", s/k, k}'
done
```

---

> **사용자 결정 (2026-08-12, 최신)**:
> 1. **Hard v76을 canonical v77 baseline으로 승격.** v30 S2 결정은 역사적 기록이다.
> 2. **ICI는 기본 잠금 유지.** §50과 §86은 사용자 명시 해제에 따른 예외 평가.
> 3. **Musk 목표는 0.95 유지.**

**Read first if you are picking this up**: **§98 (v77 명명·baseline 승격)**, §97 (large-ragged), §96 (아키텍처 SSOT), §91 (Hard 선택), §89 (v76 구조/학습 경계),
§88 (v74 baseline/CT/v71–v74 판정),
§87 (DD/v70/synthetic 일반화),
§86 (canonical CV+mean 계약), §85 (v62–v66), §71 (SEAL 판정), §73–§74
(호환성/학습 경로), §79 (generic 평가/YAML).

> [!IMPORTANT]
> **방법론 경고 3건 — 다음 arm 설계 전에 읽을 것**:
> 0. **합성 val 지표(val_ce·val_AUROC)로 arm을 고르지 말 것 (§69-6).** CV-only의 합성 val AUROC는
>    ep0 0.8885 = ep49 0.8882로 평평한데 er_status는 +0.037 오른다. **판정은 er_status 50-fold로만**
>    (캐싱으로 45초). 단일 측정의 요동이 ±0.05이므로 seed 반복도 필수다.
> 1. **val_ce로 arm을 고르지 말 것.** v37 쌍은 val_ce가 확실히 좋았으나(0.3354 vs 0.3402) 50-fold는
>    **−0.0068로 나빴다**(CI가 0 제외). 200 epoch은 합성 생성기에 과적합한다.
> 2. **학습 길이가 다른 arm 간 비교는 그 자체로 교란이다** (§42-43 arm C 교훈의 재확인).
>    control은 항상 같은 epoch 수로 새로 학습할 것.

**열린 과제 (CV-only 노선, 우선순위 순)**:
① **`subspace_rank` 2·4 판정** — 진행 중, SEAL 10개 채점 자동 대기.
② **learnable 사영** — label-free 축 8개가 전부 0.68±0.03 천장이므로 **라벨이 남은 유일한
   정보원**이다. P는 1536×K(98K~197K)로 이 모델에서 가장 큰 잠재 파라미터인데 완전히 고정돼
   있다. ⚠️ CV-1이 closed-form이라 gradient가 ridge solve를 통과해야 하므로 **CV-2 쪽부터**
   붙이는 것이 안전하다(§66 ridge 제거 시 gradient 발산 전력).
③ **v40_cv_only / v38_control의 SEAL 10개 채점** — §70의 "대역폭+CV-2 = +0.0271"이 er_status
   기준이라 10개 기준의 실제 크기를 모른다. 각 20분.
④ **K=256** — 차원 유효가 §71-5로 확인됐으므로 재검토 가치(VRAM 22%로 여유). ridge-only
   진단상 K128→256은 +0.003이라 기대는 낮다.
⑤ **seed 반복** — 지금까지 arm당 1 seed. 요동이 ±0.02~0.05다.
⑥ **task별 편차 원인 규명** — 같은 TP53이 brca +0.018 / luad −0.066. ccrcc VHL은 0.4503으로
   랜덤 이하. 코호트 크기(112 vs 324)나 조직 특성으로 추정되나 미규명.
⑦ CV-2의 거리 평균 연산(`.square().mean(dim=-1)`) — rank를 올려도 MLP 입력이 스칼라 4개로
   고정되는 병목. ①이 무변화로 나오면 여기가 다음 손잡이다.

**해결·폐기**: 6-분기 아키텍처 전체(§68), v36 Q1·v37(§65), ridge ablation 계열(§66·§67),
G-2 제거 확정(§68에서 분기 통째 제거로 해소), E>1 노선(§68-5), label-free 사영 축 8개(§69).
상세 기록은 [`history.md`](history.md).

**Branches**: `main` = v30 확정 baseline + 미채택 v31 CCER-v2 재현 코드. 참고용 branch/tag 구조는
[`history.md`](history.md).
**Project**: ICF (BagPFN Single-Cell In-Context Meta-Classifier)
**Architecture Versions**: `30` 확정 baseline; `31` CCER-v2와 `32` DR-CCER 미채택(재현용 보존);
`33` MR-BagPFN은 proposal-only. 코드 기본 `bag_representation`은 `legacy` 유지.
**Purpose**: 연구실 / 집 / 노트북 간 상태를 동기화하는 SSOT living document.

---

## 0. 30초 요약 — 새 세션은 여기부터

**활성 baseline: v77 Hard orthogonal, 공식 SEAL 10-task macro 0.6873** (§98).
`CovarianceMeanLearnablePDDCTMLPModel`, 학습 파라미터 197,057개(P 196,608 + head 449).
역사적 전체 최고는 여전히 **v41_K128 CV-only 0.6940**(229 파라미터)이므로, 활성 baseline이
사상 최고보다 낮은 상태다. 지도학습 ABMIL은 0.7266(−0.0393), 상회는 2/10 task.

**지금 돌아가는 것**: 없음. v78은 기각됐다(§102-5).

**판정 방법이 §99에서 바뀌었다 — 이걸 모르면 arm 비교를 잘못한다.**
점추정 macro끼리 빼지 말고 **fold-paired Δ + bootstrap CI**를 쓸 것. GPU 불필요:

```bash
python scripts/compare_arms_paired.py --baseline <TAG> --arm <TAG>
```

**세 줄 아키텍처**: bag의 cell을 learnable P(1536×128)에 사영해 covariance를 만들고, CV(ridge)·
DD(dispersion)·CT(abundance) 세 branch가 에피소드마다 **closed-form으로** 12개 relation feature를
만들어 12→32→1 MLP가 읽는다. 분류기 weight를 저장하지 않고 ridge를 매 에피소드 다시 푼다.
현행 스펙은 [`current_architecture.md`](current_architecture.md).

**열린 과제**
1. **학습 seed 노이즈가 미측정** — 모든 arm이 seed 1회다. fold 노이즈는 §99가 처리했지만
   realization 노이즈는 pairing으로 줄일 수 없다. latent sweep의 비단조성(L16 딥)이 실효과인지
   seed 요동인지 여기서 갈린다 (§99-3).
2. **cptac_ccrcc VHL 0.4385 — 랜덤 이하**. large-ragged가 +0.0090(CI 0 제외)로 올려도 여전히
   0.45 미만. 노이즈가 아니라 체계적 부호 문제로 의심된다.
3. **cptac_ccrcc BAP1이 large-bag에서만 −0.0179로 무너진다** (§99-2).
4. ICI는 사용자 지시로 잠금.

**작업 규칙 4가지**
- 판정은 **SEAL 10개 macro**, 그것도 **fold-paired Δ + CI**로 (§99). 합성 val 지표는 checkpoint
  선택에만 쓴다 — 이 리포에서 합성이 좋아지고 SEAL이 내려간 사례가 반복됐다(v54가 최악의 예).
- **clipping 금지**, **bf16-mixed 필수**(학습·평가 양쪽), GPU는 **0–3만** 사용.
- 장시간 작업은 완전 이탈형 백그라운드로 띄우고 PID/PGID·로그 경로를 즉시 기록한다.
  프로세스는 **프로세스 그룹**으로 죽인다(wrapper PID만 kill하면 GPU가 안 풀린다).
- 다음 Action과 판정 기준이 명확하면 재확인 없이 실행하고, 각 논리 단위마다 결과·명령·산출물
  경로·판단·다음 Action을 이 문서에 갱신한 뒤 커밋한다.

⚠️ **다시 열지 말 것**: [`history.md`](history.md) §19가 닫힌 결론 10건을 모아둔다
(retrieval, 세포 선택, CCER 계열, Q-5 상수 분기, CV-2 손잡이, label-free 사영 등).

## 1. 멀티 작업공간 (연구실/집/노트북) 바톤 터치 지침

> [!IMPORTANT]
> **새 대화 세션 시작 시 Agent 초기화 원칙**:
> 1. 사용자는 매번 **새 대화 세션(New Chat Session)**으로 접속합니다.
> 2. 새로 접속한 Agent는 **`docs/` 최상위 루트의 Living md 파일 5개(`agent_handoff.md`, `current_status.md`, `current_architecture.md`, `current_experiments.md`, `README.md`)와 현행 `architecture_*_proposal.md` 1개를 최우선으로 정독**하여 전체 개발 맥락과 프로젝트 규칙을 파악합니다.
> 3. 터미널 조회가 필요한 명령어는 NVML/쉘 hang 방지를 위해 **반드시 `timeout 3s ps aux | grep python`과 같이 타임아웃**을 적용합니다.
> 4. 코드 변경 시 unittest 통과 필수:
>    `timeout 300s /home/aibio_3/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"`

---

## 2. 프로젝트 핵심 아키텍처 및 환경 명세 (Architecture **v24 확정**) — **아카이브됨**

v24 확정 시점의 아키텍처·환경 명세. 현행 스펙은 [`current_architecture.md`](current_architecture.md). 전문은 [`history.md`](history.md) §6.

## 3. 실험 현황 — **아카이브됨**

v22~v24 시대 실험 현황표(212줄). 결론은 bag-collapse family 진단으로 수렴. 전문은 [`history.md`](history.md) §6.

## 4. v22 결정: retrieval 완전 제거 (2026-07-29) — **아카이브됨**

retrieval 완전 제거 결정. ICI 규모에서 이득 없음 — 다시 열지 말 것. 전문은 [`history.md`](history.md) §5.

## 5. 실험 전략 (2026-07-29 확정) — **아카이브됨**

v22 시점 실험 전략. 전문은 [`history.md`](history.md) §1.

## 6. 다음 작업 세션 Action Plan — 구조적 변경 및 실험 목록 (아카이브됨)

구조 변경·실험 목록(당시 Action Plan). 이미 소진. 전문은 [`history.md`](history.md) §6.

## 7. 평가 프로토콜 보강 (2026-07-29) — **아카이브됨**

평가 프로토콜 보강. 검증된 불변식은 history §2에 통합. 전문은 [`history.md`](history.md) §2.

## 8. Source of Truth 파일 — **아카이브됨**

Source of Truth 파일 목록. 현행은 이 문서 상단과 [`README.md`](README.md). 전문은 [`history.md`](history.md) §0.

## 9. 2026-07-31 세션 핸드오프 — v23/v24 bag collapse family — **아카이브됨**

v23/v24 bag collapse family 세션 핸드오프. 전문은 [`history.md`](history.md) §6.

## 10. 2026-08-01 세션 핸드오프 — v24 확정, 평가 계획 폐기 — **아카이브됨**

v24 확정, 당시 평가 계획 폐기. 전문은 [`history.md`](history.md) §6.

## 11. 2026-08-02 세션 핸드오프 — v25 Medium 평가 완료(사실상 동률), Easy tier 실험 진행 중 — **아카이브됨**

v25 Medium 사실상 동률, Easy tier 진행. 전문은 [`history.md`](history.md) §6.

## 12-14. 2026-08-02 세션 — 폴더/문서·config/src·scripts·tests 정리 3단계

> 아카이브됨 (2026-08-02, 핸드오프 정리): checkpoint/log/prediction purge(53GB→3.3GB),
> 구버전 문서·config·스크립트 삭제, src/scripts/tests 참조 무결성 점검 기록. 전문:
> [`history.md`](history.md) §12-14.
>
> **하나만 아직 열려 있음**: §13의 config 삭제가 `test_d_stages_differ_only_in_selected_nuisance`를
> 깨뜨림 (`configs/trainer/learnability_d20.yaml` 삭제, §16에서 발견·미조치) — 상세는
> archive.md §13 경고 참고.

---

## 15. 2026-08-02 세션 마무리 — 정리 3단계 + v25 폐기 확정 + 브랜치 정리 — **아카이브됨**

정리 3단계 + v25 폐기 확정 + 브랜치 정리. 전문은 [`history.md`](history.md) §18.

## 16. 2026-08-02 세션 (이어짐) — v26/v27/v29 설계안 검토, 학습 없는 게이트 3종, CLS-token pooling(v26) 구현·학습 시작, 제안서 archive
## 17. 2026-08-03 세션 — v26 학습 완료 + CLS attention 진단 프로브 (24-CLS 제안 사전검정) — **아카이브됨**

v26 학습 완료 + CLS attention 진단 프로브. 전문은 [`history.md`](history.md) §7.

## 18. 2026-08-03 — E7 재검정: 지도 component-selection 상한 재확인 (Path B 관문) — **아카이브됨**

E7 재검정 — 지도 component-selection 상한 재확인(Path B 관문). 전문은 [`history.md`](history.md) §7.

## 19. 2026-08-03 — 정규화 천장 프로브: 고정 정규화가 천장을 제한하는가 (사용자 가설 검증) — **아카이브됨**

고정 정규화가 천장을 제한하는가 — 정규화 천장 프로브. 전문은 [`history.md`](history.md) §2.

## 20. 2026-08-03 — v24 no-L2 ablation: per-cell L2 정규화 제거 학습 (진행 중) — **아카이브됨**

v24 no-L2 ablation(per-cell L2 제거). 전문은 [`history.md`](history.md) §6.

## 21. 2026-08-03 — Zero-shot Musk (Musk2) MIL 벤치마크 테스트 — **아카이브됨**

Zero-shot Musk(Musk2) MIL 벤치마크. 전문은 [`history.md`](history.md) §8.

## 22. 2026-08-03 — 전략 전환: 생성기 개선 (Musk-like easy 데이터) — 가설 판정 완료 — **아카이브됨**

전략 전환 — Musk-like easy 생성기 개선, 가설 판정 완료. 전문은 [`history.md`](history.md) §8.

## 23. 2026-08-03 — Musk 0.95 로드맵: raw bag-stat token (mean/skew/kurt) 학습 중 — **아카이브됨**

Musk 0.95 로드맵 — raw bag-stat token(mean/skew/kurt). 전문은 [`history.md`](history.md) §8.

## 24. 2026-08-03 — Phase 1 IA-MIL (Instance-Attention MIL) — 판정: 음성 — **아카이브됨**

Phase 1 IA-MIL 판정 **음성**. 세포 선택은 bag 라벨로 학습 불가(네 번 닫힌 경로). 전문은 [`history.md`](history.md) §19.

## 25. 2026-08-04 — IA-MIL 폐기 + 문서/파일 정리 + 핸드오프 — **아카이브됨**

IA-MIL 폐기 + 문서/파일 정리. 전문은 [`history.md`](history.md) §19.

## 26. 2026-08-04 — Musk 전이 재진단: P1/P2 기각 + v30(CFMT) 제안 — **아카이브됨**

Musk 전이 재진단 — P1/P2 기각 + v30(CFMT) 제안. 전문은 [`history.md`](history.md) §8.

## 27. 2026-08-04 — 세션 마무리: 사용자 결정 반영, 문서 압축, config 재현성 복구 — **아카이브됨**

세션 마무리 — 문서 압축, config 재현성 복구. 전문은 [`history.md`](history.md) §9.

## 28. 2026-08-04 — v30 S1/S2 판정 과정 (B1 `poolz_l2`·B2 cardinality) — **아카이브됨, §29로 승격 완료**

v30 B1/B2 실험·판정 전체 기록(S1 `poolz`/`poolz_l2` 음성, S2 B2+B1 양성, paired bootstrap,
B1·B2 상호 필수 근거, 교차 분포 합성 무회귀)은 [`history.md`](history.md) §28로
이관되었습니다. 최종 결론 요약은 헤더 Status와 §29 참고.


---

## 29. 2026-08-04 — **v30 확정 baseline: B1 `poolz_l2` + B2 cardinality-faithful (사용자 승격 결정)** — **아카이브됨**

**v30 확정 baseline** = B1 `poolz_l2` + B2 cardinality-faithful. 두 손잡이는 상호 필수. 전문은 [`history.md`](history.md) §9.

## 30. 2026-08-04 — v31 CCTS 구현·학습 (아카이브됨)

CCTS 구현과 50-epoch 학습 기록은 후속 CCER-v2로 완전히 대체되어
[`history.md`](history.md)로 이동했다.

---

## 31. 2026-08-05 — v31 CCTS Musk 평가·진단 (아카이브됨)

CCTS Musk `0.8376`, 대형 bag `0.6032` 결과와 구현 결함 재분류 기록은
[`history.md`](history.md)로 이동했다.

---

## 32. 2026-08-05 — v31 CCER-Lite 구현·학습 (아카이브됨)

CCER-Lite 구현과 학습 기록은 contribution이 `~1.4e-4`로 사실상 비활성임을 확인한 뒤
[`history.md`](history.md)로 이동했다.

---

## 33. 2026-08-05 — v31 CCER-v2 아키텍처 구현 완료 (학습 미시작) (아카이브됨)

CCER-v2 아키텍처 구현·검증 기록. §38에서 CCER 계열 폐기 판정으로 대체. 본문은 [`history.md`](history.md)로 이동했다.

---

## 34. 2026-08-05 — v31 CCER-v2 20-epoch 학습 시작 (아카이브됨)

CCER-v2 20-epoch 학습 시작 기록. §38에서 폐기 판정. 본문은 [`history.md`](history.md)로 이동했다.

---

## 35. 2026-08-05 — CCER-v2 구현·검증·20 epoch 학습 완료 (아카이브됨)

CCER-v2 구현·20 epoch 학습 완료 기록. §38에서 폐기 판정. 본문은 [`history.md`](history.md)로 이동했다.

---

## 36. 2026-08-05 — v31 CCER-v2 Epoch 18 합성/Musk 평가 완료 (v30 Baseline 유지) (아카이브됨)

CCER-v2 epoch 18 합성/Musk 평가(v30 미달) 기록. §38에서 폐기 판정. 본문은 [`history.md`](history.md)로 이동했다.

---

## 37. 2026-08-05 — CCER-v2 결과 기반 v32 DR-CCER proposal 작성 (아카이브됨)

v32 DR-CCER proposal 작성 기록. §38에서 폐기 판정. 본문은 [`history.md`](history.md)로 이동했다.

---

## 38. 2026-08-05 — v32b DR-CCER: 비판적 검토 반영 개선안 + 구현 + Stage A 학습 시작 — **아카이브됨**

v32b DR-CCER — **CCER 계열 실증적 폐기**. branch 활성 ≠ 상보 정보. 전문은 [`history.md`](history.md) §10.

## 39. 2026-08-05 — v32b 완료 결과 평가 + v33 MR-BagPFN proposal — **아카이브됨**

v32b 결과 평가 + v33 MR-BagPFN proposal. 전문은 [`history.md`](history.md) §11.

## 40. 2026-08-05 — 기본 unittest suite compact화 — **아카이브됨**

기본 unittest suite compact화. 전문은 [`history.md`](history.md) §18.

## 41. 2026-08-05 — v33 Phase 0 구현: arm B(C) 데이터 컨트롤 + 학습 런치 — **아카이브됨**

B2b(per-bag cardinality) 구현·arm B/C config 런칭, 8× 에피소드 비대칭 주의 기록. 전문은
[`history.md`](history.md) §41.

## 42. 2026-08-06 — v33 Phase 0 arm B/C 학습 완료 + gate 평가 — **아카이브됨**

arm B(스파스 gate 미달 0.6747)·arm C(legacy 회귀 +0.0373) 50ep 완료 — **Phase 0 두
gate 모두 미달**. 전문은
[`history.md`](history.md) §42.

## 43. 2026-08-06 — arm C top-up: 8×A6000 DDP 전환 + NCCL P2P hang 수정 + 속도 기록 — **아카이브됨**

arm C top-up을 8×A6000 DDP로 재개. **NCCL P2P hang 진단/수정(`NCCL_P2P_DISABLE=1`,
런처 기본 적용)** + B200 vs A6000 8장 속도 비교(~4.3× 노드 총 처리량). 전문은
[`history.md`](history.md) §43.

---

## 44. 2026-08-06 — 패딩 배칭 (B2b `episode_batch_size>1`) + 병목 프로파일 — **아카이브됨**

ragged B2b 에피소드의 패딩 배칭 구현·검증(commit `568c5f8`, batch2에서 ~16 ep/s).
전문은
[`history.md`](history.md) §44.

## 45. 2026-08-06 — arm C top-up 중간 Musk zero-shot: 대형 bag 개선 + 소형 trade-off — **아카이브됨**

arm C 중간(ep64) Musk: **n>34 0.698→0.825 개선**, 소형(n≤4) 0.792→0.700 희생.
완주 checkpoint 재확인은 §48. 전문은
[`history.md`](history.md) §45.

## 46. 2026-08-06 — PathoBench zero-shot 평가: per-task PCA 전처리 + 결과 — **아카이브됨**

per-task PCA(1536→512) 캐시 파이프라인 구축 + sample-context 17개 task(대부분
0.5~0.68) + all-context 5개 task(개선). **§51 정정: 로컬 ccrcc CSV는 bc_therapy
복사본 오류**. 프로토콜은 이후 공식 50-fold(§52/§53)로 대체. 전문은
[`history.md`](history.md) §46.

## 47. 2026-08-06 — 새 기준 checkpoint(e125) 재평가 + 타일 수 제한 실험 — **아카이브됨**

`--context-mode all` 기본화 + `--max-tiles`/`--trials` 추가. e125(0.5142)를 새 기준으로
채택(val_ce 개선이 test로 대체로 전이), 타일 제한 스윕은 **task 의존**(LUAD는 제한이
개선). 전문은
[`history.md`](history.md) §47.

---

## 48. 2026-08-06 — arm C top-up 완주(150ep, best e125) + v33 Phase 0 평가 확정 + PathoBench v30 비교 — **아카이브됨**

arm C top-up 150ep 완주(e125). **legacy 회귀 gate 여전히 미달(+0.0412) — 과소학습
편향 가설 기각, B2b 데이터 자체가 회귀 원인. Musk n>34 개선(0.698→0.849) 유지,
PathoBench는 v30 우위(5-task 평균 +0.039). → v30 baseline 유지, arm C 미채택.**
전문은
[`history.md`](history.md) §48.

---

## 49. 2026-08-07 — 아키텍처 효율화(MLA-slot) + v34-1536 대규모 컨텍스트 학습 완주 + PathoBench 5-fold CV — **아카이브됨**

MLA-slot 효율화 + v34-1536 대규모 컨텍스트 학습 완주 + PathoBench 5-fold CV. 전문은 [`history.md`](history.md) §12.

## 50. 2026-08-07 — v34-1536 추가 평가: Musk 패딩 브리지(타일), ICI(랜덤), PathoBench 17-task 전체 CV — **아카이브됨**

v34-1536 추가 평가 — Musk 타일 패딩 브리지, ICI(랜덤), 17-task CV. 전문은 [`history.md`](history.md) §12.

## 51. 2026-08-07 — PathoBench 원본 검증: 로컬 cptac_ccrcc CSV는 bc_therapy의 잘못된 복사본 — **아카이브됨**

⚠️ 로컬 `cptac_ccrcc_*` CSV가 `bc_therapy` 복사본으로 확정 — 데이터 출처 검증 교훈. 전문은 [`history.md`](history.md) §2.

## 52. 2026-08-07 — 실제 ccrcc 평가 완료 + SEAL baseline 비교 + 공식 50-fold 평가 계획 — **아카이브됨**

실제 ccrcc 평가 + SEAL baseline 비교 + 공식 50-fold 계획. 전문은 [`history.md`](history.md) §12.

## 53. 2026-08-07 — v34 최종 확정 + 공식 50-fold 평가(SEAL 동일 프로토콜) 진행 — **아카이브됨**

v34 최종 확정 + 공식 50-fold(SEAL 동일 프로토콜) 진행. 전문은 [`history.md`](history.md) §12.

## 54-55. 2026-08-07 — 아카이빙 정리 + 리팩터링 1단계 (완료, 아카이브됨)

두 절 모두 종료된 정리 작업이라 전문을 [`history.md`](history.md)로 이관했다.
요약: §54 = 구버전 문서/config/스크립트 아카이빙 + v34 태그, §55 = AST 정적 분석으로 미사용
함수 제거. 열린 과제 없음.

---

## 56. 2026-08-07 — config 시스템 리팩터링(v34 base·default 참조·재아카이빙) + 공식 50-fold 재시작 — **아카이브됨**

> 아카이브됨 (2026-08-08, §64 정리): config 리팩터링은 완료됐고 지속되는 규칙은
> [`agent_handoff.md`](agent_handoff.md) §7(config 관리·자체 포함형 아카이빙·참조 검증)에 있다.
> 여기서 재시작한 공식 50-fold는 §57(case leakage)에 이어 §64(fp32 수치는 참고용)로 대체됐다.
> 전문: [`history.md`](history.md)

## 57. 2026-08-07 — 50-fold 재개 전 진단: 5-fold CV의 case leakage로 lscc_arid1a 0.908이 부풀려짐 — **아카이브됨**

5-fold CV의 **case leakage**로 lscc_arid1a 0.908이 부풀려짐 → 공식 50-fold 0.462가 정직한 값. 전문은 [`history.md`](history.md) §2.

## 58. 2026-08-07 — v35 설계 확정: rare-instance 제거 + context/query 공통 chunk + 대형화 — **아카이브됨**

v35 설계 확정 — rare 제거 + 공통 chunk + 대형화. 전문은 [`history.md`](history.md) §13.

## 59. 2026-08-07 — v35 제안서 비판적 재검토(rev.2) + 정확 스트리밍 구현 + v35 학습 시작 — **아카이브됨**

v35 제안서 rev.2 재검토 + 정확 스트리밍 구현. 전제 3건이 반증됐다. 전문은 [`history.md`](history.md) §13.

## 60. 2026-08-08 — v35-16384 50ep 완주 + 메모리/val plateau 진단 + v35 공식 50-fold 평가(EGFR·PIK3CA 완료) + SEAL 비교 — **아카이브됨**

v35-16384 완주 + 메모리/val plateau 진단 + 공식 50-fold 평가. 전문은 [`history.md`](history.md) §13.

## 61. 2026-08-08 — P0-b 게이트 통과 + rare branch 제거 (rev.2 step 5) — **아카이브됨**

`rare_logits=0` ablation이 |Δpooled| **0.0009 < 0.003**으로 게이트를 통과해 rare 분기를 제거했다
(`meta_enable_rare_evidence: false`, 코드 삭제가 아니라 강제 0 — ckpt 호환·가역). 이후 모든 arm이
rare-free이므로 **평가는 반드시 그 arm의 훈련 config로** 해야 한다. 전문: [`history.md`](history.md).

## 62. 2026-08-08 — v36 제안서 비판적 재검토 + P0-slots 무료 probe (Q1 확정 / Q2 보류) — **아카이브됨**

v36 제안서 재검토 + P0-slots probe. **진단은 유효하나 처방은 §65가 반증** — routing softmax가 길이 1 축에 걸려 무력(계약은 [`agent_handoff.md`](agent_handoff.md)). 전문은 [`history.md`](history.md) §14.

## 63. 2026-08-08 — current_architecture v34 개편 검토 + bf16-mixed 계약 실제 강제 — **아카이브됨**

`configs/trainer/default.yaml`이 precision을 설정하지 않아 v34/v35가 fp32로 조용히 학습됐던 것을
확인하고 bf16-mixed를 예외 없이 강제했다(`tests/test_precision_contract.py`). 계약 본문은
[`agent_handoff.md`](agent_handoff.md) §3.4에 있다. 전문: [`history.md`](history.md).

## 64. 2026-08-08 — 평가도 bf16-mixed 강제 + 폴드 단위 context 캐싱(bit-identical, 7.1×) + bc_therapy/er_status 기본 평가 확정 — **아카이브됨**

평가도 bf16-mixed 강제 + 폴드 단위 context 캐싱(bit-identical, 7.1×). 계약은 [`agent_handoff.md`](agent_handoff.md). 전문은 [`history.md`](history.md) §3.

## 65. 2026-08-09 — v36 Q1 / v37 두 arm 평가 완료: **둘 다 게이트 미달**, 40→1 압축은 원인이 아니었다 — **아카이브됨**

v36 Q1 / v37 **둘 다 게이트 미달** — 40→1 압축은 원인이 아니었다. 전문은 [`history.md`](history.md) §14.

## 66. 2026-08-09 — ridge ablation (v38): **G-2 global ridge는 무기여 / P-2·CV-1은 제거 시 학습 붕괴** — **아카이브됨**

ridge ablation(v38) — G-2 무기여 / P-2·CV-1은 제거 시 학습 붕괴. 전문은 [`history.md`](history.md) §15.

## 67. 2026-08-09 — v39 수치 안정화: **역효과**(clipping이 −0.0317) + LR 가설 반증 — **아카이브됨**

v39 수치 안정화 **역효과** — clipping이 −0.0317. clipping 금지. 전문은 [`history.md`](history.md) §15.

## 68. 2026-08-09 — 분기 기여도 진단 → **CV-only 성공**: 6개 분기 중 2개만 남겨도 동률 — **아카이브됨**

분기 기여도 진단 → **CV-only 성공**. 6개 분기 중 2개만 남겨도 동률. 전문은 [`history.md`](history.md) §16.

## 69. 2026-08-09 — covariance sketch 기저 진단: **label-free 축 8개 전부 무효**, 차원만 유효 — **아카이브됨**

covariance sketch 기저 진단 — **label-free 축 8개 전부 무효**, 차원만 유효(대역폭 고정 시). 전문은 [`history.md`](history.md) §17.

## 70. 2026-08-09 — v41: er_status 0.7303 (**+0.031**) — 이득의 정체는 차원이 아니라 대역폭·CV-2  ⚠️ **§71이 정정: 10개 task로 넓히면 SEAL 상회 주장은 성립하지 않는다** — **아카이브됨**

v41 er_status 0.7303 — 이득은 차원이 아니라 대역폭·CV-2. ⚠️ §71이 정정. 전문은 [`history.md`](history.md) §17.

## 71. 2026-08-09 — SEAL 10개 task 전면 평가: **일반화 실패**, er_status는 가장 유리한 task였다 — **아카이브됨**

SEAL 10개 전면 평가 — **일반화 실패**, er_status가 가장 유리한 task였다. 판정 기준 변경. 전문은 [`history.md`](history.md) §17.

## 72. 2026-08-09/10 — 세션 요약: CV-2 손잡이 소진, 소스 prune, 학습 2.4배 가속, 계보 B 재설계 — **아카이브됨**

세션 요약 — CV-2 손잡이 소진, 소스 prune, 학습 2.4배 가속, 계보 B 재설계. 전문은 [`history.md`](history.md) §20.

## 73. 2026-08-09 — config로 끄기만 했던 5개 분기를 소스에서 삭제 (−11,285줄) — **아카이브됨**

config로 끄던 5개 분기를 소스에서 삭제(−11,285줄). ⚠️ prune 이전 ckpt는 `ICF_pre_prune` worktree 필요. 전문은 [`history.md`](history.md) §20.

## 74. 2026-08-10 — 학습이 평가용 ragged 경로를 타고 있었다 (74.2 → 31.3 ms/step) — **아카이브됨**

학습이 평가용 ragged 경로를 타고 있었다(74.2 → 31.3 ms/step). 전문은 [`history.md`](history.md) §20.

## 75. 2026-08-10 — v42 subspace_rank 2/4: 무효 — **아카이브됨**

v42 subspace_rank 2/4 — 무효. 전문은 [`history.md`](history.md) §20.

## 76. 2026-08-10 — v43/v44 identity margin: **기각**, tanh 유지 — **아카이브됨**

v43/v44 identity margin **기각**, tanh 유지. 전문은 [`history.md`](history.md) §20.

## 77. 2026-08-10 — v45 paired_head: 동률, 그러나 라벨 대칭성을 얻었다 — **아카이브됨**

v45 paired_head — 동률이나 라벨 대칭성을 구성으로 획득. 전문은 [`history.md`](history.md) §20.

## 78. 2026-08-10 — CV-1의 dual(kernel) ridge는 옳다 — **아카이브됨**

CV-1의 dual(kernel) ridge는 옳다 — 근거는 bag 수 ≪ 특징 수, 실측 30배. 전문은 [`history.md`](history.md) §20.

## 79. 2026-08-10 — 계보 B (Encoder+Ridge): 첫 판본은 설계 오류, 재설계 후 궤적 반전 — **아카이브됨**

계보 B(Encoder+Ridge) — 첫 판본은 설계 오류, 재설계도 SEAL 하락. **문제는 일반화**. 전문은 [`history.md`](history.md) §20.

## 80. 다음 세션이 할 일

당시(2026-08-10) 기준 Action Plan으로 소진됐다. 현행 다음 작업은 **§99-5**와 **§100-5**.

## 81. 2026-08-10 — episode 내부 bag별 cardinality + zero-padding/mask — **아카이브됨**

episode 내부 bag별 cardinality + zero-padding/mask 계약. 전문은 [`history.md`](history.md) §21.

## 82. 2026-08-10 — factorized response/XOR 데이터 arm — **아카이브됨**

factorized response/XOR 데이터 arm — v57~v61 전부 v41 미달. 전문은 [`history.md`](history.md) §21.

## 83. 2026-08-10 — v61: random MLP를 orthogonal linear projection으로 교체 — **아카이브됨**

v61 — random MLP를 orthogonal linear projection으로 교체. 전문은 [`history.md`](history.md) §21.

## 84. 2026-08-10 — v62 Linear-16 + CV-1 K128 hybrid — **아카이브됨**

v62 Linear-16 + CV-1 K128 hybrid. 4,096 cap을 생성 전에 적용(OOM 교훈). 전문은 [`history.md`](history.md) §21.

## 85. 2026-08-11 — v62–v66 hybrid 결과, branch 명칭 확정, 4-pop DDP8 완료 — **아카이브됨**

v62–v66 hybrid 결과, branch 명칭 확정, 4-pop DDP8 완료. 전문은 [`history.md`](history.md) §21.

## 86. 2026-08-11 — v66 기각 + CV의 raw bag-mean 승격 — **아카이브됨**

v66 기각 + **CV의 raw bag-mean 승격**(canonical CV). 현행 스펙은 [`current_architecture.md`](current_architecture.md) F. 전문은 [`history.md`](history.md) §22.

## 87. 2026-08-11 — Dispersion Distance와 v70 relation MLP: synthetic 일반화 신호 확인 — **아카이브됨**

Dispersion Distance와 v70 relation MLP. 현행 스펙은 [`current_architecture.md`](current_architecture.md) G. 전문은 [`history.md`](history.md) §22.

## 88. 2026-08-11 — v71–v74 ablation 완료, v74 CV+DD+CT를 활성 baseline으로 확정 — **아카이브됨**

v71–v74 ablation 완료, v74 CV+DD+CT 확정. 현행 스펙은 [`current_architecture.md`](current_architecture.md) H. 전문은 [`history.md`](history.md) §22.

## 89. 2026-08-11 — v76 learnable P를 활성 baseline으로 승격 — **아카이브됨**

v76 learnable P를 활성 baseline으로 승격(fixed-P 0.6731 → 0.6748). 전문은 [`history.md`](history.md) §22.

## 90. 2026-08-12 — provisional v77-pop-residual 기각, synthetic 난이도 축 분해 — **아카이브됨**

provisional v77-pop-residual 기각(0.6750), synthetic 난이도 축 분해. 전문은 [`history.md`](history.md) §23.

## 91. 2026-08-12 — ClassSep sweep 완료: Hard 0.6873, seed 반복 전 승격 보류

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

## 92. 2026-08-12 — Active: Hard latent dimension 2/4/8/16 ablation, 8×GPU — **아카이브됨**

Hard latent dimension ablation — L2/L4/L8/L16 = 0.6775/0.6781/0.6771/0.6662, L32(=v77) 0.6873. 전문은 [`history.md`](history.md) §23.

## 93. 2026-08-12 — Active: Hard fixed 3-layer MLP-bank sweep — **아카이브됨**

Hard fixed 3-layer MLP-bank sweep — M=128~4096 최고 0.6779(M=1024), v77 미달. 전문은 [`history.md`](history.md) §23.

## 94. 2026-08-12 — Active: Hard 50:50 infinite-linear + MLP-1024 — **아카이브됨**

Hard 50:50 infinite-linear + MLP-1024 — 0.6755, 기각. 전문은 [`history.md`](history.md) §23.

## 95. 2026-08-12 — Active: Hard orthogonal + learned ridge calibration — **아카이브됨**

Hard orthogonal + learned ridge calibration — 0.6840, 기각. 전문은 [`history.md`](history.md) §23.

## 96. 2026-08-12 — architecture/handoff SSOT 정리 — **아카이브됨**

architecture/handoff SSOT 정리. 전문은 [`history.md`](history.md) §23.

## 97. 2026-08-12 — Active: Hard v76 warm-start, 2k–16k ragged training — **아카이브됨**

Hard v77 warm-start 2k–16k ragged — epoch 34 best 0.6885(+0.0012). ⚠️ 첫 launch는 CUDA prefetch 중첩으로 OOM. 전문은 [`history.md`](history.md) §23.

## 98. 2026-08-12 — Hard v76을 canonical v77 baseline으로 승격

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

## 100. 2026-08-12 — v78: DD quadratic-form gradient path (구현 완료, 미실행)

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

## 101. 2026-08-12 — 문서 압축: §2~§97 본문을 history.md로 아카이빙

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

## 102. 2026-08-12 — configs 정리: 루트 67개 → 2개

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
