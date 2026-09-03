# Current Status

- **Last Updated**: 2026-09-03 19:50 (KST)
- **Status**: CLEAN
- **Host / Node**: gnode3 (5x NVIDIA RTX A5000, 24GB VRAM)
- **Environment**: uv venv `.venv` | Python 3.12.11 (PyTorch 2.14.0+cu130, Lightning 2.6.5)
- **Active Session / Job**: None
- **Read First**: [#initial-baseline-migration](#initial-baseline-migration)

### Immediate Next Command
```bash
bash scripts/run_tests.sh
```

---

## 2026-09-03 — In-Episode LOO Dual Selection & 차원의 저주(Curse of Dimensionality) 규명 (§211)

### Completed
- **16-Branch Dual Pool 및 In-Episode LOO 평가 엔진 구현 완료**:
  - $O(1)$ Allen's PRESS(Hat Matrix $h_{ii}$) 공식 기반 LOO 평가기 및 5-GPU 병렬 오케스트레이터 완비.
- **Primary 7 과제 50-Fold 전수 실측 완료 (§211)**:
  - **초대형 성과**: `ARID1A`에서 **`0.6193` (+7.21%p 폭등, 전 모델 역대 최고치 경신!)**, `Histologic Grade`에서 **`0.7012` (+1.89%p, 0.70 장벽 돌파)**.
  - **수학적 기전 규명 (차원의 저주)**:
    - 서로 다른 차원을 가진 브랜치(CV: $32,640\text{D}$ vs DS: $32\text{D}$)를 raw LOO로 비교 시, 고차원 브랜치는 $DOF/N = 88.1\%$의 극단적 암기(Overfitting)로 인해 Context LOO가 $0.94\sim 1.000$의 '가짜 천재' 점수를 얻음.
    - 반면 동일 차원($D=32$)을 가진 $Full$ vs $Sub$ 간의 Intra-Branch LOO는 차원 편향 0%로 완벽하게 정직하게 작동함.
  - **Focal Mutation 누락 해결책 도출**:
    - KRAS 등 국소 변이의 테스트 하락을 원천 차단하기 위해, 상위 10~20% 살리언스 패치는 100% 앵커 보존하고 배경 패치만 서브샘플링하는 **Salience-Guided Selective Subsampling** 도입 필요성 규명.

### Active Research Queue (다음 연구 가설)
- **[Exp 1] Salience-Guided Selective Subsampling (국소 변이 앵커 보존)**:
  - **가설**: 살리언스 상위 패치를 앵커로 100% 보존하고 기질 패치만 서브샘플링하여, KRAS/KEAP1/PBRM1 하락을 원천 방어하면서 ARID1A(+7.2%p)와 Grade(+2.0%p)의 이점을 동시 획득.

### Code Reality vs Documentation Delta
- `scripts/test_pathobench.py`: `ICF_DS_AUG_MODE=auto_loo` 지원 추가.
- `scripts/run_ds_auto_loo_experiments.sh`: 5-GPU 병렬 오케스트레이터 추가.
- `docs/history/archive.md`: §211 기록 완료.

### Blockers & Tech Debt
- None. 131개 전 테스트 통과.

_by Antigravity on gnode3 at 2026-09-03 19:50:00_


