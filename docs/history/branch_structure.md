# 브랜치 구조 및 버전 관리 정책

**최종 갱신**: `2026-08-02`
**결정**: semver(0.1.0/0.2.0) 도입은 **폐기**. `architecture_version` 정수를 그대로 브랜치 이름으로 쓰는 방식으로 확정.

---

## 1. 브랜치 구조

> [!IMPORTANT]
> **현재 (2026-08-02)**: `main` = **v24 확정** (현재 개발선/SSOT). v25(T5-A)는 폐기 확정 —
> 태그 `v25-typed-bag-final`로 보존. 아래는 역사적 v18→v22 구조 + 현재 v23/v24/v25 요약.

```
128ff6f  ← 공통 조상 (v18 learnability ladder 시작점)
   │
   ├── v18  → b8d4f61   arch v18   [ladder 실행: 결과·스크립트·수정 11 커밋, A6000 호스트]
   │
   └── v19  → f99682b   arch v19   [shift-invariant covariance, CSP candidate A/B 선정]
          │
          └── (v21 구간: ecf6199 … d8c2b2b — 별도 브랜치 없음, 히스토리로만 존재)
                 │
                 └── v22 / main → arch v22  [retrieval 제거, 평가 프로토콜 재구축]
                       │
                       └── v23/v24 candidate line (codex/v23-bag-mean → v24 확정)
                             └── main → arch v24  [residual+bottleneck bag projection, 현재 SSOT]
                                   └── codex/v25-typed-bag → arch v25 [typed bag-preserving]
                                         └── 🗑 폐기 (2026-08-02), 태그 v25-typed-bag-final로 보존
```

| 브랜치 | `architecture_version` | 내용 |
|---|---:|---|
| `v18` | 18 | learnability ladder 실행 라인. 다른 서버(A6000)에서 관리. |
| `v19` | 19 | shift-invariant covariance 아키텍처, Candidate A/B 비교 완료 시점. |
| `v22` | 22 | retrieval 제거 + 평가 프로토콜 재구축. 참고용 보존. |
| `v24` | 24 | v24-B1 확정 지점. `main`과 동일 커밋. |
| `main` | 24 | **기본 브랜치, 현재 개발선 (SSOT)** — v24 확정. |
| `codex/v25-typed-bag` | 25 | v25(T5-A) 작업 브랜치. **폐기 확정 → 태그 `v25-typed-bag-final`로 보존 후 삭제.** |

v25 폐기 사유와 평가 수치는 [`current_status.md`](../current_status.md) §3 v25 배너 및
§11 "판정 근거 종합" 참고.

### v20, v21이 없는 이유

- **v20은 코드로 존재한 적이 없습니다.** `configs/archive/v20/*.yaml`는 v19 코드 위에서 돌던 **설정 파일 시리즈**일 뿐이며, `architecture_version`이 20이었던 커밋은 히스토리에 없습니다.
- **v21은 실재하지만 브랜치를 두지 않습니다.** `ecf6199`(19→21 bump)부터 `d8c2b2b`까지가 v21 구간이고, `fbc3ba1`에서 22로 올라갔습니다. 필요하면 `git checkout d8c2b2b`로 접근하세요. retrieval 최종 상태는 태그 **`v21-retrieval-final`** 로도 보존되어 있습니다.

### 태그

| 태그 | 대상 | 의미 |
|---|---|---|
| `arch-v18-learnability-baseline` | `128ff6f` | v18 ladder D0–D5 baseline 동결 지점 |
| `v21-retrieval-final` | v21 구간 마지막 | retrieval 계층이 살아 있던 마지막 상태 (복구용) |
| `v25-typed-bag-final` | v25 마지막 상태 | v25(T5-A) 폐기 직전 상태 (복구/참고용) |

---

## 2. 버전 관리 정책

**`architecture_version`(정수) 하나만 씁니다. semver는 도입하지 않습니다.**

`architecture_version`은 단순한 라벨이 아니라 **체크포인트 호환성 게이트**입니다:
- 모든 체크포인트 안에 `model._architecture_version` 텐서로 저장됩니다
- `ModelInterface.on_load_checkpoint`가 값이 다르면 로딩을 **거부**합니다
- 따라서 모델 구조가 바뀌어 기존 가중치를 못 읽게 될 때마다 정수로 올립니다

브랜치 이름을 이 정수에 맞추면 "어느 브랜치의 체크포인트가 어디에 로드되는가"가 이름만 보고 명확해집니다. 이것이 semver 대신 이 방식을 택한 이유입니다.

> [!WARNING]
> `architecture_version`을 문자열/semver로 바꾸면 체크포인트 게이트가 깨집니다. 정수를 유지하세요.

---

## 3. ⚠ v18 → v22 미적용 수정 2건 (2026-07-29 확인, 의도적 보류)

`v18` 브랜치는 `128ff6f`에서 갈라져 나가 **11개 커밋**이 쌓였고, 그중 **2건은 v22에도 해당하는 실제 버그 수정인데 아직 v22에 반영되지 않았습니다.** 사용자 결정으로 이번 브랜치 정리에서는 보류했습니다.

### ① `c05ff8d` — ridge-residual Cholesky backward 안정화 **(우선순위 높음)**

- **문제**: Cholesky 분해는 거의 특이(near-singular)한 행렬에서도 forward는 성공할 수 있지만, **backward에서 non-finite gradient**가 나옵니다.
- **수정 내용**: 첫 시도부터 항상 adaptive jitter를 더하도록 변경.
  ```python
  # v18 (수정됨)                          # v22 (현재, 미수정)
  for _ in range(6):                      for attempt in range(6):
      candidate = system + jitter...          candidate = system
                                              if attempt:
                                                  candidate = system + jitter...
  ```
- **v22 현재 위치**: `src/models/baseline.py:1482`
- **영향**: **단일 GPU에서도 발생 가능**. 학습 중 gradient가 non-finite가 되면 `_raise_if_nonfinite_parameters`가 잡아내지만, 그 전에 조용히 품질을 해칠 수 있습니다.

### ② `835b726` — rank-local CUDA episode 생성 (우선순위 중간, 현재 잠재)

- **문제**: CUDA current device는 thread-local이라, 중첩 generation worker가 `torch.device("cuda")`를 **전부 device 0으로 해석**합니다. DDP rank들이 같은 GPU에 생성하게 됩니다.
- **수정 내용**: `SyntheticEpisodeDataset._generation_device()` 헬퍼 추가 — bare `cuda`를 `torch.cuda.current_device()`로 해소.
- **v22 현재 상태**: `_generation_device()` 없음. `torch.device(self.generation_device)` 직접 사용.
- **영향**: **현재는 단일 GPU 운영(`NPROC_PER_NODE=1`)이라 잠재 상태**입니다. 다중 GPU DDP로 전환하는 순간 터집니다.

### ③ `7a623f2` — wandb nested key (v22 무관)

`scripts/record_learnability_run.py`만 수정하는데 **이 스크립트는 v22에 존재하지 않습니다.** v18 전용 도구이므로 이식 불필요.

### 반영하려면

```bash
git checkout v22
git cherry-pick c05ff8d          # Cholesky
git cherry-pick 835b726          # rank-local (충돌 가능 — synthetic_data.py가 많이 바뀜)
timeout 1500s /NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"
```

`835b726`은 `scripts/check_ddp_episode_data.py`, `scripts/record_learnability_run.py` 등 v22에 없는 파일도 건드리므로 `src/datasets/synthetic_data.py` 변경분만 골라 적용하는 편이 깔끔합니다.

---

## 4. 작업 규칙

- **신규 개발은 `main`(= `v22`)에서** 브랜치를 따서 진행합니다.
- 아키텍처 구조가 바뀌어 기존 체크포인트가 무효화되면 `architecture_version`을 올리고 **새 버전 브랜치**(`v23` 등)를 만듭니다.
- 구버전 브랜치(`v18`, `v19`)는 **재현·참조용으로 보존**하며 새 기능을 얹지 않습니다.
- `v18` 브랜치는 다른 서버(A6000 ladder 호스트)에서 관리되므로, 이쪽에서 강제로 밀지 마세요.
