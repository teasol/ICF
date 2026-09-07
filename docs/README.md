# Documentation Map

**Last updated**: `2026-09-07`

> 이 파일은 **어느 사실이 어느 문서의 정본인지**만 말한다. 수치·상태·결정을 여기에 적지 않는다.

---

## 1. 새 세션이 읽는 순서

| 순서 | 문서 | 무엇을 얻는가 |
|---:|:---|:---|
| 1 | [`PROJECT.md`](PROJECT.md) | 최종 목표, 현재 목표, 기준선 수치, 승격 기준, 채택 게이트 |
| 2 | [`current_status.md`](current_status.md) | 지금 무엇을 하고 있는가, 블로커, 다음 명령 |
| 3 | [`agent_handoff.md`](agent_handoff.md) | 작업 규범, RU 프로세스, 불변식, 보고 무결성 계약, 실행 환경 |
| 4 | [`closed_axes.md`](closed_axes.md) | 다시 열지 않는 축과 **그 경계** — 제안을 심사하기 전에 반드시 |
| 5 | [`current_architecture.md`](current_architecture.md) | 브랜치 수식과 파이프라인 (구조·수식을 다룰 때) |

`git log --oneline -20`으로 최근 궤적을 함께 확인한다.

---

## 2. 정본 표 (Single Source of Truth)

**하나의 사실은 정확히 한 파일에서만 선언한다.** 다른 문서는 복사하지 않고 링크한다.
`tests/test_docs_consistency.py`가 이 규칙을 검사한다.

| 사실 | 정본 |
|:---|:---|
| 기준선 AUROC · 승격 기준 · 게이트 조건 · 판정 설계와 정밀도 · 오염 검사 상수 | [`PROJECT.md`](PROJECT.md) |
| 현재 목표 · 진행 중 RU · 블로커 · 다음 명령 · 환경 실측값 | [`current_status.md`](current_status.md) |
| 닫힌 축과 경계 정의 · 경계 미확정 목록 | [`closed_axes.md`](closed_axes.md) |
| 결정의 근거 · 정정 · 철회 이력 (append-only) | [`history/archive.md`](history/archive.md) §결정 이력 |
| 작업 규범 · RU 프로세스 · 불변식 · 보고 무결성 계약 | [`agent_handoff.md`](agent_handoff.md) |
| 브랜치 정의와 수식 · 집계 규칙 | [`current_architecture.md`](current_architecture.md) |
| 연구 이력 전수 (RU-01~) | [`history/research_units_all.md`](history/research_units_all.md) |
| 현행 연구 방향 후보 큐 | [`research_directions.md`](research_directions.md) |
| Slurm 노드 스펙 · 선택 우선순위 · job 로그 경로 | [`/home/kimds/agent_rules/slurm_rules.md`](/home/kimds/agent_rules/slurm_rules.md) (저장소 밖) |
| 노드 종속 설정 (인터프리터 · GPU 수 · 경로) | [`../scripts/node_env.sh`](../scripts/node_env.sh) |
| arm 구성과 환경변수 주입 | [`../scripts/lib/arms.sh`](../scripts/lib/arms.sh) |

---

## 3. 과거 기록 (History)

| 파일 | 내용 |
|:---|:---|
| [`history/research_units_all.md`](history/research_units_all.md) | **450커밋 전수 연구 단위 총람** (RU-01~78 역사 복원 및 이후 완료 RU append, 6대 시대, 113건 인용 검증). 앞으로 `ru.py close`가 여기에 append한다 |
| [`history/research_units_all.json`](history/research_units_all.json) | 위의 기계 판독 DB |
| [`history/research_units_all_manifest.json`](history/research_units_all_manifest.json) | 0..449 전체 커밋 매니페스트 (SHA · 날짜 · 메시지 · 변경 파일) |
| [`history/era_review_notes.md`](history/era_review_notes.md) | **시대별 검토 메모 (발표 자료용)** — 시대마다 "얻은 것 / 닫은 것"과 검토 지적. 정본 아님, 해석 메모 |
| [`history/archive.md`](history/archive.md) | 세션 절 아카이브 (§190~§226) |
| [`history.md`](history.md) | v18~v184 시대의 설계·딥다이브 통합본. §19는 "다시 열면 안 되는 결론" 요약이나, **경계 정의를 갖춘 정본은 [`closed_axes.md`](closed_axes.md)다** |

폐기된 architecture/진단 테스트의 archive 정책은 [`../tests/history/README.md`](../tests/history/README.md)를 따른다.

---

## 4. 유지 규칙

- **단일 선언**: §2의 정본 표를 지킨다. 수치를 두 곳에 적지 않는다.
- **정정은 덮어쓴다**: 본문은 현행 사실만 담는다. 낡은 서술을 주석으로 누적하지 않고,
  시점 한정 사실과 번복 이력은 [`history/archive.md`](history/archive.md)의 결정 이력에만 남긴다.
- **entry는 자기완결적이다**: living 문서의 각 항목은 **이력 링크를 타고 들어가지 않아도**
  무엇을 어떻게 할지 전부 알 수 있게 쓴다. 결정 ID(`D-xxx`)와 절 번호(`§xxx`)는 **출처 표기**로만
  쓰고 규칙의 내용을 그 ID에 위임하지 않는다. 테스트가 위임 서술 형태를 검사한다.
  자기완결 단위(entry)는 파일마다 다르다.

| 파일 | entry 하나 = |
|:---|:---|
| [`PROJECT.md`](PROJECT.md) | 절(§1~§5) 하나, 그리고 현재 목표·승격 기준·게이트 표의 각 **행** |
| [`closed_axes.md`](closed_axes.md) | **축 하나(`CA-xx`)**. `기각 기전`·`경계 안`·`경계 밖`·`재개 조건`이 그 안에서 완결된다 |
| [`agent_handoff.md`](agent_handoff.md) | 규칙 하나 (적용 규칙 `R1`~`R7`, 보고 무결성 계약 항목, 호스트 유형, 표준 명령) |
| [`current_status.md`](current_status.md) | 헤더 표의 **행** 하나, 그리고 각 절의 **불릿** 하나 |
| [`current_architecture.md`](current_architecture.md) | 브랜치 하나 |
| [`research_directions.md`](research_directions.md) | 후보 하나, Phase 표의 **행** 하나 |
| 이 파일 | 정본 표의 **행** 하나, 유지 규칙 하나 |

  **기록물은 이 의무에서 제외된다** — `history/`·`reports/`·`ru/`는 시점 기록이므로 당시 서술을
  보존하고 현행 사실로 덮어쓰지 않는다.
- **`current_status.md`는 80줄을 넘기지 않는다.** 종료된 절은 `history/archive.md`로,
  결정 이력도 같은 파일 말미에 있다. 테스트가 강제한다.
- **RU 단위로 진행한다**: 착수 전 `scripts/docs/ru.py open`, 종료 시 `close`.
  절차는 [`agent_handoff.md` §1](agent_handoff.md).
- **축을 닫을 때는 경계를 함께 적는다**: `기각 기전`·`경계 안`·`경계 밖`·`재개 조건`
  네 필드가 모두 필요하다. 테스트가 강제한다.
- **절 번호는 재사용하지 않는다** (`history/archive.md`에 §199·§200이 중복 존재한다).
- **문서 변경 후**: `bash scripts/run_tests.sh test_docs_consistency.py`

### 4.1 `configs/` 루트 관리

`configs/` 루트에는 **현재 활성 파이프라인의 진입점만** 둔다.

> ⚠️ 이 규칙의 상세는 원래 `agent_handoff.md` §7에 있었으나 그 문서가 압축될 때 사라졌다 —
> **이 절이 현행 단일 출처**다 (archive 파일 헤더 200여 개가 인용하는 "agent_handoff SS7.3"도
> 같은 옛 절을 가리키는 역사적 표기다).

| config | 역할 |
|---|---|
| `train_v98_p1_reverse_1536_1gpu.yaml` | **루트에 남은 유일한 config.** 활성 경로가 로드하는 체크포인트 껍데기([`../scripts/node_env.sh`](../scripts/node_env.sh) `ICF_CONFIG` 기본값). v106+ 가 projection과 head를 덮어쓰므로 이 파일의 학습값은 마진에 닿지 않는다(§152) |

활성 baseline은 학습 파라미터가 0개이므로 루트에 학습 arm config를 새로 만들 이유가 없다.
arm 정의는 config가 아니라 [`../scripts/lib/arms.sh`](../scripts/lib/arms.sh)의 `icf_arm_v1xx`
함수 + 환경변수로 주입한다. fold/seed도 config가 아니라 `--cv`/`--seed`로 주입한다.

**2026-09-02 정리 내역 (§205)**: 학습 파라미터 계보(v77~v105) 루트 config 26개를 시대별
`configs/archive/` 폴더로 이관하고, 참조가 0인 config-group 23개와 Research Harness 전용 yaml
5개를 삭제했다(tracked config 277 → 249). 이관 폴더는 `v77_hard_orthogonal/`,
`v80_v82_seed_batch/`, `v83_linear_head/`, `v86_v93_episode_shape/`, `v94_v102_cell_value/`,
`v103_v105_head_proj/`이며, 기존 시대별 폴더와 함께 전부 `base_config` 없는 **자체 포함형**으로
보관한다. 원문은 git 이력에 보존된다.
