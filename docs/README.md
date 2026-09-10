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
| 작업 규범 · RU 프로세스 · 불변식 · 보고 무결성 계약 · 유지 규칙 | [`agent_handoff.md`](agent_handoff.md) |
| 브랜치 정의와 수식 · 집계 규칙 | [`current_architecture.md`](current_architecture.md) |
| 연구 이력 전수 (RU-01~) | [`history/research_units_all.md`](history/research_units_all.md) |
| **진행 가능한 후보 전체와 우선순위** | [`research_directions.md`](research_directions.md) §0 |
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
| [`history/history.md`](history/history.md) | v18~v184 시대의 설계·딥다이브 통합본(7,458줄). §19는 "다시 열면 안 되는 결론" 요약이나, **경계 정의를 갖춘 정본은 [`closed_axes.md`](closed_axes.md)다.** 파일 안의 `history.md §N` 자기 인용은 이 파일 자신을 가리킨다 |

폐기된 architecture/진단 테스트의 archive 정책은 [`../tests/history/README.md`](../tests/history/README.md)를 따른다.

---

## 4. 유지 규칙

문서 유지 규칙(단일 선언, 덮어쓰기, entry 자기완결성, 줄 수 제한, RU 관리 등)과 `configs/` 루트 관리 규범의 정본은 [`agent_handoff.md` §7](agent_handoff.md)이다.

