docs/ 와 talks/ 의 문서를 훑어 **낡은 서술**을 찾아 보고하라. **문서를 고치지 마라.**

배경: 2026-09-18 하루 동안 이 프로젝트의 계측 진단이 두 번 뒤집혔다. 그 결과 일부 문서가
이미 철회된 결론을 아직 담고 있을 수 있다. 무엇이 낡았는지 아무도 훑어본 적이 없다.

정본(가장 새로운 사실)은 이 파일들이다. 이들을 먼저 읽어라:
- docs/current_status.md
- talks/reports/2026-09-18_ru90_baseline_regen.md (특히 9절, 10절)
- talks/reports/2026-09-18_arm_parity.md
- talks/reports/2026-09-18_branch_redundancy.md
- talks/reports/2026-09-18_foldfit_cost.md
- docs/history/archive.md 의 D-053 항목

그다음 docs/ 와 talks/reports/ 의 **나머지** 문서를 읽고, 위 정본과 **어긋나는 서술**을
찾아라. 특히 이런 것들:
- 여섯 실행 러너가 서로 다른 구성을 돌렸다는 전제 (틀렸다. 전부 같은 7-branch 였다)
- `ICF_*` 환경변수로 arm 을 고를 수 있다는 서술 (코드가 읽지 않는다)
- 계측이 복구 불가라는 서술 (YAML 경로는 정상이다)
- 과거 arm 간 성능 차이를 구성 차이로 설명하는 서술 (적합 변동이었다)
- 확증 프로브가 예산 안에 들어간다는 전제 (fold 적합 1회가 20.0초다)

보고 형식: `talks/ops/reports/docs_stale.md`. 항목마다:
- 파일과 줄 번호
- 현재 그 문서가 하는 말 (인용)
- 어느 정본과 어떻게 어긋나는가
- 심각도: `철회 필요` / `보강 필요` / `표현만 낡음`

**절대 하지 말 것**: 문서 수정. 특히 AGENTS.md, docs/PROJECT.md, docs/closed_axes.md,
docs/history/archive.md 는 사용자 결정 기록이라 배경 작업이 고칠 수 없다. 보고서 하나만 쓴다.

없는 모순을 만들어내지 마라. 어긋난다고 확신할 수 없으면 적지 마라.
