# 질문
1. `talks/reports/2026-09-18_arm_parity.md`가 재집계에 쓴 저장 예측 파일이 branch별 마진을 담고 있는가, 아니면 최종 fold-mean AUROC만 담고 있는가?
2. D-050이 지시한 provenance 대조 절차 중 RU-100 이후 **남은 작업의 범위**는 무엇인가 — 기록 필드 정의까지인가, 대조·판정 규칙 명문화까지인가?
3. `evaluate_pure.py`를 감싸는 러너 스크립트의 현재 이름은 무엇인가 (`eval_seal_tasks.sh`·`eval_v121.sh` 외에)? 상태 문서는 `scripts/eval_v121.sh`가 폴백으로 7-branch로 수렴했다고 기록하는데, 그 폴백이 "큰 소리로 실패"하도록 고쳐진 뒤의 러너 이름이 필요하다.
4. `SH` 단독·`SJ` 단독 측정은 어느 러너로, 언제 생산됐으며, 그 산출물에 provenance가 있는가(`D-053`의 폴백 경로를 탔는지)?
5. `RU-100`의 provenance writer는 `SCREEN_ONLY` 경로를 덮는가, 덮지 않는다면 게이트 ①은 현재 유효한가?
6. 과제별 fold 수준 paired Δ의 과제 내 sd는 문서에 있는가(반론 1의 다중비교 검정에 필요)?
7. RU-100이 저장한 예측 파일에 per-branch margin이 포함되는가, 아니면 branch list·가중치·해시만인가?
8. 오염 검사 재설계의 대조 기준(reference manifest)은 무엇으로 삼는가 — config 해시 단독인가, fold manifest 해시까지 포함인가?
9. 3/7 악화 과제(`Histologic_Grade`·`progression_regression`·`PBRM1`)의 N_ctx·클래스 균형·fold manifest 값이 나머지 4 과제와 다른가?
10. RU-100이 저장한 예측 파일의 경로·형식은 무엇이며, 과제별 per-fold AUROC가 남아 있는가?
11. `PROJECT.md` §3.2의 "`SH` 단독·`SJ` 단독에서도 같은 방향으로 악화된다"는 관측은 어느 실행에서 나왔고, 그 실행에 provenance가 기록돼 있는가?
12. 오염 검사 재설계는 현재 어디까지 진행됐는가 — 규격 초안이 존재하는가, 아니면 `D-050`의 방향 선언만 있는가?
13. 게이트 ① `SCREEN_ONLY` 마진 계산 1회의 GPU-hour 비용은 얼마인가?

# 원문 응답 (검증 전)
[1]
이 구간은 예측 파일에서 fold 단위로 재계산한 값(§추가)에서 나온 것이다.
**셋 다 맞는다.** `D-050`이 `unverified-hold`로 묶은 수치들은 값 자체가 틀린 것이 아니었다. 문제는 그 값을 **어느 구성에 귀속시킬지 알 수 없었다**는 것이었고, 이번 실행은 provenance를 기록했으므로 그 문제가 없다.

[2]

[3]
`eval_v121.sh`가 **학습 설정**을 넘기고 `eval_seal_tasks.sh`가 `model:` 블록을 보고 7-branch로 **조용히 교체**했다. 나머지 다섯도 같다. **arm 이름이 다른 실행들이 서로 다른 구성을 돌린 적이 없다.**
실제 이유는 `scripts/eval_v121.sh` 등 여섯 러너가 학습 설정을 넘겨 `eval_seal_tasks.sh`의 폴백에 걸렸고, 그 결과 **arm이 다른 실행들이 전부 같은 7-branch 설정으로 수렴**했기 때문이다.
폴백은 이제 **큰 소리로 실패**한다.

[4]

[5]
측정은 `SCREEN_ONLY=1`로 앙상블에 넣지 않은 채 마진만 기록해 수행하고, §3.4 오염 검사를 통과해야 한다.
**실행 선행 조건**: 게이트 ① 실행 경로(`SCREEN_ONLY`)가 현재 무효다(`D-050`). 계측이 복구된 뒤 라벨 무관·저비용 게이트 ①을 먼저 돌린다.

[6]

[7]
`evaluate_pure.py`가 저장 예측에 provenance(branch list·가중치·config/code 해시·fold manifest 해시)를 기록하게 됐고, 5-branch(`sha256 2a2ec33a97ab3e45`) 대 7-branch(`d8c8ae4dca9b8ecf`)를 **같은 잡·같은 세션**에서 Primary 7 × 50 fold로 재실행해 Δ_macro `+0.0057` · 과제군집 CI `[-1.15%p, +2.28%p]`(0 포함) · 악화 `3/7`을 재현했다.

[8]
**재설계 방향**: 값 일치가 아니라 **provenance 기록**을 검사 대상으로 삼는다 — 산출물에 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시를 저장하고 이를 대조한다.

[9]

[10]

[11]

[12]
**재설계 방향**: 값 일치가 아니라 **provenance 기록**을 검사 대상으로 삼는다 — 산출물에 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시를 저장하고 이를 대조한다.
재설계가 완료될 때까지 오염 검사 통과를 주장하는 실행은 무효다.
**provenance 기반 재설계 결정은 유지한다.**

[13]