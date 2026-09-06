# 연구 이력 추출 030–064

이 문서는 고정 감사 패킷의 35개 커밋을 연구 질문과 당시 결정별로 11개 단위로 재구성한다. 수치와 판단은 해당 시점 문서 기록값이며, 새 실행이나 예측 재계산은 하지 않았다. 원문 SHA·경로·행·정확 인용은 동기화된 [JSON](research_units_030_064.json)에 있다.

당시 CT 포함 구성, SEAL 사용 방식, 승격 규칙은 역사적 맥락이다. 현재의 공식 비교는 CT 제외 5-branch Primary 7 기준이므로 이 기록의 기준선 수치를 현재 승격 근거로 쓰지 않는다.

| 단위 | 커밋 | 연구 질문과 관측 | 당시 결정 |
|---|---:|---|---|
| M01 | 030 | DD distance를 κ=1 ordered-typicality/weight 1로 바꾸자 v112는 SEAL 0.70432, hold-out 7 0.60181, 전체 0.66211로 기록됐다. | v112를 활성화하고 distance DD는 historical 재현용으로 남겼다. |
| M02 | 031 | full-cell CT는 22GB에서 LUAD 대형 bag OOM이 났다. own-bag 1/8, floor 64 arm은 SEAL 10을 모두 완료했고 0.70432→0.70394였다. | 성능 승리가 아니라 feasibility를 이유로 v113을 승격했다. 전체 17-task/hold-out 7은 미측정이었다. |
| M03 | 032–034 | 같은 fraction CT에서 head weight를 모두 1.0으로 바꾸자 0.70394→0.70509. CT RBF/poly KRR(0.69500/0.69475), top-k VHL(0.5076), mean+top-k(0.70067)는 하락했다. | v114를 승격하고 CT KRR/top-k는 기각하되 재현 코드는 보존했다. |
| M04 | 035–045 | 모듈을 분해하고 leading-subspace class-balanced ridge BM을 구현했다. 문서에서는 `current_status`를 개발 현황 SSOT로 배치하고 Primary 7/SEAL 역할을 정리했다. | 이 묶음은 성능 결과가 아닌 이후 branch 비교의 구현·문서 기반으로 기록한다. |
| M05 | 046 | projected 32D bag-mean ridge는 v114 0.6051→v115 0.6094, 5/7 승으로 기록됐다. | 사용자 승인으로 BM을 v115에 추가했다. |
| M06 | 047 | BD log-trace 후보는 0.6071이었고 spectral-entropy 후보는 v115 대비 +0.0025인 0.6119, 4/7 승이었다. | 당시 사용자 승인으로 spectral-entropy BD의 v116을 승격했다. 4/7은 현재 5/7 규칙과 구분한다. |
| M07 | 048–050 | DD 제거 선형 v117은 0.6191(5/7), CV+CT+BM+BD soft vote v118은 0.6205, v116 대비 +0.0086·6/7이었다. | v118을 활성화했다. SEAL/지도학습 표는 이후 선택 근거가 아닌 hold-out 역사 기록이다. |
| M08 | 051–053 | QA 포함 vote 비교에서 trimmed mean 0.6275가 기록됐고 DS 포함 v120 Primary 7은 0.6265였다. | QA/CV를 포함한 trimmed-mean 계보를 보존하고 v120을 당시 기준선으로 승격했다. |
| M09 | 054–056 | KRR 0.6125, LR 0.5874(7-branch 0.6195), Fisher 0.5709은 기각됐다. CT 단독 Primary 7은 0.6147로 기록됐다. | 네 실패 축을 종료하고 CT 단독은 앙상블을 대체하는 승격 근거가 아닌 비교 관측으로 보관했다. 문서 내 초기·후속 CT SEAL 값 충돌은 해소하지 않았다. |
| M10 | 057–062 | harness baseline 0.621643과 modularized arm 0.621614는 0.6265±0.005 band에 들고 arm 환경 parity는 통과했다. 후속 gate report는 metrics hash 불일치를 기록했다. | tolerance 통과를 engineering check로만 해석하며 bf16 drift를 성능/결정성 주장으로 쓰지 않았다. DE/SW/LOO는 이 범위에서 tooling 추가다. |
| M11 | 063–064 | v120은 `train_v98_p1_reverse_1536_1gpu.yaml` 하나를 로드한다는 정적 분석 후, 학습 config 26개를 archive로 옮기고 미참조 group/harness yaml을 정리했다. | 학습 계보는 archive와 git history에 보존하고 활성 경로와 분리했다. 당시 환경 소실로 회귀 suite 대신 정적 검증을 했다. |

## 검증

```bash
python3 /tmp/icf-research-audit/validate_terra.py \
  docs/history/research_units_030_064.json
```

검증은 030–064 coverage, 각 필수 필드, 고정 SHA, 인용 행의 `git show SHA:path` 문자열 일치를 확인한다.
