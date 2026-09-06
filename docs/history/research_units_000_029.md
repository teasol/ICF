# Research units 000–029

작성일: 2026-09-05. 이 문서는 커밋 0–29의 당시 문서 기록을 질문별로 묶은 인덱스다. 수치와 판정은 새로 계산하지 않았으며, 각 항목의 `evidence`는 원본 커밋 본문에서 확인한 문장이다. 구조화 정본은 같은 이름의 JSON이다.

## E01 — CT 부분공간과 ridge readout의 결합 (0–1)

PCA와 ridge가 함께 CT 정보를 회복하는지 검사했다. PCA32+ridge는 v107 대비 전체 +0.0037(11/17)로 두 집단에서 양수여서 v108이 됐고, weight 0.5는 부호 일치 저하로 채택하지 않았다. 결정론적 arm에서는 task t/p/CI를 근거에서 제외했다. 근거: `ebcdb835…:docs/current_status.md:5287,5352,5333`.

## E02 — DD의 가중치와 코호트 의존성 (2)

DD 제거는 SEAL을 단조 하락시키고 홀드아웃을 단조 상승시켰다. 전역 weight 선택은 보류하고 0.343을 유지했다. 근거: `bd7647bb…:docs/current_status.md:5424,5466`.

## E03 — DD LLR 보정과 상대 거리의 판정 가능성 (3–4)

log-det 보정은 fold 내 상수여서 fold-mean AUROC를 바꿀 수 없었다. query별 상대 거리도 DD 단독 성능을 낮춰 기각했다. 근거: `c61a4100…:docs/current_status.md:5526,5544`; `c85156b8…:docs/current_status.md:5584`.

## E04 — CV descriptor의 정보 분해 (5, 11)

대각 feature 제거는 유효했지만, 공분산을 상관행렬로 바꿔 대각 스케일 효과까지 지우면 full model이 하락했다. off-diagonal 공분산의 크기 가중을 보존했다. 근거: `9c1889d2…:docs/current_status.md:6192–6193`.

## E05 — CT token 생성 개선과 v109 (6–7)

FPS token의 낮은 실사용률을 k-means가 보완했다. CV off-diagonal 및 k-means CT weight 0.7의 조합을 v109로 채택했다. 근거: `f2f29ccc…:docs/current_status.md:5981`; `7456a22c…:docs/current_status.md:5826`.

## E06 — CT 표본 수·token 수·ridge λ의 분리 (8–10, 12)

full cells는 모든 token 수에서 손해였고, 32 tokens·64 cells는 +0.0051, 15/17이었다. v110은 이를 채택했으며 λ 재검증은 token 이득의 약 30%가 λ 영향임을 정정했다. 근거: `a04f0cf2…:docs/current_status.md:6037`; `131c95a8…:docs/current_status.md:6100`; `fb91a069…:docs/current_status.md:6265`.

## E07 — 서버 이동용 실행 환경 정비 (13)

node 설정을 중앙화하고 handoff·architecture를 v110 계보에 맞춰 갱신했다. 성능 주장이나 승격은 없는 운영 기록이다. 근거: `69360766…:docs/agent_handoff.md:1`.

## E08 — 대체 CT dictionary와 full-cell hierarchical v111 (14–24)

random·density 계열을 기각한 뒤, full-cell/full-abundance hierarchical PCA32/K256을 v111로 채택했다. v110보다 macro는 낮았지만 sampling 불변성에 대한 사용자 선택이었다. CT 분기는 종료했다. 근거: `65a20d9c…:docs/current_status.md:7056,7072,7076`.

## E09 — DD ordered×typicality 후보의 구현과 수정 재실행 (25–29)

v111의 DD만 바꾸는 후보를 구현하고 17-task 평가를 launch했다. 첫 실행은 selector 변수 충돌로 fold 전 실패했으며 수정 후 재실행됐다. 0–29 범위에는 결과가 없어서 v111을 유지했고 index 30 결과와 연결해야 한다. 근거: `80f17db7…:docs/current_status.md:7127`; `cec1a154…:docs/current_status.md:7145,7151`.
