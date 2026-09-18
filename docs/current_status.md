# Current Status

> **이 파일은 "지금 무엇을 하고 있는가"만 담는다. 100줄을 넘기지 않는다.**
> 수치·기준은 [`PROJECT.md`](PROJECT.md), 닫힌 축은 [`closed_axes.md`](closed_axes.md),
> 결정 이력은 [결정 이력](history/archive.md)가 정본이다.
> 종료된 RU의 상세는 [`history/research_units_all.md`](history/research_units_all.md)로 이관한다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-18 00:20 KST |
| **Status** | **WIP** · `RU-97` GF 단독 탐색 **음성으로 종료 방향**. 규범·정체성 3건 개정(`D-046`\~`D-048`). 협의체 5회차 완료 |
| **Host / Node** | `NEXGEM` · NVIDIA B200 8장(각 183,359 MiB) · `sbatch`·`squeue` 없음 |
| **Environment** | uv venv `.venv` · Python 3.12.3 · PyTorch 2.14.0+cu130 · `scikit-learn` 1.8.0 · `pytest` 9.1.1 |
| **GPU 배정** | **GPU 4 = 실험 전용.** GPU 5·6·7 = 로컬 Qwen 서버(포트 8003/8001/8000, 협의체 좌석용). GPU 0-3은 타 사용자 |
| **Active Job** | 없음 |
| **회귀 테스트** | `bash scripts/run_tests.sh` 189 tests · 1 failed (16 skipped) · 109초. **실패 1건은 기존 결함**(아래 참조) |

---

## 최근 결정 3건

- **`D-046`** 에이전트 규범을 **좌석 기반**으로 재편. 고정 5역할 폐지, 직은 소집자·종결권자·기록자
  셋만 지속. 권한은 사안 위험도(`T0`/`T1`/`T2`)에 붙는다. 집행부는 `scripts/council.py`.
- **`D-047`** `SEAL 10`의 경계는 **열람 금지가 아니라 선택 금지**다. 평가 실행·기록은 허용하되,
  그 결과로 모델·후보·구조를 고르는 것이 부정행위다. **개선 근거는 `Primary 7` 뿐이다.**
- **`D-048`** "학습 파라미터 0개" 정체성 **폐기**, `CA-14` `ABOLISHED`. 목표는 `ABMIL`을 성능으로
  넘어서는 것. 결정론이 깨지므로 **모든 비교에 적합 분산을 포함**하는 것이 새 의무다.

## `RU-97` GF 단독 탐색 — 음성 (판정은 미확정)

탐색 공간 `M ∈ {8,16,32}` × 화이트닝 `{없음, d128, d256, d512}` × 표본 `{raw, balanced}`
**안에서 `Primary 7` macro를 구분하는 축은 없었다.**

| 구성 | 적합 4회 평균 | 적합 sd | 승격 기준(`+0.0030`) 대비 |
|---|---:|---:|---:|
| `m8_raw` | 0.5863 | 0.0021 | 0.7배 |
| `m32_raw` | 0.5866 | 0.0050 | 1.7배 |
| `whiten_d256_m8` | 0.5871 | 0.0083 | 2.8배 |

- `M=32 − M=8`은 적합 분산 포함 시 **`+0.0003` `[-0.0067, +0.0073]`** 으로 0을 포함한다.
  단일 적합에서 보였던 `+0.0099`는 **seed 42가 그 구성 평균보다 1.5 sd 높았던 뽑기**였다.
- 화이트닝은 유효 성분 붕괴(1.0대)를 해소하고 차원을 6배 줄이지만 macro를 올리지 않고
  **적합 분산을 4배 키운다**.
- 분산분해: seed 주효과는 전체의 `0~2.9%`로 작으나 seed sd 자체가 기준의 최대 2.8배다.
  `m32_raw`는 **fold×seed 잔차가 23.9%** 로 seed 주효과보다 크다 — 재적합이 fold 순서를
  뒤섞으며, fold를 더 모아도 줄지 않는다.
- 상세와 무효화·정정 이력: [`talks/reports/2026-09-17_ru97_gf_primary7.md`](../talks/reports/2026-09-17_ru97_gf_primary7.md) §8\~§10.
- **기전은 미확인이고 `hold-out 미검증`이다.** 이는 이 구성 공간의 결과이지 GF 접근 전체의
  불가능이 아니다.

## 협의체 운영

`scripts/council.py`로 회차를 소집한다. 좌석은 회차 카드로 구동하며 카드는 결과 전에 쓴다.

```bash
.venv/bin/python scripts/council.py plan --round-id C-YYYYMMDD-n --question "..." \
  --seats P,R,A,S --endpoints "http://127.0.0.1:8000/v1/chat/completions,..." \
  --model Qwen3.8-27B > talks/council/<id>_card.json
# composition_rationale 기재 (없으면 소집 거부)
.venv/bin/python scripts/council.py check talks/council/<id>_card.json
.venv/bin/python scripts/council.py run   talks/council/<id>_card.json
```

회차 `C-20260917-1`\~`5` 완료(전부 기권 0). 산출물은 `talks/council/<회차 ID>/`.
**좌석 산출물은 로컬 모델 생성물이며 검증되지 않았다.** 인용 수치는 정본과 대조한 뒤에만 쓴다.

### 운영 규약 — 감독자와 회차

**회차는 Claude가 직접 띄운다. 감독자는 회차를 발사하지 않는다.** 앞선 회차의 결과를 읽고
질문을 다시 쓰는 텀이 협의체가 자기 말에 동의하는 것을 막는 유일한 장치다. 그 텀을 없애
GPU 가동률을 올리지 않는다. 대신 쉬는 시간에는 **회차와 무관한 짧은 일상 업무**를 돌린다.

```bash
bash scripts/ops/run_round.sh talks/council/<회차>_card.json   # 회차는 반드시 이것으로
bash scripts/ops/check_supervisor.sh    # 턴 종료 시 실행 (run_round.sh는 내부에서 이미 호출)
cat talks/ops/state.md                  # 노드 상태 (감독자가 유지, 이 파일만 읽으면 된다)
```

**회차는 `run_round.sh`를 백그라운드 작업으로 띄운다. `nohup`·`&`로 분리하지 않는다.**
이 스크립트는 회차가 끝날 때까지 블록하므로, 작업 완료 알림이 곧 회차 종료 알림이 된다.
따로 감시를 걸 필요가 없고, 따라서 거는 것을 잊을 수도 없다. 2026-09-18에 회차가 09:12에
끝났는데 09:40까지 아무것도 돌지 않은 것은 시간별 깨어남만 남기고 회차별 감시를 없앴기
때문이다.

- 감독자는 60초마다 스냅샷하고, **아무것도 안 돌 때만** `talks/ops/queue/`의 `routine`
  항목을 발사한다. `kind: "council"` 항목은 설계상 거부한다.
- 심박(`talks/ops/heartbeat`)이 180초 이상 멈추면 죽은 것이다. `check_supervisor.sh`가
  pid 파일로 판정하고 되살린다. **`pgrep -f`를 쓰지 않는다** — 패턴이 검사 스크립트 자신의
  명령줄과 일치해 셸을 죽인 사고가 이 저장소에서 세 번 있었다.
- `state.md`의 실측 절반은 코드가 생성하고 모델은 서술만 쓴다. 모델 호출 실패 시 직전
  서술에 노후 표시가 붙는다.

### Immediate Next Command

```bash
cd /NHNHOME/WORKSPACE/kimds/ICF
# 6회차: 첫 학습 파라미터 후보의 사전 등록 설계 (D-048 이후 첫 후보)
.venv/bin/python scripts/council.py run talks/council/C-20260918-6_card.json
```

---

## 미해결 (Open Issues)

- **적합 분산을 포함한 승격 판정 절차가 미승인이다.** 회차 `C-20260917-5`가 조건부 설계안을
  냈다(적합 평균 Δ를 estimand로, best-seed 선택 금지, `+0.0030` 유지하되 총 SE 병기,
  `R=8`은 잠정 보정). **사용자 승인 전까지 학습 파라미터 후보의 승격 판정은 보류한다.**
- **라벨로 학습한 margin을 라벨 무관 게이트 ①에 넣을 수 없다.** `D-048`이 연 절차 구멍이며
  아직 메워지지 않았다.
- **LSAK 사양이 문서에 없다.** `L`/`S` 원출, 차원, bandwidth, 정규화, margin 스칼라 환원,
  `max |r|`·`eff.rank` 집계 단위가 모두 미기재. 회차 `C-20260917-3`이 설계를 완성하지 못했다.
- **RU 번호 공백**: 대장이 `RU-90` → `RU-97`로 건너뛴다. `RU-91`\~`96`이 대장에도 열린 카드에도
  없어 `test_docs_consistency.py` 1건이 실패한다. 없는 기록을 지어낼 수 없어 미수정으로 둔다.
- `CA-R1` 다중 강도는 `closed_axes.md` §3-J 사용자 경계 판정 대기.
- 모든 후보 판정은 `SEAL 10` hold-out 미검증 상태를 유지한다.

[작성자: Claude Code / 소집자 겸 종결권자(사용자 지시에 의한 겸임) / claude-opus-5 (effort: 미확인) · 2026-09-18 00:20 KST]
