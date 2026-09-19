# 세션 인계 — 2026-09-19 09:40 KST · 실험은 Slurm으로 옮겨졌고 LLM은 터널로 쓴다

정본은 [`current_status.md`](current_status.md)다. 이 파일은 **이 전환에서 다음 세션이
모르면 막히는 것만** 담는다. 이전 판(2026-09-19 06:40, NHN 기계에서 작성)을 대체한다.

## 0. 이 기계가 어디인가 — 대소문자가 다르면 다른 기계다

| 표기 | 정체 | 유형 |
|:---|:---|:---|
| **`nexgem`** (소문자) | **지금 이 기계.** Slurm 로그인 노드. GPU 없음, `sbatch`/`squeue`/`sinfo` 있음. 20 CPU · 93 GB | 제출 노드형 |
| `NEXGEM` (대문자) | NHN Cloud 컨테이너. B200 8장, 딥시크가 GPU 4\~7 점유. `sbatch` 없음 | 직접 실행형 |

Slurm 규칙 정본은 **`/home/kimds/.agents/rules/slurm_rules.md`** 다.
(이전 판이 "이 기계에 없다"고 적은 그 파일이다. `agent_rules/`가 아니라 `.agents/rules/`.)
§1.1이 위 함정을 이미 문서화해 두었다.

- 파티션은 `batch` 하나. CPU 노드 `node1\~5`, GPU 노드 `gnode1\~6`.
- `--gres=gpu:<type>:N`은 동작하지 않는다. 노드는 `--nodelist`로 지정한다.
- job 로그는 `slurm_outputs/YYYY-MM-DD/HHMM/`, **절대경로로, 제출 전 `mkdir -p`.**
- 회귀 스위트도 로그인 노드에서 돌리지 말고 CPU 노드로 제출한다.

## 1. 딥시크에 닿는 법 — 저장소에 적힌 주소로는 안 된다

> **2026-09-19 15:30 정정 — 터널 불필요.** NEXGEM이 **tailscale로 직접 닿게** 됐다.
> `talks/ops/llm.local.json`을 `http://100.97.255.47:8000/v1/chat/completions`로 바꿨고,
> ssh도 `kimds@100.97.255.47`(포트 **22**, `~/NEXGEM_key`)다. 옛 주소
> `59.150.32.1:46401`(`ssh nhn`)은 지금 이 기계에서 no route이고, `10.34.5.16`·
> `100.134.17.129`도 여전히 안 닿는다. 아래 터널 설명은 09:40 시점 기록으로 보존한다.

`talks/ops/llm.json`의 `10.34.5.16`·`100.134.17.129`는 **이 기계에서 `No route to host`** 다.
서로 다른 tailnet이다. 닿는 경로는 **SSH 터널뿐**이다.

```bash
ssh -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -L 127.0.0.1:8000:127.0.0.1:8000 \
    -L 192.168.100.100:8000:127.0.0.1:8000 nhn
curl -s http://192.168.100.100:8000/v1/models   # 확인
```

`~/.ssh/config`에 `Host nhn` = `59.150.32.1:**46401**`, 키 `~/NEXGEM_key`가 이미 있다.
(`46501`은 열려 있지 않다.) `192.168.100.100` 바인딩이라 **계산 노드에서도 닿는다**
(`node1`에서 HTTP 200 실측).

**터널은 ssh 프로세스와 함께 죽는다.** 세션을 시작하면 먼저 살아 있는지 확인하라.
`loginctl enable-linger kimds`는 sudo 없이 통과했으므로 `systemctl --user` 유닛으로
영속화할 수 있다(아직 안 만들었다).

주소는 코드가 아니라 파일이 정한다. `talks/ops/llm.local.json`(git-ignore)이
`llm.json`을 이긴다. `llm_env.describe()`가 어느 파일을 읽었는지 이름으로 밝힌다.
위임용 worktree에는 `routine_opencode.sh`가 이 파일을 복사해 넘긴다.

**함정**: 이 모델은 사소한 질문에도 `reasoning_tokens`를 먼저 쓴다. `max_tokens`가 작으면
`finish_reason`이 `stop`인데도 `message.content`가 **에러 없이 `None`** 이다. 넉넉히 줘라.

## 2. 지금 돌고 있는 것

- **`tmux` 세션 `queue_monitor`** — `http://100.65.212.1:8899` (tailnet 안에서만).
  vLLM `/metrics`의 running·waiting·초당 생성 토큰, 주기 잡무 대기, 최근 실행·실패를 2초마다 갱신.
  로그 `/home/kimds/tmp/queue_monitor.log`. 딥시크가 작성, 소집자가 확인·병합.
- **RU-98 array job `156357`** — 아래 §3.

**스크래치는 `/tmp`가 아니라 `/home/kimds/tmp`를 쓴다**(사용자 지시). 상주 프로세스는 tmux로 띄운다 —
`nohup ... &`는 권한 분류기에 막히지만 tmux는 통과한다.

## 3. RU-98 — 교차 기계 재현성, 4/7까지 전부 완전 일치

사전 등록: `docs/ru/RU-98.json` (커밋 `efcc222`, **결과 보기 전에 기준 고정**).
참조값은 2026-09-18 NHN `NEXGEM` GPU 4의 5-branch Primary 7
([`arm_parity`](../talks/reports/2026-09-18_arm_parity.md)).

| 과제 | 참조(B200) | 이번(`gnode5` A6000) | Δ | 소요 |
|:---|---:|---:|---:|---:|
| `ucla_lung/progression_regression` | `0.7891` | `0.7891` | `0.0000` | 72s |
| `cptac_pda/SMAD4_mutation` | `0.4426` | `0.4426` | `0.0000` | 612s |
| `cptac_ccrcc/PBRM1_mutation` | `0.5546` | `0.5546` | `0.0000` | 1047s |
| `cptac_lscc/Histologic_Grade` | `0.6772` | `0.6772` | `0.0000` | 1264s |
| `cptac_lscc/ARID1A_mutation` | `0.5507` | 진행 중 | | |
| `cptac_lscc/KEAP1_mutation` | `0.6038` | 진행 중 | | |
| `cptac_luad/KRAS_mutation` | `0.7004` | 진행 중 | | |
| macro | `0.6169` | | | |

**남은 3개 수확 방법** (job이 끝났으면 바로 읽힌다):

```bash
LOGDIR=/home/kimds/ICF/slurm_outputs/2026-09-19/0853_ru98
for f in "$LOGDIR"/*.out; do
  t=$(grep -m1 "^task=" $f | cut -d= -f2)
  a=$(grep -m1 "fold-mean AUROC:" $f | awk '{print $3}')
  n=$(grep -m1 "^host=" $f | cut -d= -f2)
  g=$(grep -m1 NVIDIA $f | cut -d, -f1)
  [ -n "$a" ] && echo "$t $a $n $g"
done
```

**사전 등록한 판정 기준** (과제별 `|ΔAUROC|` 최댓값):
`< 0.0005` 4자리가 장비를 건너 유지됨 / `0.0005\~0.0030` 값 앵커는 장비 간 성립 불가하나
승격 판정은 안전 / `≥ 0.0030` 장비가 판정을 뒤집을 수 있음.

**이미 말할 수 있는 것**: 앵커 두 과제(`SMAD4` `0.4426`, `PBRM1` `0.5546`)가 소수 4자리까지
재현됐다. 따라서 `PROJECT.md` §3.4 앵커(`0.4421`/`0.5553`)와의 `5e-4`\~`7e-4` 어긋남은
**장비가 원인이 아니다.** 원인은 여전히 미확인이다.

**비용 추정이 크게 빗나갔다.** 기록된 `20.0`초/fold는 이 클러스터에 맞지 않는다 —
A6000에서 과제당 50 fold가 `72\~1264`초(과제 크기에 따라 `1.4\~25`초/fold)다.
예산을 `20.0`초/fold로 잡은 문서가 있으면 고쳐야 한다.

**교란은 없다.** 착수 전에 확인했다.
- 데이터 동일성: feature 파일 수(304/1139/242/112/245), 표본 `.h5` sha256, 7과제 fold manifest
  **내용** 해시가 두 기계에서 전부 일치.
  (파일 경로까지 해시하면 무조건 달라 보인다 — 내용만 해시하라.)
- 코드 동등성: 패리티 실행 시점 `830c973`과 HEAD의 집계 margin이 이 기계에서 비트 동일
  (`5branch 454a2effff16b2db`, `7branch bf8f25ee1ff0ab9d`). 그 사이 `src/` 변경
  `e2b0d79`·`bda3503`(미검토 자동 편집)은 집계 경로에 영향 없음.

### 직전 판의 데이터 경고는 해소됐다

NHN 쪽 세션이 `e604b9c`에서 **"코드는 lustre라 따라가지만 feature 데이터는 로컬 xfs라
계산 노드에서 안 보일 수 있다 — 그 전에는 어떤 제출도 의미가 없다"**고 경고했다.
그 경고는 NHN 기계 기준이었고, **이 클러스터에서는 해당되지 않는다.** 실측:
`data/repro_labels_folds/features/*`가 `/data-hdd/archive/public/HE/pathobench/...`로
심볼릭돼 있고 계산 노드에서 정상적으로 읽힌다(RU-98 job 4건이 실제로 완주했다).
NHN에서 계산 노드로 나갈 때는 그 경고가 여전히 유효하다.

## 4. 오늘 고친 것 — 안전망이 고장나 있었다

**`numeric_fingerprint.py --check`가 아무것도 비교하지 않고 통과해 왔다.**
`7b53973`이 키에 `@device`를 붙였는데(`5branch` → `5branch@cpu`) 기준선 파일은 옛 이름
그대로여서 교집합이 비었고 `return 0`으로 갔다. 이제 크게 실패한다(`20a2236`).

**기계가 바뀌면 지문 해시는 달라진다.** 이 기계 CPU 해시는
`ea2aa5fa3e8d5d03` / `888d4f7d282626ce` / `736a16f92a93987e`이고 기준선과 다르다.
1fcccb6 시절 코드를 여기서 돌려도 같은 값이 나오므로 **차이는 기계에서 온다**(집계 margin
절대 `1.5e-06`·상대 `4.6e-05`). 무효가 아니라 **비트 비교 불가**다.
기준선을 기계별로 나눌지는 **아직 안 정했다.**

`pytest`가 어느 requirements에도 선언돼 있지 않아 `test_gf_background_gmm.py`가 이 기계에서
collection ERROR를 냈다. 선언했다(`23f3ae2`). 회귀는 **214 tests, 실패 1건**
(`RU-91`\~`96` 대장 공백 — 기존 결함).

## 5. 새 규범 — `D-055` (사용자 결정, 모든 에이전트에 적용)

**완벽함보다 빠른 진행이 훨씬 중요하다.** 일단 휴리스틱하게 진행하고 결과가 나온 뒤 검증한다.
목표를 달성하지도 못한 채 모든 경우를 차근차근 확인하며 가는 방식은 완벽하게 아무것도
진행되지 않는 프로젝트로 수렴한다. `AGENTS.md` §7.0에 있고 §7의 나머지보다 우선한다.

남기는 것은 둘 — **빠르게 간 것과 검증한 것을 구분해 적는 일**, **확증에서 판정 기준을 결과
전에 고정하는 일**. 속도를 위해 생략하는 것은 절차이지 사실의 구분이 아니다.

실무 규칙: 계획 항목마다 **"이게 없으면 지금 질문에 답을 못 하나?"** 아니면 뺀다.

## 6. 다음에 할 일 — 딥시크 큐 (사용자 검토 대기)

**딥시크가 노는 근본 원인은 감독자가 이 기계에서 안 돌기 때문이다.** 모니터가 주기 잡무 6건
(`seat-thinking`·`open-questions`·`docs-consistency`·`provenance-scan`·`regression-suite`·`lit-index`)을
전부 `미실행`로 보고한다. **`scripts/ops/supervisor.py`를 tmux로 띄우는 것이 개별 작업보다 먼저다.**

기존 위임 작업 9개 중 8개는 09-18에 처리됐고 미착수는 `perf_transfer` 하나다.

제안한 순서(사용자 승인 전):
1. `ru_ledger_backfill` — `RU-91`\~`96`을 git 이력(`4e83fb7`·`50f81ad`·`56dc176`·`d68aeb6`)에서
   복원해 대장에 채운다. **지금 회귀 실패 1건의 원인.** 복원 못 하면 지어내지 말 것.
   **(2026-09-19 완료**: `RU-91`·`92`·`93`·`95`·`96`은 보고서·`D-043`·`D-044`, `RU-94`는
   `D-044`(S1/S2/S3 후보 폐기)에서 대장·총람으로 이관. 사전 등록 필드는 `미보존`으로 명시.
   `RU-94` 전용 보고서는 없고 `ru94_rerun` 데이터는 삭제되어 가장 얇은 복원이다.**)**
2. `queue_monitor_fix` — 모니터의 `tasks` 미착수/착수됨이 `None`으로 나온다.
3. `undeclared_deps` — import 하는데 requirements에 없는 패키지 전수 점검.
4. `provenance_check` — `D-050` 방향 구현(branch list·가중치·코드 해시·데이터 버전·fold
   manifest 해시 저장·대조). 값 앵커를 대체하는 물건이라 값어치가 크다.
5. `baseline_per_host` — 지문 기준선을 호스트별로 분리.
6. `pca_subsample_spec` — 회차 16의 v1 안이 기록되지 않았다.
7. `perf_transfer` — **RU-98이 끝난 뒤에.** 수치를 바꿀 수 있다.
8. `handoff_refresh` — 문서 동기화.

## 7. 사용자 결정 대기

1. **지문 기준선을 기계별로 나눌 것인가**, 그리고 별도로 **결론 재현 허용 오차**를 판정 규칙에
   넣을 것인가(RU-98 결과가 입력).
2. 예산 상한 — 클러스터로 옮겨 `4` GPU-h 제약의 의미가 달라졌다. fold 비용이 기록보다
   훨씬 싸다.
3. `closed_axes` G(AKS 등방성)·H(LID 부분추출) 판정.
4. 다중 후보 비교 보정 규칙.
5. 모니터를 systemd user 유닛으로 영속화할지(linger는 켜 두었다).

## 8. 걸려 넘어지기 쉬운 곳

- **`/tmp` 쓰지 말 것.** `/home/kimds/tmp`.
- 상주 프로세스는 **tmux**로. `nohup ... &`는 막힌다.
- `pgrep -f`로 자기 프로세스를 찾지 마라. 이 저장소에서 셸이 세 번 죽었다.
- 실험은 블로킹으로 띄워라(`sbatch --wait` 또는 `srun`). 밖에서 띄워 5시간을 날린 적이 있다.
- 파일 목록 해시는 **경로가 아니라 내용**을 해시하라.
- `Co-Authored-By`를 쓰지 않는다. 실제 작성 주체를 단독 서명한다.

[작성자: Claude Code / Platform Agent / claude-opus-5 (effort: 미확인) · 2026-09-19 09:40 KST]
