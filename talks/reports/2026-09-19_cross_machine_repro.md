# RU-98 교차 기계 재현성 실행 보고

## 범위

같은 코드·설정·데이터의 5-branch Primary 7 평가를 NHN `NEXGEM`의 B200과 Slurm
클러스터 GPU 노드에서 비교했다. 사전 등록 질문·기준·예산·중단 조건은
[`RU-98.json`](../../docs/ru/RU-98.json)에 있다. `SEAL 10`은 열지 않았다.

## 실행 provenance

- Slurm array: `156357`, 원소 `0`\~`6` 전부 `COMPLETED`, `ExitCode=0:0`
- 실행 커밋: `efcc222b2c1212e13bef79272c463350795e4d7b`
- 설정: `configs/baseline/v121_active.yaml`, sha256 접두사 `2a2ec33a97ab3e45`
- 환경: PyTorch `2.14.0+cu130`; `gnode3` 1건, `gnode5` 6건
- GPU: `gnode3`의 NVIDIA RTX A5000 1건, `gnode5`의 RTX A6000 6건;
  드라이버 `595.91.07`
- 원시 로그: `slurm_outputs/2026-09-19/0853_ru98/`
- 예측 파일: `predictions/xmachine/`

## 관측 결과

| 과제 | B200 참조 | Slurm GPU | 표시 정밀도 Δ | 경과 시간 |
|:---|---:|---:|---:|---:|
| `cptac_lscc/ARID1A_mutation` | `0.5507` | `0.5507` | `0.0000` | `1617s` |
| `cptac_lscc/Histologic_Grade` | `0.6772` | `0.6772` | `0.0000` | `1264s` |
| `cptac_lscc/KEAP1_mutation` | `0.6038` | `0.6038` | `0.0000` | `2352s` |
| `cptac_luad/KRAS_mutation` | `0.7004` | `0.7004` | `0.0000` | `1925s` |
| `cptac_pda/SMAD4_mutation` | `0.4426` | `0.4426` | `0.0000` | `612s` |
| `ucla_lung/progression_regression` | `0.7891` | `0.7891` | `0.0000` | `72s` |
| `cptac_ccrcc/PBRM1_mutation` | `0.5546` | `0.5546` | `0.0000` | `1047s` |
| **macro** | **`0.6169`** | **`0.6169`** | **`0.0000`** | — |

참조값 출처는 [`2026-09-18_arm_parity.md`](2026-09-18_arm_parity.md)다. 양쪽 과제별
AUROC가 같은 소수 넷째 자리 구간으로 반올림됐으므로, 각 과제의 실제 차이는 절대값
`0.0001` 미만이다. 따라서 사전 등록한 첫 기준인 `|ΔAUROC| < 0.0005`에 들어간다.

## 해석 경계

- 실행 관측: 이 코드·설정·데이터에서는 B200과 A5000/A6000 사이에 4자리 AUROC 결론이 유지됐다.
- 사전 기준 대입: 앵커의 `5e-4`\~`7e-4` 어긋남은 장비 차이로 설명되지 않는다.
- 미확인: 앵커 어긋남의 실제 원인과 더 넓은 하드웨어·소프트웨어 조합의 일반화 가능성.
- 공식 연구 판정과 RU 종료는 Reasoning 담당이 수행한다. 이 문서는 Platform 실행 보고다.

[작성자: OpenAI Codex / Platform Agent / GPT-5 (effort: 미확인) · 2026-09-19 09:33 KST]
