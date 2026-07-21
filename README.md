# BagPFN

Architecture v18은 context에서 만든 8개 density + 4개 rare population anchor로 모든
bag을 정렬하고, slot마다 center/spread/rare-state token을 보존합니다. Meta classifier는
class context를 평균 하나로 줄이지 않고 8개 memory token으로 압축한 뒤 query와
cross-attention하며, query의 모든 cell에서 1/5/10/20% class-conditioned rare evidence를
추출합니다. 학습 loss는 final CE/ranking과 routing balance만 사용합니다.
ICI 평가는 donor의 모든 cell을 padding 없이 사용합니다.

## 실행 환경과 데이터

Python과 `torchrun`은 활성화된 가상환경의 `PATH`에서 찾습니다. 별도 실행 파일을
사용하려면 `PYTHON_BIN` 또는 `TORCHRUN_BIN` 환경변수로 지정할 수 있습니다.

ICI 데이터는 Git에 포함되지 않으며 저장소 루트의 다음 위치에 배치합니다.

```text
data/
├── ICI_CVOnly_scConcept_512/
│   └── SEED42/...
├── ICI_GSE285888_scConcept_512.pt
└── ICI_GSE285888_scConcept_512_info.csv
```

```bash
# 최소 학습 가능성 확인
./main_minimum.sh

# shared-background 중간 난이도 학습
./main_medium.sh

# hard synthetic 학습
./main.sh

# ICI 평가
./test.sh checkpoints/<run>/<dataset>/<checkpoint>.ckpt
```

Interactive 학습 스크립트(`main.sh`, `main_medium.sh`, `main_minimum.sh`)는
기본적으로 `nohup + setsid`로 터미널과 분리됩니다. SSH 또는 VS Code terminal을
닫아도 학습이 유지되며, 실행 시 출력되는 PID·training log·launcher log로 상태를
확인할 수 있습니다.
검증 성능이 top-k를 갱신하지 못하더라도 `last.ckpt`는 매 validation epoch의 최신
optimizer/scheduler/loop 상태로 덮어씁니다.

```bash
# 새 detached 학습
./main_medium.sh

# checkpoint부터 detached 재개
CKPT_PATH=checkpoints/<run>/medium/last.ckpt ./main_medium.sh

# 디버깅할 때만 foreground 실행
ICF_FOREGROUND=1 ./main_medium.sh
```

구조와 minimum benchmark 설명은 `MODEL_ARCHITECTURE_KO.md`를 참고하십시오.

Slurm 작업은 저장소 루트에서 `sbatch main_slurm.sh` 또는
`sbatch test_slurm.sh ...`로 제출하십시오.
