# ICF Directory-based Inbox System

에이전트 및 사용자 간 비동기 작업 전달과 추적을 위한 파일 큐 시스템입니다.
세션 간 작업 지연, 백그라운드 위임 시 분실 방지, 오프라인 작업 인계를 지원합니다.

## 디렉토리 구조
- `talks/inbox/<role>/`: 해당 역할이 처리해야 할 대기 중인 작업 티켓 (`.md`)
- `talks/archive/inbox/<role>/`: 완료 처리된 작업 티켓 보관소
- `talks/inbox/TICKET_TEMPLATE.md`: 티켓 템플릿

## 지원 역할 (Roles)
- `orca`: Reasoning Agent (설계 및 판정)
- `owl`: Idea Agent (가설 및 대안 생성)
- `lime`: Coding Agent (구현 및 기술 검증)
- `document`: Document Agent (문서 최신화 및 정합성)
- `platform`: Platform Agent (운영 및 작업 연결)
- `kimds` (alias: `lead`): User / Project Lead

## CLI 사용법 (`scripts/inbox.py`)
```bash
# 1. 작업 티켓 전송
python scripts/inbox.py send --to lime --from orca --title "v122_shape_eval" --topic "docs/ru/RU-91.json" --desc "- [ ] 50-fold evaluation"

# 2. 대기 작업 목록 조회
python scripts/inbox.py list lime

# 3. 대기 작업 존재 여부 빠른 확인 (작업 있음: exit 0, 없음: exit 1)
python scripts/inbox.py check lime

# 4. 작업 완료 처리 (archive로 이동)
python scripts/inbox.py done talks/inbox/lime/2026-09-11_...md
```
