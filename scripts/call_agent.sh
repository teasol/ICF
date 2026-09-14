#!/usr/bin/env bash
# /home/kimds/.local/bin/call_agent.sh
# Dedicated Claude Agent Runner with persistent Session UUIDs.
# Supports ICF (Orca, Owl, Lime) and TIRANOS (Kite, Pear).

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

show_help() {
    cat << 'HELP'
Usage: call_agent.sh <agent-name> "<prompt>"

Agents by Project & Session UUID:
  [ICF] (/home/kimds/ICF)
    orca : Reasoning Agent (gpt-6-astra, effort: medium)   [UUID: 01a09b39-5d49-71a2-a71d-456a9745badc, Codex]
    owl  : Idea Agent      (claude-sonnet-5, effort: medium) [UUID: 87f23128-4c12-4bb8-ab44-7a0c5657c881, Claude]
    lime : Coding Agent    (claude-sonnet-5, effort: medium) [UUID: a63a3b5d-ac45-4c84-8ff4-d75872ac585b, Claude]

  [TIRANOS] (/home/kimds/TIRANOS)
    kite : Idea Agent      (claude-sonnet-5, effort: medium) [UUID: 797e15f4-b8e4-44a2-a27f-6694096c842c, Claude]
    pear : Coding Agent    (claude-sonnet-5, effort: medium) [UUID: 71385ef1-b993-4bd6-a39f-3fa24bd39b80, Claude]

Examples:
  call_agent.sh orca "Review recent proposals and decide RU-89 scope"
  call_agent.sh owl  "Propose 3 alternative mechanisms for task identification"
  call_agent.sh lime "Run regression tests and verify test_sj_branch.py"
  call_agent.sh kite "Propose boundary loss alternatives for DiT"
  call_agent.sh pear "Implement bisection volume ODE test"
HELP
}

if [[ $# -lt 2 ]]; then
    show_help
    exit 1
fi

RAW_TARGET="$1"
shift
PROMPT="$*"

TARGET="$(echo "$RAW_TARGET" | tr '[:upper:]' '[:lower:]')"

# Resolve generic roles (idea, coding, main) based on current working directory
CURRENT_DIR="$(pwd -P)"
if [[ "$TARGET" == "idea" || "$TARGET" == "coding" || "$TARGET" == "main" ]]; then
    if [[ "$CURRENT_DIR" == *"/TIRANOS"* ]]; then
        case "$TARGET" in
            idea)   TARGET="kite" ;;
            coding) TARGET="pear" ;;
            main)
                echo "Error: Navi is the Main Agent for TIRANOS (interactive Codex). call skill is not configured for Navi." >&2
                exit 1
                ;;
        esac
    else
        case "$TARGET" in
            main)   TARGET="orca" ;;
            idea)   TARGET="owl" ;;
            coding) TARGET="lime" ;;
        esac
    fi
fi

ENGINE="claude"

case "$TARGET" in
    # ---- ICF Project ----
    orca)
        PROJECT_DIR="/home/kimds/ICF"
        ENGINE="codex"
        SESSION_ID="01a09b39-5d49-71a2-a71d-456a9745badc"
        MODEL="gpt-6-astra"
        EFFORT="medium"
        TOKEN_FILE="$HOME/.gittoken_icf"
        ;;
    owl)
        PROJECT_DIR="/home/kimds/ICF"
        ENGINE="claude"
        SESSION_ID="87f23128-4c12-4bb8-ab44-7a0c5657c881"
        MODEL="claude-sonnet-5"
        EFFORT="medium"
        TOKEN_FILE="$HOME/.gittoken_icf"
        ;;
    lime)
        PROJECT_DIR="/home/kimds/ICF"
        ENGINE="claude"
        SESSION_ID="a63a3b5d-ac45-4c84-8ff4-d75872ac585b"
        MODEL="claude-sonnet-5"
        EFFORT="medium"
        TOKEN_FILE="$HOME/.gittoken_icf"
        ;;

    # ---- TIRANOS Project ----
    kite)
        PROJECT_DIR="/home/kimds/TIRANOS"
        ENGINE="claude"
        SESSION_ID="797e15f4-b8e4-44a2-a27f-6694096c842c"
        MODEL="claude-sonnet-5"
        EFFORT="medium"
        TOKEN_FILE="$HOME/.gittoken_tiranos"
        ;;
    pear)
        PROJECT_DIR="/home/kimds/TIRANOS"
        ENGINE="claude"
        SESSION_ID="71385ef1-b993-4bd6-a39f-3fa24bd39b80"
        MODEL="claude-sonnet-5"
        EFFORT="medium"
        TOKEN_FILE="$HOME/.gittoken_tiranos"
        ;;
    navi)
        echo "Error: Navi is non-Claude (GPT). call skill is not configured for Navi." >&2
        exit 1
        ;;
    *)
        echo "Error: Unknown agent '$RAW_TARGET'." >&2
        show_help
        exit 1
        ;;
esac

cd "$PROJECT_DIR"

# Ensure Git credentials use appropriate token
if [[ -f "$TOKEN_FILE" ]] && ! git config --local --get-all credential.https://github.com.helper 2>/dev/null | grep -q "$(basename "$TOKEN_FILE")"; then
    git config --local --replace-all credential.https://github.com.helper "store --file $TOKEN_FILE"
fi

if [[ "$ENGINE" == "codex" ]]; then
    CODEX_BIN=""
    if [[ -x "$HOME/.local/bin/codex" ]]; then
        CODEX_BIN="$HOME/.local/bin/codex"
    elif command -v codex >/dev/null 2>&1; then
        CODEX_BIN="$(command -v codex)"
    else
        echo "Error: codex binary not found." >&2
        exit 1
    fi
    exec "$CODEX_BIN" exec resume \
        -c model_reasoning_effort="$EFFORT" \
        "$SESSION_ID" \
        "$PROMPT"
else
    CLAUDE_BIN=""
    if [[ -x "$HOME/.local/bin/claude" ]]; then
        CLAUDE_BIN="$HOME/.local/bin/claude"
    elif command -v claude >/dev/null 2>&1; then
        CLAUDE_BIN="$(command -v claude)"
    else
        echo "Error: claude binary not found in ~/.local/bin or PATH." >&2
        exit 1
    fi

    # Execute Claude via persistent Session ID (--resume with --session-id fallback)
    # Note: permission-mode set to bypassPermissions per user authorization
    if ! "$CLAUDE_BIN" -p \
        --resume "$SESSION_ID" \
        --model "$MODEL" \
        --effort "$EFFORT" \
        --settings '{"autoMemoryEnabled":false}' \
        --permission-mode bypassPermissions \
        "$PROMPT"; then
        # Fallback to --session-id if resume fails
        exec "$CLAUDE_BIN" -p \
            --session-id "$SESSION_ID" \
            --model "$MODEL" \
            --effort "$EFFORT" \
            --settings '{"autoMemoryEnabled":false}' \
            --permission-mode bypassPermissions \
            "$PROMPT"
    fi
fi
