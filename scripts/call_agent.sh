#!/usr/bin/env bash
# scripts/call_agent.sh
# Invoke ICF research agents (Orca, Owl, Lime) via Claude Code CLI.

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Ensure Git uses ~/.gittoken_icf
# Ensure Git credentials use ~/.gittoken_icf if not already configured
if [[ -f "$HOME/.gittoken_icf" ]] && ! git config --local --get-all credential.https://github.com.helper | grep -q "gittoken_icf"; then
    git config --local --replace-all credential.https://github.com.helper "store --file $HOME/.gittoken_icf"
fi

CLAUDE_BIN=""
if [[ -x "$HOME/.local/bin/claude" ]]; then
    CLAUDE_BIN="$HOME/.local/bin/claude"
elif command -v claude >/dev/null 2>&1; then
    CLAUDE_BIN="$(command -v claude)"
else
    echo "Error: claude binary not found in ~/.local/bin or PATH." >&2
    exit 1
fi

show_help() {
    cat << EOF
Usage: bash scripts/call_agent.sh <agent-name> "<prompt>" [options]

Agents:
  orca | main-agent    Main Agent (model: claude-opus-5, effort: high)
  owl  | idea-agent    Idea Agent (model: claude-sonnet-5, effort: medium)
  lime | coding-agent  Coding Agent (model: claude-sonnet-5, effort: medium)

Examples:
  bash scripts/call_agent.sh owl "Propose 3 alternative mechanisms for task identification"
  bash scripts/call_agent.sh lime "Run regression tests and verify test_tier1_branches.py"
  bash scripts/call_agent.sh orca "Review recent proposals and decide RU-89 scope"
EOF
}

if [[ $# -lt 2 ]]; then
    show_help
    exit 1
fi

AGENT_TARGET="$1"
shift
PROMPT="$*"

case "$AGENT_TARGET" in
    orca|main-agent)
        AGENT="orca"
        MODEL="claude-opus-5"
        EFFORT="high"
        ;;
    owl|idea-agent)
        AGENT="owl"
        MODEL="claude-sonnet-5"
        EFFORT="medium"
        ;;
    lime|coding-agent)
        AGENT="lime"
        MODEL="claude-sonnet-5"
        EFFORT="medium"
        ;;
    *)
        echo "Unknown agent: $AGENT_TARGET" >&2
        show_help
        exit 1
        ;;
esac

"$CLAUDE_BIN" -p     --agent "$AGENT"     --model "$MODEL"     --effort "$EFFORT"     --permission-mode acceptEdits     "$PROMPT"
