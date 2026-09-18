#!/usr/bin/env bash
# Download a paper to disk without routing it through the orchestrator's context.
#
# The orchestrator is the only participant with internet access, but it is also
# the most expensive place to read a 40-page paper: a WebFetch puts the whole
# text into a context that then carries it for the rest of the session. curl
# writes to disk instead, so the orchestrator spends tokens on the URL and the
# byte count, and the local model does the reading.
#
#   bash scripts/ops/lit_fetch.sh <url> <slug>
#   bash scripts/ops/lit_fetch.sh https://arxiv.org/abs/2606.06458 icmil
#
# arXiv ids are expanded to the HTML full text, falling back to the abstract
# page when no HTML rendering exists (older or PDF-only submissions).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
URL="${1:?usage: lit_fetch.sh <url> <slug>}"
SLUG="${2:?usage: lit_fetch.sh <url> <slug>}"
RAW="talks/lit/raw/${SLUG}.txt"
META="talks/lit/raw/${SLUG}.source"

mkdir -p talks/lit/raw
fetch() { curl -sL --max-time 120 -A "icf-lit-bot/1.0" "$1" -o "$2"; }

TMP="$(mktemp)"
TARGET="$URL"
if [[ "$URL" =~ arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5}) ]]; then
  ID="${BASH_REMATCH[1]}"
  # Size, not a keyword, decides whether the HTML rendering exists: arXiv serves
  # a short placeholder page for PDF-only submissions, and keyword checks match
  # the paper's own prose. 50 KB separates the two by an order of magnitude.
  if fetch "https://arxiv.org/html/${ID}" "$TMP" \
     && [ "$(wc -c < "$TMP")" -gt 51200 ]; then
    TARGET="https://arxiv.org/html/${ID}"
  else
    fetch "$URL" "$TMP"
  fi
else
  fetch "$URL" "$TMP"
fi

[ -s "$TMP" ] || { echo "내려받기 실패: $URL" >&2; rm -f "$TMP"; exit 1; }

# Many venues serve only PDF. pdftotext -layout keeps table columns adjacent,
# which matters because the numbers we care about live in results tables and a
# reflowed table turns them into an unreadable stream.
if head -c 5 "$TMP" | grep -q "%PDF"; then
  pdftotext -layout -q "$TMP" "${TMP}.txt" 2>/dev/null || {
    echo "PDF 변환 실패: $URL" >&2; rm -f "$TMP"; exit 1; }
  mv "${TMP}.txt" "$RAW"
  rm -f "$TMP"
  printf 'url: %s\nfetched_from: %s (PDF)\nfetched_at: %s\n' \
    "$URL" "$TARGET" "$(date '+%F %T %Z')" > "$META"
  echo "저장: ${RAW} ($(wc -c < "$RAW") bytes, $(wc -w < "$RAW") words) · PDF ${TARGET}"
  exit 0
fi

.venv/bin/python - "$TMP" "$RAW" <<'PY'
import re, sys, html
src, dst = sys.argv[1], sys.argv[2]
t = open(src, encoding="utf-8", errors="replace").read()
t = re.sub(r"(?is)<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", t)
t = re.sub(r"(?s)<[^>]+>", " ", t)
t = html.unescape(t)
t = re.sub(r"[ \t]+", " ", t)
t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t)
open(dst, "w", encoding="utf-8").write(t.strip())
PY
rm -f "$TMP"
printf 'url: %s\nfetched_from: %s\nfetched_at: %s\n' "$URL" "$TARGET" "$(date '+%F %T %Z')" > "$META"
echo "저장: ${RAW} ($(wc -c < "$RAW") bytes, $(wc -w < "$RAW") words) · 출처 ${TARGET}"
