#!/usr/bin/env python3
"""Build a searchable index of the literature briefs.

A brief that cannot be found is a brief that was not written. The digests pile
up one file per paper with no way to ask "what did we read about variance?", so
this walks the corpus and writes one table: slug, title, source, date, and the
first relevance line the digest itself produced.

Everything here is extracted by code. The model wrote the briefs; it does not
get to write the index too, because an index is exactly where a confident
mis-summary would be hardest to notice.

Raw papers stay out of git (hundreds of KB each, re-fetchable from the recorded
URL). The briefs are committed -- they are small, and they are the part that
took a GPU to produce.

    .venv/bin/python scripts/ops/lit_index.py
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIT = PROJECT_ROOT / "talks/lit"
RAW = LIT / "raw"
OUT = LIT / "INDEX.md"
KST = timezone(timedelta(hours=9))


def source_meta(slug: str) -> dict[str, str]:
    f = RAW / f"{slug}.source"
    if not f.is_file():
        return {}
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def first_relevance(text: str) -> str:
    """The digest's own first bullet under the relevance heading, verbatim."""
    m = re.search(r"##\s*5\.[^\n]*\n+(.*?)(?=\n##\s|\Z)", text, re.S)
    if not m:
        return "—"
    for line in m.group(1).splitlines():
        line = line.strip()
        if line.startswith("-"):
            return line.lstrip("- ").strip()[:160]
    return "—"


def title_of(slug: str, text: str) -> str:
    raw = RAW / f"{slug}.txt"
    if raw.is_file():
        for line in raw.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if 10 < len(line) < 120 and not line.startswith(("arXiv", "License", "Title")):
                return line
    return slug


def main() -> None:
    rows, pending = [], []
    for digest in sorted(LIT.glob("*_digest.md")):
        slug = digest.name[: -len("_digest.md")]
        text = digest.read_text(encoding="utf-8")
        meta = source_meta(slug)
        rows.append({
            "slug": slug,
            "title": title_of(slug, text),
            "url": meta.get("url", "—"),
            "fetched": meta.get("fetched_at", "—")[:10],
            "words": len((RAW / f"{slug}.txt").read_text(encoding="utf-8").split())
                     if (RAW / f"{slug}.txt").is_file() else 0,
            "relevance": first_relevance(text),
        })
    have = {r["slug"] for r in rows}
    for raw in sorted(RAW.glob("*.txt")):
        if raw.stem not in have:
            pending.append(raw.stem)

    stamp = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    lines = [
        "# 문헌 색인 (자동 생성)", "",
        f"- 갱신: {stamp} · 브리프 {len(rows)}건 · 요약 대기 {len(pending)}건",
        "- 생성: `scripts/ops/lit_index.py` (코드 추출만, 모델이 쓰지 않음)",
        "- **모든 브리프는 외부 주장이며 우리의 관측이 아니다**(`D-047`). 게이트·승격의 근거가",
        "  될 수 없고, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.",
        "- 원문은 `talks/lit/raw/`에 두고 git에 넣지 않는다. 기록된 URL로 다시 받을 수 있다.", "",
        "| 브리프 | 제목 | 원문 | 수집 | 분량 | 이 프로젝트와의 관련 (브리프 §5 첫 줄) |",
        "|---|---|---|---|---:|---|",
    ]
    for r in rows:
        url = f"[원문]({r['url']})" if r["url"].startswith("http") else "—"
        lines.append(
            f"| [`{r['slug']}`]({r['slug']}_digest.md) | {r['title']} | {url} | "
            f"{r['fetched']} | {r['words']:,}단어 | {r['relevance']} |")
    if pending:
        lines += ["", "## 요약 대기", ""] + [f"- `{s}`" for s in pending]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"색인: {OUT.relative_to(PROJECT_ROOT)} · 브리프 {len(rows)}건 · 대기 {len(pending)}건")


if __name__ == "__main__":
    main()
