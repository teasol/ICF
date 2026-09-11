#!/usr/bin/env python3
"""
ICF Directory-based Inbox CLI (Maildir style)
Zero-dependency helper script for agent/human task dispatch, inspection, and completion.
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import shutil
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TALKS_DIR = os.path.join(BASE_DIR, "talks")
INBOX_DIR = os.path.join(TALKS_DIR, "inbox")
ARCHIVE_DIR = os.path.join(TALKS_DIR, "archive", "inbox")

ROLES = ["orca", "owl", "lime", "document", "platform", "kimds", "lead"]


def normalize_role(role: str) -> str:
    r = role.lower().lstrip("@")
    return "kimds" if r == "lead" else r


def get_kst_now() -> datetime.datetime:
    # KST (UTC + 9 hours)
    return datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.strip().lower())
    return re.sub(r"[-\s]+", "_", text)[:40]


def cmd_list(args) -> None:
    role = normalize_role(args.role)
    target_dir = os.path.join(INBOX_DIR, role)
    if not os.path.exists(target_dir):
        print(f"Unknown or non-existent inbox: {role}", file=sys.stderr)
        sys.exit(1)

    files = [f for f in sorted(os.listdir(target_dir)) if f.endswith(".md") and not f.startswith(".")]
    if not files:
        print(f"No pending tasks for @{role}.")
        return

    print(f"Pending tasks for @{role} ({len(files)} items):")
    for idx, fname in enumerate(files, 1):
        fpath = os.path.join(target_dir, fname)
        title = fname
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                for line in fp:
                    if line.startswith("# "):
                        title = line.strip("# \r\n")
                        break
        except Exception:
            pass
        print(f"  [{idx}] {fname} — {title}")


def cmd_check(args) -> None:
    role = normalize_role(args.role)
    target_dir = os.path.join(INBOX_DIR, role)
    if not os.path.exists(target_dir):
        sys.exit(1)

    files = [f for f in os.listdir(target_dir) if f.endswith(".md") and not f.startswith(".")]
    if files:
        print(f"FOUND {len(files)} pending task(s)")
        sys.exit(0)
    else:
        sys.exit(1)


def cmd_send(args) -> None:
    to_role = normalize_role(args.to_role)
    from_role = normalize_role(args.from_role)
    target_dir = os.path.join(INBOX_DIR, to_role)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    now = get_kst_now()
    timestamp_str = now.strftime("%Y-%m-%d_%H%M")
    date_display = now.strftime("%Y-%m-%d %H:%M KST")
    slug = slugify(args.title) if args.title else "task"
    filename = f"{timestamp_str}_from_{from_role}_{slug}.md"
    filepath = os.path.join(target_dir, filename)

    content = f"""# Task: {args.title}

- **Date**: {date_display}
- **From**: @{from_role}
- **To**: @{to_role}
- **Related Topic / RU**: `{args.topic if args.topic else "N/A"}`

## 📌 작업 목표 및 지침
{args.desc if args.desc else "- [ ] " + args.title}

## 📦 기대 산출물 및 회신 위치
- 관련 결과(수치, 로그, 변경 파일, 보고서)를 기록 또는 전달한 후, `python scripts/inbox.py done {filepath}`를 실행하여 완료 처리합니다.
"""
    with open(filepath, "w", encoding="utf-8") as fp:
        fp.write(content)

    print(f"Created ticket: {filepath}")


def cmd_done(args) -> None:
    filepath = os.path.abspath(args.ticket_path)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    parent_dir = os.path.dirname(filepath)
    role = os.path.basename(parent_dir)
    filename = os.path.basename(filepath)

    archive_target_dir = os.path.join(ARCHIVE_DIR, role)
    os.makedirs(archive_target_dir, exist_ok=True)
    dest_path = os.path.join(archive_target_dir, filename)

    shutil.move(filepath, dest_path)
    print(f"Archived ticket:\n  Source: {filepath}\n  Dest:   {dest_path}")


def main():
    parser = argparse.ArgumentParser(description="ICF Directory Inbox CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List pending tasks for a role")
    p_list.add_argument("role", choices=ROLES, help="Role name (orca, owl, lime, document, platform, kimds, lead)")
    p_list.set_defaults(func=cmd_list)

    # check
    p_check = subparsers.add_parser("check", help="Fast-fail check (exit 0 if tasks, exit 1 if empty)")
    p_check.add_argument("role", choices=ROLES, help="Role name")
    p_check.set_defaults(func=cmd_check)

    # send
    p_send = subparsers.add_parser("send", help="Send a task ticket to an inbox")
    p_send.add_argument("--to", dest="to_role", required=True, choices=ROLES, help="Target role")
    p_send.add_argument("--from", dest="from_role", required=True, help="Sender role/name")
    p_send.add_argument("--title", required=True, help="Short task title")
    p_send.add_argument("--topic", default="", help="Related topic file or RU card")
    p_send.add_argument("--desc", default="", help="Task details / checklist")
    p_send.set_defaults(func=cmd_send)

    # done
    p_done = subparsers.add_parser("done", help="Archive a completed ticket")
    p_done.add_argument("ticket_path", help="Path to the ticket markdown file")
    p_done.set_defaults(func=cmd_done)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
