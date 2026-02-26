#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import tiktoken


def iter_files(root: Path, exts: set[str]) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit token counts for public files.")
    ap.add_argument("root", nargs="?", default=".", help="Repository root")
    ap.add_argument("--max", type=int, default=10000, help="Token limit")
    ap.add_argument(
        "--ext",
        action="append",
        default=[".txt", ".html", ".json", ".xml"],
        help="File extension to include (repeatable)",
    )
    ap.add_argument("--json", dest="json_out", help="Write detailed report JSON")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    exts = {e if e.startswith(".") else f".{e}" for e in args.ext}
    enc = tiktoken.get_encoding("cl100k_base")

    rows: list[dict] = []
    for p in iter_files(root, exts):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        tokens = len(enc.encode(text))
        rows.append(
            {
                "path": p.relative_to(root).as_posix(),
                "tokens": tokens,
                "bytes": p.stat().st_size,
                "ext": p.suffix.lower(),
            }
        )

    rows.sort(key=lambda x: x["tokens"], reverse=True)
    over = [r for r in rows if r["tokens"] > args.max]

    print(f"files_total\t{len(rows)}")
    print(f"over_limit\t{len(over)}")
    print(f"max_tokens\t{rows[0]['tokens'] if rows else 0}")
    print("top_over_limit")
    for r in over[:100]:
        print(f"{r['tokens']}\t{r['bytes']}\t{r['path']}")

    if args.json_out:
        out = {
            "root": str(root),
            "limit": args.max,
            "files_total": len(rows),
            "over_limit": len(over),
            "max_tokens": rows[0]["tokens"] if rows else 0,
            "over_files": over,
        }
        Path(args.json_out).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return 1 if over else 0


if __name__ == "__main__":
    raise SystemExit(main())
