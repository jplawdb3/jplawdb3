#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def tok_len(text: str) -> int:
    return len(ENC.encode(text))


def json_text(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def split_text_by_tokens(content: str, target: int) -> List[str]:
    lines = content.splitlines(keepends=True)
    if not lines:
        return [""]

    parts: List[str] = []
    cur = ""

    def flush() -> None:
        nonlocal cur
        if cur:
            parts.append(cur)
            cur = ""

    for line in lines:
        if not cur:
            if tok_len(line) <= target:
                cur = line
                continue
            ltoks = ENC.encode(line)
            for i in range(0, len(ltoks), target):
                parts.append(ENC.decode(ltoks[i : i + target]))
            continue

        candidate = cur + line
        if tok_len(candidate) <= target:
            cur = candidate
            continue

        flush()
        if tok_len(line) <= target:
            cur = line
        else:
            ltoks = ENC.encode(line)
            for i in range(0, len(ltoks), target):
                parts.append(ENC.decode(ltoks[i : i + target]))

    flush()
    return parts


def _split_single_block(block: str) -> List[str]:
    m = re.match(r"(<div[^>]*>)([\s\S]*?)(</div>)$", block)
    if not m:
        return [block]
    head, body, tail = m.group(1), m.group(2), m.group(3)
    paras = re.findall(r"<p[^>]*>[\s\S]*?</p>", body)
    if len(paras) <= 1:
        toks = ENC.encode(body)
        chunks: List[str] = []
        for i in range(0, len(toks), 3000):
            chunk = ENC.decode(toks[i : i + 3000])
            chunks.append(f"{head}<p>{html.escape(chunk)}</p>{tail}")
        return chunks or [block]
    return [f"{head}{p}{tail}" for p in paras]


def split_html_by_blocks(content: str, target: int, max_tokens: int) -> List[str]:
    if tok_len(content) <= max_tokens:
        return [content]

    pattern = re.compile(r"<div data-paragraph=\"[^\"]+\"[\s\S]*?</div>", re.S)
    matches = list(pattern.finditer(content))
    if not matches:
        toks = ENC.encode(content)
        chunks = [ENC.decode(toks[i : i + target]) for i in range(0, len(toks), target)]
        return chunks

    prefix = content[: matches[0].start()]
    suffix = content[matches[-1].end() :]
    blocks = [m.group(0) for m in matches]

    expanded: List[str] = []
    for b in blocks:
        single = prefix + b + suffix
        if tok_len(single) > max_tokens:
            expanded.extend(_split_single_block(b))
        else:
            expanded.append(b)

    parts: List[str] = []
    cur_blocks: List[str] = []
    for block in expanded:
        test = prefix + "".join(cur_blocks + [block]) + suffix
        if cur_blocks and tok_len(test) > target:
            parts.append(prefix + "".join(cur_blocks) + suffix)
            cur_blocks = [block]
        else:
            cur_blocks.append(block)
    if cur_blocks:
        parts.append(prefix + "".join(cur_blocks) + suffix)

    fixed: List[str] = []
    for p in parts:
        if tok_len(p) <= max_tokens:
            fixed.append(p)
        else:
            toks = ENC.encode(p)
            fixed.extend(ENC.decode(toks[i : i + target]) for i in range(0, len(toks), target))
    return fixed


def write_parts(out_dir: Path, ext: str, parts: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, part in enumerate(parts, start=1):
        (out_dir / f"part-{i:03d}{ext}").write_text(part, encoding="utf-8")


def convert_text_tree(db_root: Path, target: int, max_tokens: int) -> None:
    text_root = db_root / "text"
    files = [p for p in text_root.glob("*/*.txt") if p.is_file()]
    for p in files:
        code = p.parent.name
        item = p.stem
        content = p.read_text(encoding="utf-8", errors="ignore")
        parts = split_text_by_tokens(content, target)
        if any(tok_len(x) > max_tokens for x in parts):
            raise RuntimeError(f"text split failed: {p}")
        out_dir = p.parent / item
        write_parts(out_dir, ".txt", parts)
        p.unlink()


def convert_html_tree(db_root: Path, target: int, max_tokens: int) -> None:
    enh_root = db_root / "enhanced"
    files = [
        p
        for p in enh_root.glob("*/*.html")
        if p.is_file() and p.name != "index.html"
    ]
    for p in files:
        code = p.parent.name
        item = p.stem
        content = p.read_text(encoding="utf-8", errors="ignore")
        parts = split_html_by_blocks(content, target, max_tokens)
        if any(tok_len(x) > max_tokens for x in parts):
            raise RuntimeError(f"html split failed: {p}")
        out_dir = p.parent / item
        write_parts(out_dir, ".html", parts)
        p.unlink()


def build_manifest(db_root: Path, domain: str) -> None:
    data_parts = db_root / "data" / "parts" / domain
    if data_parts.exists():
        for p in sorted(data_parts.rglob("*.json"), reverse=True):
            p.unlink()

    text_root = db_root / "text"
    html_root = db_root / "enhanced"

    keys = set()
    for code_dir in text_root.iterdir():
        if code_dir.is_dir():
            for item_dir in code_dir.iterdir():
                if item_dir.is_dir():
                    keys.add((code_dir.name, item_dir.name))
    for code_dir in html_root.iterdir():
        if code_dir.is_dir():
            for item_dir in code_dir.iterdir():
                if item_dir.is_dir() and item_dir.name != "index.html":
                    keys.add((code_dir.name, item_dir.name))

    for code, item in sorted(keys):
        tdir = text_root / code / item
        hdir = html_root / code / item
        tparts = sorted(x.name for x in tdir.glob("part-*.txt")) if tdir.is_dir() else []
        hparts = sorted(x.name for x in hdir.glob("part-*.html")) if hdir.is_dir() else []
        obj = {
            "code": code,
            "id": item,
            "text_parts": len(tparts),
            "html_parts": len(hparts),
            "text_parts_files": tparts,
            "html_parts_files": hparts,
            "text_part_url_template": f"text/{code}/{item}/part-{{part}}.txt",
            "html_part_url_template": f"enhanced/{code}/{item}/part-{{part}}.html",
        }
        out = data_parts / code / f"{item}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, obj)


def rebuild_enhanced_index_law(db_root: Path) -> None:
    laws = json.loads((db_root / "data" / "laws.json").read_text(encoding="utf-8"))
    title_map = {x["code"]: x["title"] for x in laws.get("laws", [])}

    for code_dir in sorted((db_root / "enhanced").iterdir()):
        if not code_dir.is_dir():
            continue
        code = code_dir.name
        title = title_map.get(code, code)
        html_text = f"""<!doctype html>
<html lang=\"ja\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title} ({code})</title>
</head>
<body>
  <h1>{title}</h1>
  <p>このページは軽量版です。本文は part URL を利用してください。</p>
  <ul>
    <li><a href=\"../../llms.txt\">llms.txt</a></li>
    <li><a href=\"../../quickstart.txt\">quickstart.txt</a></li>
    <li><a href=\"../../data/resolve_lite/{code}.json\">resolve_lite/{code}.json</a></li>
  </ul>
  <p>本文テンプレート: <code>../../text/{code}/{{article}}/part-{{part}}.txt</code></p>
  <p>根拠テンプレート: <code>./{{article}}/part-{{part}}.html</code></p>
</body>
</html>
"""
        (code_dir / "index.html").write_text(html_text, encoding="utf-8")


def rebuild_enhanced_index_tsutatsu(db_root: Path) -> None:
    idx = json.loads((db_root / "data" / "resolve_lite" / "index.json").read_text(encoding="utf-8"))
    docs = idx.get("docs", {})
    title_map = {k: v.get("title", k) for k, v in docs.items()}

    for code_dir in sorted((db_root / "enhanced").iterdir()):
        if not code_dir.is_dir():
            continue
        code = code_dir.name
        title = title_map.get(code, code)
        html_text = f"""<!doctype html>
<html lang=\"ja\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title} ({code})</title>
</head>
<body>
  <h1>{title}</h1>
  <p>このページは軽量版です。本文は part URL を利用してください。</p>
  <ul>
    <li><a href=\"../../llms.txt\">llms.txt</a></li>
    <li><a href=\"../../quickstart.txt\">quickstart.txt</a></li>
    <li><a href=\"../../data/resolve_lite/{code}.json\">resolve_lite/{code}.json</a></li>
  </ul>
  <p>本文テンプレート: <code>../../text/{code}/{{item_id}}/part-{{part}}.txt</code></p>
  <p>根拠テンプレート: <code>./{{item_id}}/part-{{part}}.html</code></p>
</body>
</html>
"""
        (code_dir / "index.html").write_text(html_text, encoding="utf-8")


def update_resolve_templates_law(db_root: Path) -> None:
    rdir = db_root / "data" / "resolve_lite"
    idx_path = rdir / "index.json"
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    idx["article_url_template"] = "enhanced/{law_code}/{article}/part-{part}.html"
    idx["text_url_template"] = "text/{law_code}/{article}/part-{part}.txt"
    idx["parts_index_template"] = "data/parts/law/{law_code}/{article}.json"
    write_json(idx_path, idx)

    for p in rdir.glob("*.json"):
        if p.name == "index.json":
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        data["article_url_template"] = "enhanced/{law_code}/{article}/part-{part}.html"
        data["text_url_template"] = "text/{law_code}/{article}/part-{part}.txt"
        data["parts_index_template"] = "data/parts/law/{law_code}/{article}.json"
        write_json(p, data)


def update_resolve_templates_tsutatsu(db_root: Path) -> None:
    rdir = db_root / "data" / "resolve_lite"
    idx_path = rdir / "index.json"
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    idx["item_url_template"] = "enhanced/{doc_code}/{item_id}/part-{part}.html"
    idx["text_url_template"] = "text/{doc_code}/{item_id}/part-{part}.txt"
    idx["parts_index_template"] = "data/parts/tsutatsu/{doc_code}/{item_id}.json"
    write_json(idx_path, idx)

    for p in rdir.glob("*.json"):
        if p.name == "index.json":
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        data["item_url_template"] = "enhanced/{doc_code}/{item_id}/part-{part}.html"
        data["text_url_template"] = "text/{doc_code}/{item_id}/part-{part}.txt"
        data["parts_index_template"] = "data/parts/tsutatsu/{doc_code}/{item_id}.json"
        write_json(p, data)


def split_json_object_entries(
    static_obj: dict, entries: List[Tuple[str, dict]], key_name: str, target: int
) -> List[dict]:
    parts: List[dict] = []
    cur: Dict[str, dict] = {}

    for k, v in entries:
        candidate = dict(static_obj)
        candidate[key_name] = dict(cur)
        candidate[key_name][k] = v
        if cur and tok_len(json_text(candidate)) > target:
            obj = dict(static_obj)
            obj[key_name] = dict(cur)
            parts.append(obj)
            cur = {k: v}
        else:
            cur[k] = v
    if cur:
        obj = dict(static_obj)
        obj[key_name] = dict(cur)
        parts.append(obj)
    return parts


def split_large_jsons(root: Path, max_tokens: int, target: int) -> None:
    law_db = root / "ai-law-db"
    tsu_db = root / "ai-tsutatsu-db"

    # Remove monolith JSONs (policy: abolish)
    for rel in [
        "ai-law-db/data/resolve.json",
        "ai-law-db/data/resolve.min.json",
        "ai-law-db/data/resolve_lite.json",
    ]:
        p = root / rel
        if p.exists():
            p.unlink()

    # Split ai-law-db/data/resolve_meta/*.json if needed
    rmeta = law_db / "data" / "resolve_meta"
    for p in sorted(rmeta.glob("*.json")):
        if p.name == "index.json":
            continue
        raw = p.read_text(encoding="utf-8", errors="ignore")
        if tok_len(raw) <= max_tokens:
            continue
        data = json.loads(raw)
        articles = data.get("articles", {})
        static = {k: v for k, v in data.items() if k != "articles"}
        entries = sorted(articles.items(), key=lambda x: x[0])
        parts = split_json_object_entries(static, entries, "articles", target)

        parts_dir = law_db / "data" / "resolve_meta_parts" / p.stem
        parts_dir.mkdir(parents=True, exist_ok=True)
        part_index = []
        for i, part in enumerate(parts, start=1):
            fn = f"part-{i:03d}.json"
            out = parts_dir / fn
            write_json(out, part)
            part_index.append({"file": fn, "count": len(part.get("articles", {}))})

        idx_obj = {
            "law_code": p.stem,
            "parts_count": len(part_index),
            "parts": part_index,
            "part_url_template": f"data/resolve_meta_parts/{p.stem}/part-{{part}}.json",
        }
        write_json(parts_dir / "index.json", idx_obj)

        stub = dict(static)
        stub["articles"] = {}
        stub["split"] = True
        stub["parts_index"] = f"data/resolve_meta_parts/{p.stem}/index.json"
        write_json(p, stub)

    # Split large ai-tsutatsu-db/data/resolve_lite/*.json (items list)
    rlite = tsu_db / "data" / "resolve_lite"
    for p in sorted(rlite.glob("*.json")):
        if p.name == "index.json":
            continue
        raw = p.read_text(encoding="utf-8", errors="ignore")
        if tok_len(raw) <= max_tokens:
            continue
        data = json.loads(raw)
        items = data.get("items", [])
        static = {k: v for k, v in data.items() if k != "items"}

        parts_dir = tsu_db / "data" / "resolve_lite_parts" / p.stem
        parts_dir.mkdir(parents=True, exist_ok=True)

        parts: List[dict] = []
        cur: List[str] = []
        for it in items:
            cand = dict(static)
            cand["items"] = cur + [it]
            if cur and tok_len(json_text(cand)) > target:
                obj = dict(static)
                obj["items"] = list(cur)
                parts.append(obj)
                cur = [it]
            else:
                cur.append(it)
        if cur:
            obj = dict(static)
            obj["items"] = list(cur)
            parts.append(obj)

        part_index = []
        for i, part in enumerate(parts, start=1):
            part["part"] = i
            fn = f"part-{i:03d}.json"
            write_json(parts_dir / fn, part)
            part_index.append({"file": fn, "count": len(part.get("items", []))})

        idx_obj = {
            "doc_code": p.stem,
            "parts_count": len(part_index),
            "parts": part_index,
            "part_url_template": f"data/resolve_lite_parts/{p.stem}/part-{{part}}.json",
        }
        write_json(parts_dir / "index.json", idx_obj)

        stub = dict(static)
        stub["items"] = []
        stub["split"] = True
        stub["parts_index"] = f"data/resolve_lite_parts/{p.stem}/index.json"
        write_json(p, stub)


def rewrite_docs(root: Path) -> None:
    law_qs = """# ai-law-db quickstart (10k制約版)

- 原則: 1URL <= 10,000 tokens
- 本文テンプレート: `text/{law_code}/{article}/part-{part}.txt`
- 根拠テンプレート: `enhanced/{law_code}/{article}/part-{part}.html#p{n}`

## 手順
1) `data/law_aliases.json` で `law_code` を確定
2) `data/resolve_lite/{law_code}.json` で `article` を確定
3) `data/parts/law/{law_code}/{article}.json` で分割数を確認
4) `part-001` から順に読む

## 備考
- 旧 `text/{law_code}/{article}.txt` は廃止
- 旧 `enhanced/{law_code}/{article}.html` は廃止
"""
    law_llms = """# AI向け 日本法（10k制約版）

- 入口: `quickstart.txt`
- 最軽量索引: `data/resolve_lite/index.json`
- 条文タイトル索引: `data/resolve_meta_corp/index.json`
- 分割索引: `data/parts/law/{law_code}/{article}.json`

## URL
- text: `text/{law_code}/{article}/part-{part}.txt`
- html: `enhanced/{law_code}/{article}/part-{part}.html`

## 方針
- 1URL <= 10,000 tokens
- 軽い索引で絞る -> 本文 part へ最短到達
"""

    tsu_qs = """# ai-tsutatsu-db quickstart (10k制約版)

- 原則: 1URL <= 10,000 tokens
- 本文テンプレート: `text/{doc_code}/{item_id}/part-{part}.txt`
- 根拠テンプレート: `enhanced/{doc_code}/{item_id}/part-{part}.html#p{n}`

## 手順
1) `data/doc_aliases.json` で `doc_code` を確定
2) `data/resolve_lite/{doc_code}.json` で `item_id` を確定
3) `data/parts/tsutatsu/{doc_code}/{item_id}.json` で分割数を確認
4) `part-001` から順に読む

## 備考
- 旧 `text/{doc_code}/{item_id}.txt` は廃止
- 旧 `enhanced/{doc_code}/{item_id}.html` は廃止
"""
    tsu_llms = """# AI向け 日本税務通達（10k制約版）

- 入口: `quickstart.txt`
- 最軽量索引: `data/resolve_lite/index.json`
- 分割索引: `data/parts/tsutatsu/{doc_code}/{item_id}.json`

## URL
- text: `text/{doc_code}/{item_id}/part-{part}.txt`
- html: `enhanced/{doc_code}/{item_id}/part-{part}.html`

## 方針
- 1URL <= 10,000 tokens
- 軽量索引 -> 本文 part の順で読む
"""

    root_llms3 = """# jplawdb3 llms3

## 設計原則
- 1 URL <= 10,000 tokens
- 軽い索引で絞る -> 本文 part へ最短到達
- 巨大単一JSONは公開しない

## 入口
- `ai-law-db/quickstart.txt`
- `ai-tsutatsu-db/quickstart.txt`

## 主要テンプレート
- law text: `ai-law-db/text/{law_code}/{article}/part-{part}.txt`
- tsutatsu text: `ai-tsutatsu-db/text/{doc_code}/{item_id}/part-{part}.txt`
"""

    (root / "ai-law-db" / "quickstart.txt").write_text(law_qs, encoding="utf-8")
    (root / "ai-law-db" / "llms.txt").write_text(law_llms, encoding="utf-8")
    (root / "ai-tsutatsu-db" / "quickstart.txt").write_text(tsu_qs, encoding="utf-8")
    (root / "ai-tsutatsu-db" / "llms.txt").write_text(tsu_llms, encoding="utf-8")
    (root / "llms3.txt").write_text(root_llms3, encoding="utf-8")

    readme = root / "README.md"
    txt = readme.read_text(encoding="utf-8", errors="ignore")
    if "## Breaking Changes" not in txt:
        txt += "\n\n## Breaking Changes\n\n- URL 上限を 1URL <= 10,000 tokens に統一。\n"
        txt += "- 旧単一本文URL（`.../{id}.txt`, `.../{id}.html`）は part URL へ破壊的変更。\n"
        txt += "- `ai-law-db/data/resolve.json` 等の巨大JSONは廃止。\n"
        readme.write_text(txt, encoding="utf-8")


def build_sitemaps(db_root: Path, base_url: str, target_tokens: int, max_tokens: int) -> None:
    # Collect public files
    files: List[Path] = []
    for p in db_root.rglob("*"):
        if not p.is_file():
            continue
        if "/sitemaps/" in p.as_posix():
            continue
        if p.name == "sitemap.xml":
            continue
        if p.suffix.lower() not in {".html", ".txt", ".json", ".xml"}:
            continue
        files.append(p)
    files.sort()

    urls = [f"{base_url}/{p.relative_to(db_root).as_posix()}" for p in files]
    urls.insert(0, f"{base_url}/")

    sitemaps_dir = db_root / "sitemaps"
    if sitemaps_dir.exists():
        for old in sorted(sitemaps_dir.glob("*.xml")):
            old.unlink()
    sitemaps_dir.mkdir(parents=True, exist_ok=True)

    head = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n"
    foot = "</urlset>\n"

    shards: List[str] = []
    current: List[str] = []

    def xml_for(lines: List[str]) -> str:
        return head + "".join(lines) + foot

    for u in urls:
        line = f"  <url><loc>{u}</loc></url>\n"
        cand = xml_for(current + [line])
        if current and tok_len(cand) > target_tokens:
            shard_name = f"sitemap-{len(shards)+1:04d}.xml"
            (sitemaps_dir / shard_name).write_text(xml_for(current), encoding="utf-8")
            shards.append(shard_name)
            current = [line]
        else:
            current.append(line)

    if current:
        shard_name = f"sitemap-{len(shards)+1:04d}.xml"
        (sitemaps_dir / shard_name).write_text(xml_for(current), encoding="utf-8")
        shards.append(shard_name)

    idx_head = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<sitemapindex xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n"
    idx_foot = "</sitemapindex>\n"
    idx_lines = [
        f"  <sitemap><loc>{base_url}/sitemaps/{name}</loc></sitemap>\n" for name in shards
    ]
    idx_xml = idx_head + "".join(idx_lines) + idx_foot
    if tok_len(idx_xml) > max_tokens:
        raise RuntimeError(f"sitemap index too large: {db_root}")

    (db_root / "sitemap.xml").write_text(idx_xml, encoding="utf-8")

    # Hard guard
    for name in shards:
        xml = (sitemaps_dir / name).read_text(encoding="utf-8")
        if tok_len(xml) > max_tokens:
            raise RuntimeError(f"sitemap shard too large: {name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Refactor datasets to <=10k tokens per URL")
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--max", type=int, default=10000)
    ap.add_argument("--target", type=int, default=9000)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    law_db = root / "ai-law-db"
    tsu_db = root / "ai-tsutatsu-db"

    convert_text_tree(law_db, args.target, args.max)
    convert_html_tree(law_db, args.target, args.max)
    convert_text_tree(tsu_db, args.target, args.max)
    convert_html_tree(tsu_db, args.target, args.max)

    build_manifest(law_db, "law")
    build_manifest(tsu_db, "tsutatsu")

    update_resolve_templates_law(law_db)
    update_resolve_templates_tsutatsu(tsu_db)

    split_large_jsons(root, args.max, args.target)

    rebuild_enhanced_index_law(law_db)
    rebuild_enhanced_index_tsutatsu(tsu_db)

    rewrite_docs(root)

    build_sitemaps(law_db, "https://jplawdb3.github.io/jplawdb3/ai-law-db", args.target, args.max)
    build_sitemaps(tsu_db, "https://jplawdb3.github.io/jplawdb3/ai-tsutatsu-db", args.target, args.max)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
