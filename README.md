# jplawdb3

Japanese Law Database for AI Agents — 日本法令データベース（AIエージェント向け）

## Overview

AI エージェント（Claude Code 等）が日本の税法・法令を高速に検索・参照するためのローカルデータベースシステム。

## Status

🚧 Under construction


## Breaking Changes

- URL 上限を 1URL <= 10,000 tokens に統一。
- 旧単一本文URL（`.../{id}.txt`, `.../{id}.html`）は part URL へ破壊的変更。
- `ai-law-db/data/resolve.json` 等の巨大JSONは廃止。
