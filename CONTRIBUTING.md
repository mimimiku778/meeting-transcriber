# Contributing

## 開発環境

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # 本体＋開発ツール (pytest / ruff / pre-commit)
pre-commit install           # 任意: コミット時に ruff を自動実行
```

speakrs バックエンドを使う場合は `native/speakrs-diarizer` を `cargo build --release`
（`brew install openblas` が必要）。

## よく使うコマンド（Makefile）

| コマンド | 内容 |
|---|---|
| `make dev` | 開発依存をインストール（`pip install -e ".[dev]"`）|
| `make test` | pytest（torch/mlx 不要の純ロジック）|
| `make lint` | ruff check |
| `make format` | ruff の自動修正＋整形 |
| `make install` | `install.sh`（MCP登録／スキル配置／CLI）|

## コードスタイル

- ruff（`line-length = 120`、`select = E/F/I/B/UP`）。`make lint` が緑であること。
- 日本語コメントで「なぜ」を書く。型ヒントを付ける。
- 進捗ログは `print(..., flush=True)` で出す（`transcribe --watch` が tail する設計）。

## テストの方針

実際の文字起こし・diarization・声紋 embedding は GPU/モデルを要し CI で回せないため、
**純ロジック**（案件ストアの判定/学習、決定的正規化、話者ヒントのスコアリング）を `tests/` でカバーする。
embedding 計算はモックに差し替える（`tests/test_voiceprint.py` 参照）。CI は numpy・pyyaml だけで回る。

## ⚠️ 公開リポジトリの鉄則: 実名を入れない

実案件の固有名詞（顧客名・人名・社内略号・案件名）を、コード・コメント・例示・テンプレートに
**絶対に入れない**。例は中立名（山田／鈴木／A社／B社／myteam）のみ。実データ（組織図・声紋・用語）は
`~/.claude/meeting-contexts/` と `~/.claude/voiceprints/`（いずれもリポジトリ外）に置く。
**commit 前に必ず grep で実名混入を確認すること。**
