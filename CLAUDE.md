# Meeting Transcriber

Web会議の録画動画から話者識別付き議事録を作成するMCPサーバー & CLIツール。

> ⚠️ **公開リポジトリ**: 実案件の固有名詞（顧客名・人名・社内略号・案件名）を絶対に入れない。
> コード・コメント・例示・テンプレートはすべて中立名（山田/鈴木/A社/B社/myteam 等）のみ。
> 実データ（組織図・声紋・用語）は `~/.claude/meeting-contexts/` と `~/.claude/voiceprints/`
> （いずれもリポジトリ外）に置く。commit 前に実名が混入していないか必ず grep で確認すること。

## 目的

OBS等で録画したGoogle Meet/Teams/Discord等の会議動画を入力として:
1. mlx-Whisperで文字起こし（Apple Silicon最適化）
2. SpeechBrainで話者識別
3. 動画フレームから参加者名を取得して置換
4. 議事録を出力

## 構成

```
src/meeting_transcriber/
  server.py        # MCPサーバー（ツール提供）
  cli.py           # CLIエントリーポイント
  transcriber.py   # Whisperによる文字起こし
  diarization_speakrs.py # 話者識別（speakrs / CoreML 既定）
  diarization_v2.py # 話者識別（pyannote.audio）
  voiceprint.py    # 声紋 登録/識別＋クラスタ類似度・混在検出・区間照合（~/.claude/voiceprints/、MPS既定）
  context_store.py # 案件コンテキストの永続ストア＋自動判定＋修正ベース学習（~/.claude/meeting-contexts/）
  context_loader.py # 案件yaml→ASR固有名詞/正規化/議事録プロンプト展開
  frame_extractor.py # 動画フレーム抽出＋OCR

skills/
  transcribe-meeting.md  # Claude Code用スキル（ゼロお膳立て・自動判定・修正ベース学習）

templates/project.context.template.yaml  # 案件コンテキストの雛形
install.sh       # セットアップスクリプト
```

## 育てる2ストア（声紋と案件を同一 slug で連結）

- `~/.claude/voiceprints/<slug>.json` … 話者声紋。`enroll`/`identify`/区間照合。MPS既定（速度）。
- `~/.claude/meeting-contexts/<slug>.yaml` … 案件知識（組織図・ロスター・用語・帰属ルール・
  判定シグナル・議事録の取捨 minutes_preferences）。`identify_project`/`upsert_project_context`。
- いずれもローカルのみ・リポジトリ外。固有名詞は公開リポジトリに入れない。

## MCPツール

- `transcribe_meeting` - 動画から文字起こし+話者識別（`project`/`voiceprint_profile` で実名化＋
  話者同一性ヒントを `<transcript>_speakers.json` に出力）
- `enroll_voiceprints` - 発話者N→実名マッピングから声紋を登録/更新（議事録の後に回す）
- `extract_video_frame` - 指定秒のフレームを抽出＋OCR
- `extract_video_frames` - 複数時刻を動画1回オープンで一括抽出＋OCR並列（複数枚はこちらが高速）
- `update_speaker_names` - 発話者名を置換
- `read_transcript` - 文字起こし結果を読み込み
- `finalize_meeting_files` - タイトル付きディレクトリへ整理
- `identify_project` - 動画パス/フレームOCR/声紋から案件を自動判定
- `resolve_speakers` - 話者同一性ヒント（過分割の統合候補/過少分割の混在検出/区間照合）を算出
- `upsert_project_context` - ユーザーの議事録修正を案件ストアへ書き戻して育てる（修正ベース学習）
- `list_projects` - 登録済み案件一覧

## 開発

```bash
# 仮想環境
source .venv/bin/activate

# 依存関係
pip install -e .

# CLI実行
transcribe /path/to/video.mov
```
