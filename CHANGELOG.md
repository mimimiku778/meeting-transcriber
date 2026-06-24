# Changelog

このプロジェクトの主な変更点を記録する。形式は [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/)、
バージョニングは [Semantic Versioning](https://semver.org/lang/ja/) に従う。

## [0.2.0]

### Added
- **案件コンテキストストア**（`~/.claude/meeting-contexts/<slug>.yaml`、声紋と同一 slug で連結）。
  動画パス・フレームOCR・声紋一致から案件を自動判定し、ユーザーの議事録修正を書き戻して育てる
  （修正ベース学習）。MCPツール `identify_project` / `resolve_speakers` / `upsert_project_context` / `list_projects`。
- **話者同一性の解決**: 声紋クラスタ類似度（過分割＝同一人物が別ラベルに分裂した統合候補）・
  混在検出（過少分割＝1ラベルに別人混在）・登録済み声紋による区間単位の実名照合。
  `transcribe_meeting` / `resolve_speakers` が `<transcript>_speakers.json` を出力。
- `extract_video_frames`: 複数フレームを動画1回オープンで一括抽出＋OCR並列。
- `tests/`（pytest・torch/mlx 非依存の純ロジック）、ruff/pytest 設定、Makefile、CI、pre-commit。

### Changed
- 声紋 embedding を MPS 既定化＋1パス化で高速化。
- enroll の話者分離を transcribe と同じ speakrs に統一（高速化＋話者ラベル整合）。
- 音声抽出を動画 stem 単位でキャッシュ（transcribe/enroll/resolve の重複抽出を排除）。
- README を刷新（ゼロお膳立て＋修正ベース学習の設計を明記）。

### Fixed
- `Inference` に文字列モデル名を渡すと `'str' object has no attribute 'eval'` で落ちるバグを修正
  （`Model` を明示ロードして渡す）。

### Removed
- 従来版 SpeechBrain（v1）話者識別の記載と、未使用の ECAPA-TDNN モデルを削除。

## [0.1.0]

- 初版: mlx-Whisper 文字起こし＋ speakrs/pyannote 話者識別＋動画フレーム OCR＋声紋エンロールメント。
