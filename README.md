# Meeting Transcriber

macOS (Apple Silicon) 専用のClaude Codeスキル + MCPサーバー。

OBSなどで録画したGoogle Meet / Teams / Zoom / Discord等の会議動画から、**話者識別付きの文字起こし**を生成し、議事録・要約・Q&A抽出など指定したフォーマットでMarkdown文書を作成する。

## 特徴

- **高精度話者識別**: pyannote.audio (v2) でGPU (MPS) を活用した話者識別。従来版 (SpeechBrain) も選択可
- **Apple Silicon最適化**: mlx-Whisperで高速な音声認識、pyannote.audioはMPS (Metal) で高速推論
- **画面キャプチャ**: 動画フレームを画像として抽出し、Claudeが視覚的に内容を理解。Apple Vision OCRで参加者名や共有画面のテキストも高速・高精度に抽出

## 使用技術

| 技術 | 用途 |
|------|------|
| mlx-Whisper | 音声認識（Apple Silicon最適化） |
| pyannote.audio | 話者識別 v2（高精度、MPS対応）← デフォルト |
| SpeechBrain | 話者識別 v1（ECAPA-TDNN、従来版） |
| macOS Vision | 画面OCR |
| ffmpeg | 音声抽出 |

## 使い方

### Claude Code（対話的に議事録作成）

```
/transcribe-meeting
```

起動すると3つのモードから選択:

```
A. 新しい動画を文字起こしする
B. 既存の議事録を編集・更新する
C. 過去の議事録を検索・まとめ生成する
```

**A. 新規文字起こし:**
動画指定 → モデル選択 → 文字起こし実行 → 自動でタイトル付きフォルダに整理 → 発話者名を画面から取得して置換 → 議事録生成

**B. 既存編集:**
フォルダ指定 → 議事録の修正/再生成/誤字修正/フレーム追加抽出

**C. 検索:**
複数の会議を横断してキーワード検索 → 該当箇所と動画/音声のタイムスタンプを特定 → 複数議事録からまとめ生成

### CLI（文字起こしのみ）

```bash
transcribe /path/to/video.mov                          # デフォルト(medium + simple-diarizer)
transcribe /path/to/video.mov --diarization-v2          # pyannote.audio v2で高精度話者識別
transcribe /path/to/video.mov --diarization-v2 --speakers 3  # 話者数指定でさらに精度向上
transcribe /path/to/video.mov -m small-4bit             # 最速
```

出力: `video_transcript.txt`（話者識別付きテキスト）

## モデル

| モデル | サイズ | 特徴 |
|--------|--------|------|
| small-4bit | ~120MB | 最速 |
| small | 466MB | 高速 |
| medium | 1.5GB | バランス（デフォルト） |
| large-v3 | 3GB | 最高精度 |

※ M4 Max 64GBでmediumモデル使用時、1時間の動画で約5〜10分程度

## インストール

```bash
git clone <repo-url>
cd meeting-transcriber
./install.sh
```

自動で以下を設定:
- ffmpeg（Homebrew）
- Python依存パッケージ（venv内）
- Claude Code MCPサーバー + スキル
- `transcribe` CLIコマンド

## アップデート

```bash
cd ~/.claude/mcp-servers/meeting-transcriber
git pull
./install.sh
```

Claude Codeを再起動してMCPサーバーを反映。

## アンインストール

```bash
# MCPサーバー削除
claude mcp remove meeting-transcriber -s user

# スキル削除
rm ~/.claude/commands/transcribe-meeting.md

# CLIコマンド削除
rm ~/.local/bin/transcribe

# リポジトリ削除（必要に応じて）
rm -rf /path/to/meeting-transcriber
```

## 構成

```
~/.claude/commands/transcribe-meeting.md  # スキル定義
    ↓ 呼び出し
MCPサーバー (meeting-transcriber)
    ├── transcribe_meeting     # 文字起こし + 話者識別
    ├── extract_video_frame    # フレーム抽出 + OCR
    ├── update_speaker_names   # 発話者名置換
    ├── read_transcript        # 文字起こし読み込み
    └── finalize_meeting_files # ファイル整理
```

## ファイル構成

```
src/meeting_transcriber/
  server.py          # MCPサーバー
  cli.py             # CLIエントリーポイント
  transcriber.py     # mlx-Whisper文字起こし
  diarization.py     # SpeechBrain話者識別（v1・従来版）
  diarization_v2.py  # pyannote.audio話者識別（v2・高精度）
  frame_extractor.py # OpenCV + Vision OCR
```
