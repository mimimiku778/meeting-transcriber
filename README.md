# Meeting Transcriber

Google Meet、Teams、DiscordなどのWeb会議をOBS等でスクリーンレコードした動画から、話者識別付き議事録を作成するツール。

## 機能

1. 動画から音声を抽出し、Whisperで文字起こし
2. pyannote-audioで話者を自動識別（発話者1、発話者2...）
3. 動画のフレームから参加者名を取得し、発話者名を置換
4. 対話的に議事録を作成

## 動作フロー

```
ユーザー: 「/path/to/meeting.mp4 の議事録を作成して」
    ↓
[文字起こし + 話者識別]
    ↓
出力: /path/to/meeting_transcript.txt
    発話者1 (00:15)
    それでは会議を始めます。

    発話者2 (00:23)
    よろしくお願いします。
    ↓
[次のアクション選択]
  A. 議事録を作成 → テンプレート確認、誤字修正、名前確認
  B. 文字起こし確認
  C. 発話者名を更新
    ↓
[名前確認時: 動画フレームから参加者名を取得]
    ↓
[議事録生成]
```

## 必要環境

- macOS (Apple Silicon)
- Python 3.10+
- Homebrew

## 依存関係

`install.sh` が以下を自動インストール:

| パッケージ | 用途 |
|-----------|------|
| ffmpeg | 動画から音声抽出 |
| mlx-whisper | 音声→テキスト変換（Apple Silicon最適化） |
| pyannote.audio | 話者識別 |
| torch / torchaudio | 機械学習基盤 |
| opencv-python | 動画フレーム抽出 |

`install.sh` が自動設定:

- Claude Code: MCPサーバー + `/transcribe-meeting` スキル
- Claude Desktop: MCPサーバー
- CLI: `transcribe` コマンド

## セットアップ

### 1. Hugging Face権限（3つとも必須）

各リンクで「Access repository」をクリック:

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-community-1

### 2. Hugging Faceトークン取得

https://huggingface.co/settings/tokens

### 3. インストール

```bash
git clone <repo-url>
cd meeting-transcriber
./install.sh
```

ffmpeg、Python依存パッケージは自動インストールされます。

### 4. HF_TOKEN設定

```bash
cp .env.example .env
# .env を編集してトークンを設定
```

または `install.sh` 実行時に入力すると自動で `.env` が作成されます。

## 使い方

### CLI

```bash
transcribe /path/to/video.mov
transcribe /path/to/video.mov --speakers 3  # 話者数指定
```

### Claude Code

```
/transcribe-meeting /path/to/video.mov
```

### Claude Desktop

再起動後:
```
「/path/to/meeting.mp4 の議事録を作成して」
```

## アンインストール

```bash
# Claude Code
claude mcp remove meeting-transcriber -s user
rm ~/.claude/commands/transcribe-meeting.md

# Claude Desktop: claude_desktop_config.json から meeting-transcriber を削除

# CLI
rm ~/.local/bin/transcribe
```

## ライセンス

MIT
