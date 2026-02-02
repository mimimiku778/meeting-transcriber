# Meeting Transcriber

Google Meet、Teams、DiscordなどのWeb会議をOBS等でスクリーンレコードした動画から、話者識別付き議事録を作成するツール。

## 機能

1. 動画から音声を抽出し、mlx-Whisperで文字起こし（Apple Silicon最適化）
2. SpeechBrainで話者を自動識別（発話者1、発話者2...）
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
| simple-diarizer | 話者識別（SpeechBrain ECAPA-TDNN） |
| torch / torchaudio | 機械学習基盤 |
| opencv-python | 動画フレーム抽出 |

`install.sh` が自動設定:

- Claude Code: MCPサーバー + `/transcribe-meeting` スキル
- Claude Desktop: MCPサーバー
- CLI: `transcribe` コマンド

## セットアップ

### 1. インストール

```bash
git clone <repo-url>
cd meeting-transcriber
./install.sh
```

ffmpeg、Python依存パッケージは自動インストールされます。

## 使い方

### CLI

```bash
# 最高精度（デフォルト: large-v3モデル）
transcribe /path/to/video.mov

# 話者数を指定（精度向上）
transcribe /path/to/video.mov --speakers 3

# 速度優先モード
transcribe /path/to/video.mov --fast

# モデル指定
transcribe /path/to/video.mov -m turbo  # 高速、精度やや低
transcribe /path/to/video.mov -m medium # バランス型
```

### モデル一覧

| モデル | 精度 | 速度 | サイズ |
|--------|------|------|--------|
| tiny | ★☆☆☆☆ | 最速 | 75MB |
| base | ★★☆☆☆ | 速い | 142MB |
| small | ★★★☆☆ | 普通 | 466MB |
| medium | ★★★★☆ | やや遅い | 1.5GB |
| turbo | ★★★★☆ | 速い | 1.6GB |
| large-v3 | ★★★★★ | 遅い | 3GB |

※ デフォルトは `large-v3`（最高精度）

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
