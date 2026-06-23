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
transcribe /path/to/video.mov                          # デフォルト(large-v3-turbo + pyannote v2)
transcribe /path/to/video.mov --context dir/project.context.yaml  # 案件用語集の注入＋決定的正規化
transcribe /path/to/video.mov --speakers 3             # 話者数指定でさらに精度向上
transcribe /path/to/video.mov --diarization-v1         # 従来simple-diarizer（フォールバック）
transcribe /path/to/video.mov -m small-4bit            # 最速

# 再文字起こし不要で、既存transcriptに決定的正規化だけ適用
transcribe --normalize dir/xxx_transcript.txt --context dir/project.context.yaml
```

出力: `video_transcript.txt`（話者識別付きテキスト）

## モデル

| モデル | サイズ | 特徴 |
|--------|--------|------|
| small-4bit | ~120MB | 最速 |
| small | 466MB | 高速 |
| medium | 1.5GB | 軽量 |
| large-v3 | 3GB | 最高精度・低速 |
| **large-v3-turbo** | ~1.6GB | **large並み精度をmedium並みの速度で（デフォルト）** |

※ 日本語会議では large-v3-turbo が精度・速度のバランスで最良（雑音・複数話者環境のローカルモデル中トップ）。
  4bit量子化は日本語で精度劣化が大きいため非推奨。

## 案件コンテキストと帰属精度（重要）

議事録の「決定/宿題/課題」の区別や担当（自社/クライアント/エンドユーザー）がズレる主因は、
**誰がどの会社かを音声/文字だけからは復元できない**こと。会議ディレクトリに `project.context.yaml`
（雛形: `templates/project.context.template.yaml`）を置くと:

- 固有名詞（社名・略号・人名・専門用語）が ASR の `initial_prompt` に注入される
- 文字起こし後に**決定的(辞書)正規化**（曖昧さの無い表記ゆれのみ）が適用される
- 議事録生成時、Claude が組織図・話者ロスター・帰属ルールに従い、各項目に
  **担当会社(enum) / 種別(決定·宿題·課題) / 根拠(話者+時刻+逐語引用)** を付け、
  確証の無い帰属は推測せず UNKNOWN として差し出す（スキル Step A-3〜A-5）

**組織構造・話者↔会社は動画フレーム（参加者パネルの会社プレフィックス、共有資料のロゴ/宛先）
の確定事実から作る。略号を音から推測で読み替えない。**

## 話者識別と HF_TOKEN

既定の pyannote v2 はgatedモデル（`speaker-diarization-community-1`、未取得時は `3.1` にフォールバック）。
初回はモデルのキャッシュが必要で、未キャッシュ環境では huggingface.co で利用規約に同意し
`HF_TOKEN` を環境変数で渡す（MCP登録時は `-e HF_TOKEN=...`）。既にキャッシュ済みならトークン無しで動作する。
MPS は既定で無効（タイムスタンプ崩れの報告があるため）。使う場合は `MEETING_DIARIZER_DEVICE=mps`。

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
  transcriber.py     # mlx-Whisper文字起こし（turbo + ハルシネーション抑制 + 用語集注入）
  context_loader.py  # project.context.yaml → 用語集/決定的正規化/組織コンテキスト展開
  diarization.py     # SpeechBrain話者識別（v1・フォールバック）
  diarization_v2.py  # pyannote.audio話者識別（v2・既定・word単位多数決）
  frame_extractor.py # OpenCV + Vision OCR
templates/
  project.context.template.yaml  # 案件コンテキストの雛形
```
