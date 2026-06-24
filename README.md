# Meeting Transcriber

macOS (Apple Silicon) 専用の Claude Code スキル + MCP サーバー + CLI。
**会議の録画動画を放り込むだけ**で、話者識別付きの文字起こしと、決定事項・宿題・課題を根拠付きで整理した議事録を生成する。

## 入力 → 成果物

- **入力**: OBS 等で録画した Web 会議動画（Google Meet / Teams / Zoom / Discord …）。準備するのは**動画ファイルだけ**。
- **成果物**: 動画と同じ場所に整理されたフォルダ `<動画名>_<会議タイトル>/` が作られ、中に:
  - `*_transcript_*.txt` — 話者識別付き文字起こし（タイムスタンプ付き）
  - `*_minutes_*.md` — 議事録。**決定事項 / 宿題（担当会社・期限）/ 課題**を、各項目に話者・時刻・逐語引用の根拠付きで
  - `frames/` — 抽出した画面フレーム＋OCR
  - `*_speakers.json` — 話者同一性ヒント（過分割の統合候補・過少分割の混在検出・区間照合）

## 設計思想：ゼロお膳立て＋修正ベース学習

**精度を上げる方法はひとつ — 出てきた議事録を直すこと。** 事前にチューニングや設定はしない。
成果物（議事録）へ修正フィードバックを返すと、それがそのままナレッジ（声紋・組織図・用語・取捨の好み）
として溜まり、次回以降の文字起こし精度・話者の自動実名化・議事録の質になる。**使うほど賢くなる。**

事前に用意するのは動画だけ。**案件知識（組織図・話者・用語・議事録の好み）も話者の声紋も、事前に設定しない。**

- 初回は動画パス・フレーム・声紋から**案件を自動判定**（無ければ自動で新規作成）。
- 曖昧な所だけユーザーに訊き、**ユーザーが議事録(文面)を直したら、その修正を案件ストアに書き戻して学習**する。
- 会議を重ねるほど、固有名詞のASR精度・話者の自動実名化・議事録の取捨が**勝手に育つ**。

支えるのは声紋と同じ「育てる」2ストア（どちらもローカル・リポジトリ外）。同一 `<slug>` で連結（同じ顔ぶれ＝同じ案件）:

| ストア | 中身 |
|---|---|
| `~/.claude/voiceprints/<slug>.json` | 話者の声紋（特徴ベクトル）。`発話者N → 実名` の自動付与 |
| `~/.claude/meeting-contexts/<slug>.yaml` | 組織図・話者ロスター・用語・帰属ルール・議事録の取捨・案件判定シグナル |

## 動作フロー

```
動画 ─▶ ①音声抽出(ffmpeg・stem単位キャッシュ) ─▶ ②文字起こし(mlx-Whisper)
     ─▶ ③話者分離(speakrs/CoreML) ─▶ ④話者同一性ヒント(声紋: 過分割の統合/混在検出/区間照合)
     ─▶ ⑤フレーム抽出+OCR(一括・並列) ─▶ ⑥議事録生成(Claude: 案件ストアの組織図/帰属ルール/取捨を適用)
     ─▶ ⑦ユーザーの修正を案件ストアへ学習(upsert)　［声紋登録は議事録の後 or 省略］
```

1次の文字起こしの精度には拘らない。**話者の取り違え（同一人物が別ラベルに分裂／別人が1ラベルに混在）は、声紋ヒントを使って議事録(成果物)側で正す**設計。

## 使用技術

| 技術 | 用途 |
|---|---|
| mlx-Whisper (large-v3-turbo) | 音声認識（Apple Silicon / Metal） |
| speakrs (Rust / CoreML) | 話者分離 ← **既定・最速**。pyannote.audio も選択可（`--diarizer pyannote`） |
| pyannote `embedding` (MPS) | 声紋（話者の特徴ベクトル）・クラスタ類似度・区間照合 |
| macOS Vision | 画面フレームの OCR（並列実行） |
| ffmpeg | 音声抽出 |
| MCP | Claude Code から各ツールを呼び出し |

## 使い方

### Claude Code（推奨・議事録まで自動）

```
/transcribe-meeting <動画パス>
```

動画を渡すだけで、案件判定 → 文字起こし → 話者解決 → フレーム確認 → 議事録生成まで自動で進む。出てきた議事録を直せばその修正から学習する。

### CLI（文字起こしのみ）

```bash
transcribe /path/to/video.mov                      # 既定: large-v3-turbo + speakrs
transcribe /path/to/video.mov --project myteam     # 案件ストア＋声紋を自動適用＋話者ヒント出力
transcribe /path/to/video.mov --diarizer pyannote  # 話者分離を pyannote に切替
transcribe --resolve-speakers /path/to/video.mov --project myteam  # 話者同一性ヒントだけ算出
transcribe --normalize dir/xxx_transcript.txt --project myteam     # 既存transcriptに正規化のみ
transcribe --watch     # 進行監視     transcribe --kill   # 強制終了 & MCP再起動
```

出力: `video_transcript.txt`（話者識別付きテキスト）＋ `video_speakers.json`（話者ヒント）。

### モデル

| モデル | 特徴 |
|---|---|
| **large-v3-turbo**（既定） | large並み精度を medium 並み速度で（~1.6GB）|
| large-v3 | 最高精度・低速（3GB）|
| medium / small | 軽量・試し用 |

※ 4bit量子化は日本語で精度劣化が大きいため非推奨。

## アーキテクチャ / ファイル構成

```
~/.claude/commands/transcribe-meeting.md   # スキル（自律フロー・ゼロお膳立て）
        ↓ 呼ぶ
MCPサーバー (meeting-transcriber)
  transcribe_meeting / extract_video_frame(s) / read_transcript / update_speaker_names /
  finalize_meeting_files / enroll_voiceprints /
  identify_project / resolve_speakers / upsert_project_context / list_projects

src/meeting_transcriber/
  server.py              # MCPサーバー
  cli.py                 # CLIエントリーポイント
  transcriber.py         # mlx-Whisper（turbo＋ハルシネーション抑制＋用語集注入＋音声キャッシュ）
  diarization_speakrs.py # 話者分離（speakrs / CoreML・既定）
  diarization_v2.py      # 話者分離（pyannote.audio・選択）
  voiceprint.py          # 声紋 登録/識別＋クラスタ類似度/混在検出/区間照合（MPS）
  context_store.py       # 案件ストア（自動判定・修正ベース学習・~/.claude/meeting-contexts/）
  context_loader.py      # 案件yaml → ASR用語/決定的正規化/議事録プロンプト展開
  frame_extractor.py     # OpenCV 抽出 + Vision OCR（一括・並列）
templates/project.context.template.yaml   # 案件コンテキストの雛形
native/speakrs-diarizer/                   # speakrs CLI（Rust・要 cargo build）
```

## インストール / 必要トークン

```bash
git clone <repo-url> && cd meeting-transcriber && ./install.sh
```

`./install.sh` が ffmpeg（Homebrew）・Python依存（venv）・MCP登録・`transcribe` CLI・スキルを自動設定する。
speakrs を使うには `native/speakrs-diarizer` を `cargo build --release`（`brew install openblas` が必要）。

- **HF_TOKEN**: 声紋（`pyannote/embedding`）と pyannote 話者分離は gated モデル。初回のみ huggingface.co で利用規約に同意し `HF_TOKEN` を環境変数で渡す（MCP登録時 `-e HF_TOKEN=...`）。キャッシュ後はオフライン可。**speakrs 既定の話者分離には不要**。
- **デバイス**: 声紋embedding は MPS 既定（`MEETING_VOICEPRINT_DEVICE=cpu/mps/auto`）。pyannote 話者分離は cpu 既定（`MEETING_DIARIZER_DEVICE=mps` で変更）。

## アップデート / アンインストール

```bash
# アップデート
cd ~/.claude/mcp-servers/meeting-transcriber && git pull && ./install.sh   # 後 Claude Code 再起動

# アンインストール
claude mcp remove meeting-transcriber -s user
rm ~/.claude/commands/transcribe-meeting.md ~/.local/bin/transcribe
rm -rf ~/.claude/voiceprints ~/.claude/meeting-contexts   # 学習データも消す場合
```
