# Meeting Transcriber

Web会議の録画動画から話者識別付き議事録を作成するMCPサーバー & CLIツール。

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
  diarization.py   # 話者識別（simple-diarizer / v1フォールバック）
  diarization_v2.py # 話者識別（pyannote.audio / v2既定）
  voiceprint.py    # 声紋エンロールメント/識別（~/.claude/voiceprints/）
  frame_extractor.py # 動画フレーム抽出

skills/
  transcribe-meeting.md  # Claude Code用スキル定義

install.sh       # セットアップスクリプト
```

## MCPツール

- `transcribe_meeting` - 動画から文字起こし+話者識別（`voiceprint_profile` 指定で声紋による実名化）
- `enroll_voiceprints` - 発話者N→実名マッピングから声紋を登録/更新
- `extract_video_frame` - 指定秒のフレームを抽出
- `update_speaker_names` - 発話者名を置換
- `read_transcript` - 文字起こし結果を読み込み

## 開発

```bash
# 仮想環境
source .venv/bin/activate

# 依存関係
pip install -e .

# CLI実行
transcribe /path/to/video.mov
```
