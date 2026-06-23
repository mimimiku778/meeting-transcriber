#!/usr/bin/env python3
"""CLI tool for meeting transcription."""

# Suppress all warnings before any imports
import warnings
warnings.filterwarnings("ignore")

import argparse
import subprocess
import sys
from pathlib import Path

from .transcriber import transcribe_video, format_timestamp, DEFAULT_MODEL
from .context_loader import load_context, asr_glossary, apply_normalization

LOG_FILE = Path("/tmp/meeting-transcriber.log")


def watch_progress():
    """Watch MCP server log with tail -f."""
    print("📡 ログを監視中... (Ctrl+C で終了)")
    print()
    try:
        subprocess.run(["tail", "-f", str(LOG_FILE)])
    except KeyboardInterrupt:
        pass


def should_skip_process(pid: int, current_pid: int) -> bool:
    """Check if a process should be skipped (watch, kill, tail, or current process)."""
    if pid == current_pid:
        return True
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            cmdline = result.stdout.strip()
            # Skip watch processes
            if "--watch" in cmdline or " -w " in cmdline or cmdline.endswith(" -w"):
                return True
            # Skip kill processes (including this one)
            if "--kill" in cmdline or " -k " in cmdline or cmdline.endswith(" -k"):
                return True
            # Skip tail processes (used by --watch)
            if cmdline.startswith("tail "):
                return True
    except Exception:
        pass
    return False


def kill_all_transcribe():
    """Kill all transcribe processes and restart MCP servers."""
    import os
    import signal
    from datetime import datetime

    current_pid = os.getpid()
    killed_transcribe = []
    killed_mcp = []

    # "transcribe" pattern matches the CLI process itself.
    # should_skip_process() filters out --watch, --kill, and tail processes.
    patterns = [
        ("transcribe", killed_transcribe),
        ("meeting-transcriber", killed_mcp),
        ("mcp-server", killed_mcp),
    ]

    all_killed = set()
    for pattern, killed_list in patterns:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            pids = result.stdout.strip().split("\n")
            for pid_str in pids:
                if pid_str:
                    pid = int(pid_str)
                    if pid in all_killed:
                        continue
                    if should_skip_process(pid, current_pid):
                        continue
                    try:
                        os.kill(pid, signal.SIGKILL)
                        killed_list.append(pid)
                        all_killed.add(pid)
                    except ProcessLookupError:
                        pass

    # Write to log file so --watch can see it
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_messages = [f"\n{'='*50}", f"[{timestamp}] 強制終了が実行されました"]
    if killed_transcribe:
        log_messages.append(f"  文字起こしプロセス: {len(killed_transcribe)} 件終了")
    if killed_mcp:
        log_messages.append(f"  MCPサーバー: {len(killed_mcp)} 件終了（自動再起動します）")
    log_messages.append("=" * 50 + "\n")

    with open(LOG_FILE, "a") as f:
        f.write("\n".join(log_messages))

    # Report results to terminal
    if killed_transcribe:
        print(f"文字起こしプロセス終了: {len(killed_transcribe)} 件 (PID: {', '.join(map(str, killed_transcribe))})")
    else:
        print("実行中の文字起こしプロセスはありません")

    if killed_mcp:
        print(f"MCPサーバー終了: {len(killed_mcp)} 件 (PID: {', '.join(map(str, killed_mcp))})")
        print("💡 MCPサーバーはClaude Codeが自動的に再起動します")
    else:
        print("実行中のMCPサーバーはありません")


def main():
    parser = argparse.ArgumentParser(
        description="会議動画から話者識別付き文字起こしを生成"
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        help="動画ファイルのパス"
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="MCPサーバーの進行状況を監視"
    )
    parser.add_argument(
        "--kill", "-k",
        action="store_true",
        help="全ての文字起こしプロセスを強制終了"
    )
    parser.add_argument(
        "-o", "--output",
        help="出力ファイルのパス（省略時は動画と同じディレクトリに _transcript.txt）"
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        choices=["small-4bit", "small", "medium", "large-v3", "large-v3-turbo"],
        help=f"Whisperモデルサイズ (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--context",
        help="案件コンテキスト(project.context.yaml)のパス。固有名詞をASRに注入し、"
             "文字起こし後に決定的(辞書)正規化を適用する"
    )
    parser.add_argument(
        "--normalize",
        metavar="TRANSCRIPT",
        help="既存の文字起こしファイルに --context の決定的正規化のみを適用して保存し終了"
             "（再文字起こし不要の再処理用）"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="速度優先モード（精度設定を緩和）"
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="話者識別をスキップ"
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="話者数を指定（精度向上）"
    )
    parser.add_argument(
        "--diarization-v2",
        action="store_true",
        help="（既定）pyannote.audio ベースの高精度話者識別。現在はこちらが標準"
    )
    parser.add_argument(
        "--diarization-v1",
        action="store_true",
        help="従来の simple-diarizer を使用（注: huggingface_hub 1.x では非互換の場合あり）"
    )

    args = parser.parse_args()

    # Watch mode
    if args.watch:
        watch_progress()
        return

    # Kill mode
    if args.kill:
        kill_all_transcribe()
        return

    # Normalize-only mode: 既存transcriptに決定的正規化を適用して終了
    if args.normalize:
        if not args.context:
            parser.error("--normalize には --context が必要です")
        transcript_path = Path(args.normalize)
        if not transcript_path.exists():
            print(f"エラー: ファイルが見つかりません: {transcript_path}", file=sys.stderr)
            sys.exit(1)
        context = load_context(args.context)
        text = transcript_path.read_text(encoding="utf-8")
        text, report = apply_normalization(text, context)
        transcript_path.write_text(text, encoding="utf-8")
        print(f"正規化適用: {transcript_path}")
        print("\n".join(report) if report else "置換対象なし")
        return

    # Normal transcription mode requires video_path
    if not args.video_path:
        parser.error("動画ファイルのパスを指定してください（または --watch で進行状況を監視）")

    video_path = Path(args.video_path).resolve()
    if not video_path.exists():
        print(f"エラー: ファイルが見つかりません: {video_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_transcript.txt"
    else:
        output_path = Path(output_path)

    max_accuracy = not args.fast
    mode = "速度優先" if args.fast else "最高精度"

    # v2(pyannote)を既定とし、--diarization-v1 指定時のみ従来版
    use_v2 = not args.diarization_v1
    diarization_mode = "pyannote.audio v2" if use_v2 else "simple-diarizer (v1)"

    # 案件コンテキスト（任意）: ASR固有名詞 + 文字起こし後の決定的正規化
    context = load_context(args.context) if args.context else None
    glossary = asr_glossary(context) if context else None

    print(f"動画: {video_path}", flush=True)
    print(f"出力: {output_path}", flush=True)
    print(f"モデル: {args.model} ({mode})", flush=True)
    print(f"話者識別: {diarization_mode}", flush=True)
    if context:
        print(f"案件コンテキスト: {args.context}（固有名詞 {len(glossary)} 件）", flush=True)
    print(flush=True)

    # Step 1: Transcribe
    print("1/3 音声抽出・文字起こし中...", flush=True)
    whisper_result, audio_path = transcribe_video(str(video_path), args.model, max_accuracy, glossary)
    print(f"    完了 ({len(whisper_result.get('segments', []))} セグメント)", flush=True)

    # Import diarization module (v2=pyannote が既定)
    if use_v2:
        from .diarization_v2 import load_diarization_pipeline, diarize_audio, assign_speakers_to_segments
    else:
        from .diarization import load_diarization_pipeline, diarize_audio, assign_speakers_to_segments

    # Step 2: Speaker diarization
    if args.no_diarization:
        print("2/3 話者識別: スキップ")
        segments_with_speakers = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
                "speaker": "発話者",
            }
            for seg in whisper_result.get("segments", [])
        ]
    else:
        print("2/3 話者識別中...", flush=True)
        print("    モデル読み込み中...", flush=True)
        pipeline = load_diarization_pipeline()
        print("    解析中...", flush=True)
        diarization_segments = diarize_audio(audio_path, pipeline, num_speakers=args.speakers)
        segments_with_speakers = assign_speakers_to_segments(whisper_result, diarization_segments)
        unique_speakers = sorted(set(seg["speaker"] for seg in segments_with_speakers if seg["speaker"] != "不明"))
        print(f"    完了 (話者: {', '.join(unique_speakers)})", flush=True)

    # Step 3: Format and save
    print("3/3 テキスト生成中...", flush=True)
    output_lines = []
    current_speaker = None
    current_text_parts = []
    current_start = None

    for segment in segments_with_speakers:
        if segment["speaker"] != current_speaker:
            if current_speaker is not None and current_text_parts:
                timestamp = format_timestamp(current_start)
                text = "".join(current_text_parts).strip()
                output_lines.append(f"{current_speaker} ({timestamp})")
                output_lines.append(text)
                output_lines.append("")

            current_speaker = segment["speaker"]
            current_text_parts = [segment["text"]]
            current_start = segment["start"]
        else:
            current_text_parts.append(segment["text"])

    if current_speaker is not None and current_text_parts:
        timestamp = format_timestamp(current_start)
        text = "".join(current_text_parts).strip()
        output_lines.append(f"{current_speaker} ({timestamp})")
        output_lines.append(text)
        output_lines.append("")

    output_text = "\n".join(output_lines)

    # 決定的(辞書)正規化: 曖昧さの無い表記ゆれのみ機械置換（文脈依存はClaude校正へ）
    if context:
        output_text, norm_report = apply_normalization(output_text, context)
        if norm_report:
            print(f"    正規化: {', '.join(norm_report)}", flush=True)

    output_path.write_text(output_text, encoding="utf-8")

    print(f"    完了", flush=True)
    print(flush=True)
    print(f"出力ファイル: {output_path}", flush=True)


if __name__ == "__main__":
    main()
