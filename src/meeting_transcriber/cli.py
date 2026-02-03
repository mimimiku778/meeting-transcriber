#!/usr/bin/env python3
"""CLI tool for meeting transcription."""

# Suppress all warnings before any imports
import warnings
warnings.filterwarnings("ignore")

import argparse
import subprocess
import sys
from pathlib import Path

from .transcriber import transcribe_video, format_timestamp
from .diarization import load_diarization_pipeline, diarize_audio, assign_speakers_to_segments

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

    # Patterns to search for (don't include generic "transcribe" to avoid killing watch/kill)
    patterns = [
        ("mlx_whisper", killed_transcribe),
        ("simple_diarizer", killed_transcribe),
        ("speechbrain", killed_transcribe),
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
                        os.kill(pid, signal.SIGTERM)
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
        default="medium",
        choices=["small-4bit", "small", "medium", "large-v3"],
        help="Whisperモデルサイズ (default: medium)"
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

    args = parser.parse_args()

    # Watch mode
    if args.watch:
        watch_progress()
        return

    # Kill mode
    if args.kill:
        kill_all_transcribe()
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

    print(f"動画: {video_path}")
    print(f"出力: {output_path}")
    print(f"モデル: {args.model} ({mode})")
    print()

    # Step 1: Transcribe
    print("1/3 音声抽出・文字起こし中...")
    whisper_result, audio_path = transcribe_video(str(video_path), args.model, max_accuracy)
    print(f"    完了 ({len(whisper_result.get('segments', []))} セグメント)")

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
        print("2/3 話者識別中...")
        pipeline = load_diarization_pipeline()
        diarization_segments = diarize_audio(audio_path, pipeline, num_speakers=args.speakers)
        segments_with_speakers = assign_speakers_to_segments(whisper_result, diarization_segments)
        unique_speakers = sorted(set(seg["speaker"] for seg in segments_with_speakers if seg["speaker"] != "不明"))
        print(f"    完了 (話者: {', '.join(unique_speakers)})")

    # Step 3: Format and save
    print("3/3 テキスト生成中...")
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
    output_path.write_text(output_text, encoding="utf-8")

    print(f"    完了")
    print()
    print(f"出力ファイル: {output_path}")


if __name__ == "__main__":
    main()
