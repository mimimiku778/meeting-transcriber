#!/usr/bin/env python3
"""CLI tool for meeting transcription."""

# Suppress all warnings before any imports
import warnings
warnings.filterwarnings("ignore")

import argparse
import sys
from pathlib import Path

# Load .env from package directory
from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)

from .transcriber import transcribe_video, format_timestamp
from .diarization import load_diarization_pipeline, diarize_audio, assign_speakers_to_segments


def main():
    parser = argparse.ArgumentParser(
        description="会議動画から話者識別付き文字起こしを生成"
    )
    parser.add_argument(
        "video_path",
        help="動画ファイルのパス"
    )
    parser.add_argument(
        "-o", "--output",
        help="出力ファイルのパス（省略時は動画と同じディレクトリに _transcript.txt）"
    )
    parser.add_argument(
        "-m", "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisperモデルサイズ (default: medium)"
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

    video_path = Path(args.video_path).resolve()
    if not video_path.exists():
        print(f"エラー: ファイルが見つかりません: {video_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_transcript.txt"
    else:
        output_path = Path(output_path)

    print(f"動画: {video_path}")
    print(f"出力: {output_path}")
    print(f"モデル: {args.model}")
    print()

    # Step 1: Transcribe
    print("1/3 音声抽出・文字起こし中...")
    whisper_result, audio_path = transcribe_video(str(video_path), args.model)
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
