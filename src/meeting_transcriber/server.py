"""MCP Server for meeting transcription."""

# Load .env from package directory
from pathlib import Path
from dotenv import load_dotenv
_package_dir = Path(__file__).parent.parent.parent
load_dotenv(_package_dir / ".env")

# Disable MPS for pyannote-audio compatibility (must be before torch import)
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import json
import re
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .transcriber import transcribe_video, format_timestamp
from .diarization import load_diarization_pipeline, diarize_audio, assign_speakers_to_segments
from .frame_extractor import extract_frame, get_video_duration

# Create server instance
server = Server("meeting-transcriber")

# Cache for diarization pipeline
_diarization_pipeline = None


def get_diarization_pipeline():
    """Get or create diarization pipeline (lazy loading)."""
    global _diarization_pipeline
    if _diarization_pipeline is None:
        _diarization_pipeline = load_diarization_pipeline()
    return _diarization_pipeline


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="transcribe_meeting",
            description="会議の動画ファイルから文字起こしを作成します。音声を抽出し、Whisperで文字起こしを行い、pyannote-audioで話者識別を行います。",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "動画ファイルのパス（絶対パス）"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "出力テキストファイルのパス（省略時は動画と同じディレクトリに _transcript.txt として保存）"
                    },
                    "model": {
                        "type": "string",
                        "description": "Whisperモデルサイズ (tiny, base, small, medium, large)",
                        "default": "medium"
                    }
                },
                "required": ["video_path"]
            }
        ),
        Tool(
            name="extract_video_frame",
            description="動画から指定した秒数のフレームを抽出し、base64エンコードされた画像として返します。参加者の名前を確認するために使用できます。",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "動画ファイルのパス（絶対パス）"
                    },
                    "timestamp_seconds": {
                        "type": "number",
                        "description": "フレームを抽出する時間（秒）"
                    }
                },
                "required": ["video_path", "timestamp_seconds"]
            }
        ),
        Tool(
            name="update_speaker_names",
            description="文字起こしテキストファイル内の「発話者N」を実際の名前に置換します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "transcript_path": {
                        "type": "string",
                        "description": "文字起こしテキストファイルのパス"
                    },
                    "speaker_mapping": {
                        "type": "object",
                        "description": "発話者IDから名前へのマッピング（例: {\"発話者1\": \"田中\", \"発話者2\": \"佐藤\"}）",
                        "additionalProperties": {"type": "string"}
                    }
                },
                "required": ["transcript_path", "speaker_mapping"]
            }
        ),
        Tool(
            name="read_transcript",
            description="保存済みの文字起こしテキストファイルを読み込みます。",
            inputSchema={
                "type": "object",
                "properties": {
                    "transcript_path": {
                        "type": "string",
                        "description": "文字起こしテキストファイルのパス"
                    }
                },
                "required": ["transcript_path"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "transcribe_meeting":
            return await handle_transcribe_meeting(arguments)
        elif name == "extract_video_frame":
            return await handle_extract_video_frame(arguments)
        elif name == "update_speaker_names":
            return await handle_update_speaker_names(arguments)
        elif name == "read_transcript":
            return await handle_read_transcript(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_transcribe_meeting(arguments: dict) -> list[TextContent]:
    """Handle transcribe_meeting tool call."""
    video_path = arguments["video_path"]
    output_path = arguments.get("output_path")
    model = arguments.get("model", "medium")

    if output_path is None:
        video_p = Path(video_path)
        output_path = str(video_p.parent / f"{video_p.stem}_transcript.txt")

    # Transcribe video
    whisper_result, audio_path = transcribe_video(video_path, model)

    # Perform speaker diarization
    pipeline = get_diarization_pipeline()
    diarization_segments = diarize_audio(audio_path, pipeline)

    # Assign speakers to segments
    segments_with_speakers = assign_speakers_to_segments(whisper_result, diarization_segments)

    # Format output
    output_lines = []
    current_speaker = None
    current_text_parts = []
    current_start = None

    for segment in segments_with_speakers:
        if segment["speaker"] != current_speaker:
            # Save previous speaker's text
            if current_speaker is not None and current_text_parts:
                timestamp = format_timestamp(current_start)
                text = "".join(current_text_parts).strip()
                output_lines.append(f"{current_speaker} ({timestamp})")
                output_lines.append(text)
                output_lines.append("")

            # Start new speaker
            current_speaker = segment["speaker"]
            current_text_parts = [segment["text"]]
            current_start = segment["start"]
        else:
            current_text_parts.append(segment["text"])

    # Don't forget the last speaker
    if current_speaker is not None and current_text_parts:
        timestamp = format_timestamp(current_start)
        text = "".join(current_text_parts).strip()
        output_lines.append(f"{current_speaker} ({timestamp})")
        output_lines.append(text)
        output_lines.append("")

    # Write output file
    output_text = "\n".join(output_lines)
    Path(output_path).write_text(output_text, encoding="utf-8")

    # Get unique speakers
    unique_speakers = sorted(set(seg["speaker"] for seg in segments_with_speakers if seg["speaker"] != "不明"))

    return [TextContent(
        type="text",
        text=f"文字起こしが完了しました。\n\n"
             f"出力ファイル: {output_path}\n"
             f"検出された発話者: {', '.join(unique_speakers)}\n"
             f"セグメント数: {len(segments_with_speakers)}\n\n"
             f"発話者の名前を更新するには、extract_video_frame ツールで名前が表示されているフレームを確認し、"
             f"update_speaker_names ツールで名前を置換してください。"
    )]


async def handle_extract_video_frame(arguments: dict) -> list[TextContent]:
    """Handle extract_video_frame tool call."""
    video_path = arguments["video_path"]
    timestamp_seconds = arguments["timestamp_seconds"]

    # Get video duration for validation
    duration = get_video_duration(video_path)

    # Extract frame
    base64_image = extract_frame(video_path, timestamp_seconds)

    return [TextContent(
        type="text",
        text=f"フレームを抽出しました（{timestamp_seconds}秒時点、動画長: {duration:.1f}秒）\n\n"
             f"data:image/png;base64,{base64_image}"
    )]


async def handle_update_speaker_names(arguments: dict) -> list[TextContent]:
    """Handle update_speaker_names tool call."""
    transcript_path = arguments["transcript_path"]
    speaker_mapping = arguments["speaker_mapping"]

    path = Path(transcript_path)
    if not path.exists():
        return [TextContent(type="text", text=f"ファイルが見つかりません: {transcript_path}")]

    content = path.read_text(encoding="utf-8")

    # Replace speaker names
    replacements_made = []
    for old_name, new_name in speaker_mapping.items():
        pattern = re.compile(re.escape(old_name) + r'(?=\s*\()')
        matches = pattern.findall(content)
        if matches:
            content = pattern.sub(new_name, content)
            replacements_made.append(f"{old_name} -> {new_name} ({len(matches)}箇所)")

    # Write updated content
    path.write_text(content, encoding="utf-8")

    if replacements_made:
        return [TextContent(
            type="text",
            text=f"発話者名を更新しました。\n\n置換内容:\n" + "\n".join(replacements_made)
        )]
    else:
        return [TextContent(
            type="text",
            text="置換対象が見つかりませんでした。発話者名を確認してください。"
        )]


async def handle_read_transcript(arguments: dict) -> list[TextContent]:
    """Handle read_transcript tool call."""
    transcript_path = arguments["transcript_path"]

    path = Path(transcript_path)
    if not path.exists():
        return [TextContent(type="text", text=f"ファイルが見つかりません: {transcript_path}")]

    content = path.read_text(encoding="utf-8")

    return [TextContent(type="text", text=content)]


async def main():
    """Main entry point."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


def run_test():
    """Simple test to verify imports work."""
    print("Testing imports...")
    print(f"  - Server: {server.name}")
    print("  - Transcriber module: OK")
    print("  - Diarization module: OK")
    print("  - Frame extractor module: OK")
    print("All imports successful!")
    print("\nNote: Full functionality test requires a video file.")


def run():
    """Entry point for console script."""
    import asyncio
    asyncio.run(main())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test()
    else:
        run()
