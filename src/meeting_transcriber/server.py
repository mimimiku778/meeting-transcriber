"""MCP Server for meeting transcription."""

import asyncio
import re
import subprocess
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .frame_extractor import extract_frame, get_video_duration

LOG_FILE = Path("/tmp/meeting-transcriber.log")

server = Server("meeting-transcriber")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="transcribe_meeting",
            description="会議の動画ファイルから文字起こしを作成します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "動画ファイルのパス（絶対パス）"},
                    "output_path": {"type": "string", "description": "出力ファイルのパス（省略時は動画と同じディレクトリ）"},
                    "model": {"type": "string", "description": "Whisperモデル (small/medium/large/large-v3/turbo)", "default": "medium"}
                },
                "required": ["video_path"]
            }
        ),
        Tool(
            name="extract_video_frame",
            description="動画から指定秒のフレームを抽出しJPEG画像として保存します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "動画ファイルのパス"},
                    "timestamp_seconds": {"type": "number", "description": "抽出する時間（秒）"}
                },
                "required": ["video_path", "timestamp_seconds"]
            }
        ),
        Tool(
            name="update_speaker_names",
            description="文字起こし内の「発話者N」を実際の名前に置換します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "transcript_path": {"type": "string", "description": "文字起こしファイルのパス"},
                    "speaker_mapping": {"type": "object", "description": "発話者IDから名前へのマッピング", "additionalProperties": {"type": "string"}}
                },
                "required": ["transcript_path", "speaker_mapping"]
            }
        ),
        Tool(
            name="read_transcript",
            description="文字起こしファイルを読み込みます。",
            inputSchema={
                "type": "object",
                "properties": {
                    "transcript_path": {"type": "string", "description": "文字起こしファイルのパス"}
                },
                "required": ["transcript_path"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "transcribe_meeting":
            return await handle_transcribe_meeting(arguments)
        elif name == "extract_video_frame":
            return handle_extract_video_frame(arguments)
        elif name == "update_speaker_names":
            return handle_update_speaker_names(arguments)
        elif name == "read_transcript":
            return handle_read_transcript(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_transcribe_meeting(arguments: dict) -> list[TextContent]:
    video_path = arguments["video_path"]
    output_path = arguments.get("output_path")
    model = arguments.get("model", "medium")

    cmd = ["transcribe", video_path, "-m", model]
    if output_path:
        cmd.extend(["-o", output_path])

    LOG_FILE.write_text("")
    with open(LOG_FILE, "w") as log_file:
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, bufsize=1)
        process.wait()

    if output_path is None:
        output_path = str(Path(video_path).parent / f"{Path(video_path).stem}_transcript.txt")

    log_content = LOG_FILE.read_text()
    return [TextContent(type="text", text=f"完了\n出力: {output_path}\n\n{log_content}")]


def handle_extract_video_frame(arguments: dict) -> list[TextContent]:
    video_path = arguments["video_path"]
    timestamp_seconds = arguments["timestamp_seconds"]
    duration = get_video_duration(video_path)
    output_path, ocr_texts = extract_frame(video_path, timestamp_seconds)

    ocr_section = ""
    if ocr_texts:
        ocr_section = "\n\n## 画面内テキスト（OCR）\n" + "\n".join(ocr_texts)

    return [TextContent(type="text", text=f"フレーム抽出完了（{timestamp_seconds}秒、動画長: {duration:.1f}秒）\n\n画像パス: {output_path}{ocr_section}\n\nClaudeにこの画像を見せるには、Readツールでパスを読み込んでください。")]


def handle_update_speaker_names(arguments: dict) -> list[TextContent]:
    transcript_path = arguments["transcript_path"]
    speaker_mapping = arguments["speaker_mapping"]

    path = Path(transcript_path)
    if not path.exists():
        return [TextContent(type="text", text=f"ファイルが見つかりません: {transcript_path}")]

    content = path.read_text(encoding="utf-8")
    replacements = []
    for old_name, new_name in speaker_mapping.items():
        pattern = re.compile(re.escape(old_name) + r'(?=\s*\()')
        matches = pattern.findall(content)
        if matches:
            content = pattern.sub(new_name, content)
            replacements.append(f"{old_name} -> {new_name} ({len(matches)}箇所)")

    path.write_text(content, encoding="utf-8")
    return [TextContent(type="text", text="置換完了\n" + "\n".join(replacements) if replacements else "置換対象なし")]


def handle_read_transcript(arguments: dict) -> list[TextContent]:
    path = Path(arguments["transcript_path"])
    if not path.exists():
        return [TextContent(type="text", text=f"ファイルが見つかりません: {path}")]
    return [TextContent(type="text", text=path.read_text(encoding="utf-8"))]


def run():
    asyncio.run(main())


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    run()
