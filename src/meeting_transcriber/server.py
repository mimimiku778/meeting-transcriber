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
                    "model": {"type": "string", "description": "Whisperモデル (small-4bit/small/medium/large-v3)", "default": "medium"}
                },
                "required": ["video_path"]
            }
        ),
        Tool(
            name="extract_video_frame",
            description="動画から指定秒のフレームを抽出しbase64画像として返します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "動画ファイルのパス"},
                    "timestamp_seconds": {"type": "number", "description": "抽出する時間（秒）"},
                    "output_dir": {"type": "string", "description": "整理されたディレクトリのパス（指定時はframes/サブディレクトリに保存）"}
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
        ),
        Tool(
            name="finalize_meeting_files",
            description="文字起こしファイルをタイトル付きディレクトリに整理します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "元の動画ファイルのパス"},
                    "title": {"type": "string", "description": "会議のタイトル（ディレクトリ名とファイル名に使用、日本語可）"},
                    "transcript_path": {"type": "string", "description": "既存の文字起こしファイルのパス（省略時は自動検出）"}
                },
                "required": ["video_path", "title"]
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
        elif name == "finalize_meeting_files":
            return handle_finalize_meeting_files(arguments)
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
    output_dir = arguments.get("output_dir")
    duration = get_video_duration(video_path)
    output_path, ocr_texts, ocr_path = extract_frame(video_path, timestamp_seconds, output_dir)

    ocr_section = ""
    if ocr_texts:
        ocr_section = "\n\n## 画面内テキスト（OCR）\n" + "\n".join(ocr_texts)

    saved_info = f"画像パス: {output_path}"
    if ocr_path:
        saved_info += f"\nOCRテキスト: {ocr_path}"

    return [TextContent(type="text", text=f"フレーム抽出完了（{timestamp_seconds}秒、動画長: {duration:.1f}秒）\n\n{saved_info}{ocr_section}\n\nClaudeにこの画像を見せるには、Readツールでパスを読み込んでください。")]


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


def handle_finalize_meeting_files(arguments: dict) -> list[TextContent]:
    video_path = Path(arguments["video_path"])
    title = arguments["title"]
    transcript_path = arguments.get("transcript_path")

    if not video_path.exists():
        return [TextContent(type="text", text=f"動画ファイルが見つかりません: {video_path}")]

    video_stem = video_path.stem
    video_dir = video_path.parent

    # タイトルからファイル名に使えない文字を除去
    safe_title = re.sub(r'[<>:"/\\|?*]', '', title).strip()
    if not safe_title:
        return [TextContent(type="text", text="タイトルが無効です")]

    # ディレクトリ作成
    output_dir = video_dir / f"{video_stem}_{safe_title}"
    output_dir.mkdir(exist_ok=True)

    # 文字起こしファイルの移動
    if transcript_path is None:
        transcript_path = video_dir / f"{video_stem}_transcript.txt"
    else:
        transcript_path = Path(transcript_path)

    new_transcript_path = output_dir / f"{video_stem}_transcript_{safe_title}.txt"
    new_minutes_path = output_dir / f"{video_stem}_minutes_{safe_title}.md"

    if transcript_path.exists():
        content = transcript_path.read_text(encoding="utf-8")
        new_transcript_path.write_text(content, encoding="utf-8")
        transcript_path.unlink()  # 元ファイルを削除
        moved_msg = f"文字起こしファイルを移動しました: {new_transcript_path}"
    else:
        moved_msg = f"文字起こしファイルが見つかりません: {transcript_path}"

    return [TextContent(type="text", text=f"""整理完了

ディレクトリ: {output_dir}
{moved_msg}
議事録ファイルパス: {new_minutes_path}

議事録を作成する場合は、上記パスに保存してください。""")]


def run():
    asyncio.run(main())


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    run()
