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
                    "model": {"type": "string", "description": "Whisperモデル (small-4bit/small/medium/large-v3/large-v3-turbo)", "default": "large-v3-turbo"},
                    "diarizer": {"type": "string", "enum": ["speakrs", "pyannote"], "description": "話者識別バックエンド（既定: speakrs = Apple Silicon CoreMLで最速・pyannote同等精度）。pyannoteは同モデルだがCPUで低速", "default": "speakrs"},
                    "context_path": {"type": "string", "description": "案件コンテキスト(project.context.yaml)のパス。固有名詞をASRに注入し文字起こし後に決定的正規化を適用"},
                    "voiceprint_profile": {"type": "string", "description": "声紋プロファイル名（例: myteam）。~/.claude/voiceprints/<名前>.json と照合し『発話者N』を登録済みの実名に自動置換する。未登録・低信頼の話者は発話者Nのまま"},
                    "project": {"type": "string", "description": "案件スラッグ。~/.claude/meeting-contexts/<slug>.yaml を案件コンテキストとして使い、同名の声紋プロファイルも自動照合する（声紋と案件ストアを同一slugで連結）。話者同一性ヒントを <transcript>_speakers.json に出力"}
                },
                "required": ["video_path"]
            }
        ),
        Tool(
            name="enroll_voiceprints",
            description="人間が確定した『発話者N→実名』マッピングから各実名の声紋を登録/更新します。会議を重ねるほど精度が上がります。",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "登録元の動画ファイルのパス（絶対パス）"},
                    "voiceprint_profile": {"type": "string", "description": "声紋プロファイル名（例: myteam）。~/.claude/voiceprints/<名前>.json に保存される"},
                    "speaker_mapping": {"type": "object", "description": "発話者ラベル→実名のマッピング 例 {\"発話者1\":\"山田\",\"発話者2\":\"鈴木\"}", "additionalProperties": {"type": "string"}}
                },
                "required": ["video_path", "voiceprint_profile", "speaker_mapping"]
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
        ),
        Tool(
            name="identify_project",
            description="動画パス・フレームOCR語・声紋一致から『どの案件か』を自動判定し、候補を確信度降順で返します。ゼロお膳立て運用の入口。該当が無ければ新規案件として進めてよい。",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "動画ファイルのパス（ディレクトリ/ファイル名のキーワード照合に使う）"},
                    "ocr_terms": {"type": "array", "items": {"type": "string"}, "description": "冒頭フレームのOCRや参加者パネルから拾った語（組織名・タイトル語）。signals と照合する"},
                    "voiceprint_matches": {"type": "object", "description": "声紋で一致した案件ごとの人数 {slug: 一致人数}（任意・最強シグナル）", "additionalProperties": {"type": "integer"}},
                    "extra_text": {"type": "string", "description": "追加の手掛かりテキスト（文字起こし冒頭など・任意）"}
                }
            }
        ),
        Tool(
            name="resolve_speakers",
            description="動画の話者同一性ヒント（声紋クラスタ類似度＝過分割の統合候補／混在検出＝過少分割／登録済み声紋による区間単位の実名照合）を算出し、サイドカーJSONに保存して返します。1次の文字起こしは直さず、議事録(成果物)側で話者帰属を正すための材料。",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "動画ファイルのパス"},
                    "project": {"type": "string", "description": "案件スラッグ（同名の声紋プロファイルを使い、混在ラベルを区間単位で実名照合する）"},
                    "voiceprint_profile": {"type": "string", "description": "声紋プロファイル名（project 未指定時の直接指定）"}
                },
                "required": ["video_path"]
            }
        ),
        Tool(
            name="upsert_project_context",
            description="ユーザーが議事録(文面)で確定・修正した情報を案件ストア(~/.claude/meeting-contexts/<slug>.yaml)へマージして育てる（声紋enrollの案件版・修正ベース学習の書き戻し）。組織図/話者ロスター/用語/帰属ルール/議事録の取捨(minutes_preferences)/判定シグナル等を部分更新。人間確定が常に優先。",
            inputSchema={
                "type": "object",
                "properties": {
                    "slug": {"type": "string", "description": "案件スラッグ（声紋プロファイルと同一。無ければ新規作成）"},
                    "updates": {"type": "object", "description": "部分更新。キー例: organization/speaker_roster/glossary/topic_kinds/minutes_preferences/asr_prompt_terms/attribution_rules/signals/meeting/normalization"},
                    "meeting_dir": {"type": "string", "description": "指定すると更新後ストアを project.context.yaml としてその会議ディレクトリへ焼き込む（任意）"}
                },
                "required": ["slug", "updates"]
            }
        ),
        Tool(
            name="list_projects",
            description="登録済みの案件ストア一覧（slug・タイトル・登場人物・組織・学習回数）を返します。",
            inputSchema={"type": "object", "properties": {}}
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
        elif name == "enroll_voiceprints":
            return await handle_enroll_voiceprints(arguments)
        elif name == "identify_project":
            return handle_identify_project(arguments)
        elif name == "resolve_speakers":
            return await handle_resolve_speakers(arguments)
        elif name == "upsert_project_context":
            return handle_upsert_project_context(arguments)
        elif name == "list_projects":
            return handle_list_projects(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_transcribe_meeting(arguments: dict) -> list[TextContent]:
    video_path = arguments["video_path"]
    output_path = arguments.get("output_path")
    model = arguments.get("model", "large-v3-turbo")
    context_path = arguments.get("context_path")

    # 話者識別バックエンド: diarizer 明示 > speakrs(既定)
    diarizer = arguments.get("diarizer") or "speakrs"
    voiceprint_profile = arguments.get("voiceprint_profile")
    project = arguments.get("project")

    cmd = ["transcribe", video_path, "-m", model, "--diarizer", diarizer]
    if output_path:
        cmd.extend(["-o", output_path])
    if context_path:
        cmd.extend(["--context", context_path])
    if voiceprint_profile:
        cmd.extend(["--voiceprints", voiceprint_profile])
    if project:
        cmd.extend(["--project", project])

    LOG_FILE.write_text("")
    with open(LOG_FILE, "w") as log_file:
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, bufsize=1)
        process.wait()

    if output_path is None:
        output_path = str(Path(video_path).parent / f"{Path(video_path).stem}_transcript.txt")

    log_content = LOG_FILE.read_text()
    return [TextContent(type="text", text=f"完了\n出力: {output_path}\n\n{log_content}")]


async def handle_enroll_voiceprints(arguments: dict) -> list[TextContent]:
    video_path = arguments["video_path"]
    profile = arguments["voiceprint_profile"]
    mapping = arguments["speaker_mapping"]

    import json as _json
    cmd = ["transcribe", video_path, "--voiceprints", profile, "--enroll", _json.dumps(mapping, ensure_ascii=False)]

    LOG_FILE.write_text("")
    with open(LOG_FILE, "w") as log_file:
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, bufsize=1)
        process.wait()

    log_content = LOG_FILE.read_text()
    return [TextContent(type="text", text=f"声紋登録\nプロファイル: {profile}\n\n{log_content}")]


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


def handle_identify_project(arguments: dict) -> list[TextContent]:
    from .context_store import identify_project
    candidates = identify_project(
        video_path=arguments.get("video_path"),
        ocr_terms=arguments.get("ocr_terms"),
        voiceprint_matches=arguments.get("voiceprint_matches"),
        extra_text=arguments.get("extra_text"),
    )
    if not candidates:
        return [TextContent(type="text", text="該当案件なし（新規案件として進めてください）。\n[]")]
    import json as _json
    lines = ["案件候補（確信度降順）:"]
    for c in candidates:
        s = c["summary"]
        lines.append(f"- {c['slug']}（score {c['score']}）: {', '.join(c['reasons'])}"
                     f" / 組織 {s.get('orgs')} / 人 {s.get('people')} / 学習{s.get('enroll_count')}回")
    lines.append("\n```json\n" + _json.dumps(candidates, ensure_ascii=False, indent=2) + "\n```")
    return [TextContent(type="text", text="\n".join(lines))]


async def handle_resolve_speakers(arguments: dict) -> list[TextContent]:
    video_path = arguments["video_path"]
    project = arguments.get("project")
    voiceprint_profile = arguments.get("voiceprint_profile")

    cmd = ["transcribe", video_path, "--resolve-speakers"]
    if project:
        cmd.extend(["--project", project])
    elif voiceprint_profile:
        cmd.extend(["--voiceprints", voiceprint_profile])

    LOG_FILE.write_text("")
    with open(LOG_FILE, "w") as log_file:
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, bufsize=1)
        process.wait()

    sidecar = str(Path(video_path).with_name(Path(video_path).stem + "_speakers.json"))
    log_content = LOG_FILE.read_text()
    return [TextContent(type="text", text=f"話者ヒント算出\nサイドカー: {sidecar}\n\n{log_content}")]


def handle_upsert_project_context(arguments: dict) -> list[TextContent]:
    from .context_store import merge_into_project, export_to_meeting_dir
    slug = arguments["slug"]
    updates = arguments.get("updates") or {}
    data = merge_into_project(slug, updates)
    msg = [f"案件ストア更新: {slug}（学習{data.get('enroll_count')}回目）"]
    if arguments.get("meeting_dir"):
        out = export_to_meeting_dir(slug, arguments["meeting_dir"])
        msg.append(f"会議ディレクトリへ焼き込み: {out}")
    return [TextContent(type="text", text="\n".join(msg))]


def handle_list_projects(arguments: dict) -> list[TextContent]:
    from .context_store import list_projects
    projects = list_projects()
    if not projects:
        return [TextContent(type="text", text="登録済み案件なし")]
    lines = ["登録済み案件:"]
    for p in projects:
        lines.append(f"- {p['slug']}: {p.get('title','')} / 組織 {p.get('orgs')} / 人 {p.get('people')}"
                     f" / 議題 {p.get('kinds')} / 学習{p.get('enroll_count')}回 / 更新 {p.get('updated')}")
    return [TextContent(type="text", text="\n".join(lines))]


def run():
    asyncio.run(main())


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    run()
