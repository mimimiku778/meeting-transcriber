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
from .context_loader import load_context, asr_glossary, apply_normalization, expected_speakers

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


def resolve_diarizer_backend(args) -> str:
    """話者識別バックエンドを決定する。既定は speakrs（Apple Silicon CoreMLで最速）。

    優先順: --diarizer 明示 > --diarization-v2(pyannote) > speakrs(既定)。
    """
    explicit = getattr(args, "diarizer", None)
    if explicit:
        return explicit
    if getattr(args, "diarization_v2", False):
        return "pyannote"
    return "speakrs"


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
        "--voiceprints",
        metavar="PROFILE",
        default=None,
        help="声紋プロファイル名（例: myteam）。~/.claude/voiceprints/<PROFILE>.json と照合し"
             "『発話者N』を登録済みの実名に自動置換する。未登録・低信頼の話者は発話者Nのまま"
    )
    parser.add_argument(
        "--project",
        metavar="SLUG",
        default=None,
        help="案件スラッグ。~/.claude/meeting-contexts/<SLUG>.yaml を --context として使い、"
             "同名の声紋プロファイルも自動で照合する（声紋と案件ストアを同一slugで連結）"
    )
    parser.add_argument(
        "--no-speaker-hints",
        action="store_true",
        help="話者同一性ヒント（声紋クラスタ類似度・統合候補・混在検出）の算出をスキップ"
    )
    parser.add_argument(
        "--resolve-speakers",
        action="store_true",
        help="文字起こしせず、話者同一性ヒント（声紋クラスタ類似度・統合候補・混在検出・区間照合）だけを"
             "算出して <video>_speakers.json に保存して終了する"
    )
    parser.add_argument(
        "--enroll",
        metavar="JSON",
        default=None,
        help="声紋登録モード。{'発話者1':'山田',...} のマッピング(JSON文字列 or ファイルパス)を渡すと、"
             "--voiceprints のプロファイルに各実名の声紋を登録/更新して終了する"
    )
    parser.add_argument(
        "--diarizer",
        choices=["speakrs", "pyannote"],
        default=None,
        help="話者識別バックエンド: speakrs(既定/最速・Apple Silicon CoreML) / "
             "pyannote(同モデル・CPUで低速)。未指定は speakrs"
    )
    parser.add_argument(
        "--diarization-v2",
        action="store_true",
        help="pyannote.audio バックエンドを使用（= --diarizer pyannote のエイリアス）"
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

    # Enroll-only mode: 動画＋『発話者N→実名』マッピングから声紋を登録して終了
    if args.enroll:
        if not args.voiceprints:
            parser.error("--enroll には --voiceprints（プロファイル名）が必要です")
        if not args.video_path:
            parser.error("--enroll には動画ファイルのパスが必要です")
        enroll_video = Path(args.video_path).resolve()
        if not enroll_video.exists():
            print(f"エラー: ファイルが見つかりません: {enroll_video}", file=sys.stderr)
            sys.exit(1)
        # マッピングは JSON文字列 or ファイルパス
        import json as _json
        raw = args.enroll
        if Path(raw).exists():
            raw = Path(raw).read_text(encoding="utf-8")
        try:
            mapping = _json.loads(raw)
        except Exception as e:
            parser.error(f"--enroll のJSONを解釈できません: {e}")

        from .transcriber import ensure_audio
        # diarization は transcribe と同じバックエンド（既定 speakrs）に揃える。
        # pyannote(CPU)で再diarizationすると遅い上、transcribeと話者ラベルの採番がズレて
        # 声紋が誤対応する。speakrs に揃えることで高速化＋ラベル整合を両立する。
        backend = resolve_diarizer_backend(args)
        if backend == "pyannote":
            from .diarization_v2 import load_diarization_pipeline, diarize_audio
        else:
            from .diarization_speakrs import load_diarization_pipeline, diarize_audio
        from .voiceprint import enroll, db_path
        print("声紋登録: 音声準備中（キャッシュ再利用）...", flush=True)
        audio_path = ensure_audio(str(enroll_video))
        print("声紋登録: 話者識別中...", flush=True)
        context = load_context(args.context) if args.context else None
        num_speakers = args.speakers if args.speakers is not None else expected_speakers(context)
        pipeline = load_diarization_pipeline()
        diar = diarize_audio(audio_path, pipeline, num_speakers=num_speakers)
        report = enroll(args.voiceprints, audio_path, mapping, diar)
        print(f"声紋登録完了: {db_path(args.voiceprints)}")
        for name, info in report.items():
            print(f"  {name}: {info}")
        return

    # Resolve-speakers mode: 文字起こしせず、話者同一性ヒントだけ算出してサイドカーへ保存
    if args.resolve_speakers:
        if not args.video_path:
            parser.error("--resolve-speakers には動画ファイルのパスが必要です")
        rv = Path(args.video_path).resolve()
        if not rv.exists():
            print(f"エラー: ファイルが見つかりません: {rv}", file=sys.stderr)
            sys.exit(1)

        # 案件 → 声紋プロファイル/コンテキスト解決（--project 指定時）
        voiceprint_profile = args.voiceprints
        context = None
        if args.project:
            from .context_store import store_path, load_project
            proj = load_project(args.project)
            if proj:
                if not voiceprint_profile:
                    voiceprint_profile = proj.get("voiceprint_profile") or args.project
                context = load_context(str(store_path(args.project)))

        from .transcriber import ensure_audio
        import json as _json
        print("話者ヒント: 音声準備中（キャッシュ再利用）...", flush=True)
        audio_path = ensure_audio(str(rv))

        backend = resolve_diarizer_backend(args)
        if backend == "pyannote":
            from .diarization_v2 import load_diarization_pipeline, diarize_audio
        else:
            from .diarization_speakrs import load_diarization_pipeline, diarize_audio
        num_speakers = args.speakers if args.speakers is not None else expected_speakers(context)
        print("話者ヒント: 話者識別中...", flush=True)
        pipeline = load_diarization_pipeline()
        diar = diarize_audio(audio_path, pipeline, num_speakers=num_speakers)

        from .voiceprint import cluster_similarity, identify, identify_segments
        print("話者ヒント: 声紋クラスタ解析中...", flush=True)
        resolve = cluster_similarity(audio_path, diar)
        if voiceprint_profile:
            try:
                name_map = identify(voiceprint_profile, audio_path, diar, auto_update=False)
                resolve["identified"] = {k: v for k, v in name_map.items() if v}
            except Exception as e:
                print(f"  声紋識別スキップ（{type(e).__name__}）", file=sys.stderr)
            mixed_labels = [w["label"] for w in resolve.get("mixed_warnings", [])]
            if mixed_labels:
                try:
                    resolve["segment_relabel"] = identify_segments(
                        voiceprint_profile, audio_path, diar, labels=mixed_labels
                    )
                except Exception:
                    pass

        out = Path(args.output) if args.output else (rv.parent / f"{rv.stem}_speakers.json")
        out.write_text(_json.dumps(resolve, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"話者ヒント保存: {out}", flush=True)
        if resolve.get("merge_suggestions"):
            print("  統合候補(同一の可能性): " + ", ".join("+".join(m["labels"]) for m in resolve["merge_suggestions"]), flush=True)
        if resolve.get("mixed_warnings"):
            print("  混在の可能性(過少分割): " + ", ".join(w["label"] for w in resolve["mixed_warnings"]), flush=True)
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

    # 話者識別バックエンド: speakrs(既定/最速) > pyannote > simple
    backend = resolve_diarizer_backend(args)
    diarization_mode = {
        "speakrs": "speakrs (CoreML)",
        "pyannote": "pyannote.audio",
    }[backend]

    # 案件コンテキスト（任意）: ASR固有名詞 + 文字起こし後の決定的正規化
    # --project <slug> 指定時は案件ストア(~/.claude/meeting-contexts/<slug>.yaml)を context として使い、
    # 同一slugの声紋プロファイルも既定で有効にする（声紋と案件ストアを slug で連結）。
    context_path = args.context
    voiceprint_profile = args.voiceprints
    if args.project:
        from .context_store import store_path, load_project
        proj = load_project(args.project)
        if proj is None:
            print(f"    案件『{args.project}』はまだ未登録（初回として進めます）", flush=True)
        else:
            if not context_path:
                context_path = str(store_path(args.project))
            if not voiceprint_profile:
                voiceprint_profile = proj.get("voiceprint_profile") or args.project
    context = load_context(context_path) if context_path else None
    glossary = asr_glossary(context) if context else None

    print(f"動画: {video_path}", flush=True)
    print(f"出力: {output_path}", flush=True)
    print(f"モデル: {args.model} ({mode})", flush=True)
    print(f"話者識別: {diarization_mode}", flush=True)
    if context:
        src = f"案件:{args.project}" if args.project else context_path
        print(f"案件コンテキスト: {src}（固有名詞 {len(glossary)} 件）", flush=True)
    print(flush=True)

    # Step 1: Transcribe
    print("1/3 音声抽出・文字起こし中...", flush=True)
    whisper_result, audio_path = transcribe_video(str(video_path), args.model, max_accuracy, glossary)
    print(f"    完了 ({len(whisper_result.get('segments', []))} セグメント)", flush=True)

    # Import diarization module（backend に応じて。いずれも同じ3関数を提供）
    if backend == "pyannote":
        from .diarization_v2 import load_diarization_pipeline, diarize_audio, assign_speakers_to_segments
    else:
        from .diarization_speakrs import load_diarization_pipeline, diarize_audio, assign_speakers_to_segments

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
        # 話者数ヒント: --speakers 明示 > 案件コンテキスト(expected_speakers/roster件数)
        num_speakers = args.speakers if args.speakers is not None else expected_speakers(context)
        print("2/3 話者識別中...", flush=True)
        if num_speakers:
            print(f"    話者数ヒント: {num_speakers}（過分割を抑制）", flush=True)
        print("    モデル読み込み中...", flush=True)
        pipeline = load_diarization_pipeline()
        print("    解析中...", flush=True)
        diarization_segments = diarize_audio(audio_path, pipeline, num_speakers=num_speakers)

        # 声紋識別（任意）: --voiceprints/--project 指定時、diarizationクラスタを声紋DBと照合して実名化
        name_map = None
        if voiceprint_profile:
            try:
                from .voiceprint import identify
                name_map = identify(voiceprint_profile, audio_path, diarization_segments, auto_update=True)
                hit = {k: v for k, v in name_map.items() if v}
                print(f"    声紋識別: {hit if hit else '一致なし（発話者Nのまま）'}", flush=True)
            except Exception as e:
                print(f"    声紋識別スキップ（{type(e).__name__}: {str(e)[:80]}）", file=sys.stderr)

        segments_with_speakers = assign_speakers_to_segments(whisper_result, diarization_segments, name_map=name_map)
        unique_speakers = sorted(set(seg["speaker"] for seg in segments_with_speakers if seg["speaker"] != "不明"))
        print(f"    完了 (話者: {', '.join(unique_speakers)})", flush=True)

        # 話者同一性ヒント（柱2）: 1次結果は直さず、議事録(成果物)側で過分割/過少分割を正すための材料。
        # 声紋クラスタ類似度＋混在検出を算出し、声紋が育っていれば混在ラベルを区間単位で実名照合する。
        # 結果はログに出し、サイドカー <transcript>_speakers.json に保存して skill から参照させる。
        if not args.no_speaker_hints:
            try:
                from .voiceprint import cluster_similarity, identify_segments
                resolve = cluster_similarity(audio_path, diarization_segments)
                if name_map:
                    resolve["identified"] = {k: v for k, v in name_map.items() if v}
                mixed_labels = [w["label"] for w in resolve.get("mixed_warnings", [])]
                if voiceprint_profile and mixed_labels:
                    resolve["segment_relabel"] = identify_segments(
                        voiceprint_profile, audio_path, diarization_segments, labels=mixed_labels
                    )
                merges = resolve.get("merge_suggestions", [])
                mixed = resolve.get("mixed_warnings", [])
                if merges:
                    summary = "; ".join(f"{'+'.join(m['labels'])}~{m['score']}" for m in merges)
                    print(f"    話者統合候補（同一の可能性）: {summary}", flush=True)
                if mixed:
                    print(f"    混在の可能性（過少分割）: {', '.join(w['label'] for w in mixed)}", flush=True)
                import json as _json
                sidecar = output_path.with_name(output_path.stem + "_speakers.json")
                sidecar.write_text(_json.dumps(resolve, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"    話者ヒント: {sidecar}", flush=True)
            except Exception as e:
                print(f"    話者ヒント算出スキップ（{type(e).__name__}: {str(e)[:80]}）", file=sys.stderr)

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
