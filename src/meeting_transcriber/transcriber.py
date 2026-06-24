"""Whisper-based transcription module using mlx-whisper for Apple Silicon."""

import subprocess
import tempfile
from pathlib import Path

import mlx_whisper

MLX_MODELS = {
    "small-4bit": "mlx-community/whisper-small-mlx-4bit",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    # fp16フル精度・約1.6GB。日本語はlarge-v2同等精度をmedium並みの速度で出す（推奨デフォルト）
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

DEFAULT_MODEL = "large-v3-turbo"

# 句読点・語尾・相槌を例示して日本語の文体を安定させるベースプロンプト。
# Whisperのpromptは「命令」ではなく「文体/語彙の例示」として働く点に注意。
_BASE_PROMPT = (
    "これは日本語のビジネス会議の文字起こしです。"
    "話者は敬語で話します。発言は「です。」「ます。」で終わります。"
    "相槌「はい。」「なるほど。」や言い淀み「えーと、」が含まれます。"
)


def _build_prompt(glossary: list[str] | None, limit: int = 224) -> str:
    """ベース文 + 案件用語を 224トークン以内に末尾配置で合成する。

    入りきらない場合は末尾（＝低優先）の語から削る。high-priority語は
    glossary の先頭に置くこと。トークナイザが無い環境では文字数で近似する。
    """
    if not glossary:
        return _BASE_PROMPT
    try:
        from mlx_whisper.tokenizer import get_tokenizer
        tok = get_tokenizer(multilingual=True, language="ja", task="transcribe")
        def fits(text: str) -> bool:
            return len(tok.encode(" " + text.strip())) <= limit
    except Exception:
        # フォールバック: 日本語は概ね1文字>=1トークンなので文字数で安全側に近似
        def fits(text: str) -> bool:
            return len(text) <= limit

    terms = [t for t in glossary if t]
    while terms:
        candidate = _BASE_PROMPT + " 固有名詞: " + "、".join(terms) + "。"
        if fits(candidate):
            return candidate
        terms.pop()
    return _BASE_PROMPT


def extract_audio(video_path: str, output_path: str) -> None:
    cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", output_path]
    subprocess.run(cmd, check=True, capture_output=True)


def ensure_audio(video_path: str, audio_path: str | None = None) -> str:
    """16kHz mono wav を用意して返す。既存wavが動画より新しければ再抽出しない（高速化）。

    transcribe→enroll→resolve_speakers と同じ動画を複数回処理する際、ffmpeg抽出の重複を
    避けるためのキャッシュ。パスは動画stem基準で決定的なので各処理から共有される。
    """
    video = Path(video_path)
    if audio_path is None:
        audio_dir = Path(tempfile.gettempdir()) / "meeting_transcriber"
        audio_dir.mkdir(exist_ok=True)
        audio_path = audio_dir / f"{video.stem}.wav"
    audio = Path(audio_path)
    try:
        if audio.exists() and audio.stat().st_size > 0 and audio.stat().st_mtime >= video.stat().st_mtime:
            return str(audio)
    except OSError:
        pass
    extract_audio(str(video), str(audio))
    return str(audio)


def transcribe_audio(
    audio_path: str,
    model_name: str = DEFAULT_MODEL,
    max_accuracy: bool = True,
    glossary: list[str] | None = None,
) -> dict:
    """音声を文字起こしする。

    max_accuracy=True では MEMORY.md の日本語ベストプラクティスに準拠した
    ハルシネーション抑制パラメータを適用する。glossary を渡すと initial_prompt
    に固有名詞を注入する（ただし下記の制約に注意）。

    重要な制約: condition_on_previous_text=False のため initial_prompt は
    冒頭セグメントにしか効かず会議後半へ伝播しない。固有名詞の確実な正規化は
    ASR後の決定的置換 + Claude校正（議事録生成スキル側）で担保する設計。
    """
    model_repo = MLX_MODELS.get(model_name, MLX_MODELS[DEFAULT_MODEL])

    if not max_accuracy:
        return mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_repo,
            language="ja",
            word_timestamps=True,
            verbose=False,
        )

    return mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_repo,
        language="ja",
        word_timestamps=True,
        verbose=False,
        initial_prompt=_build_prompt(glossary),
        # 閾値違反時のみ昇温して再デコード（標準のフォールバック挙動）
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        condition_on_previous_text=False,    # 最重要: 繰り返しハルシネーションの伝播を断つ
        compression_ratio_threshold=1.8,     # 2.4 -> 1.8（より厳格に棄却）
        logprob_threshold=-0.5,              # -1.0 -> -0.5（低確信度を棄却）
        no_speech_threshold=0.3,             # 0.6 -> 0.3（無音区間を積極スキップ）
        hallucination_silence_threshold=2.0, # 0.5 -> 2.0（無音区間の幻聴除去）
    )


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def transcribe_video(
    video_path: str,
    model_name: str = DEFAULT_MODEL,
    max_accuracy: bool = True,
    glossary: list[str] | None = None,
) -> tuple[dict, str]:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    audio_path = ensure_audio(str(video_path))
    result = transcribe_audio(audio_path, model_name, max_accuracy, glossary)

    return result, str(audio_path)
