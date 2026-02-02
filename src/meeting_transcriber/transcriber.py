"""Whisper-based transcription module using mlx-whisper for Apple Silicon."""

import subprocess
import tempfile
from pathlib import Path

import mlx_whisper


# Model mapping for mlx-whisper (accuracy: tiny < base < small < medium < large < large-v3)
MLX_MODELS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",  # 速度重視（精度はlarge-v3より若干劣る）
}

# 日本語会議用の初期プロンプト（専門用語や話し言葉の認識精度向上）
JAPANESE_MEETING_PROMPT = """これは日本語の会議の文字起こしです。
話者は「はい」「えーと」「あの」などの相槌や言い淀みを使います。
丁寧語と敬語が使われます。"""


def extract_audio(video_path: str, output_path: str) -> None:
    """Extract audio from video file using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def transcribe_audio(audio_path: str, model_name: str = "large-v3", max_accuracy: bool = True) -> dict:
    """
    Transcribe audio using mlx-whisper (Apple Silicon optimized).

    Args:
        audio_path: Path to audio file
        model_name: Model size (tiny/base/small/medium/large/large-v3/turbo)
        max_accuracy: Use maximum accuracy settings (slower but more accurate)
    """
    model_repo = MLX_MODELS.get(model_name, MLX_MODELS["large-v3"])

    # 最大精度設定
    if max_accuracy:
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_repo,
            language="ja",
            word_timestamps=True,
            verbose=False,
            # 精度向上設定
            initial_prompt=JAPANESE_MEETING_PROMPT,
            temperature=0,  # グリーディデコード（確定的、精度向上）
            condition_on_previous_text=True,  # 前のテキストを考慮（文脈の一貫性）
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            hallucination_silence_threshold=0.5,  # 無音時のハルシネーション防止
        )
    else:
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_repo,
            language="ja",
            word_timestamps=True,
            verbose=False,
        )

    return result


def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def transcribe_video(video_path: str, model_name: str = "large-v3", max_accuracy: bool = True) -> tuple[dict, str]:
    """
    Transcribe video file.

    Args:
        video_path: Path to video file
        model_name: Whisper model size
        max_accuracy: Use maximum accuracy settings

    Returns:
        tuple: (whisper_result, audio_path)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Extract audio to temp file
    audio_dir = Path(tempfile.gettempdir()) / "meeting_transcriber"
    audio_dir.mkdir(exist_ok=True)
    audio_path = audio_dir / f"{video_path.stem}.wav"

    extract_audio(str(video_path), str(audio_path))

    # Transcribe
    result = transcribe_audio(str(audio_path), model_name, max_accuracy)

    return result, str(audio_path)
