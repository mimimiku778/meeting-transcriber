"""Whisper-based transcription module using mlx-whisper for Apple Silicon."""

import subprocess
import tempfile
from pathlib import Path

import mlx_whisper


# Model mapping for mlx-whisper
MLX_MODELS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large": "mlx-community/whisper-large-v3-mlx",
}


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


def transcribe_audio(audio_path: str, model_name: str = "medium") -> dict:
    """Transcribe audio using mlx-whisper (Apple Silicon optimized)."""
    model_repo = MLX_MODELS.get(model_name, MLX_MODELS["medium"])

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_repo,
        language="ja",
        word_timestamps=True,
        verbose=False
    )

    return result


def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def transcribe_video(video_path: str, model_name: str = "medium") -> tuple[dict, str]:
    """
    Transcribe video file.

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
    result = transcribe_audio(str(audio_path), model_name)

    return result, str(audio_path)
