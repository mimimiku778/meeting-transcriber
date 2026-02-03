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
}

JAPANESE_MEETING_PROMPT = """これは日本語の会議の文字起こしです。
話者は「はい」「えーと」「あの」などの相槌や言い淀みを使います。
丁寧語と敬語が使われます。"""


def extract_audio(video_path: str, output_path: str) -> None:
    cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", output_path]
    subprocess.run(cmd, check=True, capture_output=True)


def transcribe_audio(audio_path: str, model_name: str = "medium", max_accuracy: bool = True) -> dict:
    model_repo = MLX_MODELS.get(model_name, MLX_MODELS["large-v3"])

    if max_accuracy:
        return mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_repo,
            language="ja",
            word_timestamps=True,
            verbose=False,
            initial_prompt=JAPANESE_MEETING_PROMPT,
            temperature=0,
            condition_on_previous_text=True,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            hallucination_silence_threshold=0.5,
        )
    else:
        return mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_repo,
            language="ja",
            word_timestamps=True,
            verbose=False,
        )


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def transcribe_video(video_path: str, model_name: str = "medium", max_accuracy: bool = True) -> tuple[dict, str]:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    audio_dir = Path(tempfile.gettempdir()) / "meeting_transcriber"
    audio_dir.mkdir(exist_ok=True)
    audio_path = audio_dir / f"{video_path.stem}.wav"

    extract_audio(str(video_path), str(audio_path))
    result = transcribe_audio(str(audio_path), model_name, max_accuracy)

    return result, str(audio_path)
