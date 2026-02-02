"""Speaker diarization module using pyannote-audio."""

import os
from pathlib import Path

# Force CPU for pyannote-audio (MPS doesn't support sparse tensors)
# Must be set before torch import
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

# Monkey-patch MPS availability to force CPU usage
# pyannote-audio checks this internally and will use MPS if available
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

from pyannote.audio import Pipeline


def load_diarization_pipeline() -> Pipeline:
    """Load pyannote speaker diarization pipeline."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required for pyannote-audio")

    # Note: pyannote-audio uses sparse tensor ops not supported on MPS
    # Force CPU for compatibility on Apple Silicon
    device = torch.device("cpu")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )
    pipeline.to(device)

    return pipeline


def diarize_audio(audio_path: str, pipeline: Pipeline = None, num_speakers: int = None) -> list[dict]:
    """
    Perform speaker diarization on audio file.

    Args:
        audio_path: Path to audio file
        pipeline: Diarization pipeline (loaded if None)
        num_speakers: Number of speakers (optional, improves accuracy)

    Returns:
        list of dicts with 'start', 'end', 'speaker' keys
    """
    if pipeline is None:
        pipeline = load_diarization_pipeline()

    # Load audio using torchaudio (avoids torchcodec AudioDecoder issue)
    import torchaudio
    waveform, sample_rate = torchaudio.load(audio_path)

    # pyannote expects dict with 'waveform' and 'sample_rate'
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    # Pass num_speakers if specified
    if num_speakers is not None:
        result = pipeline(audio_input, num_speakers=num_speakers)
    else:
        result = pipeline(audio_input)

    # Handle both DiarizeOutput (new API) and Annotation (old API)
    if hasattr(result, 'speaker_diarization'):
        # New API: DiarizeOutput dataclass
        diarization = result.speaker_diarization
    else:
        # Old API: direct Annotation object
        diarization = result

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    return segments


def assign_speakers_to_segments(whisper_result: dict, diarization_segments: list[dict]) -> list[dict]:
    """
    Assign speaker labels to Whisper segments based on diarization.

    Args:
        whisper_result: Whisper transcription result
        diarization_segments: List of diarization segments

    Returns:
        List of segments with speaker assignments
    """
    result_segments = []

    # Create a mapping of speaker IDs to numbered labels
    unique_speakers = sorted(set(seg["speaker"] for seg in diarization_segments))
    speaker_map = {spk: f"発話者{i+1}" for i, spk in enumerate(unique_speakers)}

    for segment in whisper_result.get("segments", []):
        seg_start = segment["start"]
        seg_end = segment["end"]
        seg_mid = (seg_start + seg_end) / 2

        # Find the diarization segment that best matches this segment
        best_speaker = None
        best_overlap = 0

        for diar_seg in diarization_segments:
            # Calculate overlap
            overlap_start = max(seg_start, diar_seg["start"])
            overlap_end = min(seg_end, diar_seg["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar_seg["speaker"]

        # If no overlap found, use the segment containing the midpoint
        if best_speaker is None:
            for diar_seg in diarization_segments:
                if diar_seg["start"] <= seg_mid <= diar_seg["end"]:
                    best_speaker = diar_seg["speaker"]
                    break

        speaker_label = speaker_map.get(best_speaker, "不明") if best_speaker else "不明"

        result_segments.append({
            "start": seg_start,
            "end": seg_end,
            "text": segment["text"].strip(),
            "speaker": speaker_label,
            "original_speaker_id": best_speaker
        })

    return result_segments
