"""Speaker diarization module using SpeechBrain (simple-diarizer)."""

import os
from pathlib import Path
from simple_diarizer.diarizer import Diarizer

# Global diarizer instance (lazy loading)
_diarizer = None


def load_diarization_pipeline() -> Diarizer:
    """Load SpeechBrain-based diarizer (faster than pyannote)."""
    global _diarizer
    if _diarizer is None:
        # Set model cache directory to user's home to avoid read-only filesystem issues
        cache_dir = Path.home() / ".cache" / "meeting-transcriber" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Change to cache dir temporarily to ensure models are saved there
        original_cwd = os.getcwd()
        try:
            os.chdir(cache_dir)
            # Use ECAPA-TDNN embeddings with spectral clustering
            _diarizer = Diarizer(
                embed_model='ecapa',
                cluster_method='sc'  # spectral clustering
            )
        finally:
            os.chdir(original_cwd)
    return _diarizer


def diarize_audio(audio_path: str, pipeline: Diarizer = None, num_speakers: int = None) -> list[dict]:
    """
    Perform speaker diarization on audio file.

    Args:
        audio_path: Path to audio file
        pipeline: Diarizer instance (loaded if None)
        num_speakers: Number of speakers (optional, improves accuracy)

    Returns:
        list of dicts with 'start', 'end', 'speaker' keys
    """
    if pipeline is None:
        pipeline = load_diarization_pipeline()

    # Diarize
    if num_speakers is not None:
        segments = pipeline.diarize(audio_path, num_speakers=num_speakers)
    else:
        # Auto-detect number of speakers
        segments = pipeline.diarize(audio_path, threshold=0.5)

    # Convert to our format
    result = []
    for seg in segments:
        result.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": f"SPEAKER_{seg['label']}"
        })

    return result


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
