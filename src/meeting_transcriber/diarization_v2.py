"""Speaker diarization module using pyannote.audio (high-accuracy)."""

import torch
import torchaudio
from pyannote.audio import Pipeline

# Global pipeline instance (lazy loading)
_pipeline = None


def _get_device() -> torch.device:
    """Select best available device: MPS > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    torch.set_num_threads(8)
    return torch.device("cpu")


def load_diarization_pipeline() -> Pipeline:
    """Load pyannote.audio speaker-diarization-3.1 pipeline."""
    global _pipeline
    if _pipeline is None:
        device = _get_device()
        _pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        _pipeline.to(device)
        # Increase segmentation step: 0.1 (default) -> 0.25 (~2.5x faster, minimal accuracy loss)
        _pipeline._segmentation.step = _pipeline._segmentation.duration * 0.25
        print(f"    デバイス: {device}", flush=True)
    return _pipeline


def diarize_audio(audio_path: str, pipeline: Pipeline = None, num_speakers: int = None) -> list[dict]:
    """
    Perform speaker diarization on audio file using pyannote.audio.

    Args:
        audio_path: Path to audio file
        pipeline: Pipeline instance (loaded if None)
        num_speakers: Number of speakers (optional, improves accuracy)

    Returns:
        list of dicts with 'start', 'end', 'speaker' keys
    """
    if pipeline is None:
        pipeline = load_diarization_pipeline()

    params = {}
    if num_speakers is not None:
        params["num_speakers"] = num_speakers

    # Load audio and convert to 16kHz mono (pyannote's expected format)
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        sample_rate = 16000
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    result = pipeline(audio_input, **params)

    # pyannote.audio 4.x returns DiarizeOutput; 3.x returns Annotation
    annotation = getattr(result, "speaker_diarization", result)

    return [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


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
