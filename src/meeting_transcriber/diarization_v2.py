"""Speaker diarization module using pyannote.audio (high-accuracy).

speaker-diarization-community-1 を優先（取り違えが3.1比で大幅減・商用可CC-BY-4.0）。
未取得環境では3.1へフォールバック。いずれもgatedモデルのため初回はHF_TOKEN必要だが、
キャッシュ済みならオフラインでトークン無しロード可。
"""

import os
from pathlib import Path

import torch
import torchaudio
from pyannote.audio import Pipeline

# Global pipeline instance (lazy loading)
_pipeline = None

# 優先順: community-1（高精度・商用可） -> 3.1（従来）
_MODEL_CANDIDATES = [
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
]


def _get_device() -> torch.device:
    """既定はcpu（pyannote+MPSはタイムスタンプ崩れの報告があるため）。
    MEETING_DIARIZER_DEVICE=mps で明示的にMPSを使う（検証扱い）。"""
    want = os.environ.get("MEETING_DIARIZER_DEVICE", "cpu").lower()
    if want == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    torch.set_num_threads(8)
    return torch.device("cpu")


def load_diarization_pipeline() -> Pipeline:
    """pyannote.audio の話者分離パイプラインをロードする。"""
    global _pipeline
    if _pipeline is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        errors = []
        for model_id in _MODEL_CANDIDATES:
            try:
                # token=None でもキャッシュ済みならロード可（オフライン）
                _pipeline = Pipeline.from_pretrained(model_id, token=token)
                if _pipeline is not None:
                    _pipeline.to(_get_device())
                    print(f"    話者識別モデル: {model_id} / デバイス: {_get_device()}", flush=True)
                    return _pipeline
            except Exception as e:  # noqa: BLE001
                errors.append(f"{model_id}: {type(e).__name__}: {str(e)[:120]}")
        raise RuntimeError(
            "pyannote話者識別モデルをロードできません。gatedモデルのため、初回は "
            "huggingface.co で利用規約に同意し HF_TOKEN を環境変数で渡してください"
            "（MCP登録時は -e HF_TOKEN=...）。詳細:\n  " + "\n  ".join(errors)
        )
    return _pipeline


def diarize_audio(audio_path: str, pipeline: Pipeline = None, num_speakers: int = None) -> list[dict]:
    """話者分離を実行し [{'start','end','speaker'}] を返す。"""
    if pipeline is None:
        pipeline = load_diarization_pipeline()

    params = {}
    if num_speakers is not None:
        params["num_speakers"] = num_speakers

    # 16kHz mono に整える（pyannoteの期待形式）
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        sample_rate = 16000
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    result = pipeline(audio_input, **params)

    # pyannote 4.x は DiarizeOutput、3.x は Annotation を返す
    annotation = getattr(result, "speaker_diarization", result)

    return [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def _speaker_at(t: float, diarization_segments: list[dict]) -> str | None:
    """時刻 t を含む（最も近い）diarizationセグメントの話者を返す。"""
    for d in diarization_segments:
        if d["start"] <= t <= d["end"]:
            return d["speaker"]
    # 含むものが無ければ最も近いセグメント
    best, best_dist = None, None
    for d in diarization_segments:
        dist = 0 if d["start"] <= t <= d["end"] else min(abs(t - d["start"]), abs(t - d["end"]))
        if best_dist is None or dist < best_dist:
            best, best_dist = d["speaker"], dist
    return best


def assign_speakers_to_segments(
    whisper_result: dict,
    diarization_segments: list[dict],
    name_map: dict[str, str] | None = None,
) -> list[dict]:
    """Whisperセグメントへ話者を割当てる。

    word単位の中点投票（多数決）で話者を決める。これにより1セグメントに
    質問→相槌→受け が混ざるケースの話者取り違えを軽減する。
    word情報が無い場合はセグメント中点で割当てる（従来方式へフォールバック）。

    name_map（声紋識別の結果 {'発話者1':'山田',...}）が渡された話者は『発話者N』の
    代わりに実名を表示する。未識別(None)や未指定はそのまま『発話者N』。
    """
    result_segments = []

    unique_speakers = sorted(set(seg["speaker"] for seg in diarization_segments))
    speaker_map = {}
    for i, spk in enumerate(unique_speakers):
        label = f"発話者{i+1}"
        speaker_map[spk] = (name_map.get(label) or label) if name_map else label

    for segment in whisper_result.get("segments", []):
        seg_start = segment["start"]
        seg_end = segment["end"]

        words = segment.get("words") or []
        votes: dict[str, float] = {}
        for w in words:
            ws, we = w.get("start"), w.get("end")
            if ws is None or we is None:
                continue
            mid = (ws + we) / 2
            spk = _speaker_at(mid, diarization_segments)
            if spk is not None:
                # 単語長で重み付け（長い単語ほど話者の手掛かりが強い）
                votes[spk] = votes.get(spk, 0.0) + max(0.01, we - ws)

        if votes:
            best_speaker = max(votes, key=votes.get)
        else:
            # フォールバック: 最大オーバーラップ → セグメント中点
            best_speaker, best_overlap = None, 0.0
            for d in diarization_segments:
                overlap = max(0.0, min(seg_end, d["end"]) - max(seg_start, d["start"]))
                if overlap > best_overlap:
                    best_overlap, best_speaker = overlap, d["speaker"]
            if best_speaker is None:
                best_speaker = _speaker_at((seg_start + seg_end) / 2, diarization_segments)

        speaker_label = speaker_map.get(best_speaker, "不明") if best_speaker else "不明"

        result_segments.append({
            "start": seg_start,
            "end": seg_end,
            "text": segment["text"].strip(),
            "speaker": speaker_label,
            "original_speaker_id": best_speaker,
        })

    return result_segments
