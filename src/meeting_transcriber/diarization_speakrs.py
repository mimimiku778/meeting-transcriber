"""Speaker diarization via speakrs (Rust / Apple Silicon CoreML).

speakrs (https://github.com/avencera/speakrs) は pyannote community-1 のパイプライン
（segmentation + VBx + PLDA）を Rust で再実装したもの。Apple Silicon の CoreML 上で
pyannote 同等精度（VoxConverse 7.1% DER）かつ桁違いに高速（〜500x realtime）。

本モジュールは `native/speakrs-diarizer` でビルドした薄い CLI ラッパーを subprocess で
呼び出し、RTTM 出力を [{'start','end','speaker'}] へ変換する。pyannote 版（diarization_v2）
と同じインターフェース（load_diarization_pipeline / diarize_audio / assign_speakers_to_segments）
を提供するため、cli.py からは差し替えるだけで使える。

ビルド:
    cd native/speakrs-diarizer
    PKG_CONFIG_PATH=/opt/homebrew/opt/openblas/lib/pkgconfig \
    LIBRARY_PATH=/opt/homebrew/opt/openblas/lib \
    cargo build --release
"""

import os
import subprocess
import tempfile
from pathlib import Path

import torchaudio

# 話者割当ては backend 非依存なので pyannote 版の実装を再利用する
from .diarization_v2 import assign_speakers_to_segments  # noqa: F401

# リポジトリ同梱バイナリの既定パス（src/meeting_transcriber/ から見て repo ルート）
_DEFAULT_BIN = (
    Path(__file__).resolve().parents[2] / "native" / "speakrs-diarizer" / "target" / "release" / "speakrs-diarizer"
)


def _resolve_binary() -> Path:
    """speakrs-diarizer バイナリのパスを解決する。MEETING_SPEAKRS_BIN で上書き可。"""
    override = os.environ.get("MEETING_SPEAKRS_BIN")
    binary = Path(override) if override else _DEFAULT_BIN
    if not binary.exists():
        raise RuntimeError(
            "speakrs-diarizer バイナリが見つかりません: "
            f"{binary}\n  native/speakrs-diarizer で `cargo build --release` を実行してください"
            "（openblas が必要: brew install openblas）。"
        )
    return binary


def binary_available() -> bool:
    """speakrs-diarizer バイナリがビルド済みで使えるか。cli の自動フォールバック判定用。"""
    try:
        _resolve_binary()
        return True
    except RuntimeError:
        return False


def load_diarization_pipeline() -> dict:
    """speakrs は別プロセスなので「パイプライン」= バイナリパス + 実行モード。

    実行モードは MEETING_SPEAKRS_MODE で切替（coreml=既定 / coreml-fast / cpu）。
    coreml-fast は 2秒窓で高速だが話者切替の時刻精度はやや粗い。
    """
    binary = _resolve_binary()
    mode = os.environ.get("MEETING_SPEAKRS_MODE", "coreml").lower()
    if mode not in ("coreml", "coreml-fast", "cpu"):
        mode = "coreml"
    print(f"    話者識別: speakrs ({mode}) / {binary.name}", flush=True)
    return {"binary": str(binary), "mode": mode}


def _to_16k_mono_wav(audio_path: str, dst: str) -> None:
    """任意の音声を 16kHz mono / 16-bit PCM WAV に整える（speakrs/hound が読む形式）。"""
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    torchaudio.save(dst, waveform, 16000, encoding="PCM_S", bits_per_sample=16)


def _parse_rttm(rttm: str) -> list[dict]:
    """RTTM テキストを [{'start','end','speaker'}] へ。

    行形式: SPEAKER <file> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>
    """
    segments = []
    for line in rttm.splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        dur = float(parts[4])
        speaker = parts[7]
        segments.append({"start": start, "end": start + dur, "speaker": speaker})
    segments.sort(key=lambda s: s["start"])
    return segments


def diarize_audio(audio_path: str, pipeline: dict = None, num_speakers: int = None) -> list[dict]:
    """speakrs で話者分離し [{'start','end','speaker'}] を返す。

    num_speakers: speakrs(VBx) はクラスタ数を自動推定するため未使用（ヒント不要）。
    引数は pyannote 版と互換にするために受け取るだけ。
    """
    if pipeline is None:
        pipeline = load_diarization_pipeline()

    with tempfile.TemporaryDirectory() as tmp:
        wav = str(Path(tmp) / "audio16k.wav")
        _to_16k_mono_wav(audio_path, wav)
        proc = subprocess.run(
            [pipeline["binary"], wav, pipeline["mode"]],
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"speakrs-diarizer 失敗 (exit {proc.returncode}): {proc.stderr.strip()}")
    return _parse_rttm(proc.stdout)
