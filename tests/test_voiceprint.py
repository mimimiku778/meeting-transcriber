"""声紋ユーティリティ＋話者同一性ヒント（過分割の統合候補・過少分割の混在検出）のテスト。

埋め込み計算(_embed_segments)はモックに差し替え、torch/pyannote 無しでロジックを検証する。
"""

import numpy as np

from meeting_transcriber import voiceprint as vp


def test_cosine():
    a = np.array([1.0, 0.0, 0.0])
    assert vp._cosine(a, a) == 1.0
    assert vp._cosine(a, np.array([0.0, 1.0, 0.0])) == 0.0
    assert vp._cosine(a, np.zeros(3)) == 0.0  # ゼロベクトルは0


def test_label_sort_key():
    assert vp._label_sort_key("発話者12") == 12
    assert vp._label_sort_key("発話者1") == 1
    assert vp._label_sort_key("不明") == 10**9


def test_label_to_cluster_numbering():
    diar = [{"start": 0, "end": 1, "speaker": "SPEAKER_02"}, {"start": 1, "end": 2, "speaker": "SPEAKER_00"}]
    # unique を昇順採番 → 発話者1=SPEAKER_00, 発話者2=SPEAKER_02
    assert vp._label_to_cluster(diar) == {"発話者1": "SPEAKER_00", "発話者2": "SPEAKER_02"}


def _make_diar():
    """発話者1=A, 2=B, 3=A前半/C後半(混在), 4=C, 5=A, 6=A(=5と同一人物)。"""
    return [
        {"start": 0, "end": 3, "speaker": "S1"},
        {"start": 3, "end": 6, "speaker": "S1"},
        {"start": 6, "end": 9, "speaker": "S2"},
        {"start": 9, "end": 12, "speaker": "S2"},
        {"start": 12, "end": 30, "speaker": "S3"},
        {"start": 30, "end": 48, "speaker": "S3"},
        {"start": 48, "end": 51, "speaker": "S5"},
        {"start": 51, "end": 70, "speaker": "S6"},
        {"start": 70, "end": 90, "speaker": "S7"},
    ]


def test_cluster_similarity_merge_and_mixed(monkeypatch):
    rng = np.random.RandomState(0)
    A, B, C = rng.randn(64), rng.randn(64), rng.randn(64)
    diar = _make_diar()
    truth = {"S1": A, "S2": B, "S5": C, "S6": A, "S7": A}

    def fake_embed(audio_path, segments):
        s, e = segments[0]
        spk = next((d["speaker"] for d in diar if d["start"] == s and d["end"] == e), "S1")
        base = (A if s < 24 else C) if spk == "S3" else truth.get(spk, A)
        return base + 0.02 * rng.randn(64)

    monkeypatch.setattr(vp, "_embed_segments", fake_embed)
    res = vp.cluster_similarity("dummy.wav", diar)

    # 過分割: 発話者5 と 発話者6 が同一人物として統合候補に挙がる
    merged_pairs = {frozenset(m["labels"]) for m in res["merge_suggestions"]}
    assert frozenset({"発話者5", "発話者6"}) in merged_pairs

    # 過少分割: 発話者3 が混在の疑いとして警告される
    mixed_labels = {w["label"] for w in res["mixed_warnings"]}
    assert "発話者3" in mixed_labels


def test_cluster_similarity_detect_mixed_off(monkeypatch):
    rng = np.random.RandomState(1)
    base = rng.randn(64)
    diar = [{"start": 0, "end": 5, "speaker": "S1"}, {"start": 5, "end": 10, "speaker": "S1"}]
    monkeypatch.setattr(vp, "_embed_segments", lambda a, s: base + 0.01 * rng.randn(64))
    res = vp.cluster_similarity("dummy.wav", diar, detect_mixed=False)
    assert res["mixed_warnings"] == []
