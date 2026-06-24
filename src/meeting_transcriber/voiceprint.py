"""話者声紋（speaker embedding）によるエンロールメントと識別。

pyannote の埋め込みモデルで各話者の声紋ベクトルを作り、ローカルのJSON声紋DBに保存する。
新しい会議では diarization 各クラスタの声紋をDBと照合し、実名を自動付与する。

設計の要点:
- 名前は常に「人間が確定した正解ラベル」。声紋はそれを後追いで学習する補助。
- identify が外しても人間の確定が勝ち、その確定で声紋を矯正（再enroll）できる自己修正ループ。
- 保存するのは生音声ではなく特徴ベクトル（embedding）。1話者あたり数KBのJSON。

声紋DBの形式(JSON):
{
  "profile": "myteam",
  "speakers": {
    "山田": {"embedding": [...512 floats...], "enroll_count": 3},
    "鈴木": {...}
  }
}

保存先: ~/.claude/voiceprints/<profile>.json （ローカルのみ。生体情報的データを外に出さない）
"""

import json
import os
from pathlib import Path

import numpy as np

# 埋め込みモデル（gated。初回のみ HF_TOKEN 必要、キャッシュ後はオフライン可）
_EMBEDDING_MODEL = "pyannote/embedding"
_inference = None

# 識別の既定パラメータ
DEFAULT_THRESHOLD = 0.50   # コサイン類似度がこれ未満なら UNKNOWN（発話者Nのまま）
DEFAULT_MARGIN = 0.10      # 1位と2位の差がこれ未満なら曖昧として UNKNOWN
AUTO_UPDATE_MARGIN = 0.15  # 自動平均更新は「閾値＋このマージン」以上の高信頼時のみ
MIN_SEGMENT_SEC = 0.6      # これより短い区間は声紋計算に使わない


def voiceprints_dir() -> Path:
    """声紋DBの保存ディレクトリ（~/.claude/voiceprints/）。"""
    base = Path(os.environ.get("MEETING_VOICEPRINTS_DIR", Path.home() / ".claude" / "voiceprints"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def db_path(profile: str) -> Path:
    safe = "".join(c for c in profile if c.isalnum() or c in "-_") or "default"
    return voiceprints_dir() / f"{safe}.json"


def load_db(profile: str) -> dict:
    path = db_path(profile)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"profile": profile, "speakers": {}}


def save_db(db: dict, profile: str) -> Path:
    path = db_path(profile)
    path.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _get_inference():
    """pyannote 埋め込み推論器を遅延ロードする（window='whole' で区間→単一ベクトル）。"""
    global _inference
    if _inference is None:
        import torch
        from pyannote.audio import Inference
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        device = torch.device("cpu")
        try:
            _inference = Inference(_EMBEDDING_MODEL, window="whole", use_auth_token=token, device=device)
        except TypeError:
            # 新しめの pyannote は use_auth_token を受けない場合がある
            _inference = Inference(_EMBEDDING_MODEL, window="whole", device=device)
    return _inference


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _cluster_segments(diar_segments: list[dict]) -> dict[str, list[tuple[float, float]]]:
    """diarization の生ラベル(SPEAKER_xx)ごとに [(start,end)] を集約する。"""
    clusters: dict[str, list[tuple[float, float]]] = {}
    for d in diar_segments:
        if d["end"] - d["start"] < MIN_SEGMENT_SEC:
            continue
        clusters.setdefault(d["speaker"], []).append((d["start"], d["end"]))
    return clusters


def _embed_segments(audio_path: str, segments: list[tuple[float, float]]) -> np.ndarray | None:
    """区間群の声紋を長さ重み平均で1ベクトルにまとめる。"""
    from pyannote.core import Segment
    inference = _get_inference()
    vecs, weights = [], []
    # 長い区間を優先（話者の手掛かりが強い）。最大8区間まで使えば十分。
    for start, end in sorted(segments, key=lambda s: s[1] - s[0], reverse=True)[:8]:
        try:
            emb = np.asarray(inference.crop(audio_path, Segment(start, end))).reshape(-1)
        except Exception:
            continue
        if emb.size == 0 or not np.isfinite(emb).all():
            continue
        vecs.append(emb)
        weights.append(end - start)
    if not vecs:
        return None
    w = np.asarray(weights)
    return np.average(np.vstack(vecs), axis=0, weights=w)


def _label_to_cluster(diar_segments: list[dict]) -> dict[str, str]:
    """『発話者N』表記 → diarization生ラベル(SPEAKER_xx) の対応。

    assign_speakers_to_segments と同じソート順（unique_speakers昇順）で採番する。
    """
    unique = sorted(set(d["speaker"] for d in diar_segments))
    return {f"発話者{i+1}": spk for i, spk in enumerate(unique)}


def enroll(profile: str, audio_path: str, mapping: dict[str, str], diar_segments: list[dict]) -> dict:
    """人間が確定したマッピングから各実名の声紋を登録/平均更新する。

    mapping: {'発話者1': '山田', '発話者2': '鈴木', ...}（人間がフレームで確定したもの）
    既存声紋があれば enroll_count を重みにした移動平均で更新する。
    戻り値: {実名: {'status': 'new'|'updated', 'count': n}}
    """
    db = load_db(profile)
    speakers = db.setdefault("speakers", {})
    label2cluster = _label_to_cluster(diar_segments)
    clusters = _cluster_segments(diar_segments)

    report = {}
    for label, name in mapping.items():
        spk = label2cluster.get(label, label)  # 'SPEAKER_xx' 直指定も許容
        segs = clusters.get(spk)
        if not segs:
            report[name] = {"status": "skipped", "reason": "no_segments"}
            continue
        new_emb = _embed_segments(audio_path, segs)
        if new_emb is None:
            report[name] = {"status": "skipped", "reason": "embed_failed"}
            continue

        if name in speakers:
            old = np.asarray(speakers[name]["embedding"])
            cnt = int(speakers[name].get("enroll_count", 1))
            merged = (old * cnt + new_emb) / (cnt + 1)
            speakers[name] = {"embedding": merged.tolist(), "enroll_count": cnt + 1}
            report[name] = {"status": "updated", "count": cnt + 1}
        else:
            speakers[name] = {"embedding": new_emb.tolist(), "enroll_count": 1}
            report[name] = {"status": "new", "count": 1}

    save_db(db, profile)
    return report


def identify(
    profile: str,
    audio_path: str,
    diar_segments: list[dict],
    threshold: float = DEFAULT_THRESHOLD,
    margin: float = DEFAULT_MARGIN,
    auto_update: bool = False,
) -> dict[str, str | None]:
    """diarization各クラスタを声紋DBと照合し『発話者N』→実名 の対応を返す。

    戻り値: {'発話者1': '山田' or None, ...}（Noneは UNKNOWN＝発話者Nのまま）
    auto_update=True のとき、高信頼(類似度>=threshold+AUTO_UPDATE_MARGIN かつ
    2位とのmargin十分)で識別できた話者の声紋を平均更新する（誤学習を避けるため高信頼時のみ）。
    """
    db = load_db(profile)
    speakers = db.get("speakers", {})
    label2cluster = _label_to_cluster(diar_segments)
    clusters = _cluster_segments(diar_segments)

    if not speakers:
        return {label: None for label in label2cluster}

    names = list(speakers.keys())
    refs = {n: np.asarray(speakers[n]["embedding"]) for n in names}

    result: dict[str, str | None] = {}
    updated = False
    for label, spk in label2cluster.items():
        segs = clusters.get(spk)
        emb = _embed_segments(audio_path, segs) if segs else None
        if emb is None:
            result[label] = None
            continue
        scored = sorted(((_cosine(emb, refs[n]), n) for n in names), reverse=True)
        best_score, best_name = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= threshold and (best_score - second) >= margin:
            result[label] = best_name
            if auto_update and best_score >= threshold + AUTO_UPDATE_MARGIN:
                old = refs[best_name]
                cnt = int(speakers[best_name].get("enroll_count", 1))
                merged = (old * cnt + emb) / (cnt + 1)
                speakers[best_name] = {"embedding": merged.tolist(), "enroll_count": cnt + 1}
                refs[best_name] = merged
                updated = True
        else:
            result[label] = None

    if updated:
        save_db(db, profile)
    return result


def rename_speaker(profile: str, old_name: str, new_name: str) -> bool:
    """声紋DBの話者名をリネームする（紐づけ誤りの手修正用）。"""
    db = load_db(profile)
    speakers = db.get("speakers", {})
    if old_name not in speakers:
        return False
    speakers[new_name] = speakers.pop(old_name)
    save_db(db, profile)
    return True


def remove_speaker(profile: str, name: str) -> bool:
    """声紋DBから話者を削除する（誤登録の取り消し用）。"""
    db = load_db(profile)
    speakers = db.get("speakers", {})
    if name not in speakers:
        return False
    del speakers[name]
    save_db(db, profile)
    return True
