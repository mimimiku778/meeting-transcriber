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
DEFAULT_THRESHOLD = 0.50  # コサイン類似度がこれ未満なら UNKNOWN（発話者Nのまま）
DEFAULT_MARGIN = 0.10  # 1位と2位の差がこれ未満なら曖昧として UNKNOWN
AUTO_UPDATE_MARGIN = 0.15  # 自動平均更新は「閾値＋このマージン」以上の高信頼時のみ
MIN_SEGMENT_SEC = 0.6  # これより短い区間は声紋計算に使わない

# クラスタ同一性（過分割マージ）の判定パラメータ。
# DB照合の DEFAULT_THRESHOLD より高めにする: 別人を誤統合すると議事録の話者帰属が壊れ、
# その損害は「分かれたまま」より大きいので保守的に倒す（疑わしきは統合しない）。
MERGE_THRESHOLD = 0.78  # 発話者ペアの声紋がこれ以上似ていれば「同一の可能性」候補
MERGE_HIGH_CONF = 0.86  # これ以上は confidence=high（ほぼ同一）
# 1ラベル内（過少分割で別人混在の疑い）の検出: クラスタ内区間の声紋が平均からこれ未満に
# ばらつくと「混在の可能性」を警告する（自動分割はせず Claude/人間の判断材料に留める）。
MIXED_COHESION = 0.55
# クラスタの代表ベクトルと混在検出に共用する、1クラスタあたりの個別区間サンプル数。
# 多いほど精度↑だが embedding 呼び出しが増えて遅くなる。速度優先で控えめに。
CLUSTER_PROBE_SEGMENTS = 4


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


def _embedding_device():
    """声紋embeddingの計算デバイス。既定 auto（Apple Silicon の MPS があれば使う）。

    diarization は MPS でタイムスタンプが崩れる報告があり cpu 既定だが、embedding は
    window='whole' で区間→単一ベクトルを出すだけ（タイムスタンプ非依存）なので、その問題が
    当てはまらず MPS を安全に使える。声紋登録/識別が遅い主因が CPU 固定だったため auto に。
    MEETING_VOICEPRINT_DEVICE=cpu/mps/auto で上書き可。
    """
    import torch

    want = os.environ.get("MEETING_VOICEPRINT_DEVICE", "auto").lower()
    if want != "cpu" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_inference():
    """pyannote 埋め込み推論器を遅延ロードする（window='whole' で区間→単一ベクトル）。

    Inference に文字列のモデル名を渡すと、一部の pyannote 版では内部で model.eval() が
    文字列に対して呼ばれ 'str' object has no attribute 'eval' になる。必ず Model を
    明示ロードして渡す（token 引数名・device 引数の有無は版差があるので順に試す）。
    """
    global _inference
    if _inference is None:
        from pyannote.audio import Inference, Model

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        device = _embedding_device()
        print(f"    声紋embedding デバイス: {device}", flush=True)

        model = None
        last_err = None
        for kw in ({"use_auth_token": token}, {"token": token}, {}):
            try:
                model = Model.from_pretrained(_EMBEDDING_MODEL, **kw)
                break
            except TypeError as e:
                last_err = e
                continue
        if model is None:
            raise last_err or RuntimeError(f"埋め込みモデルをロードできません: {_EMBEDDING_MODEL}")

        try:
            _inference = Inference(model, window="whole", device=device)
        except TypeError:
            # device 引数が無い版は to() でフォールバック
            _inference = Inference(model, window="whole")
            try:
                model.to(device)
            except Exception:
                pass
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
            raw = inference.crop(audio_path, Segment(start, end))
            if hasattr(raw, "detach"):  # torch.Tensor(MPS/CUDA含む) は CPU numpy へ
                raw = raw.detach().to("cpu").numpy()
            emb = np.asarray(raw).reshape(-1)
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
    return {f"発話者{i + 1}": spk for i, spk in enumerate(unique)}


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


def _label_sort_key(label: str) -> int:
    """『発話者12』→ 12。数字が無ければ大きな値で末尾へ。"""
    digits = "".join(c for c in label if c.isdigit())
    return int(digits) if digits else 10**9


def _cluster_unit_embeddings(
    audio_path: str, diar_segments: list[dict], max_segs: int = CLUSTER_PROBE_SEGMENTS
) -> dict[str, dict]:
    """各『発話者N』クラスタの長い区間を最大 max_segs 個だけ個別に声紋化する（embedding 1パス）。

    この個別ベクトル群から、代表ベクトル(平均=クラスタ同士の比較用)と結束度(混在検出用)の
    両方を導けるので、用途ごとに embedding を二重に呼ばずに済む（呼び出し回数を大幅削減）。
    戻り値: {label: {"mean": np.ndarray, "units": [np.ndarray, ...]}}
    """
    label2cluster = _label_to_cluster(diar_segments)
    clusters = _cluster_segments(diar_segments)
    out: dict[str, dict] = {}
    for label, spk in label2cluster.items():
        segs = clusters.get(spk) or []
        chosen = sorted(segs, key=lambda s: s[1] - s[0], reverse=True)[:max_segs]
        units, weights = [], []
        for s, e in chosen:
            v = _embed_segments(audio_path, [(s, e)])
            if v is not None:
                units.append(v)
                weights.append(e - s)
        if not units:
            continue
        mean = np.average(np.vstack(units), axis=0, weights=np.asarray(weights))
        out[label] = {"mean": mean, "units": units}
    return out


def cluster_embeddings(audio_path: str, diar_segments: list[dict]) -> dict[str, np.ndarray]:
    """『発話者N』ラベル → クラスタ代表声紋ベクトル（後方互換の薄いラッパー）。"""
    return {label: u["mean"] for label, u in _cluster_unit_embeddings(audio_path, diar_segments).items()}


def cluster_similarity(audio_path: str, diar_segments: list[dict], detect_mixed: bool = True) -> dict:
    """この動画内の『発話者N』同士の声紋類似度・統合候補・混在疑いを返す。

    議事録段階の「話者同一性の解決」ヒント。diarization の過分割（同一人物が複数ラベルに
    割れる）・過少分割（別人が1ラベルに混ざる）を、1次結果を直さずに成果物側で正すための材料。
    代表ベクトルと結束度を同じ個別ベクトル群から1パスで算出するため軽い。

    戻り値:
      {
        "labels": ["発話者1", ...],
        "pairs": [{"a": "発話者5", "b": "発話者6", "score": 0.91}, ...],   # 全ペアの類似度（降順）
        "merge_suggestions": [{"labels": ["発話者5","発話者6"], "score": 0.91, "confidence": "high"}],
        "mixed_warnings": [{"label": "発話者3", "min_cohesion": 0.41, "segments": 4}],
      }
    """
    units = _cluster_unit_embeddings(audio_path, diar_segments)
    labels = sorted(units.keys(), key=_label_sort_key)

    pairs: list[dict] = []
    suggestions: list[dict] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            score = round(float(_cosine(units[a]["mean"], units[b]["mean"])), 3)
            pairs.append({"a": a, "b": b, "score": score})
            if score >= MERGE_THRESHOLD:
                suggestions.append(
                    {
                        "labels": [a, b],
                        "score": score,
                        "confidence": "high" if score >= MERGE_HIGH_CONF else "medium",
                    }
                )
    pairs.sort(key=lambda p: p["score"], reverse=True)
    suggestions.sort(key=lambda s: s["score"], reverse=True)

    mixed: list[dict] = []
    if detect_mixed:
        for label in labels:
            us = units[label]["units"]
            if len(us) < 2:
                continue
            # クラスタ内区間の『ペア間最小コサイン』。同一話者なら高い(>0.6)が、別人が混ざると
            # その別人区間との類似度が落ちて低くなる。平均への距離より混在に敏感。
            pair_cos = [_cosine(us[i], us[j]) for i in range(len(us)) for j in range(i + 1, len(us))]
            low = min(pair_cos)
            if low < MIXED_COHESION:
                mixed.append({"label": label, "min_cohesion": round(float(low), 3), "segments": len(us)})

    return {"labels": labels, "pairs": pairs, "merge_suggestions": suggestions, "mixed_warnings": mixed}


def identify_segments(
    profile: str,
    audio_path: str,
    diar_segments: list[dict],
    labels: list[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    margin: float = DEFAULT_MARGIN,
) -> list[dict]:
    """区間単位で声紋DBと照合し、過少分割（1ラベルに別人混在）を実名へ振り直す提案を返す。

    クラスタ単位の identify と違い、ラベル内の各発話区間を個別に照合する。声紋が育っていれば
    『発話者3』の中に混ざった山田/鈴木を、区間（=時刻）ごとに正しい実名へ割り当てられる。
    その時刻で文字起こし(word単位タイムスタンプ)と突き合わせれば、文面側を分離して直せる。

    labels: 対象ラベルを絞る（混在警告が出たラベルのみに渡すと速い）。None なら全ラベル。
    戻り値: [{label, start, end, name, score}]（高信頼な区間だけ）。声紋未登録なら []。
    """
    db = load_db(profile)
    speakers = db.get("speakers", {})
    if not speakers:
        return []
    names = list(speakers.keys())
    refs = {n: np.asarray(speakers[n]["embedding"]) for n in names}

    label2cluster = _label_to_cluster(diar_segments)
    cluster2label = {c: lbl for lbl, c in label2cluster.items()}
    target_clusters = (
        {label2cluster[lbl] for lbl in labels if lbl in label2cluster} if labels else set(label2cluster.values())
    )

    out: list[dict] = []
    for d in diar_segments:
        if d["speaker"] not in target_clusters:
            continue
        if d["end"] - d["start"] < MIN_SEGMENT_SEC:
            continue
        emb = _embed_segments(audio_path, [(d["start"], d["end"])])
        if emb is None:
            continue
        scored = sorted(((_cosine(emb, refs[n]), n) for n in names), reverse=True)
        best_score, best_name = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= threshold and (best_score - second) >= margin:
            out.append(
                {
                    "label": cluster2label.get(d["speaker"], d["speaker"]),
                    "start": round(float(d["start"]), 1),
                    "end": round(float(d["end"]), 1),
                    "name": best_name,
                    "score": round(float(best_score), 3),
                }
            )
    return out


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
