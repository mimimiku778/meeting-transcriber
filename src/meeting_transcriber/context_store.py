"""案件コンテキストの永続ストア（声紋と同じ『育てる』仕組みの双子）。

声紋(voiceprint.py)が ~/.claude/voiceprints/<slug>.json に話者の特徴ベクトルを貯めて
会議を重ねるほど実名化が当たるようになるのと同じ発想で、案件の「組織図・話者ロスター・
用語・帰属ルール・議事録の取捨の好み」を ~/.claude/meeting-contexts/<slug>.yaml に貯める。

設計の要点（ユーザー要件）:
- ゼロお膳立て: ユーザーは動画を入れるだけ。案件は signals で自動判定する。
- 修正ベース学習: 先に確認して止めるのではなく、まず最善推定で成果物を出し、ユーザーが
  議事録(文面)を直したら、その確定を merge_into_project でストアへ反映して育てる。
- 声紋と同一 slug で連結: <slug>.yaml ⇔ voiceprints/<slug>.json。話者一致が案件判定の主シグナル。
- リポジトリ外(~/.claude 配下)に置く: 固有名詞・組織情報を公開リポジトリへ漏らさない
  （声紋ベクトルを外に出さないのと同じ理由）。

ストアYAMLの形（project.context.template.yaml の上位互換 + 学習メタ）:
  slug, voiceprint_profile, enroll_count, updated
  signals: {dir_keywords, org_terms, title_terms}      # 案件自動判定用
  meeting, organization, speaker_roster, asr_prompt_terms, glossary,
  attribution_rules, topic_kinds, minutes_preferences, normalization
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


# ---- 場所・slug -----------------------------------------------------------

def contexts_dir() -> Path:
    """案件ストアの保存ディレクトリ（~/.claude/meeting-contexts/）。"""
    base = Path(os.environ.get("MEETING_CONTEXTS_DIR", Path.home() / ".claude" / "meeting-contexts"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def safe_slug(slug: str) -> str:
    """ファイル名に使える slug へ正規化する。

    voiceprint.db_path と同一規則にする（英数・日本語・-・_ を残す）。これにより
    案件 <slug>.yaml と声紋 <slug>.json が必ず同じ名前で対応し、相互参照が崩れない。
    """
    safe = "".join(c for c in (slug or "") if c.isalnum() or c in "-_") or "default"
    return safe


def store_path(slug: str) -> Path:
    return contexts_dir() / f"{safe_slug(slug)}.yaml"


def _today() -> str:
    return datetime.date.today().isoformat()


# ---- I/O ------------------------------------------------------------------

def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("pyyaml が必要です: pip install pyyaml")


def load_project(slug: str) -> dict | None:
    """案件ストアを読み込む。無ければ None。"""
    path = store_path(slug)
    if not path.exists():
        return None
    _require_yaml()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or None


def save_project(data: dict, slug: str | None = None) -> Path:
    """案件ストアを書き出す。slug 未指定なら data['slug'] を使う。"""
    _require_yaml()
    slug = slug or data.get("slug")
    if not slug:
        raise ValueError("slug が必要です")
    data["slug"] = safe_slug(slug)
    data.setdefault("voiceprint_profile", data["slug"])
    data["updated"] = _today()
    path = store_path(slug)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return path


def list_projects() -> list[dict]:
    """全案件の要約を返す（skill が候補提示に使う）。"""
    out: list[dict] = []
    for path in sorted(contexts_dir().glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        meeting = data.get("meeting", {}) or {}
        out.append({
            "slug": data.get("slug", path.stem),
            "title": meeting.get("title", ""),
            "people": [p.get("name") for p in (data.get("speaker_roster", []) or []) if p.get("name")],
            "orgs": [o.get("id") for o in (data.get("organization", []) or []) if o.get("id")],
            "kinds": [k.get("kind") for k in (data.get("topic_kinds", []) or []) if k.get("kind")],
            "enroll_count": data.get("enroll_count", 0),
            "updated": data.get("updated", ""),
        })
    return out


# ---- 案件自動判定（identify相当）-----------------------------------------

def _signals(data: dict) -> dict:
    sig = data.get("signals", {}) or {}
    return {
        "dir_keywords": [s for s in (sig.get("dir_keywords") or []) if s],
        "org_terms": [s for s in (sig.get("org_terms") or []) if s],
        "title_terms": [s for s in (sig.get("title_terms") or []) if s],
    }


def identify_project(
    video_path: str | None = None,
    ocr_terms: list[str] | None = None,
    voiceprint_matches: dict[str, int] | None = None,
    extra_text: str | None = None,
) -> list[dict]:
    """利用可能なシグナルで全案件をスコアリングし、候補を確信度降順で返す。

    シグナル（強い順）:
      - voiceprint_matches: {slug: 一致話者数}（声紋＝最強。同じ顔ぶれ＝同じ案件）
      - signals.org_terms が ocr_terms / extra_text に出現（フレームの組織名・ロゴ）
      - signals.dir_keywords が video_path に出現（ディレクトリ/ファイル名）
      - signals.title_terms が ocr_terms / extra_text に出現（会議タイトル）

    返り値: [{slug, score, reasons:[...], summary:{...}}], スコア降順。
    判定UXは呼び出し側（cli/skill）に委ねる: 最善候補で止めず進み、外れたら修正ベースで学習。
    """
    path_text = (str(video_path) or "").lower() if video_path else ""
    hay_terms = " ".join((ocr_terms or [])).lower()
    hay_extra = (extra_text or "").lower()
    haystack = f"{hay_terms}\n{hay_extra}"
    vmatch = voiceprint_matches or {}

    results: list[dict] = []
    for path in sorted(contexts_dir().glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        slug = data.get("slug", path.stem)
        sig = _signals(data)
        score = 0.0
        reasons: list[str] = []

        n_voice = int(vmatch.get(slug, 0))
        if n_voice:
            score += 3.0 * n_voice
            reasons.append(f"声紋一致 {n_voice}名")

        for term in sig["org_terms"]:
            if term and term.lower() in haystack:
                score += 2.0
                reasons.append(f"組織語『{term}』")
        for kw in sig["dir_keywords"]:
            if kw and kw.lower() in path_text:
                score += 2.0
                reasons.append(f"パス語『{kw}』")
        for term in sig["title_terms"]:
            if term and term.lower() in haystack:
                score += 1.0
                reasons.append(f"タイトル語『{term}』")

        if score > 0:
            meeting = data.get("meeting", {}) or {}
            results.append({
                "slug": slug,
                "score": round(score, 2),
                "reasons": reasons,
                "summary": {
                    "title": meeting.get("title", ""),
                    "orgs": [o.get("id") for o in (data.get("organization", []) or []) if o.get("id")],
                    "people": [p.get("name") for p in (data.get("speaker_roster", []) or []) if p.get("name")],
                    "enroll_count": data.get("enroll_count", 0),
                },
            })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ---- 学習（enroll/merge 相当）--------------------------------------------

def _upsert_list(existing: list, updates: list, key: str) -> list:
    """key で同一視するリストを upsert する。人間確定(updates)が既存フィールドを上書きする。"""
    existing = list(existing or [])
    index = {str(e.get(key)).strip(): e for e in existing if isinstance(e, dict) and e.get(key)}
    for u in updates or []:
        if not isinstance(u, dict) or not u.get(key):
            continue
        k = str(u[key]).strip()
        if k in index:
            index[k].update({kk: vv for kk, vv in u.items() if vv not in (None, "", [])})
        else:
            existing.append(dict(u))
            index[k] = existing[-1]
    return existing


def _union_strs(existing: list, updates: list) -> list:
    out = list(existing or [])
    for s in updates or []:
        if s and s not in out:
            out.append(s)
    return out


def merge_into_project(slug: str, updates: dict, *, create_if_missing: bool = True) -> dict:
    """ユーザーが確定した情報を案件ストアへマージして育てる（声紋 enroll の案件版）。

    updates は部分的でよい。受け付けるキー:
      organization(list, key=id), speaker_roster(list, key=name), glossary(list, key=term),
      topic_kinds(list, key=kind), minutes_preferences(list, key=rule),
      asr_prompt_terms(list[str]), attribution_rules(list[str]),
      signals{dir_keywords,org_terms,title_terms}(list[str]), meeting(dict, 浅い上書き),
      normalization(dict, 浅いマージ)
    人間確定が常に勝つ（既存フィールドを上書き）。enroll_count を1増やし updated を更新する。
    """
    data = load_project(slug)
    if data is None:
        if not create_if_missing:
            raise FileNotFoundError(f"案件ストアがありません: {slug}")
        data = {"slug": safe_slug(slug), "voiceprint_profile": safe_slug(slug), "enroll_count": 0}

    if "organization" in updates:
        data["organization"] = _upsert_list(data.get("organization"), updates["organization"], "id")
    if "speaker_roster" in updates:
        data["speaker_roster"] = _upsert_list(data.get("speaker_roster"), updates["speaker_roster"], "name")
    if "glossary" in updates:
        data["glossary"] = _upsert_list(data.get("glossary"), updates["glossary"], "term")
    if "topic_kinds" in updates:
        data["topic_kinds"] = _upsert_list(data.get("topic_kinds"), updates["topic_kinds"], "kind")
    if "minutes_preferences" in updates:
        prefs = updates["minutes_preferences"]
        for p in prefs or []:
            if isinstance(p, dict):
                p.setdefault("added", _today())
        data["minutes_preferences"] = _upsert_list(data.get("minutes_preferences"), prefs, "rule")

    for str_key in ("asr_prompt_terms", "attribution_rules"):
        if str_key in updates:
            data[str_key] = _union_strs(data.get(str_key), updates[str_key])

    if "signals" in updates:
        cur = data.get("signals", {}) or {}
        for sk in ("dir_keywords", "org_terms", "title_terms"):
            if sk in (updates["signals"] or {}):
                cur[sk] = _union_strs(cur.get(sk), updates["signals"][sk])
        data["signals"] = cur

    if "meeting" in updates:
        cur = data.get("meeting", {}) or {}
        cur.update({k: v for k, v in (updates["meeting"] or {}).items() if v not in (None, "")})
        data["meeting"] = cur

    if "normalization" in updates:
        cur = data.get("normalization", {}) or {}
        upd = updates["normalization"] or {}
        if "deterministic" in upd:
            cur["deterministic"] = _upsert_list(cur.get("deterministic"), upd["deterministic"], "correct")
        for lk in ("drop_phrases", "context_notes"):
            if lk in upd:
                cur[lk] = _union_strs(cur.get(lk), upd[lk])
        data["normalization"] = cur

    data["enroll_count"] = int(data.get("enroll_count", 0)) + 1
    save_project(data, slug)
    return data


# ---- 会議ディレクトリへの焼き込み（既存 --context パイプライン互換）-------

def export_to_meeting_dir(slug: str, meeting_dir: str | Path) -> Path:
    """案件ストアの内容を会議ディレクトリへ project.context.yaml として書き出す。

    これにより既存の `transcribe ... --context <project.context.yaml>` がそのまま効く
    （ASR固有名詞注入＋決定的正規化＋議事録の組織図/帰属ルール）。正はストア側、
    会議ディレクトリのはその回のスナップショット。
    """
    data = load_project(slug)
    if data is None:
        raise FileNotFoundError(f"案件ストアがありません: {slug}")
    _require_yaml()
    out = Path(meeting_dir) / "project.context.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return out
