"""案件コンテキスト(project.context.yaml)の単一ソースから、

  (a) ASR用 initial_prompt の固有名詞リスト
  (b) 決定的(辞書)置換マップ … 曖昧さの無い表記ゆれのみ
  (c) 議事録生成プロンプトに渡す 組織図/話者ロスター/帰属ルール のmarkdown

を機械的に展開するローダ。

設計思想:
- 「誰がどの会社か」「決定/宿題/課題の帰属」は音声/文字だけでは復元できない。
  動画フレーム(参加者パネル+共有資料)で確定した事実を YAML に固定し、
  ASRと議事録生成の双方へ注入することで帰属ズレを根本から断つ。
- 略語の意味反転に注意。例: 文字起こしの「DCさん」は自社(DGCircus)を正しく指す。
  決定的置換に DC を含めてはならない。曖昧な略語は context_notes で Claude に判定させる。
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def load_context(context_path: str | Path | None) -> dict | None:
    """project.context.yaml を読み込む。無ければ None。"""
    if not context_path:
        return None
    path = Path(context_path)
    if not path.exists():
        return None
    if yaml is None:
        raise RuntimeError("pyyaml が必要です: pip install pyyaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def asr_glossary(context: dict | None, limit: int = 40) -> list[str]:
    """initial_prompt 用の固有名詞リスト(高優先=先頭)を返す。

    優先順位: 明示の asr_prompt_terms -> 組織の正式名/略号 -> 話者氏名 -> 用語の正表記。
    """
    if not context:
        return []
    terms: list[str] = []

    def add(value: str | None) -> None:
        if value and value not in terms:
            terms.append(value)

    for t in context.get("asr_prompt_terms", []) or []:
        add(t)
    for org in context.get("organization", []) or []:
        add(org.get("id"))
        add(org.get("canonical"))
    for sp in context.get("speaker_roster", []) or []:
        name = sp.get("name")
        if name and "未確定" not in name and "要" not in name:
            add(name)
    for g in context.get("glossary", []) or []:
        add(g.get("term"))

    return terms[:limit]


def _replacement_pairs(context: dict | None) -> list[tuple[str, str]]:
    """(wrong, correct) の決定的置換ペアを集める。曖昧語は含めない。"""
    if not context:
        return []
    pairs: list[tuple[str, str]] = []

    # 1) glossary の aliases -> term
    for g in context.get("glossary", []) or []:
        correct = g.get("term")
        for wrong in g.get("aliases", []) or []:
            if correct and wrong:
                pairs.append((wrong, correct))

    # 2) normalization.deterministic の {correct, wrong:[...]}
    norm = context.get("normalization", {}) or {}
    for rule in norm.get("deterministic", []) or []:
        correct = rule.get("correct")
        for wrong in rule.get("wrong", []) or []:
            if correct and wrong:
                pairs.append((wrong, correct))

    # 長い表記から先に置換して部分一致の取りこぼし/二重置換を防ぐ
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def apply_normalization(text: str, context: dict | None) -> tuple[str, list[str]]:
    """決定的(辞書)置換＋既知ハルシネーション削除を適用する。文脈依存の判断はしない。

    Returns: (置換後テキスト, ["wrong -> correct (N箇所)", ...])
    """
    if not context or not text:
        return text, []
    report: list[str] = []

    # 既知ハルシネーション定型句の削除（配信の締め挨拶等。常に削除して安全な語のみ）
    norm = context.get("normalization", {}) or {}
    for phrase in norm.get("drop_phrases", []) or []:
        if phrase and phrase in text:
            n = text.count(phrase)
            text = text.replace(phrase, "")
            report.append(f"[削除] {phrase} ({n}箇所)")

    for wrong, correct in _replacement_pairs(context):
        if wrong == correct:
            continue
        n = text.count(wrong)
        if n:
            text = text.replace(wrong, correct)
            report.append(f"{wrong} -> {correct} ({n}箇所)")
    return text, report


def minutes_context_markdown(context: dict | None) -> str:
    """議事録生成プロンプトに差し込む 組織図/ロスター/帰属ルール のmarkdown。"""
    if not context:
        return ""
    lines: list[str] = []
    meeting = context.get("meeting", {}) or {}
    if meeting:
        lines.append("## 会議メタ情報")
        if meeting.get("title"):
            lines.append(f"- タイトル: {meeting['title']}")
        if meeting.get("kind"):
            lines.append(f"- 種別: {meeting['kind']}")
        present = meeting.get("attendees_present")
        if present:
            lines.append(f"- 出席(会社): {', '.join(present)}")
        lines.append("")

    orgs = context.get("organization", []) or []
    if orgs:
        lines.append("## 組織構造（フレーム由来の確定事実。owner帰属はこれに従う）")
        for org in orgs:
            oid = org.get("id", "?")
            canon = org.get("canonical", "")
            side = org.get("side", "")
            role = org.get("role", "")
            lines.append(f"- **{oid}** = {canon}（{side}）: {role}")
        lines.append("")

    roster = context.get("speaker_roster", []) or []
    if roster:
        lines.append("## 話者ロスター（話者ラベル↔氏名↔会社）")
        for sp in roster:
            label = sp.get("label", "?")
            name = sp.get("name", "")
            company = sp.get("company", "")
            role = sp.get("role", "")
            extra = f" / {role}" if role else ""
            lines.append(f"- {label} = {name}（{company}）{extra}")
        lines.append("")

    rules = context.get("attribution_rules", []) or []
    if rules:
        lines.append("## 帰属ルール（厳守）")
        for r in rules:
            lines.append(f"- {r}")
        lines.append("")

    notes = (context.get("normalization", {}) or {}).get("context_notes", []) or []
    if notes:
        lines.append("## 表記・帰属の要注意点（額面置換禁止・文脈で判定）")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    return "\n".join(lines).strip()
