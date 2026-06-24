"""案件コンテキスト(project.context.yaml)の単一ソースから、

  (a) ASR用 initial_prompt の固有名詞リスト
  (b) 決定的(辞書)置換マップ … 曖昧さの無い表記ゆれのみ
  (c) 議事録生成プロンプトに渡す 組織図/話者ロスター/帰属ルール のmarkdown

を機械的に展開するローダ。

設計思想:
- 「誰がどの会社か」「決定/宿題/課題の帰属」は音声/文字だけでは復元できない。
  動画フレーム(参加者パネル+共有資料)で確定した事実を YAML に固定し、
  ASRと議事録生成の双方へ注入することで帰属ズレを根底から断つ。
- 略語の意味反転に注意。例: ある略号が文脈によって別の会社を指すことがある。
  そうした曖昧な略号は決定的置換に含めず、context_notes で Claude に判定させる。
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


def expected_speakers(context: dict | None) -> int | None:
    """話者数のヒントを返す。meeting.expected_speakers 優先、無ければ
    speaker_roster の件数。diarizationの過分割を防ぐために num_speakers へ渡す。"""
    if not context:
        return None
    meeting = context.get("meeting", {}) or {}
    n = meeting.get("expected_speakers")
    if isinstance(n, int) and n > 0:
        return n
    roster = context.get("speaker_roster", []) or []
    return len(roster) if roster else None


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
        lines.append("## 話者ロスター（氏名↔会社。話者ラベルは声紋/フレームで都度解決）")
        for sp in roster:
            name = sp.get("name", "")
            company = sp.get("company", "")
            role = sp.get("role", "")
            label = sp.get("label")  # ストア由来は label を持たない（案件横断で固定できないため）
            head = f"{label} = {name}" if label else name
            extra = f" / {role}" if role else ""
            lines.append(f"- {head}（{company}）{extra}")
        lines.append("")

    kinds = context.get("topic_kinds", []) or []
    if kinds:
        lines.append("## 議題種別ごとの注意点")
        for k in kinds:
            kind = k.get("kind", "?")
            notes = k.get("notes", "")
            lines.append(f"- **{kind}**: {notes}")
        lines.append("")

    prefs = context.get("minutes_preferences", []) or []
    if prefs:
        lines.append("## 議事録の取捨（この案件で学習済み。何を残し何を書かないか）")
        for p in prefs:
            rule = p.get("rule", "")
            if not rule:
                continue
            polarity = p.get("polarity", "")
            mark = {"keep": "残す", "drop": "書かない"}.get(polarity, "")
            tag = f"［{mark}］" if mark else ""
            lines.append(f"- {tag}{rule}")
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


def speaker_identity_markdown(resolve: dict | None) -> str:
    """話者同一性の解決ヒント（声紋由来・実行時データ）を議事録プロンプト用 markdown に。

    resolve は voiceprint.cluster_similarity() の戻り（＋任意で 'identified': {発話者N:実名}）。
    1次の文字起こしは直さず、この材料で議事録(成果物)側の話者帰属を正すための指示を出す。
    """
    if not resolve:
        return ""
    lines: list[str] = ["## 話者の同一性（声紋ヒント。1次結果は直さず議事録側で解決する）"]

    identified = resolve.get("identified") or {}
    if identified:
        pairs = ", ".join(f"{k}→{v}" for k, v in identified.items() if v)
        if pairs:
            lines.append(f"- 声紋で実名化できた話者: {pairs}")

    merges = resolve.get("merge_suggestions") or []
    if merges:
        lines.append("- **同一人物の可能性が高い（統合候補）**: 議事録では同じ人物として一貫表記する。")
        for m in merges:
            labels = " と ".join(m.get("labels", []))
            lines.append(f"  - {labels}（声紋類似度 {m.get('score')}・確度 {m.get('confidence')}）")

    mixed = resolve.get("mixed_warnings") or []
    if mixed:
        lines.append("- **別人が混在している可能性（過少分割）**: 下記ラベルは1人に見えて複数人かもしれない。"
                     "発言ごとに文脈・フレーム・声紋で振り分け、混在のまま1人にしない。")
        for w in mixed:
            lines.append(f"  - {w.get('label')}（クラスタ結束 {w.get('min_cohesion')}・区間 {w.get('segments')}）")

    seg = resolve.get("segment_relabel") or []
    if seg:
        lines.append("- **区間単位の声紋照合（登録済み声紋による振り直し提案）**: "
                     "混在ラベル内の各区間を実名へ。時刻で文字起こしと突き合わせて反映する。")
        for s in seg:
            lines.append(f"  - {s.get('label')} {s.get('start')}〜{s.get('end')}s → {s.get('name')}"
                         f"（類似度 {s.get('score')}）")

    if len(lines) == 1:
        return ""  # 見出しだけなら出さない
    return "\n".join(lines).strip()
