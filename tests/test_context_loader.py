"""案件yaml → ASR用語/正規化/議事録プロンプト展開の純ロジックテスト。"""

from meeting_transcriber import context_loader as cl


def test_expected_speakers_prefers_meeting_then_roster():
    assert cl.expected_speakers({"meeting": {"expected_speakers": 4}}) == 4
    assert cl.expected_speakers({"speaker_roster": [{"name": "山田"}, {"name": "鈴木"}]}) == 2
    assert cl.expected_speakers(None) is None


def test_asr_glossary_priority_and_limit():
    ctx = {
        "asr_prompt_terms": ["最優先語"],
        "organization": [{"id": "A社", "canonical": "A社株式会社"}],
        "speaker_roster": [{"name": "山田"}, {"name": "(要確認)"}],
        "glossary": [{"term": "専門用語"}],
    }
    terms = cl.asr_glossary(ctx)
    assert terms[0] == "最優先語"  # asr_prompt_terms が先頭
    assert "A社" in terms and "A社株式会社" in terms
    assert "山田" in terms
    assert "(要確認)" not in terms  # 未確定ラベルは除外
    assert cl.asr_glossary(ctx, limit=2) == ["最優先語", "A社"]


def test_apply_normalization_replaces_and_drops():
    ctx = {
        "glossary": [{"term": "IABハブ", "aliases": ["いあぶはぶ"]}],
        "normalization": {
            "deterministic": [{"correct": "A社", "wrong": ["えーしゃ"]}],
            "drop_phrases": ["ご視聴ありがとうございました"],
        },
    }
    text = "えーしゃのいあぶはぶです。ご視聴ありがとうございました"
    out, report = cl.apply_normalization(text, ctx)
    assert "A社" in out and "IABハブ" in out
    assert "ご視聴ありがとうございました" not in out
    assert report  # 置換/削除のレポートが返る


def test_apply_normalization_noop_without_context():
    out, report = cl.apply_normalization("そのまま", None)
    assert out == "そのまま" and report == []


def test_minutes_context_markdown_sections():
    ctx = {
        "organization": [{"id": "A社", "canonical": "A社(example.com)", "side": "自社", "role": "設計"}],
        "speaker_roster": [{"name": "山田", "company": "A社"}, {"label": "発話者2", "name": "鈴木", "company": "A社"}],
        "topic_kinds": [{"kind": "定例", "notes": "決定とTODOだけ"}],
        "minutes_preferences": [{"rule": "細部は省く", "polarity": "drop"}],
    }
    md = cl.minutes_context_markdown(ctx)
    assert "組織構造" in md and "A社" in md
    assert "話者ロスター" in md and "山田" in md
    assert "議題種別" in md and "定例" in md
    assert "議事録の取捨" in md and "細部は省く" in md


def test_speaker_identity_markdown():
    resolve = {
        "identified": {"発話者1": "山田"},
        "merge_suggestions": [{"labels": ["発話者5", "発話者6"], "score": 0.91, "confidence": "high"}],
        "mixed_warnings": [{"label": "発話者3", "min_cohesion": 0.41, "segments": 4}],
        "segment_relabel": [{"label": "発話者3", "start": 130.2, "end": 148.0, "name": "鈴木", "score": 0.82}],
    }
    md = cl.speaker_identity_markdown(resolve)
    assert "発話者5" in md and "発話者6" in md  # 統合候補
    assert "発話者3" in md  # 混在＋区間照合
    assert "鈴木" in md
    assert cl.speaker_identity_markdown(None) == ""
    assert cl.speaker_identity_markdown({}) == ""
