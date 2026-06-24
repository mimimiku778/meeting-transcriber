"""案件コンテキストストア（自動判定・修正ベース学習）の純ロジックテスト。"""

import pytest

from meeting_transcriber import context_store as cs


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """各テストで独立したストアディレクトリを使う。"""
    monkeypatch.setenv("MEETING_CONTEXTS_DIR", str(tmp_path))
    return cs


def test_safe_slug(store):
    assert store.safe_slug("My Team!") == "MyTeam"
    assert store.safe_slug("a-b_c") == "a-b_c"
    assert store.safe_slug("") == "default"
    # 日本語は残す（声紋 db_path と同一規則）
    assert store.safe_slug("定例") == "定例"


def test_merge_creates_and_increments(store):
    data = store.merge_into_project("demo", {"organization": [{"id": "A社", "side": "自社"}]})
    assert data["slug"] == "demo"
    assert data["voiceprint_profile"] == "demo"
    assert data["enroll_count"] == 1
    # 2回目で学習回数が増える
    data = store.merge_into_project("demo", {"glossary": [{"term": "X"}]})
    assert data["enroll_count"] == 2


def test_merge_upsert_human_override(store):
    store.merge_into_project("demo", {"speaker_roster": [{"name": "山田", "company": "A社"}]})
    # 同一 name は上書き（人間確定が勝つ）、別 name は追加
    data = store.merge_into_project(
        "demo",
        {"speaker_roster": [{"name": "山田", "company": "B社"}, {"name": "鈴木", "company": "A社"}]},
    )
    roster = {p["name"]: p["company"] for p in data["speaker_roster"]}
    assert roster == {"山田": "B社", "鈴木": "A社"}


def test_minutes_preferences_get_added_date(store):
    data = store.merge_into_project("demo", {"minutes_preferences": [{"rule": "仕様の細部は省く", "polarity": "drop"}]})
    assert data["minutes_preferences"][0]["added"]  # 日付が自動付与される


def test_signals_union(store):
    store.merge_into_project("demo", {"signals": {"dir_keywords": ["demo"]}})
    data = store.merge_into_project("demo", {"signals": {"dir_keywords": ["demo", "weekly"]}})
    assert sorted(data["signals"]["dir_keywords"]) == ["demo", "weekly"]


def test_identify_by_path(store):
    store.merge_into_project("demo", {"signals": {"dir_keywords": ["demo"]}})
    res = store.identify_project(video_path="/x/demo/v.mov")
    assert res and res[0]["slug"] == "demo"
    assert any("demo" in r for r in res[0]["reasons"])


def test_identify_by_ocr_and_voiceprint(store):
    store.merge_into_project("demo", {"signals": {"org_terms": ["A社"]}})
    by_ocr = store.identify_project(ocr_terms=["本日はA社の定例"])
    assert by_ocr and by_ocr[0]["slug"] == "demo"
    # 声紋一致は最強シグナル（高スコア）
    by_voice = store.identify_project(voiceprint_matches={"demo": 2})
    assert by_voice[0]["score"] >= by_ocr[0]["score"]


def test_identify_empty_when_no_match(store):
    store.merge_into_project("demo", {"signals": {"dir_keywords": ["demo"]}})
    assert store.identify_project(video_path="/x/unrelated/v.mov") == []


def test_list_projects(store):
    store.merge_into_project("demo", {"organization": [{"id": "A社"}], "meeting": {"title": "定例"}})
    listing = store.list_projects()
    assert listing[0]["slug"] == "demo"
    assert listing[0]["title"] == "定例"
    assert "A社" in listing[0]["orgs"]


def test_export_to_meeting_dir(tmp_path, store):
    store.merge_into_project("demo", {"organization": [{"id": "A社"}]})
    out = store.export_to_meeting_dir("demo", tmp_path)
    assert out.exists() and out.name == "project.context.yaml"
    assert "A社" in out.read_text(encoding="utf-8")


def test_export_missing_raises(store):
    with pytest.raises(FileNotFoundError):
        store.export_to_meeting_dir("nope", "/tmp")
