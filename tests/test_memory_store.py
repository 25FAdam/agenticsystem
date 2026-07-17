from email_assistant.memory.long_term import MemoryStore
from email_assistant.schemas import MemoryKind


def test_remember_and_recall(store):
    store.remember(MemoryKind.preference, "reply_tone", "concise and friendly")

    records = store.recall(kind=MemoryKind.preference, key="reply_tone")
    assert len(records) == 1
    assert records[0].value == "concise and friendly"


def test_remember_upserts_by_kind_and_key(store):
    store.remember(MemoryKind.preference, "reply_tone", "formal")
    store.remember(MemoryKind.preference, "reply_tone", "casual")

    records = store.recall(kind=MemoryKind.preference, key="reply_tone")
    assert len(records) == 1
    assert records[0].value == "casual"
    assert len(store) == 1


def test_persistence_across_instances(tmp_path):
    path = tmp_path / "memory.json"
    MemoryStore(path).remember(MemoryKind.org_fact, "office", "closed on July 24")

    reloaded = MemoryStore(path)
    assert reloaded.recall(key="office")[0].value == "closed on July 24"


def test_seed_is_loaded_when_store_is_new(seeded_store, tmp_path):
    assert len(seeded_store) >= 8
    assert (tmp_path / "memory.json").exists()


def test_search_matches_keywords(seeded_store):
    results = seeded_store.search("what are the API rate limits?")
    assert results
    assert results[0].key == "api_rate_limits"


def test_search_respects_kind_filter(seeded_store):
    results = seeded_store.search("Dana Northwind", kind=MemoryKind.contact)
    assert results
    assert all(r.kind == MemoryKind.contact for r in results)


def test_search_respects_limit(store):
    for i in range(4):
        store.remember(MemoryKind.org_fact, f"policy_{i}", f"vacation policy detail {i}")
    assert len(store.search("vacation policy", limit=2)) == 2


def test_search_no_match_returns_empty(seeded_store):
    assert seeded_store.search("zzz quantum entanglement") == []
