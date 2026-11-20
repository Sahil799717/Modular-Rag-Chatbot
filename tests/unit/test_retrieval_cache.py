
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.rag import retrieval_cache


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="RAG combines retrieval and generation.",
            metadata={"page_number": 2, "file_name": "test.pdf"}
        )
    ]


def test_cache_key_normalizes_question():
    key1 = retrieval_cache.get_retrieval_cache_key(
        "What is RAG?", "user_A"
    )

    key2 = retrieval_cache.get_retrieval_cache_key(
        "  What   is RAG?  ", "user_A"
    )

    assert key1 == key2


def test_cache_key_isolates_users_and_documents():
    key1 = retrieval_cache.get_retrieval_cache_key(
        "What is RAG?", "user_A", "doc_1"
    )

    key2 = retrieval_cache.get_retrieval_cache_key(
        "What is RAG?", "user_B", "doc_1"
    )

    key3 = retrieval_cache.get_retrieval_cache_key(
        "What is RAG?", "user_A", "doc_2"
    )

    assert len({key1, key2, key3}) == 3


def test_cache_key_includes_retrieval_settings():
    key1 = retrieval_cache.get_retrieval_cache_key(
        "What is RAG?", "user_A", k=4, top_n=3, corpus_version=0
    )

    key2 = retrieval_cache.get_retrieval_cache_key(
        "What is RAG?", "user_A", k=5, top_n=3, corpus_version=0
    )

    key3 = retrieval_cache.get_retrieval_cache_key(
        "What is RAG?", "user_A", k=4, top_n=3, corpus_version=1
    )

    assert len({key1, key2, key3}) == 3


def test_get_cached_documents_hit(monkeypatch):
    cached_documents = [
        {
            "page_content": "Cached RAG content",
            "metadata": {"page_number": 2}
        }
    ]

    cache_get = MagicMock(side_effect=[1, cached_documents])
    rerank = MagicMock()

    monkeypatch.setattr(retrieval_cache, "cache_get", cache_get)
    monkeypatch.setattr(retrieval_cache, "rerank_documents", rerank)

    result = retrieval_cache.get_cached_documents(
        question="What is RAG?",
        user_id="user_A"
    )

    assert len(result) == 1
    assert isinstance(result[0], Document)
    assert result[0].page_content == "Cached RAG content"
    assert result[0].metadata["page_number"] == 2

    rerank.assert_not_called()


def test_get_cached_documents_miss(monkeypatch, sample_documents):
    cache_get = MagicMock(side_effect=[0, None])
    cache_set = MagicMock()
    rerank = MagicMock(return_value=sample_documents)

    monkeypatch.setattr(retrieval_cache, "cache_get", cache_get)
    monkeypatch.setattr(retrieval_cache, "cache_set", cache_set)
    monkeypatch.setattr(retrieval_cache, "rerank_documents", rerank)

    result = retrieval_cache.get_cached_documents(
        question="What is RAG?",
        user_id="user_A"
    )

    assert result == sample_documents
    rerank.assert_called_once()

    serialized = cache_set.call_args.args[1]

    assert serialized[0]["page_content"] == sample_documents[0].page_content
    assert serialized[0]["metadata"]["page_number"] == 2


def test_get_cached_documents_empty_result(monkeypatch):
    cache_get = MagicMock(side_effect=[0, None])
    cache_set = MagicMock()

    monkeypatch.setattr(retrieval_cache, "cache_get", cache_get)
    monkeypatch.setattr(retrieval_cache, "cache_set", cache_set)
    monkeypatch.setattr(
        retrieval_cache,
        "rerank_documents",
        MagicMock(return_value=[])
    )

    result = retrieval_cache.get_cached_documents(
        question="Unknown question",
        user_id="user_A"
    )

    assert result == []
    cache_set.assert_not_called()


def test_get_cached_documents_cache_read_failure(
    monkeypatch, sample_documents
):
    cache_get = MagicMock(
        side_effect=[0, Exception("Redis read failed")]
    )

    rerank = MagicMock(return_value=sample_documents)

    monkeypatch.setattr(retrieval_cache, "cache_get", cache_get)
    monkeypatch.setattr(retrieval_cache, "rerank_documents", rerank)

    result = retrieval_cache.get_cached_documents(
        question="What is RAG?",
        user_id="user_A"
    )

    assert result == sample_documents
    rerank.assert_called_once()


def test_get_cached_documents_version_failure(
    monkeypatch, sample_documents
):
    cache_get = MagicMock(side_effect=Exception("Redis unavailable"))
    rerank = MagicMock(return_value=sample_documents)

    monkeypatch.setattr(retrieval_cache, "cache_get", cache_get)
    monkeypatch.setattr(retrieval_cache, "rerank_documents", rerank)

    result = retrieval_cache.get_cached_documents(
        question="What is RAG?",
        user_id="user_A"
    )

    assert result == sample_documents
    rerank.assert_called_once()


def test_get_cached_documents_cache_write_failure(
    monkeypatch, sample_documents
):
    cache_get = MagicMock(side_effect=[0, None])
    cache_set = MagicMock(side_effect=Exception("Redis write failed"))
    rerank = MagicMock(return_value=sample_documents)

    monkeypatch.setattr(retrieval_cache, "cache_get", cache_get)
    monkeypatch.setattr(retrieval_cache, "cache_set", cache_set)
    monkeypatch.setattr(retrieval_cache, "rerank_documents", rerank)

    result = retrieval_cache.get_cached_documents(
        question="What is RAG?",
        user_id="user_A"
    )

    assert result == sample_documents
    rerank.assert_called_once()
    cache_set.assert_called_once()


def test_make_json_safe():
    value = {
        "items": (1, 2, 3),
        "nested": {"values": [4, 5]}
    }

    result = retrieval_cache.make_json_safe(value)

    assert result == {
        "items": [1, 2, 3],
        "nested": {"values": [4, 5]}
    }
