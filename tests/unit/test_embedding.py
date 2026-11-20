
from unittest.mock import MagicMock

import pytest

from app.services.embedding_service import CachedOllamaEmbeddings


@pytest.fixture
def mock_embedding_service(monkeypatch):
    from app.services import embedding_service

    model = MagicMock()
    service = CachedOllamaEmbeddings(model)

    cache = {}

    def mock_cache_get(key):
        return cache.get(key)

    def mock_cache_set(key, value):
        cache[key] = value

    monkeypatch.setattr(embedding_service, "cache_get", mock_cache_get)
    monkeypatch.setattr(embedding_service, "cache_set", mock_cache_set)

    return service, model, cache


def test_embed_query_cache_miss(mock_embedding_service):
    service, model, cache = mock_embedding_service

    model.embed_query.return_value = [0.1, 0.2, 0.3]

    result = service.embed_query("What is RAG?")

    assert result == [0.1, 0.2, 0.3]
    model.embed_query.assert_called_once_with("What is RAG?")
    assert len(cache) == 1


def test_embed_query_cache_hit(mock_embedding_service):
    service, model, cache = mock_embedding_service

    key = service._cache_key("What is RAG?", "query")
    cache[key] = [0.1, 0.2, 0.3]

    result = service.embed_query("What is RAG?")

    assert result == [0.1, 0.2, 0.3]
    model.embed_query.assert_not_called()


def test_embed_documents_cache_miss(mock_embedding_service):
    service, model, cache = mock_embedding_service

    model.embed_documents.return_value = [
        [0.1, 0.2],
        [0.3, 0.4]
    ]

    result = service.embed_documents(["Document A", "Document B"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    model.embed_documents.assert_called_once_with(
        ["Document A", "Document B"]
    )
    assert len(cache) == 2


def test_embed_documents_partial_cache_hit(mock_embedding_service):
    service, model, cache = mock_embedding_service

    key = service._cache_key("Document A", "document")
    cache[key] = [0.1, 0.2]

    model.embed_documents.return_value = [[0.3, 0.4]]

    result = service.embed_documents(["Document A", "Document B"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    model.embed_documents.assert_called_once_with(["Document B"])


def test_embed_query_cache_failure(mock_embedding_service, monkeypatch):
    service, model, _ = mock_embedding_service

    from app.services import embedding_service

    model.embed_query.return_value = [0.1, 0.2, 0.3]

    def failing_cache_get(key):
        raise Exception("Redis unavailable")

    monkeypatch.setattr(
        embedding_service, "cache_get", failing_cache_get
    )

    result = service.embed_query("What is RAG?")

    assert result == [0.1, 0.2, 0.3]
    model.embed_query.assert_called_once()
