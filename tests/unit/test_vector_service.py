
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services import vector_service
from app.core.exceptions import VectorStoreError


@pytest.fixture
def mock_qdrant(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(
        vector_service,
        "get_qdrant_client",
        lambda: client
    )
    return client


def get_filter_values(qdrant_filter):
    return {
        condition.key: condition.match.value
        for condition in qdrant_filter.must
    }


def test_get_filtered_retriever(mock_qdrant, monkeypatch):
    vector_store = MagicMock()
    monkeypatch.setattr(
        vector_service,
        "get_vector_store",
        lambda: vector_store
    )

    vector_service.get_filtered_retriever(
        user_id="user_A",
        document_id="doc_123",
        k=5
    )

    kwargs = vector_store.as_retriever.call_args.kwargs
    search_kwargs = kwargs["search_kwargs"]

    assert kwargs["search_type"] == "similarity"
    assert search_kwargs["k"] == 5
    assert get_filter_values(search_kwargs["filter"]) == {
        "metadata.user_id": "user_A",
        "metadata.document_id": "doc_123"
    }


def test_get_filtered_retriever_requires_user_id():
    with pytest.raises(VectorStoreError):
        vector_service.get_filtered_retriever(user_id="")


def test_delete_document(mock_qdrant, monkeypatch):
    monkeypatch.setattr(
        vector_service,
        "cache_increment",
        MagicMock()
    )

    vector_service.delete_document(
        user_id="user_A",
        document_id="doc_123"
    )

    kwargs = mock_qdrant.delete.call_args.kwargs

    assert kwargs["collection_name"] == vector_service.settings.QDRANT_COLLECTION
    assert kwargs["wait"] is True
    assert get_filter_values(kwargs["points_selector"]) == {
        "metadata.user_id": "user_A",
        "metadata.document_id": "doc_123"
    }

    vector_service.cache_increment.assert_called_once_with(
        "rag:version:user:user_A"
    )


def test_delete_document_requires_ids():
    with pytest.raises(VectorStoreError):
        vector_service.delete_document(
            user_id="user_A",
            document_id=""
        )


def test_document_exists(mock_qdrant):
    mock_qdrant.count.return_value = SimpleNamespace(count=3)

    result = vector_service.document_exists("user_A", "doc_123")

    assert result is True

    kwargs = mock_qdrant.count.call_args.kwargs

    assert get_filter_values(kwargs["count_filter"]) == {
        "metadata.user_id": "user_A",
        "metadata.document_id": "doc_123"
    }
    assert kwargs["exact"] is True


def test_document_does_not_exist(mock_qdrant):
    mock_qdrant.count.return_value = SimpleNamespace(count=0)

    result = vector_service.document_exists("user_A", "doc_123")

    assert result is False


def test_document_hash_exists(mock_qdrant):
    mock_qdrant.count.return_value = SimpleNamespace(count=2)

    result = vector_service.document_hash_exists(
        "user_A",
        "doc_123",
        "abc123"
    )

    assert result is True

    kwargs = mock_qdrant.count.call_args.kwargs

    assert get_filter_values(kwargs["count_filter"]) == {
        "metadata.user_id": "user_A",
        "metadata.document_id": "doc_123",
        "metadata.file_hash": "abc123"
    }


def test_list_user_documents(mock_qdrant):
    mock_qdrant.scroll.return_value = (
        [
            SimpleNamespace(
                payload={
                    "metadata": {
                        "document_id": "doc_123",
                        "file_name": "report.pdf",
                        "file_hash": "abc123"
                    }
                }
            ),
            SimpleNamespace(
                payload={
                    "metadata": {
                        "document_id": "doc_123",
                        "file_name": "report.pdf",
                        "file_hash": "abc123"
                    }
                }
            )
        ],
        None
    )

    result = vector_service.list_user_documents("user_A")

    assert len(result) == 1
    assert result[0]["document_id"] == "doc_123"
    assert result[0]["user_id"] == "user_A"
    assert result[0]["file_name"] == "report.pdf"
    assert result[0]["chunks"] == 2

    kwargs = mock_qdrant.scroll.call_args.kwargs
    assert get_filter_values(kwargs["scroll_filter"]) == {
        "metadata.user_id": "user_A"
    }


def test_list_user_documents_pagination(mock_qdrant):
    page_1 = [
        SimpleNamespace(
            payload={
                "metadata": {
                    "document_id": "doc_1",
                    "file_name": "first.pdf",
                    "file_hash": "hash_1"
                }
            }
        )
    ]

    page_2 = [
        SimpleNamespace(
            payload={
                "metadata": {
                    "document_id": "doc_2",
                    "file_name": "second.pdf",
                    "file_hash": "hash_2"
                }
            }
        )
    ]

    mock_qdrant.scroll.side_effect = [
        (page_1, "next_page"),
        (page_2, None)
    ]

    result = vector_service.list_user_documents("user_A")

    assert len(result) == 2
    assert {doc["document_id"] for doc in result} == {
        "doc_1", "doc_2"
    }
    assert mock_qdrant.scroll.call_count == 2
