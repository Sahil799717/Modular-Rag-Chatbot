
from unittest.mock import MagicMock

import pytest

from app.rag import reranker
from app.core.exceptions import RetrievalError


@pytest.fixture
def mock_dependencies(monkeypatch):
    base_retriever = MagicMock()
    compressor = MagicMock()
    compression_retriever = MagicMock()

    monkeypatch.setattr(
        reranker,
        "get_multi_query_retriever",
        MagicMock(return_value=base_retriever)
    )

    monkeypatch.setattr(
        reranker,
        "FlashrankRerank",
        MagicMock(return_value=compressor)
    )

    monkeypatch.setattr(
        reranker,
        "ContextualCompressionRetriever",
        MagicMock(return_value=compression_retriever)
    )

    return base_retriever, compressor, compression_retriever


def test_get_reranked_retriever(mock_dependencies):
    base_retriever, compressor, compression_retriever = mock_dependencies

    result = reranker.get_reranked_retriever(
        user_id="user_A",
        document_id="doc_123",
        k=5,
        top_n=3
    )

    assert result is compression_retriever

    reranker.get_multi_query_retriever.assert_called_once_with(
        user_id="user_A",
        document_id="doc_123",
        k=5
    )

    reranker.FlashrankRerank.assert_called_once_with(top_n=3)

    reranker.ContextualCompressionRetriever.assert_called_once_with(
        base_compressor=compressor,
        base_retriever=base_retriever
    )


def test_get_reranked_retriever_default_values(mock_dependencies):
    reranker.get_reranked_retriever(user_id="user_A")

    reranker.FlashrankRerank.assert_called_once_with(top_n=3)

    reranker.get_multi_query_retriever.assert_called_once_with(
        user_id="user_A",
        document_id=None,
        k=4
    )


def test_get_reranked_retriever_failure(monkeypatch):
    monkeypatch.setattr(
        reranker,
        "get_multi_query_retriever",
        MagicMock(side_effect=Exception("Retriever unavailable"))
    )

    with pytest.raises(RetrievalError):
        reranker.get_reranked_retriever("user_A")


def test_rerank_documents(mock_dependencies):
    _, _, compression_retriever = mock_dependencies

    expected_documents = [
        MagicMock(),
        MagicMock()
    ]

    compression_retriever.invoke.return_value = expected_documents

    result = reranker.rerank_documents(
        question="What is RAG?",
        user_id="user_A",
        document_id="doc_123",
        k=5,
        top_n=2
    )

    assert result == expected_documents

    compression_retriever.invoke.assert_called_once_with("What is RAG?")


def test_rerank_documents_failure(mock_dependencies):
    _, _, compression_retriever = mock_dependencies

    compression_retriever.invoke.side_effect = Exception(
        "FlashRank failed"
    )

    with pytest.raises(RetrievalError):
        reranker.rerank_documents(
            question="What is RAG?",
            user_id="user_A"
        )
