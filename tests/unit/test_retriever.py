
from unittest.mock import MagicMock

import pytest

from app.rag import retriever
from app.core.exceptions import RetrievalError


@pytest.fixture
def mock_dependencies(monkeypatch):
    base_retriever = MagicMock()
    llm = MagicMock()
    multi_retriever = MagicMock()

    monkeypatch.setattr(
        retriever,
        "get_filtered_retriever",
        MagicMock(return_value=base_retriever)
    )

    monkeypatch.setattr(
        retriever,
        "get_llm",
        MagicMock(return_value=llm)
    )

    monkeypatch.setattr(
        retriever.MultiQueryRetriever,
        "from_llm",
        MagicMock(return_value=multi_retriever)
    )

    return base_retriever, llm, multi_retriever


def test_get_multi_query_retriever(mock_dependencies):
    base_retriever, llm, multi_retriever = mock_dependencies

    result = retriever.get_multi_query_retriever(
        user_id="user_A",
        document_id="doc_123",
        k=5
    )

    assert result is multi_retriever

    retriever.get_filtered_retriever.assert_called_once_with(
        user_id="user_A",
        document_id="doc_123",
        k=5
    )

    retriever.get_llm.assert_called_once()

    retriever.MultiQueryRetriever.from_llm.assert_called_once_with(
        retriever=base_retriever,
        llm=llm
    )


def test_get_multi_query_retriever_default_values(mock_dependencies):
    retriever.get_multi_query_retriever(user_id="user_A")

    retriever.get_filtered_retriever.assert_called_once_with(
        user_id="user_A",
        document_id=None,
        k=4
    )


def test_get_multi_query_retriever_failure(monkeypatch):
    monkeypatch.setattr(
        retriever,
        "get_filtered_retriever",
        MagicMock(side_effect=Exception("Qdrant unavailable"))
    )

    with pytest.raises(RetrievalError):
        retriever.get_multi_query_retriever("user_A")


def test_retrieve_documents(mock_dependencies):
    _, _, multi_retriever = mock_dependencies

    expected_documents = [
        MagicMock(),
        MagicMock()
    ]

    multi_retriever.invoke.return_value = expected_documents

    result = retriever.retrieve_documents(
        question="What is RAG?",
        user_id="user_A",
        document_id="doc_123",
        k=5
    )

    assert result == expected_documents

    multi_retriever.invoke.assert_called_once_with("What is RAG?")


def test_retrieve_documents_failure(mock_dependencies):
    _, _, multi_retriever = mock_dependencies

    multi_retriever.invoke.side_effect = Exception(
        "Retrieval failed"
    )

    with pytest.raises(RetrievalError):
        retriever.retrieve_documents(
            question="What is RAG?",
            user_id="user_A"
        )
