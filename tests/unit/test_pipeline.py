
from unittest.mock import MagicMock

import pytest

from app.rag import pipeline
from app.core.exceptions import RetrievalError


@pytest.fixture
def sample_documents():
    return [
        MagicMock(
            page_content="RAG combines retrieval and generation.",
            metadata={
                "page_number": 2,
                "file_name": "test.pdf"
            }
        ),
        MagicMock(
            page_content="RAG improves answer relevance.",
            metadata={
                "page_number": 3,
                "file_name": "test.pdf"
            }
        )
    ]


def test_run_rag_pipeline(monkeypatch, sample_documents):
    retrieval = MagicMock(return_value=sample_documents)
    generation = MagicMock(return_value="RAG combines retrieval and generation.")

    monkeypatch.setattr(pipeline, "get_cached_documents", retrieval)
    monkeypatch.setattr(pipeline, "generate_answer", generation)

    result = pipeline.run_rag_pipeline(
        question="What is RAG?",
        user_id="user_A",
        document_id="doc_123",
        k=5,
        top_n=2
    )

    assert result["question"] == "What is RAG?"
    assert result["answer"] == "RAG combines retrieval and generation."

    assert result["sources"] == [
        {
            "page_number": 2,
            "file_name": "test.pdf",
            "content": "RAG combines retrieval and generation."
        },
        {
            "page_number": 3,
            "file_name": "test.pdf",
            "content": "RAG improves answer relevance."
        }
    ]

    retrieval.assert_called_once_with(
        question="What is RAG?",
        user_id="user_A",
        document_id="doc_123",
        k=5,
        top_n=2
    )

    generation.assert_called_once_with(
        question="What is RAG?",
        documents=sample_documents
    )


def test_run_rag_pipeline_default_values(monkeypatch, sample_documents):
    monkeypatch.setattr(
        pipeline,
        "get_cached_documents",
        MagicMock(return_value=sample_documents)
    )

    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        MagicMock(return_value="Answer")
    )

    pipeline.run_rag_pipeline(
        question="What is RAG?",
        user_id="user_A"
    )

    pipeline.get_cached_documents.assert_called_once_with(
        question="What is RAG?",
        user_id="user_A",
        document_id=None,
        k=4,
        top_n=3
    )


def test_run_rag_pipeline_no_documents(monkeypatch):
    retrieval = MagicMock(return_value=[])
    generation = MagicMock()

    monkeypatch.setattr(pipeline, "get_cached_documents", retrieval)
    monkeypatch.setattr(pipeline, "generate_answer", generation)

    result = pipeline.run_rag_pipeline(
        question="What is RAG?",
        user_id="user_A"
    )

    assert result == {
        "question": "What is RAG?",
        "answer": "No relevant documents were found for this user.",
        "sources": []
    }

    generation.assert_not_called()


def test_run_rag_pipeline_retrieval_failure(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "get_cached_documents",
        MagicMock(side_effect=Exception("Retrieval failed"))
    )

    with pytest.raises(RetrievalError):
        pipeline.run_rag_pipeline(
            question="What is RAG?",
            user_id="user_A"
        )


def test_run_rag_pipeline_generation_failure(
    monkeypatch, sample_documents
):
    monkeypatch.setattr(
        pipeline,
        "get_cached_documents",
        MagicMock(return_value=sample_documents)
    )

    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        MagicMock(side_effect=Exception("Generation failed"))
    )

    with pytest.raises(Exception, match="Generation failed"):
        pipeline.run_rag_pipeline(
            question="What is RAG?",
            user_id="user_A"
        )


def test_run_rag_pipeline_missing_source_metadata(monkeypatch):
    document = MagicMock(
        page_content="Sample content",
        metadata={}
    )

    monkeypatch.setattr(
        pipeline,
        "get_cached_documents",
        MagicMock(return_value=[document])
    )

    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        MagicMock(return_value="Sample answer")
    )

    result = pipeline.run_rag_pipeline(
        question="Explain this.",
        user_id="user_A"
    )

    assert result["sources"] == [
        {
            "page_number": None,
            "file_name": None,
            "content": "Sample content"
        }
    ]
