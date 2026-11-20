
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.rag import generation
from app.core.exceptions import GenerationError



def test_generate_answer(monkeypatch):
    captured = {}

    def fake_llm(prompt):
        captured["prompt"] = prompt.to_string()

        return AIMessage(
            content="RAG combines retrieval and generation."
        )

    monkeypatch.setattr(
        generation,
        "get_llm",
        lambda: RunnableLambda(fake_llm)
    )

    documents = [
        MagicMock(
            page_content="RAG combines retrieval with generation.",
            metadata={"page_number": 2}
        ),
        MagicMock(
            page_content="It improves answer relevance.",
            metadata={"page_number": 3}
        )
    ]

    result = generation.generate_answer(
        question="What is RAG?",
        documents=documents
    )

    assert result == "RAG combines retrieval and generation."
    assert "What is RAG?" in captured["prompt"]
    assert "Page 2:" in captured["prompt"]
    assert "Page 3:" in captured["prompt"]
    assert "It improves answer relevance." in captured["prompt"]


def test_generate_answer_without_page_number(monkeypatch):
    monkeypatch.setattr(
        generation,
        "get_llm",
        lambda: RunnableLambda(
            lambda prompt: AIMessage(content="Generated answer")
        )
    )

    document = MagicMock(
        page_content="Sample document content",
        metadata={}
    )

    result = generation.generate_answer(
        question="Explain this.",
        documents=[document]
    )

    assert result == "Generated answer"


def test_generate_answer_without_documents():
    with pytest.raises(GenerationError):
        generation.generate_answer(
            question="What is RAG?",
            documents=[]
        )


def test_generate_answer_llm_failure(monkeypatch):
    monkeypatch.setattr(
        generation,
        "get_llm",
        MagicMock(side_effect=Exception("LLM unavailable"))
    )

    document = MagicMock(
        page_content="Sample content",
        metadata={"page_number": 1}
    )

    with pytest.raises(GenerationError):
        generation.generate_answer(
            question="What is RAG?",
            documents=[document]
        )


def test_generate_answer_chain_failure(monkeypatch):
    def failing_llm(prompt):
        raise Exception("Generation failed")

    monkeypatch.setattr(
        generation,
        "get_llm",
        lambda: RunnableLambda(failing_llm)
    )

    document = MagicMock(
        page_content="Sample content",
        metadata={"page_number": 1}
    )

    with pytest.raises(GenerationError):
        generation.generate_answer(
            question="What is RAG?",
            documents=[document]
        )
