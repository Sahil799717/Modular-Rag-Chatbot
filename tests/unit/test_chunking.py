
import pytest
from langchain_core.documents import Document

from app.rag.chunking import split_documents
from app.core.exceptions import ChunkingError


def test_split_documents():
    documents = [
        Document(
            page_content="This is a sample document. " * 100,
            metadata={"page": 0, "file_name": "sample.pdf"}
        )
    ]

    chunks = split_documents(documents)

    assert len(chunks) > 1
    assert all(chunk.page_content for chunk in chunks)


def test_chunk_metadata():
    documents = [
        Document(
            page_content="Sample content " * 100,
            metadata={"page": 2, "file_name": "sample.pdf"}
        )
    ]

    chunks = split_documents(documents)

    assert all(chunk.metadata["page_number"] == 3 for chunk in chunks)
    assert all(chunk.metadata["file_name"] == "sample.pdf" for chunk in chunks)


def test_chunk_ids_are_unique():
    documents = [
        Document(page_content="Sample content " * 200, metadata={"page": 0})
    ]

    chunks = split_documents(documents)

    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_empty_documents():
    with pytest.raises(ChunkingError):
        split_documents([])


def test_chunk_size():
    documents = [
        Document(
            page_content="A" * 3000,
            metadata={"page": 0}
        )
    ]

    chunks = split_documents(documents)

    assert all(
        len(chunk.page_content) <= 1000
        for chunk in chunks
    )


def test_chunk_overlap():
    documents = [
        Document(
            page_content="A" * 3000,
            metadata={"page": 0}
        )
    ]

    chunks = split_documents(documents)

    for previous, current in zip(chunks, chunks[1:]):
        assert previous.page_content[-150:] == current.page_content[:150]
