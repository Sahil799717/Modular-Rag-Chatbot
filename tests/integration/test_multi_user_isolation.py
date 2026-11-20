
from pathlib import Path
from uuid import uuid4

import pytest

from app.rag.ingestion import ingest_pdf
from app.services.vector_service import (
    delete_document,
    document_exists,
    get_filtered_retriever,
    list_user_documents,
)


@pytest.mark.integration
def test_qdrant_multi_user_isolation():
    file_path = Path(__file__).resolve().parents[2] / "Temple.pdf"

    assert file_path.exists(), "Temple.pdf fixture not found"

    user_a = f"integration-user-{uuid4().hex}"
    user_b = f"integration-user-{uuid4().hex}"

    document_a = f"integration-doc-{uuid4().hex}"
    document_b = f"integration-doc-{uuid4().hex}"

    try:
        ingest_pdf(
            file_path=str(file_path),
            user_id=user_a,
            document_id=document_a,
            original_file_name="Temple.pdf",
        )

        ingest_pdf(
            file_path=str(file_path),
            user_id=user_b,
            document_id=document_b,
            original_file_name="Temple.pdf",
        )

        # Each user should see only their own documents.
        documents_a = list_user_documents(user_a)
        documents_b = list_user_documents(user_b)

        assert len(documents_a) == 1
        assert len(documents_b) == 1

        assert documents_a[0]["document_id"] == document_a
        assert documents_b[0]["document_id"] == document_b

        # A user must not be able to access another user's document.
        assert not document_exists(user_a, document_b)
        assert not document_exists(user_b, document_a)

        # Retrieval must return only the requesting user's chunks.
        results_a = get_filtered_retriever(
            user_id=user_a,
            document_id=document_a,
        ).invoke("What is a garbhagriha?")

        results_b = get_filtered_retriever(
            user_id=user_b,
            document_id=document_b,
        ).invoke("What is a garbhagriha?")

        assert results_a
        assert results_b

        assert all(doc.metadata["user_id"] == user_a for doc in results_a)
        assert all(doc.metadata["document_id"] == document_a for doc in results_a)

        assert all(doc.metadata["user_id"] == user_b for doc in results_b)
        assert all(doc.metadata["document_id"] == document_b for doc in results_b)

        # User A attempting to delete User B's document must not affect it.
        delete_document(user_a, document_b, invalidate_cache=False)

        assert document_exists(user_b, document_b)

        # Deleting User A's own document must leave User B's data untouched.
        delete_document(user_a, document_a, invalidate_cache=False)

        assert not document_exists(user_a, document_a)
        assert document_exists(user_b, document_b)

    finally:
        delete_document(user_a, document_a, invalidate_cache=False)
        delete_document(user_b, document_b, invalidate_cache=False)
