
from pathlib import Path
from uuid import uuid4

import pytest

from app.rag.ingestion import ingest_pdf
from app.services.file_service import calculate_file_hash
from app.services.vector_service import (
    delete_document,
    document_exists,
    document_hash_exists,
    list_user_documents,
)


@pytest.mark.integration
def test_pdf_ingestion_and_duplicate_detection():
    file_path = Path(__file__).resolve().parents[2] / "Temple.pdf"

    assert file_path.exists(), "Temple.pdf fixture not found"

    user_id = f"integration-user-{uuid4().hex}"
    document_id = f"integration-doc-{uuid4().hex}"

    try:
        result = ingest_pdf(
            file_path=str(file_path),
            user_id=user_id,
            document_id=document_id,
            original_file_name="Temple.pdf",
        )

        assert result["status"] == "ingested"
        assert result["chunks"] > 0

        file_hash = calculate_file_hash(str(file_path))

        assert result["file_hash"] == file_hash
        assert document_exists(user_id, document_id)
        assert document_hash_exists(user_id, document_id, file_hash)

        documents = list_user_documents(user_id)

        assert len(documents) == 1
        assert documents[0]["document_id"] == document_id
        assert documents[0]["file_name"] == "Temple.pdf"
        assert documents[0]["chunks"] == result["chunks"]

        duplicate_result = ingest_pdf(
            file_path=str(file_path),
            user_id=user_id,
            document_id=document_id,
            original_file_name="Temple.pdf",
        )

        assert duplicate_result["status"] == "duplicate"
        assert duplicate_result["chunks"] == 0

    finally:
        delete_document(
            user_id=user_id,
            document_id=document_id,
            invalidate_cache=False,
        )
