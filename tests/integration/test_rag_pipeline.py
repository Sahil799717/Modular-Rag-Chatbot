
from uuid import uuid4
from pathlib import Path

import pytest

from app.rag.ingestion import ingest_pdf
from app.rag.pipeline import run_rag_pipeline
from app.services.vector_service import delete_document


@pytest.mark.integration
def test_rag_pipeline_with_redis_cache():
    file_path = Path(__file__).resolve().parents[2] / "Temple.pdf"

    user_id = f"integration-user-{uuid4().hex}"
    document_id = f"integration-doc-{uuid4().hex}"

    question = (
        "What is the difference between a shikhara "
        "and a vimana in Indian temple architecture?"
    )

    try:
        ingestion = ingest_pdf(
            file_path=str(file_path),
            user_id=user_id,
            document_id=document_id,
            original_file_name="Temple.pdf",
        )

        assert ingestion["status"] == "ingested"

        first_response = run_rag_pipeline(
            question=question,
            user_id=user_id,
            document_id=document_id,
        )

        assert first_response["answer"]
        assert "shikhara" in first_response["answer"].lower()
        assert "vimana" in first_response["answer"].lower()
        assert first_response["sources"]

        second_response = run_rag_pipeline(
            question=question,
            user_id=user_id,
            document_id=document_id,
        )

        assert second_response["answer"]
        assert second_response["sources"] == first_response["sources"]

    finally:
        delete_document(
            user_id=user_id,
            document_id=document_id,
            invalidate_cache=True,
        )
