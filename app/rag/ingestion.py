
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader

from app.core.logger import get_logger
from app.core.exceptions import RAGException
from app.rag.chunking import split_documents
from app.services.vector_service import (
    get_vector_store,
    delete_document,
    document_hash_exists
)
from app.services.file_service import calculate_file_hash
from app.infrastructure.cache_service import cache_increment

logger = get_logger(__name__)


def ingest_pdf(
    file_path: str,
    user_id: str,
    document_id: str,
    original_file_name: str = None
):
    if not user_id or not document_id:
        raise RAGException("user_id and document_id are required")

    try:
        file_hash = calculate_file_hash(file_path)

        if document_hash_exists(user_id, document_id, file_hash):
            logger.info("Duplicate PDF detected for user %s", user_id)
            return {
                "user_id": user_id,
                "document_id": document_id,
                "status": "duplicate",
                "chunks": 0
            }

        pages = PyPDFLoader(file_path, mode="page").load()

        for page in pages:
            page.metadata.update({
                "user_id": user_id,
                "document_id": document_id,
                "file_name": original_file_name or file_path.split("\\")[-1],
                "file_hash": file_hash
            })

        chunks = split_documents(pages)

        if not chunks:
            raise RAGException("No document chunks generated")

        vector_store = get_vector_store()

        delete_document(
            user_id=user_id,
            document_id=document_id,
            invalidate_cache=False
        )

        vector_store.add_documents(
            documents=chunks,
            ids=[str(uuid4()) for _ in chunks]
        )

        cache_increment(f"rag:version:user:{user_id}")

        logger.info("Ingested %s chunks for document %s", len(chunks), document_id)

        return {
            "user_id": user_id,
            "document_id": document_id,
            "status": "ingested",
            "chunks": len(chunks),
            "file_hash": file_hash
        }

    except Exception as e:
        logger.exception("PDF ingestion failed")
        raise RAGException("PDF ingestion failed") from e
