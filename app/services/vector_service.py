
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.core.config import settings
from app.core.logger import get_logger
from app.core.exceptions import VectorStoreError
from app.services.embedding_service import get_embedding_model
from app.infrastructure.qdrant_client import get_qdrant_client
from app.infrastructure.cache_service import cache_increment

logger = get_logger(__name__)


def get_vector_store():
    try:
        return QdrantVectorStore(
            client=get_qdrant_client(),
            collection_name=settings.QDRANT_COLLECTION,
            embedding=get_embedding_model()
        )
    except Exception as e:
        logger.exception("Vector store initialization failed")
        raise VectorStoreError("Failed to initialize vector store") from e


def get_filtered_retriever(user_id: str, document_id: str = None, k: int = 4):
    if not user_id:
        raise VectorStoreError("user_id is required")

    try:
        conditions = [
            FieldCondition(
                key="metadata.user_id",
                match=MatchValue(value=user_id)
            )
        ]

        if document_id:
            conditions.append(
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchValue(value=document_id)
                )
            )

        return get_vector_store().as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
                "filter": Filter(must=conditions)
            }
        )

    except Exception as e:
        logger.exception("Filtered retriever initialization failed")
        raise VectorStoreError("Failed to initialize filtered retriever") from e


def delete_document(
    user_id: str,
    document_id: str,
    invalidate_cache: bool = True
):
    if not user_id or not document_id:
        raise VectorStoreError("user_id and document_id are required")

    try:
        get_qdrant_client().delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.user_id",
                        match=MatchValue(value=user_id)
                    ),
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),
            wait=True
        )

        if invalidate_cache:
            cache_increment(f"rag:version:user:{user_id}")

        logger.info("Deleted document %s for user %s", document_id, user_id)

    except Exception as e:
        logger.exception("Document deletion failed")
        raise VectorStoreError("Failed to delete document") from e



def document_hash_exists(
    user_id: str,
    document_id: str,
    file_hash: str
) -> bool:
    if not user_id or not document_id or not file_hash:
        raise VectorStoreError(
            "user_id, document_id and file_hash are required"
        )

    try:
        result = get_qdrant_client().count(
            collection_name=settings.QDRANT_COLLECTION,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.user_id",
                        match=MatchValue(value=user_id)
                    ),
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(value=document_id)
                    ),
                    FieldCondition(
                        key="metadata.file_hash",
                        match=MatchValue(value=file_hash)
                    )
                ]
            ),
            exact=True
        )

        return result.count > 0

    except Exception as e:
        logger.exception("Document hash lookup failed")
        raise VectorStoreError(
            "Failed to check document hash"
        ) from e



def list_user_documents(user_id: str):
    if not user_id:
        raise VectorStoreError("user_id is required")

    try:
        client = get_qdrant_client()

        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.user_id",
                    match=MatchValue(value=user_id)
                )
            ]
        )

        documents = {}

        offset = None

        while True:
            points, offset = client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                scroll_filter=qdrant_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            for point in points:
                metadata = (point.payload or {}).get("metadata", {})
                document_id = metadata.get("document_id")

                if not document_id:
                    continue

                if document_id not in documents:
                    documents[document_id] = {
                        "document_id": document_id,
                        "user_id": user_id,
                        "file_name": metadata.get("file_name"),
                        "file_hash": metadata.get("file_hash"),
                        "chunks": 0
                    }

                documents[document_id]["chunks"] += 1

            if offset is None:
                break

        return list(documents.values())

    except Exception as e:
        logger.exception("Document listing failed")
        raise VectorStoreError("Failed to list user documents") from e


def document_exists(user_id: str, document_id: str) -> bool:
    if not user_id or not document_id:
        raise VectorStoreError("user_id and document_id are required")

    try:
        result = get_qdrant_client().count(
            collection_name=settings.QDRANT_COLLECTION,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.user_id",
                        match=MatchValue(value=user_id)
                    ),
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),
            exact=True
        )

        return result.count > 0

    except Exception as e:
        logger.exception("Document existence check failed")
        raise VectorStoreError("Failed to check document") from e
