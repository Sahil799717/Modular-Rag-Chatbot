
import hashlib
import json

from langchain_core.documents import Document

from app.core.config import settings
from app.core.logger import get_logger
from app.infrastructure.cache_service import cache_get, cache_set
from app.rag.reranker import rerank_documents

logger = get_logger(__name__)


def get_retrieval_cache_key(
    question: str,
    user_id: str,
    document_id: str = None,
    k: int = 4,
    top_n: int = 3,
    corpus_version: int = 0
):
    payload = {
        "question": " ".join(question.split()),
        "user_id": user_id,
        "document_id": document_id,
        "embedding_model": settings.EMBEDDING_MODEL,
        "k": k,
        "top_n": top_n,
        "corpus_version": corpus_version
    }

    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()

    return f"rag:retrieval:{digest}"


def get_cached_documents(
    question: str,
    user_id: str,
    document_id: str = None,
    k: int = 4,
    top_n: int = 3
):
    try:
        corpus_version = get_user_corpus_version(user_id)
    except Exception:
        logger.warning("Unable to read corpus version; bypassing retrieval cache")
        return rerank_documents(
            question=question,
            user_id=user_id,
            document_id=document_id,
            k=k,
            top_n=top_n
        )

    key = get_retrieval_cache_key(
        question, user_id, document_id, k, top_n, corpus_version
    )

    try:
        cached = cache_get(key)

        if cached is not None:
            logger.info("Retrieval cache hit")
            return [Document(**doc) for doc in cached]

    except Exception:
        logger.warning("Retrieval cache read failed")

    logger.info("Retrieval cache miss")

    documents = rerank_documents(
        question=question,
        user_id=user_id,
        document_id=document_id,
        k=k,
        top_n=top_n
    )

    # Cache only non-empty retrieval results
    if documents:
        serialized = [
            {
                "page_content": doc.page_content,
                "metadata": make_json_safe(doc.metadata)
            }
            for doc in documents
        ]

        try:
            cache_set(key, serialized)
        except Exception:
            logger.warning("Retrieval cache write failed")
    else:
        logger.warning("No documents retrieved; skipping cache")

    return documents


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if hasattr(value, "tolist"):
        return make_json_safe(value.tolist())

    if hasattr(value, "item"):
        return make_json_safe(value.item())

    return value


def get_user_corpus_version(user_id: str):
    key = f"rag:version:user:{user_id}"
    version = cache_get(key)
    return version if version is not None else 0
