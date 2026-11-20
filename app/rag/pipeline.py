
import time

from app.core.logger import get_logger
from app.core.exceptions import RetrievalError
from app.rag.retrieval_cache import get_cached_documents
from app.rag.generation import generate_answer

logger = get_logger(__name__)


def run_rag_pipeline(
    question: str,
    user_id: str,
    document_id: str = None,
    k: int = 4,
    top_n: int = 3
):
    pipeline_start = time.perf_counter()

    try:
        retrieval_start = time.perf_counter()

        documents = get_cached_documents(
            question=question,
            user_id=user_id,
            document_id=document_id,
            k=k,
            top_n=top_n
        )

        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    except Exception as e:
        logger.exception("Document retrieval failed")
        raise RetrievalError("Failed to retrieve documents") from e

    if not documents:
        total_ms = (time.perf_counter() - pipeline_start) * 1000

        logger.info(
            "RAG pipeline completed | retrieval=%.2f ms | generation=0 ms | total=%.2f ms",
            retrieval_ms,
            total_ms
        )

        return {
            "question": question,
            "answer": "No relevant documents were found for this user.",
            "sources": []
        }

    generation_start = time.perf_counter()

    answer = generate_answer(
        question=question,
        documents=documents
    )

    generation_ms = (time.perf_counter() - generation_start) * 1000
    total_ms = (time.perf_counter() - pipeline_start) * 1000

    logger.info(
        "RAG pipeline completed | retrieval=%.2f ms | generation=%.2f ms | total=%.2f ms",
        retrieval_ms,
        generation_ms,
        total_ms
    )

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "page_number": doc.metadata.get("page_number"),
                "file_name": doc.metadata.get("file_name"),
                "content": doc.page_content
            }
            for doc in documents
        ]
    }
