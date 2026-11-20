
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

from app.core.logger import get_logger
from app.core.exceptions import RetrievalError
from app.rag.retriever import get_multi_query_retriever

logger = get_logger(__name__)


def get_reranked_retriever(
    user_id: str,
    document_id: str = None,
    k: int = 4,
    top_n: int = 3
):
    try:
        base_retriever = get_multi_query_retriever(
            user_id=user_id,
            document_id=document_id,
            k=k
        )

        compressor = FlashrankRerank(top_n=top_n)

        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        logger.info("FlashRank reranker initialized")
        return retriever

    except Exception as e:
        logger.exception("Reranker initialization failed")
        raise RetrievalError("Failed to initialize reranker") from e


def rerank_documents(
    question: str,
    user_id: str,
    document_id: str = None,
    k: int = 4,
    top_n: int = 3
):
    try:
        retriever = get_reranked_retriever(
            user_id=user_id,
            document_id=document_id,
            k=k,
            top_n=top_n
        )

        documents = retriever.invoke(question)

        logger.info("Reranked documents: %s", len(documents))
        return documents

    except RetrievalError:
        raise
    except Exception as e:
        logger.exception("Document reranking failed")
        raise RetrievalError("Document reranking failed") from e
