
from langchain_classic.retrievers import MultiQueryRetriever

from app.core.logger import get_logger
from app.core.exceptions import RetrievalError
from app.infrastructure.llm_gateway import get_llm
from app.services.vector_service import get_filtered_retriever

logger = get_logger(__name__)


def get_multi_query_retriever(
    user_id: str,
    document_id: str = None,
    k: int = 4
):
    try:
        base_retriever = get_filtered_retriever(
            user_id=user_id,
            document_id=document_id,
            k=k
        )

        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=get_llm()
        )

        logger.info("Multi-query retriever initialized")
        return retriever

    except Exception as e:
        logger.exception("Multi-query retriever initialization failed")
        raise RetrievalError("Failed to initialize multi-query retriever") from e


def retrieve_documents(
    question: str,
    user_id: str,
    document_id: str = None,
    k: int = 4
):
    try:
        retriever = get_multi_query_retriever(
            user_id=user_id,
            document_id=document_id,
            k=k
        )

        documents = retriever.invoke(question)

        logger.info("Multi-query retrieved %s documents", len(documents))
        return documents

    except RetrievalError:
        raise

    except Exception as e:
        logger.exception("Multi-query retrieval failed")
        raise RetrievalError("Document retrieval failed") from e
