
from app.core.logger import get_logger
from app.rag.pipeline import run_rag_pipeline

logger = get_logger(__name__)


def chat(
    question: str,
    user_id: str,
    document_id: str = None
):
    try:
        return run_rag_pipeline(
            question=question,
            user_id=user_id,
            document_id=document_id
        )

    except Exception:
        logger.exception("Chat service failed")
        raise
