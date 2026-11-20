from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logger import get_logger
from app.core.exceptions import GenerationError

logger = get_logger(__name__)


def get_llm():
    try:
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            base_url=settings.BIFROST_BASE_URL,
            api_key="bifrost",
            default_headers={
                "x-bf-vk": settings.BIFROST_VIRTUAL_KEY
            },
            temperature=0
        )

        logger.info("LLM initialized through Bifrost")
        return llm

    except Exception as e:
        logger.exception("LLM initialization failed")
        raise GenerationError("Failed to initialize LLM") from e