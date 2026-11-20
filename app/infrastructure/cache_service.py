
import json

from app.core.logger import get_logger
from app.core.config import settings
from app.infrastructure.redis_client import get_redis_client
from app.core.exceptions import RAGException

logger = get_logger(__name__)


def cache_get(key: str):
    try:
        value = get_redis_client().get(key)
        return json.loads(value) if value else None

    except Exception as e:
        logger.exception("Cache read failed")
        raise RAGException("Failed to read cache") from e



def cache_set(key: str, value, ttl: int = None):
    try:
        redis_client = get_redis_client()
        redis_client.set(
            key,
            json.dumps(value, default=json_serializer),
            ex=ttl or settings.REDIS_TTL
        )
    except Exception as e:
        logger.exception("Cache write failed")
        raise RAGException("Failed to write cache") from e


def cache_delete(key: str):
    try:
        get_redis_client().delete(key)

    except Exception as e:
        logger.exception("Cache deletion failed")
        raise RAGException("Failed to delete cache") from e


def json_serializer(value):
    if hasattr(value, "tolist"):
        return value.tolist()

    if hasattr(value, "item"):
        return value.item()

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def cache_increment(key: str):
    try:
        return get_redis_client().incr(key)
    except Exception as e:
        logger.exception("Cache version increment failed")
        raise RAGException("Failed to increment cache version") from e
