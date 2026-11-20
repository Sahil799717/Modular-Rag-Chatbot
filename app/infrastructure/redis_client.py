
from functools import lru_cache

import redis

from app.core.config import settings
from app.core.logger import get_logger
from app.core.exceptions import RAGException

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_redis_client():
    try:
        client = redis.Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True
        )
        client.ping()
        logger.info("Redis connection established")
        return client

    except Exception as e:
        logger.exception("Redis connection failed")
        raise RAGException("Failed to connect to Redis") from e
