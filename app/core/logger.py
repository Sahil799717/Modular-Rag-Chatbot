
import logging

from app.core.config import settings


logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def get_logger(name: str):
    return logging.getLogger(name)
