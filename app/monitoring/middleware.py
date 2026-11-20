
import time

from app.core.logger import get_logger

logger = get_logger(__name__)


async def request_monitoring(request, call_next):
    start = time.perf_counter()

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "HTTP request | method=%s | path=%s | status=%s | duration=%.2f ms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms
        )

        return response

    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000

        logger.exception(
            "HTTP request failed | method=%s | path=%s | duration=%.2f ms",
            request.method,
            request.url.path,
            duration_ms
        )

        raise
