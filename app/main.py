
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.routes.upload import router as upload_router
from app.api.routes.chat import router as chat_router
from app.monitoring.middleware import request_monitoring
from app.infrastructure.redis_client import get_redis_client
from app.infrastructure.qdrant_client import get_qdrant_client
from app.api.routes.auth import router as auth_router

app = FastAPI(
    title="Modular RAG Chatbot",
    description="Authenticated document-based RAG API",
    version="1.0.0"
)
app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(chat_router)

app.middleware("http")(request_monitoring)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check():
    services = {}

    try:
        services["redis"] = "healthy" if get_redis_client().ping() else "unhealthy"
    except Exception:
        services["redis"] = "unhealthy"

    try:
        get_qdrant_client()
        services["qdrant"] = "healthy"
    except Exception:
        services["qdrant"] = "unhealthy"

    is_ready = all(status == "healthy" for status in services.values())

    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={
            "status": "ready" if is_ready else "not_ready",
            "services": services
        }
    )
