
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.security import get_current_user
from app.core.logger import get_logger
from app.services.chat_service import chat

logger = get_logger(__name__)

router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: str | None = None


@router.post("/chat")
def chat_endpoint(
    request: ChatRequest,
    user_id: str = Depends(get_current_user)
):
    try:
        return chat(
            question=request.question,
            user_id=user_id,
            document_id=request.document_id
        )

    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to process chat request"
        )
