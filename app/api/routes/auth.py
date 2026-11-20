
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.security import create_access_token

router = APIRouter(prefix="/auth", tags=["Authentication"])


class DemoLoginRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)


@router.post("/demo-login")
def demo_login(request: DemoLoginRequest):
    user_id = request.user_id.strip()

    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    return {
        "access_token": create_access_token(user_id),
        "token_type": "bearer",
        "user_id": user_id
    }
