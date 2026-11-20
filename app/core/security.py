
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials



from app.core.config import settings

bearer_scheme = HTTPBearer()



def create_access_token(user_id: str):
    if not user_id:
        raise ValueError("user_id is required")

    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": user_id,
        "exp": expires_at
    }

    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token",
        headers={"WWW-Authenticate": "Bearer"}
    )

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["sub", "exp"]}
        )

        user_id = payload.get("sub")

        if not isinstance(user_id, str) or not user_id.strip():
            raise credentials_exception

        return user_id

    except jwt.InvalidTokenError:
        raise credentials_exception