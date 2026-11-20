
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    APP_NAME: str
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    OLLAMA_BASE_URL: str
    EMBEDDING_MODEL: str

    QDRANT_URL: str
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str

    BIFROST_BASE_URL: str
    BIFROST_VIRTUAL_KEY: str
    LLM_MODEL: str = "gemini-2.5-flash"

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    REDIS_URL: str
    REDIS_TTL: int = 3600

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
