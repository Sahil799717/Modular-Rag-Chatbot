
import hashlib

from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings

from app.core.config import settings
from app.core.logger import get_logger
from app.core.exceptions import EmbeddingError
from app.infrastructure.cache_service import cache_get, cache_set

logger = get_logger(__name__)


class CachedOllamaEmbeddings(Embeddings):

    def __init__(self, model):
        self.model = model

    def _cache_key(self, text, operation):
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"embedding:{settings.EMBEDDING_MODEL}:{operation}:{text_hash}"

    def embed_query(self, text):
        key = self._cache_key(text, "query")

        try:
            cached = cache_get(key)
            if cached is not None:
                return cached
        except Exception:
            logger.warning("Embedding cache read failed; using Ollama")

        embedding = self.model.embed_query(text)

        try:
            cache_set(key, embedding)
        except Exception:
            logger.warning("Embedding cache write failed")

        return embedding

    def embed_documents(self, texts):
        results = [None] * len(texts)
        missing = []

        for index, text in enumerate(texts):
            key = self._cache_key(text, "document")

            try:
                cached = cache_get(key)
            except Exception:
                cached = None

            if cached is not None:
                results[index] = cached
            else:
                missing.append((index, text, key))

        if missing:
            embeddings = self.model.embed_documents(
                [text for _, text, _ in missing]
            )

            for (index, text, key), embedding in zip(missing, embeddings):
                results[index] = embedding

                try:
                    cache_set(key, embedding)
                except Exception:
                    logger.warning("Embedding cache write failed")

        return results


def get_embedding_model():
    try:
        model = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )

        logger.info("Cached embedding model initialized")
        return CachedOllamaEmbeddings(model)

    except Exception as e:
        logger.exception("Embedding initialization failed")
        raise EmbeddingError("Failed to initialize embedding model") from e
