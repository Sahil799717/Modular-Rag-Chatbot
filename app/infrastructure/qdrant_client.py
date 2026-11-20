
from qdrant_client import QdrantClient

from app.core.config import settings
from app.core.logger import get_logger
from app.core.exceptions import VectorStoreError
from qdrant_client.models import VectorParams, Distance

logger = get_logger(__name__)




def create_collection(client, embeddings):

    collection_name = settings.QDRANT_COLLECTION

    try:
        if client.collection_exists(collection_name):
            logger.info("Qdrant collection already exists")
            return

        vector_size = len(
            embeddings.embed_query("dimension verification")
        )

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

        logger.info(
            "Qdrant collection created with %s dimensions",
            vector_size
        )

    except Exception as e:
        logger.exception("Qdrant collection creation failed")
        raise VectorStoreError(
            "Failed to create Qdrant collection"
        ) from e


def get_qdrant_client():
    try:
        api_key = settings.QDRANT_API_KEY

        client = QdrantClient(
            url=settings.QDRANT_URL,
            **(
                {"api_key": api_key.strip()}
                if api_key and api_key.strip()
                else {}
            )
        )

        client.get_collections()
        logger.info("Qdrant connection established")
        return client

    except Exception as e:
        logger.exception("Qdrant connection failed")
        raise VectorStoreError("Failed to connect to Qdrant") from e