
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logger import get_logger
from app.core.exceptions import ChunkingError

logger = get_logger(__name__)


def split_documents(documents):

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ChunkingError("No chunks generated")

        for index, chunk in enumerate(chunks):
            page = chunk.metadata.get("page")

            chunk.metadata["chunk_id"] = index

            if page is not None:
                chunk.metadata["page_number"] = page + 1

        logger.info("Generated %s document chunks", len(chunks))

        return chunks

    except ChunkingError:
        raise

    except Exception as e:
        logger.exception("Document chunking failed")
        raise ChunkingError("Failed to split documents") from e
