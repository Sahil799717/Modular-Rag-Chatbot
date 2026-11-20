
class RAGException(Exception):
    """Base exception for the RAG application."""


class PDFProcessingError(RAGException):
    """Raised when PDF processing fails."""


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""


class VectorStoreError(RAGException):
    """Raised when vector database operations fail."""


class RetrievalError(RAGException):
    """Raised when document retrieval fails."""


class GenerationError(RAGException):
    """Raised when answer generation fails."""

class ChunkingError(RAGException):
    """Raised when document chunking fails."""