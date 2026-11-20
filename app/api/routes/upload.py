
import os
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.security import get_current_user
from app.core.logger import get_logger
from app.core.exceptions import RAGException
from app.rag.ingestion import ingest_pdf
from app.services.vector_service import list_user_documents
from app.services.vector_service import document_exists, delete_document

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    document_id = str(uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")

    try:
        with open(file_path, "wb") as buffer:
            while chunk := file.file.read(1024 * 1024):
                buffer.write(chunk)

        result = ingest_pdf(
    file_path=file_path,
    user_id=user_id,
    document_id=document_id,
    original_file_name=os.path.basename(file.filename.replace("\\", "/"))
)

        return result

    except RAGException:
        if os.path.exists(file_path):
            os.remove(file_path)

        logger.exception("Document ingestion failed")
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed"
        )

    except Exception:
        if os.path.exists(file_path):
            os.remove(file_path)

        logger.exception("Document upload failed")
        raise HTTPException(
            status_code=500,
            detail="Document upload failed"
        )

    finally:
        file.file.close()





@router.get("")
def get_documents(
    user_id: str = Depends(get_current_user)
):
    try:
        documents = list_user_documents(user_id)

        return {
            "user_id": user_id,
            "total_documents": len(documents),
            "documents": documents
        }

    except Exception:
        logger.exception("Document listing failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve documents"
        )


@router.delete("/{document_id}")
def remove_document(
    document_id: str,
    user_id: str = Depends(get_current_user)
):
    try:
        if not document_exists(user_id, document_id):
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )

        delete_document(
            user_id=user_id,
            document_id=document_id
        )

        return {
            "message": "Document deleted successfully",
            "document_id": document_id
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Document deletion failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete document"
        )
