
from unittest.mock import patch

from app.core.exceptions import PDFProcessingError


def test_upload_pdf_success(client, auth_headers):
    with patch(
        "app.api.routes.upload.ingest_pdf"
    ) as mock_ingest:

        mock_ingest.return_value = {
            "document_id": "test-document",
            "chunks": 5
        }

        response = client.post(
            "/documents/upload",
            headers=auth_headers,
            files={
                "file": (
                    "sample.pdf",
                    b"fake pdf content",
                    "application/pdf"
                )
            }
        )

    assert response.status_code == 201
    assert response.json()["document_id"] == "test-document"
    mock_ingest.assert_called_once()


def test_upload_rejects_non_pdf(client, auth_headers):
    response = client.post(
        "/documents/upload",
        headers=auth_headers,
        files={
            "file": (
                "sample.txt",
                b"sample text",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are allowed"


def test_upload_requires_authentication(client):
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "sample.pdf",
                b"fake pdf content",
                "application/pdf"
            )
        }
    )

    assert response.status_code == 401



from unittest.mock import patch

from app.core.exceptions import RAGException


def test_upload_handles_ingestion_failure(client, auth_headers):
    with patch(
        "app.api.routes.upload.ingest_pdf",
        side_effect=RAGException("Ingestion failed")
    ):
        response = client.post(
            "/documents/upload",
            headers=auth_headers,
            files={
                "file": (
                    "sample.pdf",
                    b"fake pdf content",
                    "application/pdf"
                )
            }
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Document ingestion failed"


def test_upload_removes_file_after_ingestion_failure(client, auth_headers):
    with patch(
        "app.api.routes.upload.ingest_pdf",
        side_effect=RAGException("Ingestion failed")
    ):
        with patch("app.api.routes.upload.os.remove") as mock_remove:
            response = client.post(
                "/documents/upload",
                headers=auth_headers,
                files={
                    "file": (
                        "sample.pdf",
                        b"fake pdf content",
                        "application/pdf"
                    )
                }
            )

    assert response.status_code == 500
    mock_remove.assert_called_once()
