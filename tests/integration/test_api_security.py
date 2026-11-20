
from pathlib import Path
from uuid import uuid4

import pytest

from app.core.security import create_access_token
from app.rag.ingestion import ingest_pdf
from app.services.vector_service import delete_document, document_exists
from datetime import datetime, timedelta, timezone

import jwt

from app.core.config import settings


@pytest.mark.integration
def test_api_multi_user_security(client):
    file_path = Path(__file__).resolve().parents[2] / "Temple.pdf"

    assert file_path.exists(), "Temple.pdf fixture not found"

    user_a = f"integration-user-{uuid4().hex}"
    user_b = f"integration-user-{uuid4().hex}"

    document_a = f"integration-doc-{uuid4().hex}"
    document_b = f"integration-doc-{uuid4().hex}"

    headers_a = {
        "Authorization": f"Bearer {create_access_token(user_a)}"
    }
    headers_b = {
        "Authorization": f"Bearer {create_access_token(user_b)}"
    }

    try:
        ingest_pdf(
            file_path=str(file_path),
            user_id=user_a,
            document_id=document_a,
            original_file_name="Temple.pdf",
        )

        ingest_pdf(
            file_path=str(file_path),
            user_id=user_b,
            document_id=document_b,
            original_file_name="Temple.pdf",
        )

        # User A should only see their own documents.
        response = client.get("/documents", headers=headers_a)

        assert response.status_code == 200

        documents = response.json()["documents"]

        assert len(documents) == 1
        assert documents[0]["document_id"] == document_a

        # User A must not delete User B's document.
        response = client.delete(
            f"/documents/{document_b}",
            headers=headers_a,
        )

        assert response.status_code == 404
        assert document_exists(user_b, document_b)

        # User A must not retrieve User B's document through chat.
        response = client.post(
            "/chat",
            headers=headers_a,
            json={
                "question": "What is a garbhagriha?",
                "document_id": document_b,
            },
        )

        assert response.status_code == 200
        assert response.json()["sources"] == []
        assert response.json()["answer"] == (
            "No relevant documents were found for this user."
        )

        # User B should still be able to see their document.
        response = client.get("/documents", headers=headers_b)

        assert response.status_code == 200
        assert response.json()["documents"][0]["document_id"] == document_b

    finally:
        delete_document(user_a, document_a, invalidate_cache=True)
        delete_document(user_b, document_b, invalidate_cache=True)


def test_protected_endpoints_require_authentication(client):
    responses = [
        client.get("/documents"),
        client.delete("/documents/test-document"),
        client.post(
            "/chat",
            json={"question": "What is a garbhagriha?"},
        ),
    ]

    for response in responses:
        assert response.status_code == 401


def test_protected_endpoints_reject_invalid_token(client):
    headers = {
        "Authorization": "Bearer invalid-token"
    }

    assert client.get(
        "/documents", headers=headers
    ).status_code == 401

    assert client.delete(
        "/documents/test-document", headers=headers
    ).status_code == 401

    assert client.post(
        "/chat",
        headers=headers,
        json={"question": "What is a garbhagriha?"},
    ).status_code == 401



def test_protected_endpoints_reject_expired_token(client):
    payload = {
        "sub": "integration-user",
        "exp": datetime.now(timezone.utc) - timedelta(minutes=5),
    }

    token = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    headers = {
        "Authorization": f"Bearer {token}"
    }

    assert client.get(
        "/documents", headers=headers
    ).status_code == 401

    assert client.delete(
        "/documents/test-document", headers=headers
    ).status_code == 401

    assert client.post(
        "/chat",
        headers=headers,
        json={"question": "What is a garbhagriha?"},
    ).status_code == 401
