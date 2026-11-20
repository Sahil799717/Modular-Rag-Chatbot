from datetime import datetime, timedelta, timezone

import jwt

from app.core.config import settings

def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_chat_requires_authentication(client):
    response = client.post(
        "/chat",
        json={"question": "What is this document about?"}
    )

    assert response.status_code == 401


def test_chat_accepts_valid_token(client, auth_headers, monkeypatch):
    from app.api.routes import chat as chat_route

    def mock_chat(question, user_id, document_id=None):
        return {
            "question": question,
            "answer": "Test answer",
            "user_id": user_id
        }

    monkeypatch.setattr(chat_route, "chat", mock_chat)

    response = client.post(
        "/chat",
        json={"question": "What is this document about?"},
        headers=auth_headers
    )

    assert response.status_code == 200
    assert response.json()["user_id"] == "test-user"


def test_chat_rejects_invalid_token(client):
    response = client.post(
        "/chat",
        json={"question": "Test question"},
        headers={"Authorization": "Bearer invalid-token"}
    )

    assert response.status_code == 401


def test_chat_rejects_expired_token(client):
    token = jwt.encode(
        {
            "sub": "test-user",
            "exp": datetime.now(timezone.utc) - timedelta(minutes=5)
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )

    response = client.post(
        "/chat",
        json={"question": "Test question"},
        headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 401


def test_chat_rejects_token_without_subject(client):
    token = jwt.encode(
        {
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5)
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )

    response = client.post(
        "/chat",
        json={"question": "Test question"},
        headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 401


def test_chat_passes_document_id(client, auth_headers, monkeypatch):
    from app.api.routes import chat as chat_route

    def mock_chat(question, user_id, document_id=None):
        return {
            "question": question,
            "user_id": user_id,
            "document_id": document_id,
            "answer": "Test answer"
        }

    monkeypatch.setattr(chat_route, "chat", mock_chat)

    response = client.post(
        "/chat",
        json={
            "question": "Summarize this document",
            "document_id": "doc-123"
        },
        headers=auth_headers
    )

    assert response.status_code == 200
    assert response.json()["document_id"] == "doc-123"


def test_chat_rejects_empty_question(client, auth_headers):
    response = client.post(
        "/chat",
        json={"question": ""},
        headers=auth_headers
    )

    assert response.status_code == 422


def test_chat_handles_service_failure(client, auth_headers, monkeypatch):
    from app.api.routes import chat as chat_route

    def mock_chat(question, user_id, document_id=None):
        raise RuntimeError("RAG pipeline failed")

    monkeypatch.setattr(chat_route, "chat", mock_chat)

    response = client.post(
        "/chat",
        json={"question": "Test question"},
        headers=auth_headers
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to process chat request"
