
from unittest.mock import patch


def test_list_documents_success(client, auth_headers):
    documents = [
        {"document_id": "doc-1", "file_name": "sample.pdf"},
        {"document_id": "doc-2", "file_name": "report.pdf"}
    ]

    with patch(
        "app.api.routes.upload.list_user_documents",
        return_value=documents
    ) as mock_list:

        response = client.get(
            "/documents",
            headers=auth_headers
        )

    assert response.status_code == 200
    assert response.json()["total_documents"] == 2
    assert response.json()["documents"] == documents
    mock_list.assert_called_once_with("test-user")


def test_list_documents_requires_authentication(client):
    response = client.get("/documents")

    assert response.status_code == 401


def test_delete_document_success(client, auth_headers):
    with patch(
        "app.api.routes.upload.document_exists",
        return_value=True
    ) as mock_exists, patch(
        "app.api.routes.upload.delete_document"
    ) as mock_delete:

        response = client.delete(
            "/documents/doc-1",
            headers=auth_headers
        )

    assert response.status_code == 200
    assert response.json()["document_id"] == "doc-1"

    mock_exists.assert_called_once_with("test-user", "doc-1")
    mock_delete.assert_called_once_with(
        user_id="test-user",
        document_id="doc-1"
    )


def test_delete_nonexistent_document(client, auth_headers):
    with patch(
        "app.api.routes.upload.document_exists",
        return_value=False
    ) as mock_exists, patch(
        "app.api.routes.upload.delete_document"
    ) as mock_delete:

        response = client.delete(
            "/documents/doc-1",
            headers=auth_headers
        )

    assert response.status_code == 404
    mock_delete.assert_not_called()


def test_user_cannot_delete_another_users_document(
    client,
    other_user_headers
):
    with patch(
        "app.api.routes.upload.document_exists",
        return_value=False
    ) as mock_exists, patch(
        "app.api.routes.upload.delete_document"
    ) as mock_delete:

        response = client.delete(
            "/documents/user-one-document",
            headers=other_user_headers
        )

    assert response.status_code == 404

    mock_exists.assert_called_once_with(
        "other-user",
        "user-one-document"
    )

    mock_delete.assert_not_called()
