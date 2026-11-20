
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.security import create_access_token


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    token = create_access_token("test-user")

    return {
        "Authorization": f"Bearer {token}"
    }


@pytest.fixture
def other_user_headers():
    token = create_access_token("other-user")

    return {
        "Authorization": f"Bearer {token}"
    }
