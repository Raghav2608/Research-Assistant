import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.apps.app_retrieval import app 
from fastapi import status
from dotenv import load_dotenv
import os
from backend.src.backend.user_authentication.utils import validate_request

client = TestClient(app)

load_dotenv()

def dummy_validate_request():
    return

@pytest.fixture
def mock_auth():
    """Mock FastAPI authentication dependency"""
    app.dependency_overrides[validate_request] = dummy_validate_request
    yield
    app.dependency_overrides = {}

@pytest.fixture
def mock_verification():
     with patch("backend.apps.app_retrieval.verify_token", return_value={"user_id":"test"}):
        yield


@pytest.fixture
def mock_query_generator():
    """Mock query_generator"""
    with patch("backend.apps.app_retrieval.ResearchQueryGenerator.generate", return_value=["query1", "query2"]):
        yield

@pytest.fixture
def mock_fast_pipeline():
    """Mock fast pipeline response"""
    with patch("backend.apps.app_retrieval.use_fast_pipeline", return_value=["doc1", "doc2"]):
        yield

@pytest.fixture
def mock_specific_pipeline():
    """Mock specific pipeline response"""
    with patch("backend.apps.app_retrieval.use_specific_pipeline", return_value=["doc3", "doc4"]):
        yield


def test_retrieve_documents_fast_mode(mock_auth,mock_query_generator, mock_fast_pipeline,mock_verification):
    """Test successful retrieval in fast mode"""
    headers = {"Authorization": f"Bearer test"}
    request_payload = {"user_query": "What is AI?", "mode": "fast"}

    response = client.post("/retrieval", json=request_payload, headers=headers)

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"responses": ["doc1", "doc2"]}


def test_retrieve_documents_specific_mode(mock_auth, mock_query_generator, mock_specific_pipeline,mock_verification):
    """Test successful retrieval in specific mode"""
    headers = {"Authorization": "Bearer fake_token"}
    request_payload = {"user_query": "What is AI?", "mode": "specific"}

    response = client.post("/retrieval", json=request_payload, headers=headers)

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"responses": ["doc3", "doc4"]}


def test_retrieve_documents_unauthorized(mock_auth, mock_query_generator, mock_specific_pipeline):
    """Test when the user ID is not found in the token"""
    with patch("backend.apps.app_retrieval.verify_token", return_value={}):  # No user_id
        headers = {"Authorization": "Bearer fake_token"}
        request_payload = {"user_query": "What is AI?", "mode": "fast"}

        response = client.post("/retrieval", json=request_payload, headers=headers)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.json()["detail"] == "401: User ID not found in token."


def test_retrieve_documents_query_generation_error(mock_auth,mock_query_generator, mock_fast_pipeline,mock_verification):
    """Test when query generation fails"""
    with patch("backend.apps.app_retrieval.query_generator.generate", return_value="ERROR"):
        headers = {"Authorization": "Bearer fake_token"}
        request_payload = {"user_query": "What is AI?", "mode": "fast"}

        response = client.post("/retrieval", json=request_payload, headers=headers)

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"responses": "ERROR"}


def test_retrieve_documents_invalid_mode(mock_auth,mock_query_generator, mock_fast_pipeline,mock_verification):
    """Test when an invalid mode is provided"""
    headers = {"Authorization": "Bearer fake_token"}
    request_payload = {"user_query": "What is AI?", "mode": "invalid_mode"}

    response = client.post("/retrieval", json=request_payload, headers=headers)

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Invalid mode specified" in response.json()["detail"]


def test_retrieve_documents_internal_server_error(mock_auth,mock_query_generator, mock_fast_pipeline,mock_verification):
    """Test when an internal server error occurs"""
    with patch("backend.apps.app_retrieval.use_fast_pipeline", side_effect=Exception("Test error")):
        headers = {"Authorization": "Bearer fake_token"}
        request_payload = {"user_query": "What is AI?", "mode": "fast"}

        response = client.post("/retrieval", json=request_payload, headers=headers)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Test error" in response.json()["detail"]
