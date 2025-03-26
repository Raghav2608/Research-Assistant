import os
import pytest
from fastapi.testclient import TestClient
from backend.apps.app_llm_inference import app 
from unittest.mock import patch
from backend.src.backend.user_authentication.utils import validate_request

# Set up test client
client = TestClient(app)
# Mock environment variable for testing

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
     with patch("backend.apps.app_llm_inference.verify_token", return_value={"user_id":"test"}):
        yield


def test_llm_inference_success(mock_auth,mock_verification):
    """Test successful LLM inference request"""

    headers = {"Authorization": f"Bearer test"}  # Replace with a valid test token
    request_payload = {
        "user_query": "What is AI?",
        "responses": [{
                        "page_content":"AI is artificial intelligence",
                        "metadata": {
                            "title": "title",
                            "published": "published",
                            "link": "link"
                        }
                    }]
    }
    
    response = client.post("/llm_inference", json=request_payload, headers=headers)
    
    assert response.status_code == 200
    assert isinstance(response.json()["answer"],str)

def test_llm_inference_invalid_token():
    """Test LLM inference with an invalid token"""
    
    headers = {"Authorization": "Bearer invalid_token"}
    request_payload = {
        "user_query": "What is AI?",
        "responses": [{
                        "page_content":"AI is artificial intelligence",
                        "metadata": {
                            "title": "title",
                            "published": "published",
                            "link": "link"
                                    }
                    }]
    }
    
    response = client.post("/llm_inference", json=request_payload, headers=headers)
    print(response.json()["detail"])
    
    assert response.status_code == 401
    assert response.json()["detail"] == "You need to login to access this page. 401: Token is invalid."

def test_llm_inference_missing_data(mock_auth,mock_verification):
    """Test LLM inference with missing request body"""
    headers = {"Authorization": "Bearer test_token"}
    response = client.post("/llm_inference", json={}, headers=headers)
    
    assert response.status_code in [422,401]  # Unprocessable Entity due to missing fields

def test_llm_inference_internal_error(mock_auth,mock_verification):
    """Test LLM inference when an internal server error occurs"""
    with patch("backend.src.RAG.query_responder.QueryResponder.generate_answer", side_effect=Exception("Test error")):
    
        headers = {"Authorization": f"Bearer test"}
        request_payload = {
            "user_query": "What is AI?",
            "responses": [{
                            "page_content":"AI is artificial intelligence",
                            "metadata": {
                                "title": "title",
                                "published": "published",
                                "link": "link"
                            }
                        }]
        }
        
        response = client.post("/llm_inference", json=request_payload, headers=headers)
        
        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]

if __name__ == "__main__":
    pytest.main()
