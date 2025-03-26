import os
import pytest
from fastapi.testclient import TestClient
from backend.apps.app_llm_inference import app 
from unittest.mock import patch
from dotenv import load_dotenv

# Set up test client
client = TestClient(app)
# Mock environment variable for testing
os.environ["OPENAI_API_KEY"] = "test_api_key"
load_dotenv()

token = os.getenv('TEST_TOKEN')

def test_llm_inference_success():
    """Test successful LLM inference request"""

    headers = {"Authorization": f"Bearer {token}"}  # Replace with a valid test token
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

def test_llm_inference_missing_data():
    """Test LLM inference with missing request body"""
    headers = {"Authorization": "Bearer test_token"}
    response = client.post("/llm_inference", json={}, headers=headers)
    
    assert response.status_code in [422,401]  # Unprocessable Entity due to missing fields

def test_llm_inference_internal_error():
    """Test LLM inference when an internal server error occurs"""
    with patch("backend.src.RAG.query_responder.QueryResponder.generate_answer", side_effect=Exception("Test error")):
    
        headers = {"Authorization": f"Bearer {token}"}
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
