import pytest
from fastapi.testclient import TestClient
from fastapi import status
from backend.apps.app_webapp import app

# Import the actual dependency functions used in the app.
from backend.src.backend.user_authentication.utils import validate_request
from backend.src.backend.user_authentication.token_manager import verify_token

"""
Integration tests for the app_webapp FastAPI application.
These tests run end-to-end using FastAPI's TestClient and override authentication
and external calls so that endpoints can be tested without requiring real credentials.
They verify that:
  - The root, login, and register endpoints return 200 OK and contain expected HTML.
  - The whoami endpoint returns the dummy username or fails with 401 if token is missing.
  - The user_authentication endpoint returns a dummy token.
  - The query_system endpoint handles both successful responses and error conditions.
  
Additional tests simulate:
  - The "ERROR" branch in query_system (retrieval returning "ERROR").
  - An exception in query_system, ensuring that a 500 error is returned.
"""

# Define dummy dependency functions.
def dummy_validate_request():
    return

def dummy_verify_token(request):
    return {"user_id": "testuser"}

def dummy_handle_rate_limiting(request, username):
    return (False, "")

def dummy_handle_authentication(username, password, request, confirm_password=None):
    return (200, "Authenticated")

def dummy_get_token_response(username, status_code, message):
    return {"token": "dummy_token"}

# Override dependencies using FastAPI's dependency_overrides.
app.dependency_overrides[validate_request] = dummy_validate_request

# The whoami endpoint calls verify_token directly; override it in the module.
import backend.apps.app_webapp as webapp
webapp.verify_token = dummy_verify_token

# Override user_authentication_service methods.
from backend.apps.app_webapp import user_authentication_service
user_authentication_service.handle_rate_limiting = dummy_handle_rate_limiting
user_authentication_service.handle_authentication = dummy_handle_authentication
user_authentication_service.get_token_response = dummy_get_token_response

# Override template rendering to return dummy HTML.
def dummy_template_response(template_name: str, context: dict):
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=f"Dummy content for {template_name}", status_code=200)
webapp.templates.TemplateResponse = dummy_template_response

# Override external requests by monkeypatching requests.post.
import requests
def fake_requests_post(url, json, headers):
    class DummyResponse:
        def __init__(self, json_data, status_code=200):
            self._json = json_data
            self.status_code = status_code
        def json(self):
            return self._json
    if "retrieval" in url:
        # Simulate normal retrieval.
        return DummyResponse({"responses": ["dummy paper 1", "dummy paper 2"]})
    elif "llm_inference" in url:
        # Simulate LLM inference returning an answer.
        return DummyResponse({"answer": "This is a dummy answer."})
    return DummyResponse({})

@pytest.fixture(autouse=True)
def override_requests_post(monkeypatch):
    monkeypatch.setattr(requests, "post", fake_requests_post)

client = TestClient(app)

# Basic endpoint tests.
def test_root_endpoint_integration():
    response = client.get("/")
    assert response.status_code == 200
    assert "dummy content for chat.html" in response.text.lower()

def test_login_endpoint_integration():
    response = client.get("/login")
    assert response.status_code == 200
    assert "dummy content for login.html" in response.text.lower()

def test_register_endpoint_integration():
    response = client.get("/register")
    assert response.status_code == 200
    assert "dummy content for register.html" in response.text.lower()

def test_whoami_endpoint_integration():
    response = client.get("/whoami", headers={"Authorization": "dummy"})
    assert response.status_code == 200
    data = response.json()
    assert data.get("username") == "testuser"

def test_user_authentication_endpoint_integration():
    payload = {
        "username": "testuser",
        "password": "dummy_password",
        "confirm_password": "dummy_password"
    }
    response = client.post("/user_authentication", json=payload)
    assert response.status_code in [200, 201]
    data = response.json()
    assert "token" in data

# Test query_system with normal retrieval (responses not "ERROR").
def test_query_system_endpoint_integration_success():
    payload = {"user_query": "What is AI?", "mode": "default"}
    response = client.post("/query", json=payload, headers={"Authorization": "dummy"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "papers" in data
    # Expect papers list to contain dummy responses.
    assert isinstance(data["papers"], list)
    assert len(data["papers"]) > 0

# Test query_system when retrieval returns "ERROR".
def test_query_system_endpoint_integration_error_branch(monkeypatch):
    # Override fake_requests_post to simulate retrieval returning "ERROR".
    def fake_requests_post_error(url, json, headers):
        class DummyResponse:
            def __init__(self, json_data, status_code=200):
                self._json = json_data
                self.status_code = status_code
            def json(self):
                return self._json
        if "retrieval" in url:
            return DummyResponse({"responses": "ERROR"})
        elif "llm_inference" in url:
            return DummyResponse({"answer": "Dummy answer for error branch."})
        return DummyResponse({})
    
    monkeypatch.setattr(requests, "post", fake_requests_post_error)
    
    payload = {"user_query": "What is AI?", "mode": "default"}
    response = client.post("/query", json=payload, headers={"Authorization": "dummy"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    # In the error branch, papers should be an empty list.
    assert data["papers"] == []

# Test query_system exception handling: simulate an exception.
def test_query_system_endpoint_integration_exception(monkeypatch):
    def fake_requests_post_exception(url, json, headers):
        raise Exception("Simulated exception")
    monkeypatch.setattr(requests, "post", fake_requests_post_exception)
    
    payload = {"user_query": "What is AI?", "mode": "default"}
    response = client.post("/query", json=payload, headers={"Authorization": "dummy"})
    # Expect a 500 error when an exception is raised.
    assert response.status_code == 500
