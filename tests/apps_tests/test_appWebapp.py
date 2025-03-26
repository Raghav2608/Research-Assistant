# tests/test_app_webapp.py

import pytest
from fastapi.testclient import TestClient
from backend.apps.app_webapp import app, validate_request  # Adjust the import if your module is in a different location

app.dependency_overrides[validate_request] = lambda: None

client = TestClient(app)

def test_root():
    # If the root endpoint requires a valid request (for instance, via dependency injection),
    # you might need to simulate that. For now, we'll assume the request works without extra headers.
    response = client.get("/")  # Adjust the path as defined in ENDPOINT_URLS if needed.
    # Check that the status code is 200 OK and that the response contains expected content.
    assert response.status_code == 200

def test_login_page():
    response = client.get("/login")  # Adjust path if necessary
    assert response.status_code == 200
    # Instead of checking for "login.html", check for content unique to the login page.
    assert "<title>Login Form</title>" in response.text
    assert 'id="loginForm"' in response.text

def test_register_page():
    response = client.get("/register")  # Adjust path if necessary
    assert response.status_code == 200
    # Instead of checking for "register.html", check for content unique to the registration page.
    assert "<title>Register Form</title>" in response.text
    assert 'id="registerForm"' in response.text

def test_whoami_unauthenticated():
    # This endpoint requires valid authentication. With no valid token,
    # it should return a 401 Unauthorized.
    response = client.get("/whoami")
    assert response.status_code == 401 or response.status_code == 422

def test_user_authentication_invalid():
    # Example test to simulate a failed authentication.
    payload = {
        "username": "nonexistent",
        "password": "wrongpassword"
    }
    response = client.post("/user_authentication", json=payload)
    # The exact status and message may depend on your authentication service logic.
    assert response.status_code != 200

def test_query_system_error(monkeypatch):
    # For endpoints that rely on external services (e.g., the retrieval or LLM inference endpoints),
    # you can use monkeypatch to override requests.post to simulate responses.

    def fake_post(url, json, headers):
        class DummyResponse:
            def json(self):
                return {"responses": "ERROR", "answer": "Simulated answer"}
        return DummyResponse()

    monkeypatch.setattr("requests.post", fake_post)
    # Create a dummy query request payload based on your ResearchPaperQuery model
    payload = {"user_query": "test query", "mode": "default"}
    response = client.post("/query", json=payload)
    # The endpoint should process the simulated error path and return a valid answer.
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
