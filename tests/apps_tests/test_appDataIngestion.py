import pytest
from fastapi.testclient import TestClient
from fastapi import status
from backend.apps.app_data_ingestion import app, ENDPOINT_URLS

from backend.src.backend.user_authentication.utils import validate_request

"""
Integration tests for the data ingestion endpoint.
These tests run end-to-end using FastAPI's TestClient and override authentication
and the data pipeline so that the endpoint can be tested without requiring the actual pipeline.
They verify that:
  - A successful data ingestion call returns a 200 OK with the expected JSON response.
  - An exception in the data pipeline results in a 500 error.
"""
# Define a dummy validate_request function.
def dummy_validate_request():
    return

# Override the dependency.
app.dependency_overrides[validate_request] = dummy_validate_request

# Import the module to access the DataPipeline instance.
import backend.apps.app_data_ingestion as ingestion_app

# Define dummy functions to simulate the DataPipeline.run() behavior.
def dummy_run_success(user_queries):
    # Return a dummy list of entries.
    return ["entry1", "entry2", "entry3"]

def dummy_run_error(user_queries):
    # Simulate an exception in the data pipeline.
    raise Exception("Simulated pipeline failure")

client = TestClient(app)

def test_data_ingestion_success_integration(monkeypatch):
    # Override the run method to simulate a successful ingestion.
    monkeypatch.setattr(ingestion_app.data_pipeline, "run", dummy_run_success)
    
    payload = {"user_queries": ["query1", "query2"]}
    response = client.post(ENDPOINT_URLS['data_ingestion']['path'], json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "all_entries" in data
    assert "message" in data
    # Verify that the dummy run returned the expected list.
    assert data["all_entries"] == ["entry1", "entry2", "entry3"]
    # Check that the success message reflects the number of entries.
    assert "3 entries" in data["message"]

def test_data_ingestion_exception_integration(monkeypatch):
    # Override the run method to simulate an error.
    monkeypatch.setattr(ingestion_app.data_pipeline, "run", dummy_run_error)
    
    payload = {"user_queries": ["query1", "query2"]}
    response = client.post(ENDPOINT_URLS['data_ingestion']['path'], json=payload)
    # Expect a 500 Internal Server Error.
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
