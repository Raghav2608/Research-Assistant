"""
Script for testing the entire system by sending requests to the backend services.
- The script sends requests to the web app, data ingestion, retrieval, and LLM inference services 
  with unauthenticated and authenticated requests.
- The unauthenticated requests should always result in an error response.
- The authenticated requests should return the expected responses, assuming the token is valid.
"""
import set_path
import requests
import time

from backend.src.constants import ENDPOINT_URLS
from backend.src.backend.user_authentication.token_manager import TokenManager

def construct_endpoint(service:str, additional_path:str=None) -> str:
    """
    Construct the full URL for a given service with optional additional path.

    Args:
        service (str): The service to construct the endpoint for.
        additional_path (str): Optional additional path to append to the base URL.
    """
    base_url = f"http://{ENDPOINT_URLS[service]['base_url']}"
    path = ENDPOINT_URLS[service]["path"]
    if additional_path:
        path += ENDPOINT_URLS[service]["additional_paths"].get(additional_path, "")
    return base_url + path

def test_endpoint(endpoint:str, payload, headers=None) -> None:
    """
    Helper function for testing an endpoint with a given payload.
    """
    start_time = time.perf_counter()
    response = requests.post(endpoint, json=payload, headers=headers)
    end_time = time.perf_counter()

    print(f"Endpoint: {endpoint}")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print(f"API call time: {end_time - start_time:.5f} seconds\n")

if __name__ == "__main__":
    
    ENDPOINTS = {
                "web_app": construct_endpoint("web_app", "query"),
                "data_ingestion": construct_endpoint("data_ingestion"),
                "retrieval": construct_endpoint("retrieval"),
                "llm_inference": construct_endpoint("llm_inference")
                }
    
    user_query = "Are there any recent advancements in transformer models?"
    test_endpoint(ENDPOINTS["web_app"], {"user_query": user_query})
    test_endpoint(ENDPOINTS["retrieval"], {"user_query": user_query})
    test_endpoint(ENDPOINTS["data_ingestion"], {"user_queries": [user_query]})
    test_endpoint(ENDPOINTS["llm_inference"], {"user_query": user_query, "responses": []})

    # Test with authenticated requests 
    token = TokenManager().generate_token(user_id="random_user")["token"] # Temporary, should not be used in production like this
    print(f"Token: {token}")
    headers = {"Authorization": f"Bearer {token}"}

    test_endpoint(ENDPOINTS["web_app"], {"user_query": user_query}, headers=headers)
    test_endpoint(ENDPOINTS["retrieval"], {"user_query": user_query}, headers=headers)
    test_endpoint(ENDPOINTS["data_ingestion"], {"user_queries": [user_query]}, headers=headers)
    test_endpoint(ENDPOINTS["llm_inference"], {"user_query": user_query, "responses": []}, headers=headers)