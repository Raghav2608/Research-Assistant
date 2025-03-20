import requests
import time

from backend.src.constants import ENDPOINT_URLS
from backend.src.RAG.utils import clean_search_query
from backend.src.backend.user_authentication.token_manager import TokenManager

if __name__ == "__main__":
    BASE_WEBAPP_URL = "http://localhost:8000"
    SYSTEM_URL = BASE_WEBAPP_URL + ENDPOINT_URLS['web_app']['additional_paths']['query']

    DATA_INGESTION_URL = f"http://localhost:8001{ENDPOINT_URLS['data_ingestion']['path']}"
    RETRIEVAL_URL = f"http://localhost:8002{ENDPOINT_URLS['retrieval']['path']}"
    LLM_INFERENCE_URL = f"http://localhost:8003{ENDPOINT_URLS['llm_inference']['path']}"
    
    user_query = "Are there any recent advancements in transformer models?"

    # Test the system endpoint
    start_time = time.perf_counter()
    response = requests.post(SYSTEM_URL, json={"user_query": user_query})
    end_time = time.perf_counter()

    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print(f"API call time: {end_time-start_time:.5f} seconds")

    # Test the retrieval endpoint
    start_time = time.perf_counter()
    response = requests.post(RETRIEVAL_URL, json={"user_query": user_query})
    end_time = time.perf_counter()

    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print(f"API call time: {end_time-start_time:.5f} seconds")


    # Test the data ingestion endpoint
    start_time = time.perf_counter()
    response = requests.post(DATA_INGESTION_URL, json={"user_queries": [clean_search_query(user_query)]})
    end_time = time.perf_counter()
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print(f"API call time: {end_time-start_time:.5f} seconds")

    # Test the LLM inference endpoint
    start_time = time.perf_counter()
    response = requests.post(LLM_INFERENCE_URL, json={"user_query": user_query, "responses": []})
    end_time = time.perf_counter()

    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print(f"API call time: {end_time-start_time:.5f} seconds")

    # Using authenticated requests:
    token = TokenManager().generate_token(user_id="random_user")["token"]
    print(f"Token: {token}")

    headers = {"Authorization": f"Bearer {token}"}

    # Test all endpoints again
    start_time = time.perf_counter()
    response = requests.post(SYSTEM_URL, json={"user_query": user_query}, headers=headers)
    end_time = time.perf_counter()
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print(f"API call time: {end_time-start_time:.5f} seconds")

    start_time = time.perf_counter()
    response = requests.post(RETRIEVAL_URL, json={"user_query": user_query}, headers=headers)
    end_time = time.perf_counter()
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print(f"API call time: {end_time-start_time:.5f} seconds")

    start_time = time.perf_counter()
    response = requests.post(DATA_INGESTION_URL, json={"user_queries": [clean_search_query(user_query)]}, headers=headers)
    end_time = time.perf_counter()

    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print(f"API call time: {end_time-start_time:.5f} seconds")

    start_time = time.perf_counter()
    response = requests.post(LLM_INFERENCE_URL, json={"user_query": user_query, "responses": []}, headers=headers)
    end_time = time.perf_counter()

    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print(f"API call time: {end_time-start_time:.5f} seconds")