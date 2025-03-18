import requests

from backend.src.constants import ENDPOINT_URLS

if __name__ == "__main__":

    BASE_WEBAPP_URL = "http://localhost:8000"
    SYSTEM_URL = BASE_WEBAPP_URL + ENDPOINT_URLS['web_app']['additional_paths']['query']
    payload = {"user_query": "Are there any recent advancements in transformer models?"}
    
    response = requests.post(SYSTEM_URL, json=payload)

    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")