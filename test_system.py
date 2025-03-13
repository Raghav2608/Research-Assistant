import requests

from src.constants import ENDPOINT_URLS

if __name__ == "__main__":

    BASE_WEBAPP_URL = "http://localhost:8000"
    SYSTEM_URL = BASE_WEBAPP_URL + ENDPOINT_URLS['web_app']['additional_paths']['query']
    payload = {"message": "Attention mechanisms in deep learning"}
    
    response = requests.post(SYSTEM_URL, json=payload)

    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")