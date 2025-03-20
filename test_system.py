import requests
import time

from backend.src.constants import ENDPOINT_URLS

if __name__ == "__main__":
    BASE_WEBAPP_URL = "http://localhost:8000"
    SYSTEM_URL = BASE_WEBAPP_URL + ENDPOINT_URLS['web_app']['additional_paths']['query']
        
    while True:
        user_input = input("User: ")
        
        if user_input.lower() in ["exit", "quit", "end"]:
            break

        payload = {"user_query":user_input}

        start_time = time.perf_counter()
        response = requests.post(SYSTEM_URL, json=payload)
        end_time = time.perf_counter()

        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
        print(f"API call time: {end_time-start_time:.5f} seconds")