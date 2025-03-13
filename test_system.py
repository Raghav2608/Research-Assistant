import requests

if __name__ == "__main__":

    URL = "http://localhost:8000/rag_pipeline"
    payload = {"message": "Attention mechanisms in deep learning"}
    
    response = requests.post(URL, json=payload)

    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")