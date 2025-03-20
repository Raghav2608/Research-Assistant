ENDPOINT_URLS = {
    "web_app": {
        "base_url": "localhost:8000",
        "app_name": "app_webapp",
        "path": "/",
        "additional_paths": {
            "query": "/query",
            "login": "/login"
        }
    },
    "data_ingestion": {
        "base_url": "localhost:8001",
        "app_name": "app_data_ingestion",
        "path": "/data_ingestion",
    },
    "retrieval": {
        "base_url": "localhost:8002",
        "app_name": "app_retrieval",
        "path": "/retrieval",
    },
    "llm_inference": {
        "base_url": "localhost:8003",
        "app_name": "app_llm_inference",
        "path": "/llm_inference",
    }
}