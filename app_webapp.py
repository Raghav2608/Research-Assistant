import os
import uvicorn
import requests

from fastapi import FastAPI, HTTPException
from src.backend.pydantic_models import ResearchPaperQuery
from src.constants import ENDPOINT_URLS

app = FastAPI(
            title="Research Assistant API"
            )

# Root endpoint just to check if the API is running.
@app.get(ENDPOINT_URLS['web_app']['path'], summary="Root", description="Root endpoint.")
async def root():
    return {"message": "Hello from the AI Research Paper Assistant"}

# Handles research queries.
@app.post(ENDPOINT_URLS['web_app']['additional_paths']['query'], summary="Submit a research query", description="Returns an answer generated by the system.")
async def query_system(query_request:ResearchPaperQuery):
    try:
        
        # Call the data ingestion system

        DATA_INGESTION_URL = f"http://{ENDPOINT_URLS['data_ingestion']['base_url']}{ENDPOINT_URLS['data_ingestion']['path']}"
        requests.post(url=DATA_INGESTION_URL, json={"message": query_request.message})

        answer = "Successfully called the system"
        # answer = answer_with_rag.invoke(query_request.message)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("app_frontend:app", host="localhost", port=8000, reload=True)
