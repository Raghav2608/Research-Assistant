import os
import uvicorn
import requests
import logging

from fastapi import FastAPI, HTTPException, Body
from src.backend.pydantic_models import ResearchPaperQuery
from src.constants import ENDPOINT_URLS

app = FastAPI(title="Research Assistant API")
logger = logging.getLogger('uvicorn.error')

# Root endpoint just to check if the API is running.
@app.get(ENDPOINT_URLS['web_app']['path'], summary="Root", description="Root endpoint.")
async def root():
    return {"message": "Hello from the AI Research Paper Assistant"}

# Handles research queries.
@app.post(ENDPOINT_URLS['web_app']['additional_paths']['query'], summary="Submit a research query", description="Returns an answer generated by the system.")
async def query_system(query_request:ResearchPaperQuery=Body(...)):
    try:
        # Call the retrieval endpoint
        logger.info("Calling retrieval endpoint")
        RETRIEVAL_URL = f"http://{ENDPOINT_URLS['retrieval']['base_url']}{ENDPOINT_URLS['retrieval']['path']}"
        retrieval_response = requests.post(url=RETRIEVAL_URL, json={"message": query_request.message})
        responses = retrieval_response.json()["responses"]
        logger.info(f"Successfully called the retrieval endpoint. Received {len(responses)} responses.")

        logger.info("Calling LLM inference endpoint")
        LLM_INFERENCE_URL = f"http://{ENDPOINT_URLS['llm_inference']['base_url']}{ENDPOINT_URLS['llm_inference']['path']}"
        llm_response = requests.post(url=LLM_INFERENCE_URL, json={"user_query": query_request.message, "responses": responses})
        logger.info(f"Successfully called the system.")
        llm_response = llm_response.json()["answer"]
        
        return {"answer": llm_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("app_frontend:app", host="localhost", port=8000, reload=True)
