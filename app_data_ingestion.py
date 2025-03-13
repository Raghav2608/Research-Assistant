import os
import uvicorn
import logging

from fastapi import FastAPI, HTTPException
from src.backend.pydantic_models import ResearchPaperQuery
from src.constants import ENDPOINT_URLS

app = FastAPI()
logger = logging.getLogger('uvicorn.error')

@app.post(ENDPOINT_URLS['data_ingestion']['path'], description="Handles data ingestion from various sources.")
async def data_ingestion(query_request:ResearchPaperQuery):
    try:
        answer = "Successfully called data ingestion pipeline"
        logger.info(answer)
        # answer = answer_with_rag.invoke(query_request.message)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("app_data_ingestion:app", host="localhost", port=8001, reload=True)
