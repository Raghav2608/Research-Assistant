import uvicorn
import logging

from fastapi import FastAPI, HTTPException, Body

from backend.src.backend.pydantic_models import DataIngestionQuery
from backend.src.constants import ENDPOINT_URLS
from backend.src.data_ingestion.data_pipeline import DataPipeline

app = FastAPI()
logger = logging.getLogger('uvicorn.error')

data_pipeline = DataPipeline()

@app.post(ENDPOINT_URLS['data_ingestion']['path'], description="Handles data ingestion from various sources.")
async def data_ingestion(query_request:DataIngestionQuery=Body(...)):
    try:
        logger.info(f"Calling data ingestion pipeline with queries: {query_request.user_queries}")
        all_entries = data_pipeline.run(user_queries=query_request.user_queries)
        success_message = f"Successfully called data ingestion pipeline, collected {len(all_entries)} entries."
        logger.info(success_message)
        return {"all_entries": all_entries, "message": success_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("app_data_ingestion:app", host="localhost", port=8001, reload=True)
