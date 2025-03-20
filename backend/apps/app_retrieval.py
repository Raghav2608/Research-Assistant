import os
import uvicorn
import logging
import requests

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from backend.src.RAG.retrieval_engine import RetrievalEngine
from backend.src.RAG.query_generator import ResearchQueryGenerator
from backend.src.RAG.utils import clean_search_query
from backend.src.backend.pydantic_models import ResearchPaperQuery
from backend.src.constants import ENDPOINT_URLS

load_dotenv()

app = FastAPI(title="Research Assistant API")
logger = logging.getLogger('uvicorn.error')

if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("openai key not set in environment.")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise EnvironmentError("openai key not set in environment.")

query_generator = ResearchQueryGenerator(openai_api_key=OPENAI_API_KEY,session_id="foo")
retrieval_engine = RetrievalEngine(openai_api_key=OPENAI_API_KEY)

DATA_INGESTION_URL = f"http://{ENDPOINT_URLS['data_ingestion']['base_url']}{ENDPOINT_URLS['data_ingestion']['path']}"

@app.post(ENDPOINT_URLS['retrieval']['path'], description="Retrieves documents based on the user query.")
async def retrieve_documents(query_request:ResearchPaperQuery):
    try:
        logger.info("Successfully called retrieval pipeline endpoint")

        user_input = query_request.user_query

        # Generate additional queries
        additional_queries = query_generator.generate(user_input)
        print(additional_queries)

        if additional_queries == "ERROR":
            return {"responses":"ERROR"}

        # Attempt to retrieve documents the existing database
        logger.info("Attempting to retrieve documents from the existing database")
        responses = retrieval_engine.retrieve(user_queries=additional_queries)

        # Attempt to retrieve documents via data ingestion
        if responses:
            logger.info("Relevant documents found in the existing database")
        else:
            logger.info("No relevant documents found, searching for more documents")

            data_ingestion_result = requests.post(url=DATA_INGESTION_URL, json={"user_queries": additional_queries})
            all_entries = data_ingestion_result.json()["all_entries"]

            logger.info(f"Total number of retrieved entries from data ingestion: {len(all_entries)}")
            
            if len(all_entries) == 0:
                logger.info("No entries could be found for this query, please try to rephrase your query.")
            else:
                docs = retrieval_engine.convert_entries_to_docs(entries=all_entries)
                retrieval_engine.split_and_add_documents(docs=docs) # Add documents to ChromaDB (save)

            # Attempt to retrieve the documents again (should be successful this time)
            responses = retrieval_engine.retrieve(user_queries=additional_queries)
        
        logger.info(f"Responses: {responses}")
        return {"responses": responses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("app_retrieval:app", host="localhost", port=8002, reload=True)
