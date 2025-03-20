import os
import uvicorn
import requests
import logging

from fastapi import FastAPI, HTTPException, Body, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from backend.src.backend.pydantic_models import ResearchPaperQuery
from backend.src.constants import ENDPOINT_URLS
from backend.src.backend.user_authentication.utils import validate_request

app = FastAPI(title="Research Assistant API")
logger = logging.getLogger('uvicorn.error')

templates = Jinja2Templates(directory="frontend/templates_temp")

@app.get(ENDPOINT_URLS['web_app']['path'], response_class=HTMLResponse)
async def root(request:Request):
    return templates.TemplateResponse(
                                    "index.html", 
                                    {"request": request}
                                    )

@app.get(ENDPOINT_URLS['web_app']['additional_paths']['login'], response_class=HTMLResponse)
async def login(request:Request) -> HTMLResponse:
    """
    Displays the login page.

    Args:
        request (Request): The request object containing information
                           that can be used/displayed in the template.
    """
    return templates.TemplateResponse("login.html", {"request": request})

# Handles research queries.
@app.post(ENDPOINT_URLS['web_app']['additional_paths']['query'], summary="Submit a research query", description="Returns an answer generated by the system.")
async def query_system(query_request:ResearchPaperQuery=Body(...)):
    try:
        # Call the retrieval endpoint
        logger.info("Calling retrieval endpoint")
        RETRIEVAL_URL = f"http://{ENDPOINT_URLS['retrieval']['base_url']}{ENDPOINT_URLS['retrieval']['path']}"
        retrieval_response = requests.post(url=RETRIEVAL_URL, json={"user_query": query_request.user_query})
        responses = retrieval_response.json()["responses"]
        logger.info(f"Successfully called the retrieval endpoint. Received {len(responses)} responses.")

        logger.info("Calling LLM inference endpoint")
        LLM_INFERENCE_URL = f"http://{ENDPOINT_URLS['llm_inference']['base_url']}{ENDPOINT_URLS['llm_inference']['path']}"
        llm_response = requests.post(url=LLM_INFERENCE_URL, json={"user_query": query_request.user_query, "responses": responses})
        logger.info("Successfully called the system.")
        llm_response = llm_response.json()["answer"]
        
        return {"answer": llm_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("app_webapp:app", host="localhost", port=8000, reload=True)
